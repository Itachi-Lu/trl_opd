from __future__ import annotations

import copy
import csv
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trl import ModelConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.chat_template_utils import qwen3_training_chat_template
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer
from trl.models import prepare_deepspeed, prepare_fsdp

from opd_with_eval import (
    OPDScriptArguments,
    _load_dataset_splits,
    _maybe_wait_for_debugger,
    _resolve_dtype,
    _resolve_processing_class_name,
    accuracy_reward_with_fallback,
)


BUCKETS = ("all", "len_ge_8k", "len_lt_8k")
TOPKS = (8, 16, 32)
METRIC_NAMES = (
    "rho",
    "da_weight",
    "da_normalized_weight",
    "student_entropy",
    "teacher_entropy",
    "entropy_gap",
    "student_sample_logprob",
    "student_forward_sample_logprob",
    "teacher_sample_logprob",
    "teacher_student_sample_logprob_gap",
    "teacher_student_forward_logprob_gap",
    "top1_agreement",
    "overlap_ratio_top8",
    "student_overlap_mass_top8",
    "teacher_overlap_mass_top8",
    "overlap_ratio_top16",
    "student_overlap_mass_top16",
    "teacher_overlap_mass_top16",
    "overlap_ratio_top32",
    "student_overlap_mass_top32",
    "teacher_overlap_mass_top32",
)


@dataclass
class OPDMetricsScriptArguments(OPDScriptArguments):
    metrics_num_rollouts: int = field(
        default=128,
        metadata={"help": "Exact global number of generated trajectories to include in the metric curves."},
    )
    metrics_output_root: str = field(
        default="./opd_metrics",
        metadata={"help": "Root directory. Results are written under <root>/<student_model_basename>/."},
    )
    metrics_length_threshold: int = field(
        default=8192,
        metadata={"help": "Completion-token length threshold for long/short trajectory buckets."},
    )
    metrics_forward_batch_size: int | None = field(
        default=None,
        metadata={"help": "Optional local micro-batch size for student/teacher metric forwards."},
    )
    metrics_num_sample_rollouts: int = field(
        default=8,
        metadata={"help": "Maximum number of local rollout samples to write per rank for human inspection."},
    )
    metrics_sample_token_limit: int = field(
        default=0,
        metadata={
            "help": "Maximum number of token positions to render in sample HTML previews. "
            "Use 0 for the default compact preview. CSV files always keep all valid tokens."
        },
    )
    metrics_sample_topk: int = field(
        default=8,
        metadata={"help": "Top-k size for teacher/student token probability comparison in sample HTML files."},
    )


class PositionMetricAccumulator:
    def __init__(self, max_positions: int, length_threshold: int, device: torch.device):
        self.max_positions = max_positions
        self.length_threshold = length_threshold
        self.device = device
        self.counts = {bucket: torch.zeros(max_positions, device=device, dtype=torch.float32) for bucket in BUCKETS}
        self.sequence_counts = {bucket: torch.zeros((), device=device, dtype=torch.float32) for bucket in BUCKETS}
        self.sums = {
            bucket: {
                name: torch.zeros(max_positions, device=device, dtype=torch.float32) for name in METRIC_NAMES
            }
            for bucket in BUCKETS
        }

    def update(self, metrics: dict[str, torch.Tensor], mask: torch.Tensor, completion_lengths: torch.Tensor) -> None:
        if mask.numel() == 0:
            return

        token_count = min(mask.size(1), self.max_positions)
        mask = mask[:, :token_count].bool()
        completion_lengths = completion_lengths.to(device=self.device)
        selectors = {
            "all": torch.ones_like(completion_lengths, dtype=torch.bool, device=self.device),
            "len_ge_8k": completion_lengths >= self.length_threshold,
            "len_lt_8k": completion_lengths < self.length_threshold,
        }

        for bucket, selector in selectors.items():
            if selector.numel() == 0:
                continue
            self.sequence_counts[bucket] += selector.float().sum()
            token_mask = mask & selector.unsqueeze(1)
            token_mask_float = token_mask.float()
            self.counts[bucket][:token_count] += token_mask_float.sum(dim=0)

            for name, values in metrics.items():
                values = values[:, :token_count].to(device=self.device, dtype=torch.float32)
                self.sums[bucket][name][:token_count] += (values * token_mask_float).sum(dim=0)

    def reduce(self, accelerator) -> dict[str, dict[str, torch.Tensor]]:
        reduced: dict[str, dict[str, torch.Tensor]] = {}
        for bucket in BUCKETS:
            bucket_result: dict[str, torch.Tensor] = {
                "count": accelerator.reduce(self.counts[bucket], reduction="sum"),
                "sequence_count": accelerator.reduce(self.sequence_counts[bucket], reduction="sum"),
            }
            for name in METRIC_NAMES:
                bucket_result[name] = accelerator.reduce(self.sums[bucket][name], reduction="sum")
            reduced[bucket] = bucket_result
        return reduced


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _student_output_dir(root: str, student_model: str) -> Path:
    return Path(root) / Path(student_model.rstrip("/")).name


def _clone_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(row))


def _build_local_inputs(
    train_dataset,
    batch_index: int,
    local_rollouts_per_batch: int,
    global_rollouts_per_batch: int,
    group_size: int,
    process_index: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    batch_start = batch_index * global_rollouts_per_batch
    local_start = batch_start + process_index * local_rollouts_per_batch
    global_ids = torch.arange(local_start, local_start + local_rollouts_per_batch, dtype=torch.long)

    inputs = []
    dataset_len = len(train_dataset)
    for global_id in global_ids.tolist():
        prompt_group_id = global_id // group_size
        inputs.append(_clone_dataset_row(train_dataset[prompt_group_id % dataset_len]))
    return inputs, global_ids


def _slice_tensor_batch(inputs: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    output = {}
    batch_size = keep.numel()
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == batch_size:
            output[key] = value[keep.to(device=value.device)]
        else:
            output[key] = value
    return output


def _full_vocab_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    vocab_size = log_probs.size(-1)
    if vocab_size > 1:
        return entropy / math.log(vocab_size)
    return torch.zeros_like(entropy)


def _compute_metric_tensors(
    trainer: MiniLLMTrainer,
    inputs: dict[str, torch.Tensor],
    start: int,
    end: int,
    sample_topk: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    prompt_ids = inputs["prompt_ids"][start:end]
    completion_ids = inputs["completion_ids"][start:end]
    prompt_mask = inputs["prompt_mask"][start:end]
    completion_mask = inputs["completion_mask"][start:end]
    old_per_token_logps = inputs.get("old_per_token_logps")
    if old_per_token_logps is None:
        raise RuntimeError("Metrics require `old_per_token_logps`; keep `da_opd_weighting` enabled.")
    old_per_token_logps = old_per_token_logps[start:end]

    chunk_inputs = {
        "completion_mask": completion_mask,
    }
    if "tool_mask" in inputs:
        chunk_inputs["tool_mask"] = inputs["tool_mask"][start:end]
    metric_mask = trainer._get_loss_mask(chunk_inputs)
    completion_lengths = completion_mask.sum(dim=1)

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    prompt_lengths = prompt_ids.shape[1]
    shifted_labels = input_ids[:, prompt_lengths:]

    with torch.no_grad():
        student_outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        teacher_outputs = trainer.teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :] / trainer.kd_temperature
    teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :] / trainer.kd_temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    student_forward_on_labels = student_log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    teacher_on_labels = teacher_log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

    da_weights, rho = trainer._compute_da_opd_token_weights(
        teacher_token_logprobs=teacher_on_labels,
        student_token_logprobs=old_per_token_logps,
        mask=metric_mask,
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
    )
    weight_denominator = (da_weights * metric_mask.float()).sum(dim=1, keepdim=True).clamp_min(1e-12)
    normalized_weights = torch.where(metric_mask.bool(), da_weights / weight_denominator, torch.zeros_like(da_weights))

    student_entropy = _full_vocab_entropy(student_log_probs)
    teacher_entropy = _full_vocab_entropy(teacher_log_probs)

    metrics: dict[str, torch.Tensor] = {
        "rho": rho,
        "da_weight": da_weights,
        "da_normalized_weight": normalized_weights,
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "entropy_gap": torch.abs(student_entropy - teacher_entropy),
        "student_sample_logprob": old_per_token_logps,
        "student_forward_sample_logprob": student_forward_on_labels,
        "teacher_sample_logprob": teacher_on_labels,
        "teacher_student_sample_logprob_gap": teacher_on_labels - old_per_token_logps,
        "teacher_student_forward_logprob_gap": teacher_on_labels - student_forward_on_labels,
    }

    max_topk = min(32, student_log_probs.size(-1), teacher_log_probs.size(-1))
    student_topk_log_probs, student_topk_indices = student_log_probs.detach().topk(max_topk, dim=-1)
    teacher_topk_log_probs, teacher_topk_indices = teacher_log_probs.detach().topk(max_topk, dim=-1)
    metrics["top1_agreement"] = (student_topk_indices[..., 0] == teacher_topk_indices[..., 0]).float()

    for k in TOPKS:
        current_k = min(k, max_topk)
        student_indices = student_topk_indices[..., :current_k]
        teacher_indices = teacher_topk_indices[..., :current_k]
        student_log_probs_k = student_topk_log_probs[..., :current_k]
        teacher_log_probs_k = teacher_topk_log_probs[..., :current_k]

        student_overlap_mask = (student_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)).any(dim=-1)
        teacher_overlap_mask = (teacher_indices.unsqueeze(-1) == student_indices.unsqueeze(-2)).any(dim=-1)
        metrics[f"overlap_ratio_top{k}"] = student_overlap_mask.float().sum(dim=-1) / current_k
        metrics[f"student_overlap_mass_top{k}"] = (student_log_probs_k.exp() * student_overlap_mask).sum(dim=-1)
        metrics[f"teacher_overlap_mass_top{k}"] = (teacher_log_probs_k.exp() * teacher_overlap_mask).sum(dim=-1)

    topk_data = None
    if sample_topk > 0:
        sample_topk = min(sample_topk, max_topk)
        topk_data = {
            "student_topk_indices": student_topk_indices[..., :sample_topk],
            "student_topk_log_probs": student_topk_log_probs[..., :sample_topk],
            "teacher_topk_indices": teacher_topk_indices[..., :sample_topk],
            "teacher_topk_log_probs": teacher_topk_log_probs[..., :sample_topk],
        }

    return metrics, metric_mask, completion_lengths, topk_data


def _write_bucket_outputs(
    bucket_dir: Path,
    bucket: str,
    reduced: dict[str, torch.Tensor],
    config_summary: dict[str, Any],
) -> None:
    bucket_dir.mkdir(parents=True, exist_ok=True)

    counts = reduced["count"].detach().cpu()
    valid_positions = counts > 0
    positions = torch.arange(1, counts.numel() + 1)[valid_positions]

    metric_means = {}
    for name in METRIC_NAMES:
        sums = reduced[name].detach().cpu()
        means = torch.full_like(sums, float("nan"))
        means[valid_positions] = sums[valid_positions] / counts[valid_positions]
        metric_means[name] = means

    csv_path = bucket_dir / "per_position_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "count", *METRIC_NAMES])
        for pos in positions.tolist():
            idx = pos - 1
            writer.writerow(
                [
                    pos,
                    int(counts[idx].item()),
                    *[float(metric_means[name][idx].item()) for name in METRIC_NAMES],
                ]
            )

    summary = {
        **config_summary,
        "bucket": bucket,
        "rollout_count": int(reduced["sequence_count"].detach().cpu().item()),
        "positions_with_samples": int(valid_positions.sum().item()),
    }
    with (bucket_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    if positions.numel() == 0:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = positions.numpy()
    sample_count = counts[valid_positions].numpy()
    plot_series = {"sample_count": sample_count}
    plot_series.update({name: metric_means[name][valid_positions].numpy() for name in METRIC_NAMES})

    for name, y in plot_series.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, linewidth=1.2)
        ax.set_xlabel("per_token position")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(bucket_dir / f"{_sanitize_filename(name)}.png", dpi=160)
        plt.close(fig)


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _visible_text(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _bucket_names_for_length(length: int, length_threshold: int) -> list[str]:
    return ["all", "len_ge_8k" if length >= length_threshold else "len_lt_8k"]


def _select_key_indices(metric_values: dict[str, list[float]], valid_count: int) -> list[int]:
    if valid_count <= 0:
        return []

    candidates = [0, valid_count - 1]

    def add_argmax(name: str, absolute: bool = False) -> None:
        values = metric_values.get(name)
        if not values:
            return
        if absolute:
            idx = max(range(len(values)), key=lambda i: abs(values[i]))
        else:
            idx = max(range(len(values)), key=lambda i: values[i])
        candidates.append(idx)

    def add_argmin(name: str) -> None:
        values = metric_values.get(name)
        if values:
            candidates.append(min(range(len(values)), key=lambda i: values[i]))

    add_argmax("da_normalized_weight")
    add_argmax("teacher_student_sample_logprob_gap")
    add_argmin("overlap_ratio_top8")
    add_argmax("entropy_gap")
    add_argmax("rho", absolute=True)

    unique = []
    seen = set()
    for idx in candidates:
        if 0 <= idx < valid_count and idx not in seen:
            unique.append(idx)
            seen.add(idx)
    return unique


def _select_preview_indices(valid_count: int, key_indices: list[int], token_limit: int) -> list[int]:
    if valid_count <= 0:
        return []

    selected: set[int] = set()
    if token_limit > 0:
        selected.update(range(min(token_limit, valid_count)))
    else:
        selected.update(range(min(128, valid_count)))
        selected.update(range(max(0, valid_count - 64), valid_count))

    for key_idx in key_indices:
        selected.update(range(max(0, key_idx - 8), min(valid_count, key_idx + 9)))

    return sorted(selected)


def _decode_topk_row(tokenizer, token_ids: list[int], log_probs: list[float]) -> str:
    parts = []
    for token_id, log_prob in zip(token_ids, log_probs, strict=True):
        token = _visible_text(tokenizer.decode([token_id], skip_special_tokens=False))
        prob = math.exp(float(log_prob))
        parts.append(
            f"<tr><td>{token_id}</td><td><code>{html.escape(token)}</code></td><td>{prob:.4g}</td>"
            f"<td>{float(log_prob):.4g}</td></tr>"
        )
    return "\n".join(parts)


def _write_sample_files(
    output_dir: Path,
    trainer: MiniLLMTrainer,
    generated: dict[str, torch.Tensor],
    chunk_start: int,
    local_row: int,
    global_id: int,
    group_size: int,
    metrics: dict[str, torch.Tensor],
    metric_mask: torch.Tensor,
    completion_length: int,
    topk_data: dict[str, torch.Tensor] | None,
    script_args: OPDMetricsScriptArguments,
    config_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    tokenizer = trainer.processing_class.tokenizer if hasattr(trainer.processing_class, "tokenizer") else trainer.processing_class
    batch_row = chunk_start + local_row
    valid_indices_device = metric_mask[local_row].detach().bool().nonzero(as_tuple=True)[0]
    valid_indices = valid_indices_device.cpu()
    valid_count = int(valid_indices.numel())

    prompt_ids = generated["prompt_ids"][batch_row].detach().cpu()
    prompt_mask = generated["prompt_mask"][batch_row].detach().bool().cpu()
    completion_ids = generated["completion_ids"][batch_row].detach().cpu()
    completion_mask = generated["completion_mask"][batch_row].detach().bool().cpu()

    prompt_token_ids = prompt_ids[prompt_mask].tolist()
    completion_token_ids_for_text = completion_ids[completion_mask].tolist()
    metric_token_ids = completion_ids[valid_indices].tolist()

    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    completion_text = tokenizer.decode(completion_token_ids_for_text, skip_special_tokens=False)
    token_strings = tokenizer.convert_ids_to_tokens(metric_token_ids)
    decoded_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in metric_token_ids]

    metric_values = {
        name: metrics[name][local_row][valid_indices_device].detach().float().cpu().tolist() for name in METRIC_NAMES
    }
    positions = (valid_indices + 1).tolist()
    key_indices = _select_key_indices(metric_values, valid_count)
    preview_indices = _select_preview_indices(valid_count, key_indices, script_args.metrics_sample_token_limit)

    mean_weight = (
        sum(metric_values["da_weight"]) / len(metric_values["da_weight"]) if metric_values["da_weight"] else float("nan")
    )
    max_rho = max(metric_values["rho"], key=abs) if metric_values["rho"] else float("nan")
    lowest_overlap_idx = (
        min(range(valid_count), key=lambda i: metric_values["overlap_ratio_top8"][i]) if valid_count else None
    )
    lowest_overlap_position = positions[lowest_overlap_idx] if lowest_overlap_idx is not None else None

    samples_root = output_dir / "samples"
    bucket_names = _bucket_names_for_length(completion_length, script_args.metrics_length_threshold)
    prompt_group_id = global_id // group_size
    metadata_records: list[dict[str, Any]] = []

    csv_rows = []
    for i in range(valid_count):
        csv_rows.append(
            [
                positions[i],
                metric_token_ids[i],
                repr(token_strings[i]),
                decoded_tokens[i],
                *[_format_float(float(metric_values[name][i])) for name in METRIC_NAMES],
            ]
        )

    max_norm_weight = max(metric_values["da_normalized_weight"]) if metric_values["da_normalized_weight"] else 0.0
    heatmap_parts = []
    for i in preview_indices:
        token = _visible_text(decoded_tokens[i])
        value = float(metric_values["da_normalized_weight"][i])
        alpha = 0.08 if max_norm_weight <= 0 else min(0.95, 0.12 + 0.83 * value / max_norm_weight)
        heatmap_parts.append(
            "<span class=\"tok\" "
            f"style=\"background: rgba(240, 125, 67, {alpha:.3f});\" "
            f"title=\"pos={positions[i]} token_id={metric_token_ids[i]} norm_weight={value:.6g}\">"
            f"{html.escape(_visible_text(token))}</span>"
        )

    table_rows = []
    for i in preview_indices:
        table_rows.append(
            "<tr>"
            f"<td>{positions[i]}</td>"
            f"<td>{metric_token_ids[i]}</td>"
            f"<td><code>{html.escape(repr(token_strings[i]))}</code></td>"
            f"<td><code>{html.escape(_visible_text(decoded_tokens[i]))}</code></td>"
            + "".join(f"<td>{_format_float(float(metric_values[name][i]))}</td>" for name in METRIC_NAMES)
            + "</tr>"
        )

    key_sections = []
    for i in key_indices:
        topk_html = ""
        if topk_data is not None:
            position_idx = int(valid_indices[i].item())
            student_ids = topk_data["student_topk_indices"][local_row, position_idx].detach().cpu().tolist()
            student_logps = topk_data["student_topk_log_probs"][local_row, position_idx].detach().float().cpu().tolist()
            teacher_ids = topk_data["teacher_topk_indices"][local_row, position_idx].detach().cpu().tolist()
            teacher_logps = topk_data["teacher_topk_log_probs"][local_row, position_idx].detach().float().cpu().tolist()
            topk_html = f"""
            <div class="topk-grid">
              <div>
                <h4>Student top-k</h4>
                <table><thead><tr><th>id</th><th>token</th><th>prob</th><th>logp</th></tr></thead>
                <tbody>{_decode_topk_row(tokenizer, student_ids, student_logps)}</tbody></table>
              </div>
              <div>
                <h4>Teacher top-k</h4>
                <table><thead><tr><th>id</th><th>token</th><th>prob</th><th>logp</th></tr></thead>
                <tbody>{_decode_topk_row(tokenizer, teacher_ids, teacher_logps)}</tbody></table>
              </div>
            </div>
            """

        key_sections.append(
            f"""
            <section class="key-pos">
              <h3>Position {positions[i]}: <code>{html.escape(_visible_text(decoded_tokens[i]))}</code></h3>
              <p>
                rho={_format_float(float(metric_values["rho"][i]))},
                weight={_format_float(float(metric_values["da_weight"][i]))},
                norm_weight={_format_float(float(metric_values["da_normalized_weight"][i]))},
                gap={_format_float(float(metric_values["teacher_student_sample_logprob_gap"][i]))},
                overlap8={_format_float(float(metric_values["overlap_ratio_top8"][i]))}
              </p>
              {topk_html}
            </section>
            """
        )

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }
    pre { white-space: pre-wrap; background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 4px 6px; vertical-align: top; }
    th { background: #f3f4f6; position: sticky; top: 0; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }
    .card { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }
    .heatmap { line-height: 2.0; word-break: break-word; border: 1px solid #e5e7eb; padding: 10px; border-radius: 6px; }
    .tok { border-radius: 3px; padding: 2px 3px; margin: 1px; display: inline-block; }
    .topk-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .key-pos { border-top: 1px solid #e5e7eb; margin-top: 16px; padding-top: 12px; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    """

    for bucket in bucket_names:
        sample_dir = samples_root / bucket
        sample_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"sample_{global_id}"
        html_path = sample_dir / f"{base_name}.html"
        csv_path = sample_dir / f"{base_name}.tokens.csv"
        metadata_path = sample_dir / f"{base_name}.metadata.json"

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["position", "token_id", "token_repr", "decoded_token", *METRIC_NAMES])
            writer.writerows(csv_rows)

        preview_note = (
            f"Showing {len(preview_indices)} of {valid_count} metric tokens. Full token table is in "
            f"{html.escape(csv_path.name)}."
        )
        html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rollout sample {global_id}</title>
  <style>{css}</style>
</head>
<body>
  <h1>Rollout sample {global_id}</h1>
  <div class="summary">
    <div class="card"><b>Bucket</b><br>{bucket}</div>
    <div class="card"><b>Prompt group</b><br>{prompt_group_id}</div>
    <div class="card"><b>Completion length</b><br>{completion_length}</div>
    <div class="card"><b>Mean DA weight</b><br>{_format_float(mean_weight)}</div>
    <div class="card"><b>Max |rho| value</b><br>{_format_float(float(max_rho))}</div>
    <div class="card"><b>Lowest overlap8 position</b><br>{lowest_overlap_position}</div>
  </div>

  <h2>Prompt</h2>
  <pre>{html.escape(prompt_text)}</pre>

  <h2>Completion</h2>
  <pre>{html.escape(completion_text)}</pre>

  <h2>Token Heatmap</h2>
  <p>{preview_note}</p>
  <div class="heatmap">{"".join(heatmap_parts)}</div>

  <h2>Key Positions</h2>
  {"".join(key_sections)}

  <h2>Token Metrics Preview</h2>
  <details open>
    <summary>{preview_note}</summary>
    <table>
      <thead>
        <tr><th>pos</th><th>id</th><th>token repr</th><th>decoded</th>
        {"".join(f"<th>{html.escape(name)}</th>" for name in METRIC_NAMES)}</tr>
      </thead>
      <tbody>{"".join(table_rows)}</tbody>
    </table>
  </details>
</body>
</html>
"""
        html_path.write_text(html_doc, encoding="utf-8")

        metadata = {
            **config_summary,
            "bucket": bucket,
            "global_id": global_id,
            "prompt_group_id": prompt_group_id,
            "completion_length": completion_length,
            "valid_metric_tokens": valid_count,
            "html_file": str(html_path.relative_to(samples_root)),
            "csv_file": str(csv_path.relative_to(samples_root)),
            "mean_da_weight": mean_weight,
            "max_abs_rho": float(max_rho),
            "lowest_overlap8_position": lowest_overlap_position,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        metadata_records.append(metadata)

    return metadata_records


def _write_samples_index(output_dir: Path) -> None:
    samples_root = output_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    records = []
    for path in samples_root.glob("*/*.metadata.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue

    records.sort(key=lambda row: (row.get("bucket", ""), int(row.get("global_id", -1))))

    sections = []
    for bucket in BUCKETS:
        rows = [row for row in records if row.get("bucket") == bucket]
        body = []
        for row in rows:
            body.append(
                "<tr>"
                f"<td>{row['global_id']}</td>"
                f"<td>{row['prompt_group_id']}</td>"
                f"<td>{row['completion_length']}</td>"
                f"<td>{row['valid_metric_tokens']}</td>"
                f"<td>{_format_float(float(row['mean_da_weight']))}</td>"
                f"<td>{_format_float(float(row['max_abs_rho']))}</td>"
                f"<td><a href=\"{html.escape(row['html_file'])}\">html</a></td>"
                f"<td><a href=\"{html.escape(row['csv_file'])}\">csv</a></td>"
                "</tr>"
            )
        sections.append(
            f"""
            <h2>{bucket}</h2>
            <table>
              <thead><tr><th>global id</th><th>prompt group</th><th>completion len</th>
              <th>metric tokens</th><th>mean weight</th><th>max |rho|</th><th>view</th><th>csv</th></tr></thead>
              <tbody>{"".join(body)}</tbody>
            </table>
            """
        )

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 28px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
    th { background: #f3f4f6; }
    a { color: #2563eb; }
    """
    index_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rollout samples</title>
  <style>{css}</style>
</head>
<body>
  <h1>Rollout samples</h1>
  <p>Total sample entries: {len(records)}</p>
  {"".join(sections)}
</body>
</html>
"""
    (samples_root / "index.html").write_text(index_html, encoding="utf-8")


def _prepare_training_args(script_args: OPDMetricsScriptArguments, training_args: MiniLLMConfig) -> None:
    world_size = max(training_args.world_size, 1)
    total_rollouts_per_batch = script_args.groups_per_batch * script_args.group_size
    if script_args.metrics_num_rollouts <= 0:
        raise ValueError("metrics_num_rollouts must be > 0.")
    if script_args.metrics_num_sample_rollouts < 0:
        raise ValueError("metrics_num_sample_rollouts must be >= 0.")
    if script_args.metrics_sample_token_limit < 0:
        raise ValueError("metrics_sample_token_limit must be >= 0.")
    if script_args.metrics_sample_topk < 0:
        raise ValueError("metrics_sample_topk must be >= 0.")
    if script_args.group_size <= 0:
        raise ValueError("group_size must be > 0.")
    if total_rollouts_per_batch % world_size != 0:
        raise ValueError(
            "groups_per_batch * group_size must be divisible by world_size. "
            f"Got {total_rollouts_per_batch} and world_size={world_size}."
        )
    local_rollouts = total_rollouts_per_batch // world_size
    if local_rollouts % script_args.group_size != 0:
        raise ValueError(
            "Local rollouts per batch must be divisible by group_size so vLLM server mode can de-duplicate prompts. "
            f"Got local_rollouts={local_rollouts}, group_size={script_args.group_size}."
        )

    training_args.per_device_train_batch_size = local_rollouts
    training_args.gradient_accumulation_steps = 1
    training_args.generation_batch_size = total_rollouts_per_batch
    training_args.steps_per_generation = 1
    training_args.num_generations = script_args.group_size
    training_args.num_generations_eval = script_args.group_size
    training_args.per_device_eval_batch_size = script_args.metrics_forward_batch_size or 1
    training_args.rkl_advantage = False
    training_args.single_step_decomposition = False
    training_args.gamma = 0.0
    training_args.length_normalization = True
    training_args.beta = 0.0
    training_args.scale_rewards = "none"
    training_args.num_iterations = 1
    training_args.da_opd_weighting = True
    training_args.gradient_checkpointing = False
    if training_args.max_completion_length is None:
        training_args.max_completion_length = 4096

    if not script_args.enable_thinking:
        chat_template_kwargs = dict(training_args.chat_template_kwargs or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        training_args.chat_template_kwargs = chat_template_kwargs


def _prepare_metrics_student_model(trainer: MiniLLMTrainer) -> None:
    """Prepare the student model because metrics-only execution never enters Trainer.train()."""
    if trainer.is_deepspeed_enabled:
        trainer.model = prepare_deepspeed(trainer.model, trainer.accelerator)
    elif trainer.is_fsdp_enabled:
        trainer.model = prepare_fsdp(trainer.model, trainer.accelerator)
    else:
        trainer.model = trainer.accelerator.prepare_model(trainer.model, evaluation_mode=True)

    trainer.model_wrapped = trainer.model
    trainer.model.eval()


def main() -> None:
    _maybe_wait_for_debugger()

    parser = TrlParser((OPDMetricsScriptArguments, MiniLLMConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    _prepare_training_args(script_args, training_args)

    dtype = _resolve_dtype(model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
        dtype=dtype,
    )
    training_args.teacher_model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
        dtype=model_args.dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    train_dataset, _ = _load_dataset_splits(script_args)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty; cannot generate metric rollouts.")

    tokenizer_name_or_path = script_args.tokenizer_name_or_path.strip() if script_args.tokenizer_name_or_path else None
    processing_class_name = tokenizer_name_or_path or _resolve_processing_class_name(model_args.model_name_or_path)
    processing_class = AutoTokenizer.from_pretrained(
        processing_class_name,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
        truncation_side="left",
    )
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    output_dir = _student_output_dir(script_args.metrics_output_root, model_args.model_name_or_path)
    config_summary = {
        "student_model": model_args.model_name_or_path,
        "teacher_model": script_args.teacher_model_name_or_path,
        "tokenizer": processing_class_name,
        "dataset_name": script_args.dataset_name,
        "dataset_config": script_args.dataset_config,
        "dataset_train_split": script_args.dataset_train_split,
        "metrics_num_rollouts": script_args.metrics_num_rollouts,
        "length_threshold": script_args.metrics_length_threshold,
        "groups_per_batch": script_args.groups_per_batch,
        "group_size": script_args.group_size,
        "max_completion_length": training_args.max_completion_length,
        "temperature": training_args.temperature,
        "top_p": training_args.top_p,
        "da_opd_tau": training_args.da_opd_tau,
    }

    trainer = MiniLLMTrainer(
        model=model_args.model_name_or_path,
        teacher_model=script_args.teacher_model_name_or_path,
        reward_funcs=accuracy_reward_with_fallback,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=processing_class,
        peft_config=get_peft_config(model_args),
    )
    if not script_args.enable_thinking:
        trainer.chat_template = qwen3_training_chat_template
    if getattr(trainer, "use_vllm", False):
        trainer._last_loaded_step = trainer.state.global_step

    _prepare_metrics_student_model(trainer)
    trainer.teacher_model.eval()

    device = trainer.accelerator.device
    max_positions = int(training_args.max_completion_length)
    accumulator = PositionMetricAccumulator(
        max_positions=max_positions,
        length_threshold=script_args.metrics_length_threshold,
        device=device,
    )

    world_size = max(training_args.world_size, 1)
    total_rollouts_per_batch = script_args.groups_per_batch * script_args.group_size
    local_rollouts_per_batch = total_rollouts_per_batch // world_size
    num_batches = math.ceil(script_args.metrics_num_rollouts / total_rollouts_per_batch)
    forward_batch_size = script_args.metrics_forward_batch_size or training_args.per_device_train_batch_size
    forward_batch_size = max(1, min(forward_batch_size, training_args.per_device_train_batch_size))
    samples_written = 0

    for batch_index in range(num_batches):
        local_inputs, local_global_ids = _build_local_inputs(
            train_dataset=train_dataset,
            batch_index=batch_index,
            local_rollouts_per_batch=local_rollouts_per_batch,
            global_rollouts_per_batch=total_rollouts_per_batch,
            group_size=script_args.group_size,
            process_index=trainer.accelerator.process_index,
        )
        generated = trainer._generate_and_score_completions(local_inputs)
        keep = local_global_ids < script_args.metrics_num_rollouts
        if not keep.any():
            continue

        generated = _slice_tensor_batch(generated, keep)
        kept_global_ids = local_global_ids[keep]
        local_batch_size = generated["completion_ids"].size(0)
        for start in range(0, local_batch_size, forward_batch_size):
            end = min(start + forward_batch_size, local_batch_size)
            need_sample_topk = samples_written < script_args.metrics_num_sample_rollouts
            metrics, metric_mask, completion_lengths, topk_data = _compute_metric_tensors(
                trainer,
                generated,
                start,
                end,
                sample_topk=script_args.metrics_sample_topk if need_sample_topk else 0,
            )
            accumulator.update(metrics, metric_mask, completion_lengths)

            if samples_written < script_args.metrics_num_sample_rollouts:
                chunk_global_ids = kept_global_ids[start:end]
                for local_row, global_id_tensor in enumerate(chunk_global_ids):
                    if samples_written >= script_args.metrics_num_sample_rollouts:
                        break
                    completion_length = int(completion_lengths[local_row].detach().cpu().item())
                    _write_sample_files(
                        output_dir=output_dir,
                        trainer=trainer,
                        generated=generated,
                        chunk_start=start,
                        local_row=local_row,
                        global_id=int(global_id_tensor.item()),
                        group_size=script_args.group_size,
                        metrics=metrics,
                        metric_mask=metric_mask,
                        completion_length=completion_length,
                        topk_data=topk_data,
                        script_args=script_args,
                        config_summary=config_summary,
                    )
                    samples_written += 1

        trainer.accelerator.print(
            f"[opd_metrics] finished batch {batch_index + 1}/{num_batches}; "
            f"target_rollouts={script_args.metrics_num_rollouts}",
            flush=True,
        )

    reduced = accumulator.reduce(trainer.accelerator)

    if trainer.accelerator.is_main_process:
        for bucket in BUCKETS:
            _write_bucket_outputs(output_dir / bucket, bucket, reduced[bucket], config_summary)
        print(f"[opd_metrics] wrote metrics to {output_dir}", flush=True)

    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        _write_samples_index(output_dir)
        print(f"[opd_metrics] wrote sample index to {output_dir / 'samples' / 'index.html'}", flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
