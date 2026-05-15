from __future__ import annotations

import csv
import copy
import json
import math
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
    _use_qwen3_no_think,
    accuracy_reward_with_fallback,
)


DA_METHODS = ("raw", "seq", "window_avg", "ema_kl", "inverse_length")
DEFAULT_LENGTH_OPTIONS = (512, 1024, 2048, 4096)


@dataclass
class OPDHeatmapScriptArguments(OPDScriptArguments):
    heatmap_num_rollouts: int = field(
        default=5,
        metadata={"help": "Exact global number of generated trajectories to render."},
    )
    heatmap_output_root: str = field(
        default=(
            "/apdcephfs/test/jp_qy4_cephfs/apdcephfs_qy4/share_302593112/"
            "shaofanliu/projects/lzh/opd_metric_sample_out"
        ),
        metadata={"help": "Output directory for index.html, summary.json, and per-sample CSV files."},
    )
    heatmap_default_length: int = field(
        default=512,
        metadata={"help": "Default number of tokens displayed in each heatmap."},
    )
    heatmap_topks: str = field(
        default="8,16,32",
        metadata={"help": "Comma-separated top-k values for overlap/mass/local-entropy metrics."},
    )
    heatmap_forward_batch_size: int | None = field(
        default=None,
        metadata={"help": "Optional local micro-batch size for student/teacher metric forwards."},
    )


def _parse_topks(value: str) -> tuple[int, ...]:
    topks = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        topk = int(part)
        if topk <= 0:
            raise ValueError(f"heatmap_topks values must be positive, got {topk}.")
        topks.append(topk)
    if not topks:
        raise ValueError("heatmap_topks must contain at least one positive integer.")
    return tuple(sorted(set(topks)))


def _metric_names(topks: tuple[int, ...]) -> list[str]:
    names = [
        "student_entropy",
        "teacher_entropy",
        "entropy_gap",
        "rho",
        "logprob_ratio",
        "student_sample_logprob",
        "student_forward_sample_logprob",
        "teacher_sample_logprob",
        "teacher_student_sample_logprob_gap",
        "teacher_student_forward_logprob_gap",
        "top1_agreement",
    ]
    for method in DA_METHODS:
        names.append(f"da_{method}_score")
        names.append(f"da_{method}_weight")
    for topk in topks:
        names.extend(
            [
                f"overlap_ratio_top{topk}",
                f"student_overlap_mass_top{topk}",
                f"teacher_overlap_mass_top{topk}",
                f"student_top{topk}_entropy",
                f"teacher_top{topk}_entropy",
                f"entropy_gap_top{topk}",
            ]
        )
    return names


def _build_local_inputs(
    train_dataset,
    prompt_indices: list[int],
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
    for global_id in global_ids.tolist():
        prompt_group_id = global_id // group_size
        dataset_index = prompt_indices[prompt_group_id]
        inputs.append(copy.deepcopy(dict(train_dataset[dataset_index])))
    return inputs, global_ids


def _sample_prompt_indices(dataset_len: int, num_prompt_groups: int, seed: int) -> list[int]:
    if dataset_len <= 0:
        raise ValueError("Training dataset is empty; cannot sample prompts.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    if num_prompt_groups <= dataset_len:
        return torch.randperm(dataset_len, generator=generator)[:num_prompt_groups].tolist()
    return torch.randint(0, dataset_len, (num_prompt_groups,), generator=generator).tolist()


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


def _topk_local_entropy(topk_log_probs: torch.Tensor) -> torch.Tensor:
    topk_probs = topk_log_probs.exp()
    normalized = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    entropy = -(normalized * torch.log(normalized.clamp(min=1e-12))).sum(dim=-1)
    topk_size = topk_log_probs.size(-1)
    if topk_size > 1:
        return entropy / math.log(topk_size)
    return torch.zeros_like(entropy)


def _compute_all_da_metrics(
    teacher_token_logprobs: torch.Tensor,
    student_token_logprobs: torch.Tensor,
    mask: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    tau: float,
    window_size: int,
    ema_beta: float,
) -> dict[str, torch.Tensor]:
    mask_float = mask.float()
    logprob_ratio = (teacher_token_logprobs - student_token_logprobs) * mask_float
    rho = logprob_ratio.cumsum(dim=-1)

    # Keep these formulas tied to MiniLLMTrainer's DA-OPD helpers without mutating trainer state.
    scores = {
        "raw": MiniLLMTrainer._da_opd_score_raw(rho),
        "seq": MiniLLMTrainer._da_opd_score_seq(logprob_ratio, rho, mask_float),
        "window_avg": MiniLLMTrainer._da_opd_score_window_avg(rho, mask_float, window_size),
        "ema_kl": MiniLLMTrainer._da_opd_score_ema_kl(student_log_probs, teacher_log_probs, mask_float, ema_beta),
        "inverse_length": MiniLLMTrainer._da_opd_score_inverse_length(rho, mask_float),
    }

    metrics: dict[str, torch.Tensor] = {
        "rho": rho,
        "logprob_ratio": logprob_ratio,
    }
    for method in DA_METHODS:
        score = scores[method]
        metrics[f"da_{method}_score"] = score
        metrics[f"da_{method}_weight"] = torch.sigmoid(score / tau).detach() * mask_float
    return metrics


def _compute_metric_tensors(
    trainer: MiniLLMTrainer,
    inputs: dict[str, torch.Tensor],
    start: int,
    end: int,
    topks: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    prompt_ids = inputs["prompt_ids"][start:end]
    completion_ids = inputs["completion_ids"][start:end]
    prompt_mask = inputs["prompt_mask"][start:end]
    completion_mask = inputs["completion_mask"][start:end]
    old_per_token_logps = inputs.get("old_per_token_logps")
    if old_per_token_logps is None:
        raise RuntimeError("Heatmap metrics require `old_per_token_logps`; `da_opd_weighting` must be enabled.")
    old_per_token_logps = old_per_token_logps[start:end]

    chunk_inputs = {"completion_mask": completion_mask}
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
    student_logits, teacher_logits = trainer._align_logits_to_shared_model_vocab(
        student_logits, teacher_logits, labels=shifted_labels
    )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    student_forward_on_labels = student_log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    teacher_on_labels = teacher_log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

    student_entropy = _full_vocab_entropy(student_log_probs)
    teacher_entropy = _full_vocab_entropy(teacher_log_probs)

    metrics: dict[str, torch.Tensor] = {
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "entropy_gap": torch.abs(student_entropy - teacher_entropy),
        "student_sample_logprob": old_per_token_logps,
        "student_forward_sample_logprob": student_forward_on_labels,
        "teacher_sample_logprob": teacher_on_labels,
        "teacher_student_sample_logprob_gap": teacher_on_labels - old_per_token_logps,
        "teacher_student_forward_logprob_gap": teacher_on_labels - student_forward_on_labels,
    }
    metrics.update(
        _compute_all_da_metrics(
            teacher_token_logprobs=teacher_on_labels,
            student_token_logprobs=old_per_token_logps,
            mask=metric_mask,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            tau=trainer.da_opd_tau,
            window_size=trainer.da_opd_window_size,
            ema_beta=trainer.da_opd_ema_beta,
        )
    )

    max_topk = min(max(topks), student_log_probs.size(-1), teacher_log_probs.size(-1))
    student_topk_log_probs, student_topk_indices = student_log_probs.detach().topk(max_topk, dim=-1)
    teacher_topk_log_probs, teacher_topk_indices = teacher_log_probs.detach().topk(max_topk, dim=-1)
    metrics["top1_agreement"] = (student_topk_indices[..., 0] == teacher_topk_indices[..., 0]).float()

    for topk in topks:
        current_k = min(topk, max_topk)
        student_indices = student_topk_indices[..., :current_k]
        teacher_indices = teacher_topk_indices[..., :current_k]
        student_log_probs_k = student_topk_log_probs[..., :current_k]
        teacher_log_probs_k = teacher_topk_log_probs[..., :current_k]

        student_overlap_mask = (student_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)).any(dim=-1)
        teacher_overlap_mask = (teacher_indices.unsqueeze(-1) == student_indices.unsqueeze(-2)).any(dim=-1)

        student_topk_entropy = _topk_local_entropy(student_log_probs_k)
        teacher_topk_entropy = _topk_local_entropy(teacher_log_probs_k)
        metrics[f"overlap_ratio_top{topk}"] = student_overlap_mask.float().sum(dim=-1) / current_k
        metrics[f"student_overlap_mass_top{topk}"] = (student_log_probs_k.exp() * student_overlap_mask).sum(dim=-1)
        metrics[f"teacher_overlap_mass_top{topk}"] = (teacher_log_probs_k.exp() * teacher_overlap_mask).sum(dim=-1)
        metrics[f"student_top{topk}_entropy"] = student_topk_entropy
        metrics[f"teacher_top{topk}_entropy"] = teacher_topk_entropy
        metrics[f"entropy_gap_top{topk}"] = torch.abs(student_topk_entropy - teacher_topk_entropy)

    return metrics, metric_mask, completion_lengths


def _format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _json_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _visible_text(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None and math.isfinite(value)]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _arg_extreme(values: list[float | None], mode: str, absolute: bool = False) -> int | None:
    valid_indices = [idx for idx, value in enumerate(values) if value is not None and math.isfinite(value)]
    if not valid_indices:
        return None
    if mode == "max":
        return max(valid_indices, key=lambda idx: abs(values[idx]) if absolute else values[idx])
    if mode == "min":
        return min(valid_indices, key=lambda idx: values[idx])
    raise ValueError(f"Unsupported mode={mode!r}.")


def _write_sample_outputs(
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
    metric_names: list[str],
    config_summary: dict[str, Any],
    topks: tuple[int, ...],
) -> None:
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
    positions = (valid_indices + 1).tolist()

    metric_values: dict[str, list[float | None]] = {}
    for name in metric_names:
        raw_values = metrics[name][local_row][valid_indices_device].detach().float().cpu().tolist()
        metric_values[name] = [_json_float(float(value)) for value in raw_values]

    rows = []
    tokens_for_html = []
    for idx in range(valid_count):
        row_metrics = {name: metric_values[name][idx] for name in metric_names}
        rows.append(
            [
                positions[idx],
                metric_token_ids[idx],
                repr(token_strings[idx]),
                decoded_tokens[idx],
                *[_format_float(row_metrics[name]) for name in metric_names],
            ]
        )
        tokens_for_html.append(
            {
                "position": int(positions[idx]),
                "token_id": int(metric_token_ids[idx]),
                "token_repr": repr(token_strings[idx]),
                "text": _visible_text(decoded_tokens[idx]),
                "metrics": row_metrics,
            }
        )

    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"sample_{global_id}"
    csv_path = sample_dir / f"{base_name}.tokens.csv"
    data_path = sample_dir / f"{base_name}.heatmap.json"
    metadata_path = sample_dir / f"{base_name}.metadata.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "token_id", "token_repr", "decoded_token", *metric_names])
        writer.writerows(rows)

    lowest_overlap_name = f"overlap_ratio_top{min(topks)}"
    max_gap_idx = _arg_extreme(metric_values["teacher_student_sample_logprob_gap"], "max", absolute=True)
    max_abs_rho_idx = _arg_extreme(metric_values["rho"], "max", absolute=True)
    lowest_overlap_idx = _arg_extreme(metric_values[lowest_overlap_name], "min")
    summary = {
        "global_id": global_id,
        "prompt_group_id": global_id // group_size,
        "completion_length": completion_length,
        "valid_metric_tokens": valid_count,
        "csv_file": str(csv_path.relative_to(output_dir)),
        "data_file": str(data_path.relative_to(output_dir)),
        "max_abs_rho_position": positions[max_abs_rho_idx] if max_abs_rho_idx is not None else None,
        "max_abs_gap_position": positions[max_gap_idx] if max_gap_idx is not None else None,
        f"lowest_overlap_top{min(topks)}_position": positions[lowest_overlap_idx] if lowest_overlap_idx is not None else None,
    }
    for method in DA_METHODS:
        summary[f"mean_da_{method}_weight"] = _mean(metric_values[f"da_{method}_weight"])

    data = {
        **config_summary,
        **summary,
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "tokens": tokens_for_html,
    }
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    metadata_path.write_text(
        json.dumps({**config_summary, **summary}, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_index(
    output_dir: Path,
    config_summary: dict[str, Any],
    metric_names: list[str],
    num_rollouts: int,
    default_length: int,
) -> None:
    sample_dir = output_dir / "samples"
    rollouts = []
    for global_id in range(num_rollouts):
        data_path = sample_dir / f"sample_{global_id}.heatmap.json"
        if data_path.exists():
            rollouts.append(json.loads(data_path.read_text(encoding="utf-8")))

    summary = {
        **config_summary,
        "requested_rollouts": num_rollouts,
        "written_rollouts": len(rollouts),
        "metric_names": metric_names,
        "rollouts": [
            {
                key: rollout.get(key)
                for key in (
                    "global_id",
                    "prompt_group_id",
                    "completion_length",
                    "valid_metric_tokens",
                    "csv_file",
                    "max_abs_rho_position",
                    "max_abs_gap_position",
                )
            }
            for rollout in rollouts
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    page_data = {
        "config": config_summary,
        "metric_names": metric_names,
        "default_length": default_length,
        "length_options": sorted(set((*DEFAULT_LENGTH_OPTIONS, default_length))),
        "rollouts": rollouts,
    }
    data_json = json.dumps(page_data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")

    html_doc = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>OPD sample heatmaps</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2937; background: #f8fafc; }
    header { padding: 18px 24px; background: #111827; color: #f9fafb; }
    h1 { margin: 0 0 8px; font-size: 24px; font-weight: 650; }
    h2 { margin: 0; font-size: 18px; }
    h3 { margin: 20px 0 8px; font-size: 15px; }
    main { padding: 18px 24px 32px; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: #d1d5db; }
    .rollout { background: #ffffff; border: 1px solid #d7dee8; border-radius: 8px; margin: 0 0 18px; overflow: hidden; }
    .rollout-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #e5e7eb; background: #f9fafb; }
    .summary { display: flex; flex-wrap: wrap; gap: 8px 14px; font-size: 12px; color: #4b5563; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 12px 14px; border-bottom: 1px solid #e5e7eb; }
    label { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: #374151; }
    select, button { min-height: 30px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px; background: #ffffff; color: #111827; }
    button { cursor: pointer; }
    button:hover { background: #f3f4f6; }
    .heatmap { padding: 14px; line-height: 2.05; word-break: break-word; background: #ffffff; }
    .tok { display: inline-block; margin: 1px; padding: 1px 3px; border-radius: 3px; border: 1px solid transparent; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .tok.hot { border-color: #111827; box-shadow: inset 0 0 0 1px #111827; font-weight: 700; }
    .multi-filter-panel { display: none; width: 100%; padding: 10px 0 0; border-top: 1px solid #e5e7eb; }
    .multi-filter-panel.active { display: block; }
    .filter-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 8px; }
    .filter-row span { font-size: 13px; color: #4b5563; }
    .remove-filter { min-width: 30px; padding: 4px 9px; }
    .details { padding: 0 14px 14px; border-top: 1px solid #e5e7eb; }
    pre { white-space: pre-wrap; overflow-x: auto; margin: 8px 0 0; padding: 10px; border: 1px solid #e5e7eb; border-radius: 6px; background: #f8fafc; font-size: 12px; }
    a { color: #2563eb; }
    .empty { padding: 16px; color: #6b7280; }
  </style>
</head>
<body>
  <header>
    <h1>OPD sample heatmaps</h1>
    <div class="meta" id="page-meta"></div>
  </header>
  <main id="app"></main>
  <script id="rollout-data" type="application/json">__DATA_JSON__</script>
  <script>
    const pageData = JSON.parse(document.getElementById("rollout-data").textContent);
    const rangeCache = new Map();

    function fmt(value) {
      return Number.isFinite(value) ? Number(value).toPrecision(6) : "nan";
    }

    function mix(a, b, t) {
      return Math.round(a + (b - a) * t);
    }

    function colorFor(value, range) {
      if (!Number.isFinite(value)) return "#e5e7eb";
      if (!range || !Number.isFinite(range.min) || !Number.isFinite(range.max)) return "#e5e7eb";
      let ratio = range.max === range.min ? 0.5 : (value - range.min) / (range.max - range.min);
      ratio = Math.max(0, Math.min(1, ratio));
      const blue = [37, 99, 235];
      const white = [255, 255, 255];
      const red = [220, 38, 38];
      const left = ratio < 0.5;
      const t = left ? ratio * 2 : (ratio - 0.5) * 2;
      const from = left ? blue : white;
      const to = left ? white : red;
      return `rgb(${mix(from[0], to[0], t)}, ${mix(from[1], to[1], t)}, ${mix(from[2], to[2], t)})`;
    }

    function colorFromRatio(ratio) {
      if (!Number.isFinite(ratio)) return "#e5e7eb";
      ratio = Math.max(0, Math.min(1, ratio));
      const blue = [37, 99, 235];
      const white = [255, 255, 255];
      const red = [220, 38, 38];
      const left = ratio < 0.5;
      const t = left ? ratio * 2 : (ratio - 0.5) * 2;
      const from = left ? blue : white;
      const to = left ? white : red;
      return `rgb(${mix(from[0], to[0], t)}, ${mix(from[1], to[1], t)}, ${mix(from[2], to[2], t)})`;
    }

    function metricRange(metric) {
      if (rangeCache.has(metric)) return rangeCache.get(metric);
      let min = Infinity;
      let max = -Infinity;
      for (const rollout of pageData.rollouts) {
        for (const token of rollout.tokens) {
          const value = token.metrics[metric];
          if (Number.isFinite(value)) {
            min = Math.min(min, value);
            max = Math.max(max, value);
          }
        }
      }
      const range = min === Infinity ? null : { min, max };
      rangeCache.set(metric, range);
      return range;
    }

    function rankedItems(tokens, metric, direction) {
      return tokens
        .map((token, index) => ({ index, value: token.metrics[metric] }))
        .filter((item) => Number.isFinite(item.value))
        .sort((a, b) => direction === "smallest" ? a.value - b.value : b.value - a.value);
    }

    function topSet(tokens, metric, pct, direction) {
      if (pct <= 0 || direction === "off") return new Set();
      const values = rankedItems(tokens, metric, direction);
      const keep = Math.ceil(values.length * pct / 100);
      return new Set(values.slice(0, keep).map((item) => item.index));
    }

    function rangeSet(tokens, metric, direction, startPct, endPct) {
      const values = rankedItems(tokens, metric, direction);
      if (values.length === 0) return new Set();
      const start = Math.max(0, Math.min(values.length, Math.floor(values.length * startPct / 100)));
      const end = Math.max(start, Math.min(values.length, Math.ceil(values.length * endPct / 100)));
      return new Set(values.slice(start, end).map((item) => item.index));
    }

    function intersectSets(sets) {
      if (sets.length === 0) return new Set();
      const [first, ...rest] = sets;
      return new Set([...first].filter((index) => rest.every((set) => set.has(index))));
    }

    function multiFilterSet(root, tokens) {
      const rows = [...root.querySelectorAll(".filter-row")];
      const sets = [];
      for (const row of rows) {
        const metric = row.querySelector(".filter-metric").value;
        const direction = row.querySelector(".filter-direction").value;
        let startPct = Number(row.querySelector(".filter-start").value);
        let endPct = Number(row.querySelector(".filter-end").value);
        if (endPct < startPct) {
          [startPct, endPct] = [endPct, startPct];
        }
        sets.push(rangeSet(tokens, metric, direction, startPct, endPct));
      }
      return intersectSets(sets);
    }

    function rankRatioMap(tokens, metric) {
      const values = rankedItems(tokens, metric, "smallest");
      const ratios = new Map();
      const denominator = Math.max(1, values.length - 1);
      values.forEach((item, rank) => ratios.set(item.index, rank / denominator));
      return ratios;
    }

    function renderRollout(root, rolloutIndex) {
      const rollout = pageData.rollouts[rolloutIndex];
      const metric = root.querySelector(".metric-select").value;
      const lengthValue = Number(root.querySelector(".length-select").value);
      const backgroundEnabled = root.querySelector(".background-select").value === "on";
      const colorMode = root.querySelector(".color-mode-select").value;
      const highlightMode = root.querySelector(".highlight-mode-select").value;
      const count = lengthValue < 0 ? rollout.tokens.length : Math.min(lengthValue, rollout.tokens.length);
      const range = metricRange(metric);
      const rankRatios = colorMode === "rank" ? rankRatioMap(rollout.tokens, metric) : null;
      const hot = highlightMode === "multi"
        ? multiFilterSet(root, rollout.tokens)
        : highlightMode === "single"
          ? topSet(
            rollout.tokens,
            metric,
            Number(root.querySelector(".single-pct-select").value),
            root.querySelector(".single-direction-select").value,
          )
          : new Set();
      const heatmap = root.querySelector(".heatmap");
      const frag = document.createDocumentFragment();

      for (let i = 0; i < count; i += 1) {
        const token = rollout.tokens[i];
        const value = token.metrics[metric];
        const span = document.createElement("span");
        span.className = "tok" + (hot.has(i) ? " hot" : "");
        span.textContent = token.text || " ";
        if (backgroundEnabled) {
          span.style.backgroundColor = colorMode === "rank" ? colorFromRatio(rankRatios.get(i)) : colorFor(value, range);
        }
        span.title = `pos=${token.position} id=${token.token_id} ${metric}=${fmt(value)}`;
        frag.appendChild(span);
      }
      heatmap.replaceChildren(frag);

      const valueLabel = root.querySelector(".value-label");
      const colorText = colorMode === "rank" ? "within-rollout rank" : "absolute value";
      valueLabel.textContent = range ? `${metric}: ${colorText}, min ${fmt(range.min)}, max ${fmt(range.max)}` : `${metric}: no finite values`;
    }

    function makeSelect(className, options, selected) {
      const select = document.createElement("select");
      select.className = className;
      for (const option of options) {
        const el = document.createElement("option");
        el.value = option.value;
        el.textContent = option.label;
        if (String(option.value) === String(selected)) el.selected = true;
        select.appendChild(el);
      }
      return select;
    }

    function appendLabeledSelect(container, labelText, select) {
      const label = document.createElement("label");
      label.textContent = labelText;
      label.appendChild(select);
      container.appendChild(label);
      return label;
    }

    function addFilterRow(panel, metricOptions, percentOptions, onChange) {
      const rows = panel.querySelector(".filter-rows");
      const row = document.createElement("div");
      row.className = "filter-row";
      const metricSelect = makeSelect("filter-metric", metricOptions, "da_raw_weight");
      const directionSelect = makeSelect("filter-direction", [
        { value: "largest", label: "largest" },
        { value: "smallest", label: "smallest" },
      ], "largest");
      const startSelect = makeSelect("filter-start", percentOptions, 0);
      const endSelect = makeSelect("filter-end", percentOptions, 20);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-filter";
      remove.textContent = "x";
      row.appendChild(metricSelect);
      row.appendChild(directionSelect);
      row.appendChild(document.createTextNode("top"));
      row.appendChild(startSelect);
      row.appendChild(document.createTextNode("to"));
      row.appendChild(endSelect);
      row.appendChild(remove);
      rows.appendChild(row);
      for (const select of [metricSelect, directionSelect, startSelect, endSelect]) {
        select.addEventListener("change", onChange);
      }
      remove.addEventListener("click", () => {
        row.remove();
        onChange();
      });
    }

    function init() {
      const meta = document.getElementById("page-meta");
      const config = pageData.config;
      meta.textContent = [
        `student: ${config.student_model}`,
        `teacher: ${config.teacher_model}`,
        `rollouts: ${pageData.rollouts.length}`,
        `max completion: ${config.max_completion_length}`,
      ].join(" | ");

      const app = document.getElementById("app");
      if (pageData.rollouts.length === 0) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No rollout files were found.";
        app.appendChild(empty);
        return;
      }

      const metricOptions = pageData.metric_names.map((name) => ({ value: name, label: name }));
      const lengthOptions = pageData.length_options.map((value) => ({ value, label: String(value) }));
      lengthOptions.push({ value: -1, label: "all" });
      const singlePctOptions = [
        { value: 1, label: "top 1%" },
        { value: 5, label: "top 5%" },
        { value: 10, label: "top 10%" },
        { value: 20, label: "top 20%" },
        { value: 30, label: "top 30%" },
        { value: 40, label: "top 40%" },
        { value: 50, label: "top 50%" },
        { value: 60, label: "top 60%" },
        { value: 70, label: "top 70%" },
        { value: 80, label: "top 80%" },
      ];
      const percentOptions = [
        { value: 0, label: "0%" },
        { value: 1, label: "1%" },
        { value: 5, label: "5%" },
        { value: 10, label: "10%" },
        { value: 20, label: "20%" },
        { value: 30, label: "30%" },
        { value: 40, label: "40%" },
        { value: 50, label: "50%" },
        { value: 60, label: "60%" },
        { value: 70, label: "70%" },
        { value: 80, label: "80%" },
        { value: 100, label: "100%" },
      ];

      pageData.rollouts.forEach((rollout, rolloutIndex) => {
        const section = document.createElement("section");
        section.className = "rollout";
        section.innerHTML = `
          <div class="rollout-head">
            <div>
              <h2>Rollout ${rollout.global_id}</h2>
              <div class="summary">
                <span>prompt group ${rollout.prompt_group_id}</span>
                <span>completion ${rollout.completion_length}</span>
                <span>metric tokens ${rollout.valid_metric_tokens}</span>
                <span><a href="${rollout.csv_file}">CSV</a></span>
              </div>
            </div>
            <div class="value-label"></div>
          </div>
          <div class="controls"></div>
          <div class="heatmap"></div>
          <div class="details">
            <h3>Prompt</h3>
            <pre></pre>
            <h3>Completion</h3>
            <pre></pre>
          </div>
        `;

        const controls = section.querySelector(".controls");
        const metricSelect = makeSelect("metric-select", metricOptions, "da_raw_weight");
        const lengthSelect = makeSelect("length-select", lengthOptions, pageData.default_length);
        const backgroundSelect = makeSelect("background-select", [
          { value: "on", label: "on" },
          { value: "off", label: "off" },
        ], "on");
        const colorModeSelect = makeSelect("color-mode-select", [
          { value: "absolute", label: "absolute value" },
          { value: "rank", label: "within-rollout rank" },
        ], "absolute");
        const highlightModeSelect = makeSelect("highlight-mode-select", [
          { value: "off", label: "off" },
          { value: "single", label: "single metric top" },
          { value: "multi", label: "multi metric range" },
        ], "off");
        const singleDirectionSelect = makeSelect("single-direction-select", [
          { value: "largest", label: "largest" },
          { value: "smallest", label: "smallest" },
        ], "largest");
        const singlePctSelect = makeSelect("single-pct-select", singlePctOptions, 20);

        for (const [labelText, select] of [
          ["metric", metricSelect],
          ["length", lengthSelect],
          ["background", backgroundSelect],
          ["color", colorModeSelect],
          ["highlight mode", highlightModeSelect],
          ["single direction", singleDirectionSelect],
          ["single percent", singlePctSelect],
        ]) {
          appendLabeledSelect(controls, labelText, select);
          select.addEventListener("change", () => renderRollout(section, rolloutIndex));
        }

        const panel = document.createElement("div");
        panel.className = "multi-filter-panel";
        panel.innerHTML = '<div class="filter-rows"></div>';
        const addButton = document.createElement("button");
        addButton.type = "button";
        addButton.textContent = "add condition";
        panel.appendChild(addButton);
        controls.appendChild(panel);

        const syncHighlightMode = () => {
          const multi = highlightModeSelect.value === "multi";
          const single = highlightModeSelect.value === "single";
          panel.classList.toggle("active", multi);
          singleDirectionSelect.disabled = !single;
          singlePctSelect.disabled = !single;
          renderRollout(section, rolloutIndex);
        };
        highlightModeSelect.addEventListener("change", syncHighlightMode);
        addButton.addEventListener("click", () => {
          addFilterRow(panel, metricOptions, percentOptions, () => renderRollout(section, rolloutIndex));
          renderRollout(section, rolloutIndex);
        });
        addFilterRow(panel, metricOptions, percentOptions, () => renderRollout(section, rolloutIndex));

        const pres = section.querySelectorAll("pre");
        pres[0].textContent = rollout.prompt_text;
        pres[1].textContent = rollout.completion_text;
        app.appendChild(section);
        syncHighlightMode();
      });
    }

    init();
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc.replace("__DATA_JSON__", data_json), encoding="utf-8")


def _prepare_training_args(
    script_args: OPDHeatmapScriptArguments,
    training_args: MiniLLMConfig,
    qwen3_no_think: bool = False,
) -> None:
    world_size = max(training_args.world_size, 1)
    total_rollouts_per_batch = script_args.groups_per_batch * script_args.group_size
    if script_args.heatmap_num_rollouts <= 0:
        raise ValueError("heatmap_num_rollouts must be > 0.")
    if script_args.group_size <= 0:
        raise ValueError("group_size must be > 0.")
    if total_rollouts_per_batch % world_size != 0:
        raise ValueError(
            "groups_per_batch * group_size must be divisible by world_size. "
            f"Got {total_rollouts_per_batch} and world_size={world_size}."
        )

    local_rollouts = total_rollouts_per_batch // world_size
    training_args.per_device_train_batch_size = local_rollouts
    training_args.gradient_accumulation_steps = 1
    training_args.generation_batch_size = total_rollouts_per_batch
    training_args.steps_per_generation = 1
    training_args.num_generations = script_args.group_size
    training_args.num_generations_eval = script_args.group_size
    training_args.per_device_eval_batch_size = script_args.heatmap_forward_batch_size or 1
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

    if qwen3_no_think:
        chat_template_kwargs = dict(training_args.chat_template_kwargs or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        training_args.chat_template_kwargs = chat_template_kwargs


def _prepare_metrics_student_model(trainer: MiniLLMTrainer) -> None:
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

    parser = TrlParser((OPDHeatmapScriptArguments, MiniLLMConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    topks = _parse_topks(script_args.heatmap_topks)
    metric_names = _metric_names(topks)
    qwen3_no_think = _use_qwen3_no_think(script_args, model_args.model_name_or_path)
    _prepare_training_args(script_args, training_args, qwen3_no_think)

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

    train_dataset, _ = _load_dataset_splits(script_args, qwen3_no_think)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty; cannot generate heatmap rollouts.")

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

    output_dir = Path(script_args.heatmap_output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_summary = {
        "student_model": model_args.model_name_or_path,
        "teacher_model": script_args.teacher_model_name_or_path,
        "tokenizer": processing_class_name,
        "dataset_name": script_args.dataset_name,
        "dataset_config": script_args.dataset_config,
        "dataset_train_split": script_args.dataset_train_split,
        "heatmap_num_rollouts": script_args.heatmap_num_rollouts,
        "groups_per_batch": script_args.groups_per_batch,
        "group_size": script_args.group_size,
        "max_completion_length": training_args.max_completion_length,
        "temperature": training_args.temperature,
        "top_p": training_args.top_p,
        "topks": list(topks),
        "da_opd_tau": training_args.da_opd_tau,
        "da_opd_window_size": training_args.da_opd_window_size,
        "da_opd_ema_beta": training_args.da_opd_ema_beta,
        "seed": training_args.seed,
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
    if qwen3_no_think:
        trainer.chat_template = qwen3_training_chat_template
    if getattr(trainer, "use_vllm", False):
        raise ValueError("opd_sample_heatmap.py is intended for local generation; do not pass --use_vllm.")

    _prepare_metrics_student_model(trainer)
    trainer.teacher_model.eval()

    world_size = max(training_args.world_size, 1)
    total_rollouts_per_batch = script_args.groups_per_batch * script_args.group_size
    local_rollouts_per_batch = total_rollouts_per_batch // world_size
    num_batches = math.ceil(script_args.heatmap_num_rollouts / total_rollouts_per_batch)
    forward_batch_size = script_args.heatmap_forward_batch_size or training_args.per_device_train_batch_size
    forward_batch_size = max(1, min(forward_batch_size, training_args.per_device_train_batch_size))
    num_prompt_groups = math.ceil(script_args.heatmap_num_rollouts / script_args.group_size)
    prompt_indices = _sample_prompt_indices(
        dataset_len=len(train_dataset),
        num_prompt_groups=num_prompt_groups,
        seed=training_args.seed,
    )

    trainer.accelerator.wait_for_everyone()
    for batch_index in range(num_batches):
        local_inputs, local_global_ids = _build_local_inputs(
            train_dataset=train_dataset,
            prompt_indices=prompt_indices,
            batch_index=batch_index,
            local_rollouts_per_batch=local_rollouts_per_batch,
            global_rollouts_per_batch=total_rollouts_per_batch,
            group_size=script_args.group_size,
            process_index=trainer.accelerator.process_index,
        )
        generated = trainer._generate_and_score_completions(local_inputs)
        keep = local_global_ids < script_args.heatmap_num_rollouts
        if not keep.any():
            continue

        generated = _slice_tensor_batch(generated, keep)
        kept_global_ids = local_global_ids[keep]
        local_batch_size = generated["completion_ids"].size(0)
        for start in range(0, local_batch_size, forward_batch_size):
            end = min(start + forward_batch_size, local_batch_size)
            metrics, metric_mask, completion_lengths = _compute_metric_tensors(
                trainer=trainer,
                inputs=generated,
                start=start,
                end=end,
                topks=topks,
            )
            chunk_global_ids = kept_global_ids[start:end]
            for local_row, global_id_tensor in enumerate(chunk_global_ids):
                _write_sample_outputs(
                    output_dir=output_dir,
                    trainer=trainer,
                    generated=generated,
                    chunk_start=start,
                    local_row=local_row,
                    global_id=int(global_id_tensor.item()),
                    group_size=script_args.group_size,
                    metrics=metrics,
                    metric_mask=metric_mask,
                    completion_length=int(completion_lengths[local_row].detach().cpu().item()),
                    metric_names=metric_names,
                    config_summary=config_summary,
                    topks=topks,
                )

        trainer.accelerator.print(
            f"[opd_sample_heatmap] finished batch {batch_index + 1}/{num_batches}; "
            f"target_rollouts={script_args.heatmap_num_rollouts}",
            flush=True,
        )

    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        _write_index(
            output_dir=output_dir,
            config_summary=config_summary,
            metric_names=metric_names,
            num_rollouts=script_args.heatmap_num_rollouts,
            default_length=script_args.heatmap_default_length,
        )
        print(f"[opd_sample_heatmap] wrote index to {output_dir / 'index.html'}", flush=True)
        print(f"[opd_sample_heatmap] wrote summary to {output_dir / 'summary.json'}", flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
