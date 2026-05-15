"""Standalone eval pipeline: vLLM generation -> heuristic grading -> acc@k / pass@k -> CSV + wandb."""

from __future__ import annotations

import argparse
import csv
import gc
import multiprocessing as mp
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import pandas as pd
from transformers import AutoTokenizer, GenerationConfig

EVAL_SUITE_ROOT = Path(
    "/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh"
    "/eval_suite/EvalSuite"
)
EVAL_DATA_DIR = EVAL_SUITE_ROOT / "data"

DATASET_REGISTRY = {
    "aime24": {"path": "AIME24/test.parquet", "scorer": "aime"},
    "aime25": {"path": "AIME25/test.parquet", "scorer": "aime"},
    "amc23": {"path": "AMC23/test.parquet", "scorer": "aime"},
    "math500": {"path": "MATH-500/test.parquet", "scorer": "math"},
    "minerva": {"path": "Minerva/test.parquet", "scorer": "math"},
    "olympiadbench": {"path": "Olympiad-Bench/test.parquet", "scorer": "math"},
}

sys.path.insert(0, str(EVAL_SUITE_ROOT / "vendor"))
sys.path.insert(0, str(EVAL_SUITE_ROOT / "scripts"))

from grade_heuristic import score_aime_like, score_mathlike

_DP_WORKER_LLM = None
_DP_WORKER_LLM_KEY = None
_DP_WORKER_SAMPLING_PARAMS = None
_DP_WORKER_SAMPLING_KEY = None
_DP_WORKER_GPU_ID = None

QWEN3_STOP_TOKEN_IDS = [151643, 151645]


def load_samples(parquet_path: Path) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    samples = []
    for i in range(len(df)):
        raw_prompt = df.at[i, "prompt"]
        if isinstance(raw_prompt, list) and raw_prompt:
            problem_text = raw_prompt[0].get("content", "").strip()
        else:
            problem_text = str(raw_prompt).strip()

        reward_model = df.at[i, "reward_model"]
        if isinstance(reward_model, dict):
            answer = reward_model.get("ground_truth", "").strip()
        else:
            answer = str(reward_model).strip()

        samples.append({"example_id": i, "prompt": problem_text, "answer": answer})
    return samples


def generate_completions(
    llm,
    sampling_params,
    samples: list[dict],
    enable_thinking: bool,
    dp_executors: list[tuple[str, ProcessPoolExecutor]] | None,
    model_path: str,
    tokenizer_path: str | None,
    max_tokens: int,
    qwen3_no_think: bool,
) -> list[dict]:
    """Generate k completions per sample using vLLM. Returns flat list of records."""
    if dp_executors:
        return _generate_completions_dp(
            dp_executors=dp_executors,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_tokens=max_tokens,
            sampling_params=sampling_params,
            samples=samples,
            enable_thinking=enable_thinking,
            qwen3_no_think=qwen3_no_think,
        )

    if enable_thinking or not qwen3_no_think:
        messages_batch = [
            [{"role": "user", "content": s["prompt"]}]
            for s in samples
        ]
    else:
        messages_batch = [
            [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": s["prompt"]},
            ]
            for s in samples
        ]

    print(f"  Generating {sampling_params.n} completions for {len(samples)} prompts ...")
    outputs = llm.chat(messages_batch, sampling_params, use_tqdm=True)

    records = []
    for sample, output in zip(samples, outputs):
        for idx, completion in enumerate(output.outputs):
            records.append({
                "example_id": sample["example_id"],
                "prompt": sample["prompt"],
                "answer": sample["answer"],
                "completion_idx": idx,
                "response": completion.text,
            })
    return records


def _split_samples(samples: list[dict], num_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(num_shards)]
    for idx, sample in enumerate(samples):
        shards[idx % num_shards].append(sample)
    return [s for s in shards if s]


def _sampling_params_to_dict(sampling_params) -> dict:
    sampling_cfg = {
        "n": int(sampling_params.n),
        "temperature": float(sampling_params.temperature),
        "top_p": float(sampling_params.top_p),
        "max_tokens": int(sampling_params.max_tokens),
    }
    stop_token_ids = getattr(sampling_params, "stop_token_ids", None)
    if stop_token_ids:
        sampling_cfg["stop_token_ids"] = list(stop_token_ids)
    seed = getattr(sampling_params, "seed", None)
    if seed is not None:
        sampling_cfg["seed"] = int(seed)
    return sampling_cfg


def _dp_worker_generate(
    model_path: str,
    tokenizer_path: str | None,
    max_tokens: int,
    sampling_cfg: dict,
    samples: list[dict],
    enable_thinking: bool,
    qwen3_no_think: bool = False,
) -> list[dict]:
    global _DP_WORKER_LLM
    global _DP_WORKER_LLM_KEY
    global _DP_WORKER_SAMPLING_PARAMS
    global _DP_WORKER_SAMPLING_KEY
    global _DP_WORKER_GPU_ID

    from vllm import LLM, SamplingParams

    if _DP_WORKER_GPU_ID is None:
        raise RuntimeError("DP worker GPU is not initialized.")

    llm_key = (model_path, tokenizer_path or model_path, max_tokens)
    if _DP_WORKER_LLM is None or _DP_WORKER_LLM_KEY != llm_key:
        _DP_WORKER_LLM = LLM(
            model=model_path,
            tokenizer=tokenizer_path or model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_tokens + 512,
        )
        _DP_WORKER_LLM_KEY = llm_key

    sampling_key = tuple(sorted(sampling_cfg.items()))
    if _DP_WORKER_SAMPLING_PARAMS is None or _DP_WORKER_SAMPLING_KEY != sampling_key:
        _DP_WORKER_SAMPLING_PARAMS = SamplingParams(**sampling_cfg)
        _DP_WORKER_SAMPLING_KEY = sampling_key

    if enable_thinking or not qwen3_no_think:
        messages_batch = [[{"role": "user", "content": s["prompt"]}] for s in samples]
    else:
        messages_batch = [
            [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": s["prompt"]},
            ]
            for s in samples
        ]

    outputs = _DP_WORKER_LLM.chat(messages_batch, _DP_WORKER_SAMPLING_PARAMS, use_tqdm=False)

    records = []
    for sample, output in zip(samples, outputs):
        for idx, completion in enumerate(output.outputs):
            records.append({
                "example_id": sample["example_id"],
                "prompt": sample["prompt"],
                "answer": sample["answer"],
                "completion_idx": idx,
                "response": completion.text,
            })
    return records


def _dp_worker_init(gpu_id: str, worker_index: int = 0):
    global _DP_WORKER_GPU_ID
    _DP_WORKER_GPU_ID = str(gpu_id)
    try:
        # Put each DP worker in its own process group so the parent can
        # reliably terminate any vLLM child processes during teardown.
        os.setsid()
    except OSError:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = _DP_WORKER_GPU_ID
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Assign a unique RPC base port per worker to avoid TCPStore port collision
    # when multiple vLLM instances start concurrently.
    os.environ["VLLM_RPC_BASE_PORT"] = str(50000 + worker_index * 100)


def _dp_worker_warmup(
    model_path: str,
    tokenizer_path: str | None,
    max_tokens: int,
    sampling_cfg: dict,
    enable_thinking: bool,
    qwen3_no_think: bool,
) -> str:
    warmup_sampling_cfg = dict(sampling_cfg)
    warmup_sampling_cfg["n"] = 1
    warmup_sampling_cfg["temperature"] = 0.0
    warmup_sampling_cfg["top_p"] = 1.0
    warmup_sampling_cfg["max_tokens"] = min(8, max_tokens)
    _dp_worker_generate(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
        sampling_cfg=warmup_sampling_cfg,
        samples=[{"example_id": -1, "prompt": "1+1=?", "answer": "2"}],
        enable_thinking=enable_thinking,
        qwen3_no_think=qwen3_no_think,
    )
    return "ok"


def _dp_worker_cleanup() -> str:
    """Cleanup LLM object in DP worker subprocess to allow graceful shutdown."""
    global _DP_WORKER_LLM
    global _DP_WORKER_LLM_KEY
    global _DP_WORKER_SAMPLING_PARAMS
    global _DP_WORKER_SAMPLING_KEY

    if _DP_WORKER_LLM is not None:
        del _DP_WORKER_LLM
        _DP_WORKER_LLM = None
        _DP_WORKER_LLM_KEY = None

    if _DP_WORKER_SAMPLING_PARAMS is not None:
        del _DP_WORKER_SAMPLING_PARAMS
        _DP_WORKER_SAMPLING_PARAMS = None
        _DP_WORKER_SAMPLING_KEY = None

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return "cleaned"


def _get_executor_processes(executor: ProcessPoolExecutor) -> list:
    return [
        proc for proc in getattr(executor, "_processes", {}).values()
        if proc is not None
    ]


def _wait_for_processes(processes: list, timeout_s: float) -> list:
    deadline = time.monotonic() + timeout_s
    alive = []
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        proc.join(remaining)
        if proc.is_alive():
            alive.append(proc)
    return alive


def _signal_processes(processes: list, sig: signal.Signals):
    for proc in processes:
        if not proc.is_alive():
            continue
        try:
            os.killpg(proc.pid, sig)
            continue
        except Exception:
            pass

        try:
            if sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except Exception:
            pass


def _shutdown_dp_executor(gpu_id: str, executor: ProcessPoolExecutor):
    processes = _get_executor_processes(executor)
    executor.shutdown(wait=False, cancel_futures=True)

    alive = _wait_for_processes(processes, timeout_s=10.0)
    if not alive:
        print(f"DP worker on GPU {gpu_id} exited cleanly.")
        return

    alive_pids = [str(proc.pid) for proc in alive]
    print(
        f"WARNING: DP worker on GPU {gpu_id} did not exit after cleanup; "
        f"sending SIGTERM to pids {', '.join(alive_pids)}"
    )
    _signal_processes(alive, signal.SIGTERM)
    alive = _wait_for_processes(alive, timeout_s=5.0)
    if not alive:
        print(f"DP worker on GPU {gpu_id} exited after SIGTERM.")
        return

    alive_pids = [str(proc.pid) for proc in alive]
    print(
        f"WARNING: DP worker on GPU {gpu_id} still alive after SIGTERM; "
        f"sending SIGKILL to pids {', '.join(alive_pids)}"
    )
    _signal_processes(alive, signal.SIGKILL)
    alive = _wait_for_processes(alive, timeout_s=5.0)
    if alive:
        alive_pids = [str(proc.pid) for proc in alive]
        print(
            f"WARNING: DP worker on GPU {gpu_id} is still alive after SIGKILL: "
            f"{', '.join(alive_pids)}"
        )
    else:
        print(f"DP worker on GPU {gpu_id} exited after SIGKILL.")


def _generate_completions_dp(
    dp_executors: list[tuple[str, ProcessPoolExecutor]],
    model_path: str,
    tokenizer_path: str | None,
    max_tokens: int,
    sampling_params,
    samples: list[dict],
    enable_thinking: bool,
    qwen3_no_think: bool,
) -> list[dict]:
    sampling_cfg = _sampling_params_to_dict(sampling_params)
    shards = _split_samples(samples, len(dp_executors))
    print(
        f"  Generating {sampling_params.n} completions for {len(samples)} prompts "
        f"across {len(shards)} DP shard(s) ..."
    )
    futures = []
    for shard_idx, shard in enumerate(shards):
        _, executor = dp_executors[shard_idx % len(dp_executors)]
        futures.append(executor.submit(
            _dp_worker_generate,
            model_path,
            tokenizer_path,
            max_tokens,
            sampling_cfg,
            shard,
            enable_thinking,
            qwen3_no_think,
        ))

    records = []
    for shard_idx, future in enumerate(futures, start=1):
        shard_records = future.result()
        records.extend(shard_records)
        print(
            f"    Collected shard {shard_idx}/{len(futures)}: "
            f"{len(shard_records)} completions"
        )
    return records


def grade_records(
    records: list[dict],
    scorer: str,
    dataset_name: str,
    math_equal_timeout: float,
) -> list[dict]:
    total = len(records)
    progress_interval = 1000 if total >= 1000 else max(1, total)
    print(f"  Grading {total} records for {dataset_name} with scorer={scorer} ...")
    start_time = time.monotonic()
    for idx, rec in enumerate(records, start=1):
        if scorer == "aime":
            score, pred_norm, gold_norm = score_aime_like(
                response=rec["response"],
                ground_truth=rec["answer"],
            )
        else:
            score, pred_norm, gold_norm = score_mathlike(
                response=rec["response"],
                ground_truth=rec["answer"],
                math_equal_timeout=math_equal_timeout,
            )
        rec["score"] = float(score)
        rec["pred_norm"] = pred_norm
        rec["gold_norm"] = gold_norm
        if idx % progress_interval == 0 or idx == total:
            elapsed = time.monotonic() - start_time
            print(f"    Graded {idx}/{total} records ({elapsed:.1f}s elapsed)")
    return records


def compute_metrics(scores_per_example: list[list[float]], k: int) -> dict:
    if not scores_per_example:
        return {"num_examples": 0, "num_pred": 0}

    num_examples = len(scores_per_example)
    num_pred = len(scores_per_example[0]) if scores_per_example[0] else 0
    out = {"num_examples": num_examples, "num_pred": num_pred}

    if num_pred >= 1:
        out["acc@1"] = sum(1.0 if s[0] >= 1.0 else 0.0 for s in scores_per_example) / num_examples

    for kk in sorted({1, 2, 4, 8, 32, k}):
        if num_pred >= kk:
            out[f"avg@{kk}"] = sum(sum(s[:kk]) / kk for s in scores_per_example) / num_examples
            out[f"pass@{kk}"] = sum(1.0 if max(s[:kk]) >= 1.0 else 0.0 for s in scores_per_example) / num_examples

    return out


def records_to_scores(records: list[dict], k: int) -> list[list[float]]:
    by_example: dict[int, list[tuple[int, float]]] = {}
    for rec in records:
        eid = rec["example_id"]
        by_example.setdefault(eid, []).append((rec["completion_idx"], rec["score"]))

    scores_per_example = []
    for eid in sorted(by_example):
        pairs = sorted(by_example[eid], key=lambda x: x[0])
        scores_per_example.append([p[1] for p in pairs[:k]])
    return scores_per_example


def save_detail_csv(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["example_id", "prompt", "answer", "completion_idx", "response", "score", "pred_norm", "gold_norm"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"  Detail CSV saved: {path}")


def save_summary_csv(all_metrics: dict[str, dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = set()
    for m in all_metrics.values():
        all_keys.update(m.keys())
    metric_keys = sorted(all_keys - {"dataset"})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset"] + metric_keys)
        writer.writeheader()
        for ds_name, metrics in all_metrics.items():
            row = {"dataset": ds_name, **metrics}
            writer.writerow(row)
    print(f"  Summary CSV saved: {path}")


def log_to_wandb(all_metrics: dict[str, dict], model_path: str, wandb_project: str, wandb_name: str | None):
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed, skipping wandb logging.")
        return

    if wandb_name is None:
        wandb_name = Path(model_path).name

    wandb.init(project=wandb_project, name=wandb_name, config={"model_path": model_path})

    flat_metrics = {}
    for ds_name, metrics in all_metrics.items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                flat_metrics[f"eval/{ds_name}/{key}"] = value

    wandb.log(flat_metrics)

    rows = []
    for ds_name, metrics in all_metrics.items():
        rows.append({"dataset": ds_name, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    table = wandb.Table(dataframe=pd.DataFrame(rows))
    wandb.log({"eval/summary_table": table})

    wandb.finish()
    print("  wandb logging done.")


def maybe_build_tokenizer_patch_dir(model_path: Path, output_dir: Path) -> str | None:
    """
    Build a local tokenizer override dir for vLLM when tokenizer_config has
    incompatible `extra_special_tokens` schema (list vs dict).
    """
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return None

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: failed to read tokenizer_config.json: {exc}")
        return None

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return None

    tokenizer_config["extra_special_tokens"] = {
        f"extra_special_token_{idx}": token
        for idx, token in enumerate(extra_special_tokens)
    }

    patch_dir = output_dir / ".vllm_tokenizer_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ]
    for filename in tokenizer_files:
        src = model_path / filename
        if src.exists():
            shutil.copy2(src, patch_dir / filename)

    (patch_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "Detected list-type extra_special_tokens in tokenizer_config.json; "
        f"using patched tokenizer dir: {patch_dir}"
    )
    return str(patch_dir)


def _append_token_ids(target: list[int], token_id) -> None:
    if token_id is None:
        return
    token_ids = token_id if isinstance(token_id, list) else [token_id]
    for item in token_ids:
        if item is not None and item not in target:
            target.append(int(item))


def infer_stop_token_ids(model_path: Path, tokenizer_path: str | None, qwen3: bool) -> list[int] | None:
    if qwen3:
        return list(QWEN3_STOP_TOKEN_IDS)

    source = tokenizer_path or str(model_path)
    stop_token_ids: list[int] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        _append_token_ids(stop_token_ids, tokenizer.eos_token_id)
        _append_token_ids(stop_token_ids, tokenizer.pad_token_id)
    except Exception as exc:
        print(f"WARNING: failed to infer stop ids from tokenizer {source!r}: {exc}")

    try:
        generation_config = GenerationConfig.from_pretrained(str(model_path), trust_remote_code=True)
        _append_token_ids(stop_token_ids, getattr(generation_config, "eos_token_id", None))
        _append_token_ids(stop_token_ids, getattr(generation_config, "pad_token_id", None))
    except Exception:
        pass

    return stop_token_ids or None


def build_llm_and_sampling_params(
    model_path: Path,
    tokenizer_path: str | None,
    tensor_parallel_size: int,
    k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    sampling_seed: int,
    stop_token_ids: list[int] | None,
):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tokenizer=tokenizer_path or str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_tokens + 512,
    )
    sampling_kwargs = {
        "n": k,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": sampling_seed,
    }
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = stop_token_ids
    sampling_params = SamplingParams(**sampling_kwargs)
    return llm, sampling_params


def parse_visible_gpu_ids() -> list[str]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        return [x.strip() for x in cuda_visible.split(",") if x.strip()]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Standalone eval pipeline with vLLM + heuristic grading.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained student model.")
    parser.add_argument("--datasets", type=str, default="aime24,aime25,amc23,minerva,math500",
                        help="Comma-separated dataset names.")
    parser.add_argument("--k", type=int, default=32, help="Number of completions per prompt for pass@k.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs for tensor parallel. Defaults to all visible GPUs.")
    parser.add_argument("--data_parallel_size", type=int, default=1,
                        help="Number of data-parallel vLLM workers (each worker loads a full model, TP=1).")
    parser.add_argument("--math_equal_timeout", type=float, default=30.0)
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qwen3", action="store_true", default=False,
                        help="Use legacy Qwen3 eval behavior, including hard-coded Qwen3 stop token ids.")
    parser.add_argument("--wandb_project", type=str, default="opd_distill_with_eval")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name. Defaults to model dir name.")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Defaults to <model_path>/eval_results.")
    args = parser.parse_args()

    if args.tensor_parallel_size is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible.strip():
            args.tensor_parallel_size = len([x for x in cuda_visible.split(",") if x.strip()])
        else:
            import torch
            args.tensor_parallel_size = torch.cuda.device_count()

    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = maybe_build_tokenizer_patch_dir(model_path, output_dir)
    qwen3_no_think = args.qwen3 and not args.enable_thinking
    stop_token_ids = infer_stop_token_ids(model_path, tokenizer_path, args.qwen3)
    print(f"Stop token ids: {stop_token_ids if stop_token_ids is not None else 'vLLM/model default'}")
    llm = None
    sampling_params = None
    dp_executors = None
    if args.data_parallel_size > 1:
        visible_gpu_ids = parse_visible_gpu_ids()
        if not visible_gpu_ids:
            raise RuntimeError("data_parallel_size > 1 requires visible CUDA GPUs.")
        dp_workers = min(args.data_parallel_size, len(visible_gpu_ids))
        dp_gpu_ids = visible_gpu_ids[:dp_workers]
        from vllm import SamplingParams
        sampling_kwargs = {
            "n": args.k,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.sampling_seed,
        }
        if stop_token_ids:
            sampling_kwargs["stop_token_ids"] = stop_token_ids
        sampling_params = SamplingParams(**sampling_kwargs)
        sampling_cfg = _sampling_params_to_dict(sampling_params)

        dp_executors = []
        for worker_index, gpu_id in enumerate(dp_gpu_ids):
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp.get_context("spawn"),
                initializer=_dp_worker_init,
                initargs=(gpu_id, worker_index),
            )
            dp_executors.append((gpu_id, executor))

        # Warm up DP workers sequentially to avoid TCPStore port collision
        # when multiple vLLM instances initialize concurrently.
        print(f"Warming up {len(dp_executors)} DP workers sequentially ...")
        for gpu_id, executor in dp_executors:
            future = executor.submit(
                _dp_worker_warmup,
                str(model_path),
                tokenizer_path,
                args.max_tokens,
                sampling_cfg,
                args.enable_thinking,
                qwen3_no_think,
            )
            future.result()
            print(f"DP worker on GPU {gpu_id} is ready.")

        args.tensor_parallel_size = 1
    else:
        llm, sampling_params = build_llm_and_sampling_params(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            k=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            sampling_seed=args.sampling_seed,
            stop_token_ids=stop_token_ids,
        )

    dataset_names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    print(f"Model: {model_path}")
    print(f"Datasets: {dataset_names}")
    print(f"K={args.k}, temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print(f"Math equal timeout: {args.math_equal_timeout}")
    print(f"Sampling seed: {args.sampling_seed}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Data parallel size: {args.data_parallel_size}")
    print(f"Output dir: {output_dir}")
    print()

    all_metrics = {}
    for ds_name in dataset_names:
        info = DATASET_REGISTRY.get(ds_name)
        if info is None:
            print(f"WARNING: Unknown dataset {ds_name!r}, skipping. Available: {sorted(DATASET_REGISTRY)}")
            continue

        parquet_path = EVAL_DATA_DIR / info["path"]
        if not parquet_path.exists():
            print(f"WARNING: Parquet not found at {parquet_path}, skipping {ds_name}.")
            continue

        print(f"=== {ds_name} ===")
        samples = load_samples(parquet_path)
        print(f"  {len(samples)} samples loaded")

        stage_start = time.monotonic()
        print("  [1/3] Starting generation ...")
        records = generate_completions(
            llm=llm,
            sampling_params=sampling_params,
            samples=samples,
            enable_thinking=args.enable_thinking,
            dp_executors=dp_executors,
            model_path=str(model_path),
            tokenizer_path=tokenizer_path,
            max_tokens=args.max_tokens,
            qwen3_no_think=qwen3_no_think,
        )
        print(
            f"  [1/3] Generation finished: {len(records)} completions "
            f"in {time.monotonic() - stage_start:.1f}s"
        )

        stage_start = time.monotonic()
        print("  [2/3] Starting grading ...")
        records = grade_records(
            records,
            scorer=info["scorer"],
            dataset_name=ds_name,
            math_equal_timeout=args.math_equal_timeout,
        )
        print(f"  [2/3] Grading finished in {time.monotonic() - stage_start:.1f}s")

        scores_per_example = records_to_scores(records, args.k)
        metrics = compute_metrics(scores_per_example, args.k)
        all_metrics[ds_name] = metrics

        print(f"  Metrics: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in metrics.items()} }")

        stage_start = time.monotonic()
        print("  [3/3] Writing detail CSV ...")
        save_detail_csv(records, output_dir / f"{ds_name}_details.csv")
        print(f"  [3/3] Detail CSV finished in {time.monotonic() - stage_start:.1f}s")

    save_summary_csv(all_metrics, output_dir / "eval_summary.csv")

    if not args.no_wandb:
        log_to_wandb(all_metrics, str(model_path), args.wandb_project, args.wandb_name)

    if dp_executors is not None:
        print("Cleaning up DP workers ...")
        cleanup_futures = []
        for gpu_id, executor in dp_executors:
            cleanup_futures.append((gpu_id, executor.submit(_dp_worker_cleanup)))

        for gpu_id, future in cleanup_futures:
            try:
                future.result(timeout=30)
                print(f"DP worker on GPU {gpu_id} cleaned.")
            except TimeoutError:
                print(f"WARNING: Cleanup timed out for DP worker on GPU {gpu_id}.")
            except Exception as e:
                print(f"WARNING: Failed to cleanup DP worker on GPU {gpu_id}: {e}")

        for gpu_id, executor in dp_executors:
            _shutdown_dp_executor(gpu_id, executor)

    print("\nDone.")


if __name__ == "__main__":
    main()
