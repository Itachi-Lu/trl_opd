"""Smoke tests for the locally-cached DAPO-Math-17k-Processed dataset.

These tests exist to catch the failure mode that originally surfaced in
production:

    FileNotFoundError: Couldn't find any data file at
    .../trl/fan/opd/open-r1/DAPO-Math-17k-Processed.

That failure happened because compute nodes had no internet access and the
training script tried to resolve `open-r1/DAPO-Math-17k-Processed` against the
HF Hub.  The mitigation is to download the dataset to cephfs once and point
`--dataset_name` at the local path; these tests verify that the local copy is
loadable AND that the OPD training pipeline accepts it without modification.

Run locally with:
    cd trl && pytest tests/experimental/test_dapo_math_dataset.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]                     # .../lzh/trl
OPD_SCRIPT = REPO_ROOT / "fan" / "opd" / "opd_with_eval.py"

# Same physical cephfs share, two mount points:
# - production training node:  /apdcephfs_qy4/...
# - dev container (this box):  /apdcephfs/test/jp_qy4_cephfs/apdcephfs_qy4/...
# Override with env var DAPO_MATH_LOCAL_DIR if needed.
_CANDIDATE_DATA_DIRS = [
    Path(os.environ["DAPO_MATH_LOCAL_DIR"]) if "DAPO_MATH_LOCAL_DIR" in os.environ else None,
    Path("/apdcephfs_qy4/share_302593112/shaofanliu/data/DAPO-Math-17k-Processed"),
    Path("/apdcephfs/test/jp_qy4_cephfs/apdcephfs_qy4/share_302593112"
         "/shaofanliu/data/DAPO-Math-17k-Processed"),
]
LOCAL_DATA_DIR = next((p for p in _CANDIDATE_DATA_DIRS if p is not None and p.exists()), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_opd_with_eval():
    """Import `fan/opd/opd_with_eval.py` as a side-loaded module.

    We don't put `fan/opd` on PYTHONPATH globally because doing so pulls in a
    bunch of script-only imports.  Instead we side-load just the symbols we
    need to exercise the dataset-loading code path.

    On dev boxes some heavy deps (accelerate, deepspeed, vllm, ...) may be
    missing; in that case we skip rather than fail so the test suite stays
    green locally while still running fully on training nodes.
    """
    if not OPD_SCRIPT.exists():
        pytest.skip(f"opd_with_eval.py not found at {OPD_SCRIPT}")
    spec = importlib.util.spec_from_file_location("opd_with_eval", OPD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opd_with_eval"] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError, RuntimeError) as e:
        pytest.skip(f"opd_with_eval.py has unsatisfied transitive deps in this env: {e}")
    return module


def _force_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mimic the compute-node environment: no Internet, HF offline.

    If we can still load the dataset from disk under these constraints then
    we know the script will succeed on the training cluster.
    """
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def local_path() -> Path:
    if LOCAL_DATA_DIR is None:
        pytest.skip(
            "Local DAPO-Math-17k-Processed not found in any candidate path "
            f"({[str(p) for p in _CANDIDATE_DATA_DIRS if p is not None]}); "
            "set DAPO_MATH_LOCAL_DIR or run huggingface_hub.snapshot_download"
        )
    return LOCAL_DATA_DIR


def test_local_layout_matches_repo(local_path: Path) -> None:
    """The local snapshot must contain the same parquet layout as the HF repo."""
    expected = [
        local_path / "all" / "train-00000-of-00001.parquet",
        local_path / "en"  / "train-00000-of-00001.parquet",
        local_path / "cn"  / "train-00000-of-00001.parquet",
        local_path / "README.md",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    assert not missing, f"Local snapshot is incomplete; missing: {missing}"


def test_load_dataset_offline_default_config(local_path: Path, monkeypatch) -> None:
    """`load_dataset(local_path, 'all', split='train')` works with HF offline."""
    _force_offline_env(monkeypatch)
    from datasets import load_dataset

    ds = load_dataset(str(local_path), "all", split="train")
    assert ds.num_rows == 17398, f"row count drifted: {ds.num_rows}"
    expected_cols = {
        "prompt", "solution", "data_source", "source_prompt",
        "ability", "reward_model", "extra_info",
    }
    assert expected_cols.issubset(set(ds.column_names)), (
        f"missing columns; got {ds.column_names}"
    )


def test_each_config_loads(local_path: Path, monkeypatch) -> None:
    """`all` should equal `en + cn` row count; both subsets must be loadable."""
    _force_offline_env(monkeypatch)
    from datasets import load_dataset

    n_all = load_dataset(str(local_path), "all", split="train").num_rows
    n_en  = load_dataset(str(local_path), "en",  split="train").num_rows
    n_cn  = load_dataset(str(local_path), "cn",  split="train").num_rows
    assert n_all == n_en + n_cn, f"all={n_all} en={n_en} cn={n_cn}"


def test_prompts_are_non_empty_strings(local_path: Path, monkeypatch) -> None:
    _force_offline_env(monkeypatch)
    from datasets import load_dataset

    ds = load_dataset(str(local_path), "all", split="train")
    sample = ds.select(range(64))
    for row in sample:
        assert isinstance(row["prompt"], str) and len(row["prompt"]) > 0
        assert isinstance(row["reward_model"], dict)
        assert "ground_truth" in row["reward_model"]


def test_opd_pipeline_loads_local_dataset(local_path: Path, monkeypatch) -> None:
    """End-to-end: `_load_dataset_splits` + `_ensure_prompt_only_dataset` succeed
    with the local path and produce the chat-format `prompt` column the trainer
    expects.

    This is the test that would have caught the production failure: it exercises
    exactly the same code path used by `opd_with_eval.py main()`.
    """
    pytest.importorskip("torch", reason="opd_with_eval.py imports torch at module top")
    pytest.importorskip("transformers", reason="opd_with_eval.py imports transformers at module top")
    pytest.importorskip("pandas", reason="opd_with_eval.py imports pandas at module top")
    _force_offline_env(monkeypatch)
    opd = _import_opd_with_eval()

    args = opd.OPDScriptArguments(
        dataset_name=str(local_path),
        dataset_config="all",
        dataset_train_split="train",
        dataset_eval_split="none",
        enable_thinking=False,
    )
    train_ds, eval_ds = opd._load_dataset_splits(args)

    assert eval_ds is None
    assert train_ds.num_rows == 17398

    # `_ensure_prompt_only_dataset` must turn the string prompt into the
    # tinker / chat_template-friendly list-of-dict format.
    sample = train_ds[0]["prompt"]
    assert isinstance(sample, list) and len(sample) >= 1
    for turn in sample:
        assert isinstance(turn, dict)
        assert {"role", "content"}.issubset(turn.keys())
        assert isinstance(turn["content"], str)

    # When enable_thinking=False, a `/no_think` system turn must be prepended.
    roles = [t["role"] for t in sample]
    assert roles[0] == "system" and sample[0]["content"] == "/no_think", (
        f"expected /no_think system turn first, got {sample[:2]}"
    )


def test_chat_template_renders_with_real_tokenizer(local_path: Path, monkeypatch) -> None:
    """Optional: if a Qwen3 tokenizer is available locally, render the prompt
    through chat_template + check it tokenizes to a non-trivial sequence.

    This guards against subtle tokenizer-template incompatibilities (e.g.,
    `enable_thinking` flag interacting badly with the system turn we inject).
    """
    pytest.importorskip("torch", reason="opd_with_eval.py imports torch at module top")
    pytest.importorskip("transformers")
    _force_offline_env(monkeypatch)
    tok_path = "/apdcephfs_zwfy2/share_302970870/shaofanliu/models/Qwen3-0.6B"
    # Fall back to cephfs path under our mount root.
    alt = "/apdcephfs/test/jp_qy4_cephfs/apdcephfs_qy4/share_302970870/shaofanliu/models/Qwen3-0.6B"
    chosen = tok_path if os.path.exists(tok_path) else alt if os.path.exists(alt) else None
    if chosen is None:
        pytest.skip("no Qwen3 tokenizer cache available locally")

    from transformers import AutoTokenizer
    opd = _import_opd_with_eval()
    args = opd.OPDScriptArguments(
        dataset_name=str(local_path),
        dataset_config="all",
        dataset_train_split="train",
        dataset_eval_split="none",
        enable_thinking=False,
    )
    train_ds, _ = opd._load_dataset_splits(args)
    tokenizer = AutoTokenizer.from_pretrained(chosen, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        train_ds[0]["prompt"], tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assert len(rendered) > 100, f"chat template rendered too-short string: {rendered!r}"
    assert len(ids) > 32, f"tokenized prompt too short: {len(ids)} tokens"
