from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer, GenerationConfig


TOKEN_ID_KEYS = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")


def _resolve_qwen3_chat_tokenizer_name(model_name_or_path: str) -> str:
    if model_name_or_path.startswith("Qwen/Qwen3-") and model_name_or_path.endswith("-Base"):
        return model_name_or_path[: -len("-Base")]
    return model_name_or_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _infer_tokenizer_source(checkpoint_dir: Path) -> str:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not infer tokenizer source because {config_path} does not exist. "
            "Pass --tokenizer-name-or-path explicitly."
        )

    config = _read_json(config_path)
    model_name_or_path = config.get("_name_or_path") or config.get("name_or_path")
    if not isinstance(model_name_or_path, str) or not model_name_or_path:
        raise ValueError(
            f"Could not infer tokenizer source from {config_path}. "
            "Pass --tokenizer-name-or-path explicitly."
        )

    return _resolve_qwen3_chat_tokenizer_name(model_name_or_path)


def _sync_model_config(
    checkpoint_dir: Path,
    tokenizer,
    tokenizer_source: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return {}

    checkpoint_config = _read_json(config_path)
    source_config = AutoConfig.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)

    updates: dict[str, Any] = {}
    for key in TOKEN_ID_KEYS:
        value = getattr(source_config, key, None)
        if value is None and hasattr(tokenizer, key):
            value = getattr(tokenizer, key)
        if value is not None:
            checkpoint_config[key] = value
            updates[key] = value

    if updates:
        _write_json(config_path, checkpoint_config)
    return updates


def _sync_generation_config(
    checkpoint_dir: Path,
    tokenizer,
    tokenizer_source: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    generation_config_path = checkpoint_dir / "generation_config.json"
    checkpoint_generation_config = _read_json(generation_config_path) if generation_config_path.exists() else {}

    try:
        source_generation_config = GenerationConfig.from_pretrained(
            tokenizer_source, trust_remote_code=trust_remote_code
        )
    except Exception:
        source_generation_config = None

    updates: dict[str, Any] = {}
    for key in TOKEN_ID_KEYS:
        value = getattr(source_generation_config, key, None) if source_generation_config is not None else None
        if value is None and hasattr(tokenizer, key):
            value = getattr(tokenizer, key)
        if value is not None:
            checkpoint_generation_config[key] = value
            updates[key] = value

    if updates:
        _write_json(generation_config_path, checkpoint_generation_config)
    return updates


def patch_checkpoint_tokenizer(
    checkpoint_dir: Path,
    tokenizer_name_or_path: str | None = None,
    trust_remote_code: bool = False,
) -> None:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint_dir}")

    tokenizer_source = tokenizer_name_or_path or _infer_tokenizer_source(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.save_pretrained(checkpoint_dir)

    config_updates = _sync_model_config(checkpoint_dir, tokenizer, tokenizer_source, trust_remote_code)
    generation_config_updates = _sync_generation_config(checkpoint_dir, tokenizer, tokenizer_source, trust_remote_code)

    print(f"Patched checkpoint tokenizer in: {checkpoint_dir}")
    print(f"Tokenizer source: {tokenizer_source}")
    print(f"eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"bos_token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    if config_updates:
        print(f"Updated config.json token fields: {config_updates}")
    if generation_config_updates:
        print(f"Updated generation_config.json token fields: {generation_config_updates}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace tokenizer files inside a saved checkpoint directory with a Qwen3 tokenizer and align "
            "token-related config fields."
        )
    )
    parser.add_argument("checkpoint_dir", type=Path, help="Checkpoint directory to patch in place.")
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default="/apdcephfs_qy4_302593112/share_302593112/shaofanliu/projects/lzh/trl/fan/opd/outputs/opd-qwen3-1.7b-from-8b-paper/checkpoint-100",
        help=(
            "Tokenizer source to copy from. If omitted, the script tries to infer it from checkpoint config.json "
            "and maps Qwen/Qwen3-*-Base to Qwen/Qwen3-* automatically."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer/config.",
    )
    args = parser.parse_args()

    patch_checkpoint_tokenizer(
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
