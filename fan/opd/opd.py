from __future__ import annotations

import os
import sys
from types import MethodType
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trl import ModelConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.chat_template_utils import qwen3_training_chat_template
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer


def constant_reward(completions: list[str], **kwargs) -> list[float]:
    return [1.0] * len(completions)


def _attach_token_debug_dump(trainer: MiniLLMTrainer, output_dir: str) -> None:
    original_compute_loss = trainer.compute_loss
    debug_path = Path(output_dir) / "token_debug.txt"
    dumped = {"done": False}

    def compute_loss_with_debug(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not dumped["done"]:
            prompt_ids = inputs["prompt_ids"]
            completion_ids = inputs["completion_ids"]
            prompt_mask = inputs["prompt_mask"]
            completion_mask = inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            full_ids = input_ids[0].detach().cpu().tolist()
            full_mask = attention_mask[0].detach().cpu().tolist()
            completion_mask_row = completion_mask[0].detach().cpu().tolist()
            prompt_len = prompt_ids.shape[1]
            tokenizer = self.processing_class.tokenizer if hasattr(self.processing_class, "tokenizer") else self.processing_class
            token_strings = tokenizer.convert_ids_to_tokens(full_ids)
            decoded_text = tokenizer.decode(full_ids, skip_special_tokens=False)
            loss_mask = [0] * prompt_len + completion_mask_row
            loss_only_ids = [token_id for token_id, loss in zip(full_ids, loss_mask, strict=True) if loss == 1]
            decoded_loss_only_text = tokenizer.decode(loss_only_ids, skip_special_tokens=False)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("w", encoding="utf-8") as f:
                f.write("Decoded full sequence\n")
                f.write(decoded_text)
                f.write("\n\n")
                f.write("Decoded loss-masked sequence\n")
                f.write(decoded_loss_only_text)
                f.write("\n\n")
                f.write("Per-token view\n")
                f.write("idx\tinput_id\ttoken\tattention_mask\tloss_mask\n")
                for idx, (token_id, token_str, attn, loss) in enumerate(
                    zip(full_ids, token_strings, full_mask, loss_mask, strict=True)
                ):
                    safe_token = repr(token_str)
                    f.write(f"{idx}\t{token_id}\t{safe_token}\t{attn}\t{loss}\n")
            dumped["done"] = True

        return original_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    trainer.compute_loss = MethodType(compute_loss_with_debug, trainer)


@dataclass
class OPDScriptArguments:
    dataset_name: str = field(
        default="trl-lib/DeepMath-103K",
        metadata={"help": "Dataset repo or local path. Expected to contain a prompt-only split."},
    )
    dataset_config: str | None = field(
        default=None,
        metadata={"help": "Optional dataset config name passed to datasets.load_dataset."},
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Training split name or slicing expression."},
    )
    dataset_eval_split: str | None = field(
        default="test",
        metadata={"help": "Evaluation split name or slicing expression. Use `none` to disable eval loading."},
    )
    teacher_model_name_or_path: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Teacher model identifier or local path."},
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional tokenizer identifier or local path. If unset, fall back to model_name_or_path-based logic."
        },
    )
    groups_per_batch: int = field(
        default=512,
        metadata={"help": "Thinker-style number of prompt groups per batch."},
    )
    group_size: int = field(
        default=4,
        metadata={"help": "Thinker-style number of rollouts sampled per prompt group."},
    )
    wandb_project: str = field(
        default="cookbook_distillation",
        metadata={"help": "Weights & Biases project name."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Optional number of workers for dataset preprocessing."},
    )
    enable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether prompts should allow thinking mode. If false, a `/no_think` system message is prepended."
        },
    )
    tinker_single_update: bool = field(
        default=True,
        metadata={
            "help": "If true, reshape TRL batching so one rollout batch is consumed in a single optimizer update, "
            "which is closer to tinker-cookbook's default OPD dynamics."
        },
    )


def _ensure_prompt_only_dataset(
    dataset: Dataset, dataset_num_proc: int | None = None, enable_thinking: bool = True
) -> Dataset:
    column_names = set(dataset.column_names)
    if "prompt" in column_names:
        if enable_thinking:
            return dataset

        def add_no_think(example):
            prompt = example["prompt"]
            if not isinstance(prompt, list):
                return example
            if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
                if prompt[0].get("content") == "/no_think":
                    return example
                return {"prompt": [{"role": "system", "content": "/no_think"}, *prompt[1:]]}
            return {"prompt": [{"role": "system", "content": "/no_think"}, *prompt]}

        return dataset.map(add_no_think, num_proc=dataset_num_proc)

    def build_prompt(content: str) -> list[dict[str, str]]:
        if enable_thinking:
            return [{"role": "user", "content": content}]
        return [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": content},
        ]

    if "prompt" in column_names:
        return dataset

    if "question" in column_names:
        source_key = "question"
    elif "problem" in column_names:
        source_key = "problem"
    else:
        raise ValueError(
            "Dataset must contain a `prompt` column, or a `question`/`problem` column that can be converted into one. "
            f"Found columns: {sorted(column_names)}"
        )

    def make_prompt(example):
        return {"prompt": build_prompt(example[source_key])}

    return dataset.map(make_prompt, num_proc=dataset_num_proc)


def _load_dataset_splits(script_args: OPDScriptArguments) -> tuple[Dataset, Dataset | None]:
    split_names: list[str] = [script_args.dataset_train_split]
    load_eval = script_args.dataset_eval_split is not None and script_args.dataset_eval_split.lower() != "none"
    if load_eval:
        split_names.append(script_args.dataset_eval_split)

    loaded = load_dataset(script_args.dataset_name, script_args.dataset_config, split=split_names)
    if isinstance(loaded, DatasetDict):
        train_dataset = loaded[script_args.dataset_train_split]
        eval_dataset = loaded[script_args.dataset_eval_split] if load_eval else None
    elif isinstance(loaded, list):
        train_dataset = loaded[0]
        eval_dataset = loaded[1] if load_eval else None
    else:
        train_dataset = loaded
        eval_dataset = None

    train_dataset = _ensure_prompt_only_dataset(
        train_dataset, script_args.dataset_num_proc, script_args.enable_thinking
    )
    if eval_dataset is not None:
        eval_dataset = _ensure_prompt_only_dataset(
            eval_dataset, script_args.dataset_num_proc, script_args.enable_thinking
        )
    return train_dataset, eval_dataset


def _resolve_dtype(dtype_name: str | None) -> torch.dtype | str | None:
    if dtype_name in ["auto", None]:
        return dtype_name
    return getattr(torch, dtype_name)


def _resolve_processing_class_name(model_name_or_path: str) -> str:
    # Qwen3 base checkpoints share the same vocabulary with the chat model, but the chat tokenizer ships
    # the turn-ending EOS configuration (<|im_end|>) that better matches our conversational template.
    if model_name_or_path.startswith("Qwen/Qwen3-") and model_name_or_path.endswith("-Base"):
        return model_name_or_path[: -len("-Base")]
    return model_name_or_path


if __name__ == "__main__":
    parser = TrlParser((OPDScriptArguments, MiniLLMConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    os.environ.setdefault("WANDB_PROJECT", script_args.wandb_project)

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

    world_size = max(training_args.world_size, 1)
    total_generations_per_batch = script_args.groups_per_batch * script_args.group_size

    if script_args.tinker_single_update:
        if total_generations_per_batch % world_size != 0:
            raise ValueError(
                "For tinker_single_update=True, groups_per_batch * group_size "
                f"({total_generations_per_batch}) must be divisible by world_size ({world_size})."
            )
        local_train_batch = total_generations_per_batch // world_size
        training_args.per_device_train_batch_size = local_train_batch
        training_args.gradient_accumulation_steps = 1
        training_args.generation_batch_size = total_generations_per_batch
        training_args.steps_per_generation = 1
    else:
        global_train_batch = training_args.per_device_train_batch_size * world_size
        if training_args.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be >= 1 when tinker_single_update=False.")
        if training_args.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1 when tinker_single_update=False.")
        if total_generations_per_batch % global_train_batch != 0:
            raise ValueError(
                "groups_per_batch * group_size "
                f"({total_generations_per_batch}) must be divisible by the global train batch size "
                f"({global_train_batch} = per_device_train_batch_size * world_size)."
            )
        training_args.generation_batch_size = total_generations_per_batch
        training_args.steps_per_generation = total_generations_per_batch // global_train_batch

    training_args.num_generations = script_args.group_size
    if training_args.max_completion_length is None:
        training_args.max_completion_length = 4096

    if not script_args.enable_thinking:
        chat_template_kwargs = dict(training_args.chat_template_kwargs or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        training_args.chat_template_kwargs = chat_template_kwargs

    training_args.rkl_advantage = True
    training_args.single_step_decomposition = False
    training_args.gamma = 0.0
    training_args.length_normalization = True
    training_args.beta = 0.0
    training_args.scale_rewards = "none"
    training_args.num_iterations = 1
    training_args.temperature = 1.0

    train_dataset, eval_dataset = _load_dataset_splits(script_args)
    tokenizer_name_or_path = (
        script_args.tokenizer_name_or_path.strip() if script_args.tokenizer_name_or_path else None
    )
    if tokenizer_name_or_path:
        processing_class_name = tokenizer_name_or_path
    else:
        processing_class_name = _resolve_processing_class_name(model_args.model_name_or_path)
    processing_class = AutoTokenizer.from_pretrained(
        processing_class_name,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
        truncation_side="left",
    )
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    trainer = MiniLLMTrainer(
        model=model_args.model_name_or_path,
        teacher_model=script_args.teacher_model_name_or_path,
        reward_funcs=constant_reward,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=get_peft_config(model_args),
    )
    if not script_args.enable_thinking:
        trainer.chat_template = qwen3_training_chat_template
    _attach_token_debug_dump(trainer, training_args.output_dir)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
