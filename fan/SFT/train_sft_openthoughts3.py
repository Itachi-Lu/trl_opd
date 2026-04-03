#!/usr/bin/env python3
"""
SFT training script for OpenThoughts3 dataset using TRL SFTTrainer.
Supports multi-GPU distributed training with LoRA.

Extra debug features:
1. Inspect decoded full model input
2. Inspect decoded tokens that actually contribute to loss
3. Token-by-token view: token / input_id / label / contributes_to_loss
4. DEBUG_INSPECT_ONLY=1 to exit before training
"""

import os
import random

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)


def find_first_bad_example(dataset, tokenizer, debug_file, max_check=None):
    total = len(dataset) if max_check is None else min(len(dataset), max_check)

    for i in range(total):
        ex = dataset[i]
        messages = ex.get("messages", [])

        try:
            out = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=16384,
            )

            input_ids = out.get("input_ids", [])
            assistant_masks = out.get("assistant_masks", [])

            if not input_ids:
                append_to_debug_file(debug_file, f"\n[bad example found] idx={i}, reason=empty_input_ids")
                append_to_debug_file(debug_file, f"messages={messages}")
                return i

            if not assistant_masks:
                append_to_debug_file(debug_file, f"\n[bad example found] idx={i}, reason=empty_assistant_masks")
                append_to_debug_file(debug_file, f"messages={messages}")
                return i

            if sum(assistant_masks) == 0:
                append_to_debug_file(debug_file, f"\n[bad example found] idx={i}, reason=assistant_mask_sum_zero")
                append_to_debug_file(debug_file, f"messages={messages}")
                return i

            if i % 1000 == 0:
                print(f"checked {i}/{total}")

        except Exception as e:
            append_to_debug_file(debug_file, f"\n[bad example found] idx={i}, reason=exception")
            append_to_debug_file(debug_file, f"exception={type(e).__name__}: {e}")
            append_to_debug_file(debug_file, f"messages={messages}")
            return i

    append_to_debug_file(debug_file, "\nNo bad example found in checked range.")
    return None


def get_rank_world():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def append_to_debug_file(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def convert_role_values(example):
    role_mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
    }

    messages = []
    for conv in example.get("conversations", []):
        messages.append(
            {
                "role": role_mapping.get(conv.get("from"), conv.get("from")),
                "content": conv.get("value", ""),
            }
        )

    # 只保留 messages，不再保留 conversations
    return {
        "messages": messages,
        "difficulty": example.get("difficulty"),
        "source": example.get("source"),
        "domain": example.get("domain"),
    }

def is_valid_conversational_example(example):
    USER_MAX_CHARS = 20000
    # import random

    # a = random.random()   # [0, 1)
    # if a < 0.998:
    #     return False


    messages = example.get("messages", [])
    if not isinstance(messages, list) or len(messages) != 2:
        return False

    first, second = messages

    if first.get("role") != "user":
        return False
    if second.get("role") != "assistant":
        return False

    user_content = first.get("content", "")
    assistant_content = second.get("content", "")

    if not isinstance(user_content, str) or not user_content.strip():
        return False
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        return False

    # 过滤掉明显过长的 user 输入，避免 8192 token 截断后 assistant 全被截掉
    if len(user_content) > USER_MAX_CHARS:
        return False

    # 如果 assistant 里有 think 标记，要求 </think> 后面还要有真正回答
    if "</think>" in assistant_content:
        tail = assistant_content.split("</think>")[-1].strip()
        if not tail:
            return False

    return True

def print_basic_info(rank, world_size, script_args, training_args, model_args):
    if rank != 0:
        return

    print("=" * 100)
    print("SFT Training Configuration")
    print("=" * 100)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Output directory: {training_args.output_dir}")
    print(f"Max steps: {training_args.max_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"LoRA rank: {model_args.lora_r}")
    print(f"Per device batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(
        "Effective batch size: "
        f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}"
    )
    print(f"World size: {world_size}")
    print(f"Max length: {training_args.max_length}")
    print(f"Dataset num proc: {training_args.dataset_num_proc}")
    print(f"BF16: {training_args.bf16}")
    print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
    print("=" * 100)


def write_raw_samples_to_file(rank, dataset, title, debug_file, num_samples=2):
    if rank != 0:
        return

    append_to_debug_file(debug_file, "\n" + "=" * 100)
    append_to_debug_file(debug_file, title)
    append_to_debug_file(debug_file, "=" * 100)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        append_to_debug_file(debug_file, f"\n--- Raw sample {i} ---")

        if "messages" in sample:
            for j, msg in enumerate(sample["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                display_content = content[:500] + "..." if len(content) > 500 else content
                append_to_debug_file(debug_file, f"[{j}] {role}: {display_content}")
        else:
            append_to_debug_file(debug_file, str(sample))

    append_to_debug_file(debug_file, "=" * 100 + "\n")


def inspect_processed_batch_to_file(rank, trainer, tokenizer, debug_file, num_samples=2, max_show_tokens=200):
    if rank != 0:
        return

    append_to_debug_file(debug_file, "\n" + "=" * 100)
    append_to_debug_file(debug_file, "DEBUG: Inspect tokenized inputs and loss labels")
    append_to_debug_file(debug_file, "=" * 100)

    raw_examples = [trainer.train_dataset[i] for i in range(min(num_samples, len(trainer.train_dataset)))]
    batch = trainer.data_collator(raw_examples)

    input_ids_batch = batch["input_ids"]
    labels_batch = batch["labels"]
    attention_mask_batch = batch.get("attention_mask", None)

    for i in range(input_ids_batch.shape[0]):
        input_ids = input_ids_batch[i].tolist()
        labels = labels_batch[i].tolist()

        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        loss_token_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
        decoded_loss_text = tokenizer.decode(loss_token_ids, skip_special_tokens=False)

        append_to_debug_file(debug_file, f"\n{'-' * 100}")
        append_to_debug_file(debug_file, f"Sample {i}")
        append_to_debug_file(debug_file, f"{'-' * 100}")

        append_to_debug_file(debug_file, "\n[Decoded full input]")
        append_to_debug_file(debug_file, decoded_input)

        append_to_debug_file(debug_file, "\n[Decoded loss tokens only]")
        append_to_debug_file(debug_file, decoded_loss_text)

        append_to_debug_file(debug_file, "\n[First 200 input_ids]")
        append_to_debug_file(debug_file, str(input_ids[:200]))

        append_to_debug_file(debug_file, "\n[First 200 labels]")
        append_to_debug_file(debug_file, str(labels[:200]))

        if attention_mask_batch is not None:
            append_to_debug_file(debug_file, "\n[First 200 attention_mask]")
            append_to_debug_file(debug_file, str(attention_mask_batch[i].tolist()[:200]))

        append_to_debug_file(debug_file, "\n[Token-by-token view: token / id / label / contributes_to_loss]")
        max_show = min(max_show_tokens, len(input_ids))
        tokens = tokenizer.convert_ids_to_tokens(input_ids[:max_show])

        for tok, tid, lab in zip(tokens, input_ids[:max_show], labels[:max_show]):
            contributes = lab != -100
            append_to_debug_file(
                debug_file,
                f"{repr(tok):<24} id={tid:<8} label={lab:<8} loss={contributes}",
            )

    append_to_debug_file(debug_file, "=" * 100 + "\n")


def verify_assistant_mask(rank, tokenizer, debug_file):
    """
    最小验证：手工构造一条 user->assistant 样本，
    检查 assistant mask 是否真的生效。
    """
    test_messages = [
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
    ]

    out = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )

    input_ids = out.get("input_ids", [])
    assistant_masks = out.get("assistant_masks", [])

    assistant_token_ids = [
        tid for tid, m in zip(input_ids, assistant_masks) if m == 1
    ]

    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_assistant = tokenizer.decode(assistant_token_ids, skip_special_tokens=False)

    if rank == 0:
        append_to_debug_file(debug_file, "\n" + "=" * 100)
        append_to_debug_file(debug_file, "Minimal assistant-mask verification")
        append_to_debug_file(debug_file, "=" * 100)
        append_to_debug_file(debug_file, f"input_ids len: {len(input_ids)}")
        append_to_debug_file(debug_file, f"assistant_masks len: {len(assistant_masks)}")
        append_to_debug_file(debug_file, f"assistant mask sum: {sum(assistant_masks) if assistant_masks else 0}")
        append_to_debug_file(debug_file, "\n[decoded full text]")
        append_to_debug_file(debug_file, decoded_full)
        append_to_debug_file(debug_file, "\n[decoded assistant-only tokens]")
        append_to_debug_file(debug_file, decoded_assistant)
        append_to_debug_file(debug_file, "=" * 100 + "\n")

    if not input_ids:
        raise RuntimeError("Minimal assistant-mask verification failed: input_ids is empty.")
    if not assistant_masks:
        raise RuntimeError("Minimal assistant-mask verification failed: assistant_masks is empty.")
    if sum(assistant_masks) == 0:
        raise RuntimeError(
            "Minimal assistant-mask verification failed: assistant_masks sum is 0. "
            "The patched chat template is not producing assistant masks."
        )


def main(script_args, training_args, model_args):
    rank, world_size = get_rank_world()
    debug_inspect_only = os.environ.get("DEBUG_INSPECT_ONLY", "0") == "1"

    os.makedirs(training_args.output_dir, exist_ok=True)
    debug_file = os.path.join(training_args.output_dir, "debug_token_ids.txt")

    if rank == 0:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("")

    print_basic_info(rank, world_size, script_args, training_args, model_args)

    if rank == 0:
        print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    with open("./chatemplate_patch/original_chat_template.jinja", "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先做最小验证，不急着全量 filter
    verify_assistant_mask(rank, tokenizer, debug_file)

    if rank == 0:
        print("Loading model...")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    if rank == 0:
        print(f"Loading dataset: {script_args.dataset_name} ...")

    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )

    if rank == 0:
        print(f"Train dataset size: {len(dataset)}")
        if hasattr(dataset, "features"):
            print(f"Train dataset features: {dataset.features}")

    eval_dataset = None
    if training_args.eval_strategy != "no":
        try:
            eval_dataset = load_dataset(
                script_args.dataset_name,
                name=script_args.dataset_config,
                split=script_args.dataset_test_split,
            )
            if rank == 0:
                print(f"Eval dataset size: {len(eval_dataset)}")
        except Exception as e:
            if rank == 0:
                print(f"Could not load eval dataset: {e}")

    if training_args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "cookbook_distillation")

    if rank == 0:
        print("Converting role values (human->user, gpt->assistant)...")

    dataset = dataset.map(
        convert_role_values,
        num_proc=training_args.dataset_num_proc,
        remove_columns=["conversations"],
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(convert_role_values, num_proc=training_args.dataset_num_proc)

    if rank == 0:
        print(f"Train dataset size before filter: {len(dataset)}")

    dataset = dataset.filter(is_valid_conversational_example, num_proc=training_args.dataset_num_proc)

    # if rank == 0:
    #     bad_idx = find_first_bad_example(dataset, tokenizer, debug_file, max_check=None)
    #     print("first bad idx =", bad_idx)
    #     if bad_idx is not None:
    #         raise RuntimeError(f"Found bad example before SFTTrainer at idx={bad_idx}. See debug file.")

    if rank == 0:
        print(f"Train dataset size after filter: {len(dataset)}")

    if eval_dataset is not None:
        if rank == 0:
            print(f"Eval dataset size before filter: {len(eval_dataset)}")
        eval_dataset = eval_dataset.filter(is_valid_conversational_example, num_proc=training_args.dataset_num_proc)
        if rank == 0:
            print(f"Eval dataset size after filter: {len(eval_dataset)}")

    # write_raw_samples_to_file(
    #     rank,
    #     dataset,
    #     "First 2 raw samples after conversion to messages format",
    #     debug_file,
    #     num_samples=2,
    # )

    if rank == 0:
        print("Initializing SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    inspect_processed_batch_to_file(
        rank,
        trainer,
        tokenizer,
        debug_file,
        num_samples=2,
        max_show_tokens=200,
    )

    if rank == 0:
        print(f"Debug info written to: {debug_file}")

    if debug_inspect_only:
        if rank == 0:
            print("DEBUG_INSPECT_ONLY=1, exiting before training.")
        return

    if rank == 0:
        print("Starting training...")

    trainer.train()

    if rank == 0:
        print("Saving model...")
        trainer.save_model(training_args.output_dir)
        print(f"Model saved to: {training_args.output_dir}")
        print("Training completed!")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
