#!/usr/bin/env python3
"""
SFT training script for messages-format datasets (e.g., trl-lib/Capybara).
No dataset conversion or filtering is applied.
"""

import os

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
    assistant_token_ids = [tid for tid, m in zip(input_ids, assistant_masks) if m == 1]

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

    verify_assistant_mask(rank, tokenizer, debug_file)

    if rank == 0:
        print("Loading model...")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if rank == 0:
        print(f"Loading dataset: {script_args.dataset_name} ...")

    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )

    if rank == 0:
        print(f"Train dataset size (no filter): {len(dataset)}")
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
                print(f"Eval dataset size (no filter): {len(eval_dataset)}")
        except Exception as e:
            if rank == 0:
                print(f"Could not load eval dataset: {e}")

    if training_args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "cookbook_distillation")

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
