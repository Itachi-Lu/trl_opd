import argparse

from trl.experimental.gkd import GKDConfig, GKDTrainer
from datasets import load_dataset
from transformers import AutoTokenizer


def to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": str(example["answer"])},
        ]
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OPD with GKDTrainer on DeepScaleR train + AIME2024 eval.")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--train_dataset", type=str, default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--eval_dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--output_dir", type=str, default="/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/gkd_opd/gkd-opd-qwen3")

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=200)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--rollout_debug", action="store_true")
    parser.add_argument("--rollout_debug_steps", type=int, default=50)
    parser.add_argument("--rollout_debug_num_samples", type=int, default=1)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_dataset(args.train_dataset, split="train")
    eval_ds = load_dataset(args.eval_dataset, split="train")

    train_ds = train_ds.map(to_messages, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_messages, remove_columns=eval_ds.column_names)

    training_args = GKDConfig(
        output_dir=args.output_dir,
        model_init_kwargs={"dtype": "bfloat16", "use_cache": False},
        teacher_model_name_or_path=args.teacher_model,
        teacher_model_init_kwargs={"dtype": "bfloat16", "use_cache": True},
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lmbda=1.0,
        beta=1.0,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        do_eval=args.eval_strategy != "no",
        report_to=args.report_to,
        remove_unused_columns=False,
        rollout_debug=args.rollout_debug,
        rollout_debug_steps=args.rollout_debug_steps,
        rollout_debug_num_samples=args.rollout_debug_num_samples,
    )

    trainer = GKDTrainer(
        model=args.student_model,
        teacher_model=args.teacher_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
