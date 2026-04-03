from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()