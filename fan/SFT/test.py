from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-8B-Base",
    trust_remote_code=True,
)

with open("qwen3_original_full_template.jinja", "w", encoding="utf-8") as f:
    f.write(tokenizer.chat_template)

print("saved to qwen3_original_full_template.jinja")