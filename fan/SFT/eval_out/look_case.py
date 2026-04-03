python - <<'PY'
import pandas as pd

path = r"/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/eval_out/details/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/output/merged-checkpoint-6000/2026-03-23T16-25-45.241380/details_aime24|0_2026-03-23T16-25-45.241380.parquet"

df = pd.read_parquet(path)

for i, row in df.iterrows():
    doc = row["doc"]
    spec = doc.get("specific", {}) if isinstance(doc, dict) else {}

    pred = spec.get("extracted_predictions", [])
    gold = spec.get("extracted_golds", [])

    pred = pred[0] if len(pred) > 0 else None
    gold = gold[0] if len(gold) > 0 else None

    print(f"{i}\tpred={pred}\tgold={gold}")
PY


python - <<'PY'
import pandas as pd
from pprint import pprint

path = r"/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/eval_out/details/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/output/merged-checkpoint-6000/2026-03-23T16-25-45.241380/details_aime24|0_2026-03-23T16-25-45.241380.parquet"

df = pd.read_parquet(path)

for i, row in df.head(2).iterrows():
    doc = row["doc"]
    resp = row["model_response"]

    print(f"\n{'='*30} 第 {i} 题 {'='*30}")

    # 先看看 doc 里是什么结构
    if isinstance(doc, dict):
        question = doc.get("problem", doc.get("question", doc))
        answer = doc.get("answer", doc.get("solution", ""))
    else:
        question = doc
        answer = ""

    print("题目：")
    print(question)

    print("\n标准答案：")
    print(answer)

    print("\n模型答案：")
    print(resp)
PY