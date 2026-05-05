import argparse
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw jsonl file")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--test", default="data/processed/test.jsonl")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input)

    # filter minimal fields
    cleaned = []
    for r in rows:
        q = r.get("question_vi") or r.get("question")
        a = r.get("answer_vi") or r.get("answer")
        if not q or not a:
            continue
        cleaned.append({
            "id": r.get("id"),
            "question_vi": q.strip(),
            "answer_vi": a.strip(),
            "gold_docs": r.get("gold_docs", [])
        })

    train, test = train_test_split(cleaned, test_size=args.test_size, random_state=args.seed)
    save_jsonl(args.train, train)
    save_jsonl(args.test, test)

    print(f"Train: {len(train)} | Test: {len(test)}")


if __name__ == "__main__":
    main()
