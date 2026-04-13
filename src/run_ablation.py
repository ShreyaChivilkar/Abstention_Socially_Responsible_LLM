import json
import os
import argparse
from tqdm import tqdm
from datetime import datetime

from prompts import BASE_PROMPT


# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--engine", choices=["transformers", "vllm"], default="transformers")
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--num_samples", type=int, default=None,
                    help="Number of samples to run (default: all)")

args = parser.parse_args()


# ----------------------------
# Dynamic engine import
# ----------------------------
if args.engine == "transformers":
    from transformer_engine import TransformerEngine
    engine = TransformerEngine(args.model)

elif args.engine == "vllm":
    from vllm_engine import VLLMEngine
    engine = VLLMEngine(args.model)
else:
    raise ValueError(f"Unknown engine: {args.engine}")


# ----------------------------
# Prompt formatting
# ----------------------------
def format_prompt(row):
    return BASE_PROMPT.format(
        question=row["question"],
        A=row["ans0"],
        B=row["ans1"],
        C=row["ans2"],
    )


# ----------------------------
# Extract answer
# ----------------------------
def extract_answer(text):
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()


# ----------------------------
# Mapping
# ----------------------------
def map_prediction(pred, row):
    pred_lower = pred.lower()

    if any(x in pred_lower for x in ["unknown", "not", "can't"]):
        for i in range(3):
            if any(k in row[f"ans{i}"].lower() for k in ["unknown", "not", "can't"]):
                return i

    pred = pred.strip().upper()

    if pred.startswith("A"):
        return 0
    elif pred.startswith("B"):
        return 1
    elif pred.startswith("C"):
        return 2

    return -1


def is_abstain_option(row, idx):
    if idx == -1:
        return False
    text = row[f"ans{idx}"].lower()
    return any(x in text for x in ["unknown", "not", "can't"])


# ----------------------------
# Main
# ----------------------------
def run():
    with open("data/test.jsonl") as f:
        data = [json.loads(line) for line in f]

    results = []

    correct = 0
    abstain_count = 0
    commit_count = 0

    data_to_run = data if args.num_samples is None else data[:args.num_samples]

    for row in tqdm(data_to_run):
        prompt = format_prompt(row)

        output = engine.generate(prompt)

        # vLLM returns (text, logprobs)
        if args.engine == "vllm":
            raw_output, logprobs = output
        else:
            raw_output = output
            logprobs = None

        clean_output = extract_answer(raw_output)

        pred_idx = map_prediction(clean_output, row)
        label = row["label"]

        is_correct = (pred_idx == label)

        if is_correct:
            correct += 1

        if pred_idx != -1:
            if is_abstain_option(row, pred_idx):
                abstain_count += 1
            else:
                commit_count += 1

        results.append({
            "question": row["question"],
            "prediction_raw": raw_output,
            "prediction_clean": clean_output,
            "prediction_idx": pred_idx,
            "label": label,
            "correct": is_correct,
            "engine": args.engine,
            "logprobs_available": logprobs is not None
        })

    total = len(results)
    accuracy = correct / total if total else 0
    abstain_rate = abstain_count / total
    commit_rate = commit_count / total

    print("\n RESULTS")
    print(f"Engine: {args.engine}")
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Abstain rate: {abstain_rate:.3f}")
    print(f"Commit rate: {commit_rate:.3f}")

    # Save
    os.makedirs("outputs", exist_ok=True)

    safe_model = args.model.replace("/", "_")
    output_path = f"outputs/{args.engine}_{safe_model}.json"

    final_output = {
        "metadata": {
            "engine": args.engine,
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "num_samples": total
        },
        "metrics": {
            "accuracy": accuracy,
            "abstain_rate": abstain_rate,
            "commit_rate": commit_rate
        },
        "predictions": results
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n Saved to {output_path}")


if __name__ == "__main__":
    run()