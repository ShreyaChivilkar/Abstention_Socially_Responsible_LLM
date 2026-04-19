import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from prompts import BASE_PROMPT


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to run (default: all)")
    return parser.parse_args()


def load_engine(engine_name, model_name):
    if engine_name == "transformers":
        from transformer_engine import TransformerEngine
        return TransformerEngine(model_name)

    if engine_name == "vllm":
        from vllm_engine import VLLMEngine
        return VLLMEngine(model_name)

    raise ValueError(f"Unknown engine: {engine_name}")


# ----------------------------
# Dataset helpers
# ----------------------------
def unknown_pos_from_metadata(row):
    positions = []
    for i in range(3):
        answer_info = row["answer_info"][f"ans{i}"]
        assert len(answer_info) >= 2, f"Malformed answer_info for ans{i}: {answer_info}"
        if answer_info[1] == "unknown":
            positions.append(i)

    assert len(positions) == 1, f"Expected exactly one unknown option for example {row.get('example_id')}: {positions}"
    return positions[0]


def validate_row(row):
    required = [
        "example_id",
        "group_id",
        "unknown_pos",
        "context_condition",
        "category",
        "question_index",
        "question_polarity",
        "context",
        "question",
        "ans0",
        "ans1",
        "ans2",
        "answer_info",
        "additional_metadata",
        "label",
    ]
    missing = [key for key in required if key not in row]
    assert not missing, f"Missing required fields for example {row.get('example_id')}: {missing}"
    assert row["context_condition"] in {"ambig", "disambig"}, f"Bad context_condition: {row['context_condition']}"
    assert row["label"] in {0, 1, 2}, f"Bad label for example {row['example_id']}: {row['label']}"
    assert row["unknown_pos"] == unknown_pos_from_metadata(row), f"Bad unknown_pos for example {row['example_id']}"


# ----------------------------
# Prompt formatting
# ----------------------------
def format_prompt(row):
    validate_row(row)

    return BASE_PROMPT.format(
        context=row["context"],
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
def map_prediction(pred):
    pred = pred.strip().upper()

    if pred.startswith("A"):
        return 0
    elif pred.startswith("B"):
        return 1
    elif pred.startswith("C"):
        return 2

    return -1


def prediction_letter(pred_idx):
    if pred_idx == -1:
        return None
    return ["A", "B", "C"][pred_idx]


def is_abstain_option(row, idx):
    if idx == -1:
        return False
    return idx == row["unknown_pos"]


def compute_metrics(rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    abstain_count = sum(row["predicted_abstention"] for row in rows)
    commit_count = sum(row["predicted_commit"] for row in rows)
    invalid_count = sum(row["prediction_idx"] == -1 for row in rows)

    return {
        "num_examples": total,
        "accuracy": correct / total if total else 0,
        "abstain_rate": abstain_count / total if total else 0,
        "commit_rate": commit_count / total if total else 0,
        "invalid_rate": invalid_count / total if total else 0,
    }


def compute_metrics_by_context(rows):
    return {
        "overall": compute_metrics(rows),
        "ambiguous": compute_metrics([row for row in rows if row["context_condition"] == "ambig"]),
        "disambiguated": compute_metrics([row for row in rows if row["context_condition"] == "disambig"]),
    }


# ----------------------------
# Main
# ----------------------------
def run(args):
    engine = load_engine(args.engine, args.model)
    input_path = DATA_DIR / f"{args.split}.jsonl"
    assert input_path.exists(), f"Missing split file: {input_path}"

    with open(input_path) as f:
        data = [json.loads(line) for line in f]

    results = []
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

        pred_idx = map_prediction(clean_output)
        pred_letter = prediction_letter(pred_idx)
        label = row["label"]
        label_letter = prediction_letter(label)
        label_is_abstention = is_abstain_option(row, label)
        predicted_abstention = pred_idx != -1 and is_abstain_option(row, pred_idx)
        predicted_commit = pred_idx != -1 and not predicted_abstention
        is_correct = pred_idx == label

        results.append({
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "unknown_pos": row["unknown_pos"],
            "context_condition": row["context_condition"],
            "category": row["category"],
            "question_index": row["question_index"],
            "question_polarity": row["question_polarity"],
            "context": row["context"],
            "question": row["question"],
            "ans0": row["ans0"],
            "ans1": row["ans1"],
            "ans2": row["ans2"],
            "answer_info": row["answer_info"],
            "additional_metadata": row["additional_metadata"],
            "prediction_raw": raw_output,
            "prediction_clean": clean_output,
            "prediction_idx": pred_idx,
            "prediction_letter": pred_letter,
            "predicted_abstention": predicted_abstention,
            "predicted_commit": predicted_commit,
            "label": label,
            "label_letter": label_letter,
            "label_is_abstention": label_is_abstention,
            "correct": is_correct,
            "engine": args.engine,
            "logprobs_available": logprobs is not None
        })

    metrics = compute_metrics_by_context(results)

    print("\nRESULTS")
    print(f"Engine: {args.engine}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")

    for name, slice_metrics in metrics.items():
        print(f"\n{name}")
        print(f"Examples: {slice_metrics['num_examples']}")
        print(f"Accuracy: {slice_metrics['accuracy']:.3f}")
        print(f"Abstain rate: {slice_metrics['abstain_rate']:.3f}")
        print(f"Commit rate: {slice_metrics['commit_rate']:.3f}")
        print(f"Invalid rate: {slice_metrics['invalid_rate']:.3f}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    safe_model = args.model.replace("/", "_")
    output_path = OUTPUT_DIR / f"{args.split}_{args.engine}_{safe_model}.json"

    final_output = {
        "metadata": {
            "engine": args.engine,
            "model": args.model,
            "split": args.split,
            "input_path": str(input_path),
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(results)
        },
        "metrics": metrics,
        "predictions": results
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run(parse_args())
