import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from run_ablation import (  # noqa: E402
    compute_metrics,
    compute_metrics_by_context,
    is_abstain_option,
    map_prediction,
    unknown_pos_from_metadata,
)


def test_prediction_parsing_requires_option_letters():
    cases = {
        "A": 0,
        " A. The first option": 0,
        "b": 1,
        "C)": 2,
        "UNKNOWN": -1,
        "Undetermined": -1,
        "malformed text": -1,
        "": -1,
    }

    for text, expected in cases.items():
        assert map_prediction(text) == expected


def test_abstention_classification_uses_metadata_position():
    row = {
        "unknown_pos": 1,
        "answer_info": {
            "ans0": ["Person A", "group-a"],
            "ans1": ["Cannot be determined", "unknown"],
            "ans2": ["Person C", "group-c"],
        },
    }

    assert unknown_pos_from_metadata(row) == 1
    assert not is_abstain_option(row, 0)
    assert is_abstain_option(row, 1)
    assert not is_abstain_option(row, 2)
    assert not is_abstain_option(row, -1)


def test_compute_metrics_on_tiny_handcrafted_rows():
    rows = [
        {
            "context_condition": "ambig",
            "correct": True,
            "predicted_abstention": True,
            "predicted_commit": False,
            "prediction_idx": 1,
        },
        {
            "context_condition": "disambig",
            "correct": True,
            "predicted_abstention": False,
            "predicted_commit": True,
            "prediction_idx": 0,
        },
        {
            "context_condition": "ambig",
            "correct": False,
            "predicted_abstention": False,
            "predicted_commit": False,
            "prediction_idx": -1,
        },
    ]

    metrics = compute_metrics(rows)
    assert metrics["num_examples"] == 3
    assert metrics["accuracy"] == 2 / 3
    assert metrics["abstain_rate"] == 1 / 3
    assert metrics["commit_rate"] == 1 / 3
    assert metrics["invalid_rate"] == 1 / 3

    by_context = compute_metrics_by_context(rows)
    assert by_context["ambiguous"]["num_examples"] == 2
    assert by_context["ambiguous"]["accuracy"] == 1 / 2
    assert by_context["ambiguous"]["abstain_rate"] == 1 / 2
    assert by_context["ambiguous"]["invalid_rate"] == 1 / 2
    assert by_context["disambiguated"]["num_examples"] == 1
    assert by_context["disambiguated"]["accuracy"] == 1.0
    assert by_context["disambiguated"]["commit_rate"] == 1.0


def load_split(name):
    path = PROJECT_DIR / "data" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_saved_splits_have_no_group_leakage():
    splits = {name: load_split(name) for name in ["train", "dev", "test"]}
    group_ids = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}

    assert not (group_ids["train"] & group_ids["dev"])
    assert not (group_ids["train"] & group_ids["test"])
    assert not (group_ids["dev"] & group_ids["test"])


def test_saved_splits_keep_basic_context_and_polarity_balance():
    for split_name in ["train", "dev", "test"]:
        rows = load_split(split_name)
        context_counts = Counter(row["context_condition"] for row in rows)
        polarity_counts = Counter(row["question_polarity"] for row in rows)

        assert set(context_counts) == {"ambig", "disambig"}
        assert context_counts["ambig"] == context_counts["disambig"]
        assert set(polarity_counts) == {"neg", "nonneg"}
        assert polarity_counts["neg"] == polarity_counts["nonneg"]
