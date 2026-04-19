import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from training_scripts.train_lora_sft import (  # noqa: E402
    DEFAULT_RUN_NAME,
    default_output_dir,
    safe_name,
    validate_sft_row,
)


def test_lora_output_dir_uses_model_and_run_name():
    output_dir = default_output_dir("meta-llama/Meta-Llama-3.1-8B-Instruct")

    assert safe_name("meta-llama/Meta-Llama-3.1-8B-Instruct") == "meta-llama_meta-llama-3.1-8b-instruct"
    assert output_dir == (
        PROJECT_DIR
        / "outputs"
        / "sft"
        / "checkpoints"
        / f"meta-llama_meta-llama-3.1-8b-instruct__{DEFAULT_RUN_NAME}"
    )


def test_validate_sft_row_rejects_gold_answer_leak():
    row = {
        "example_id": 1,
        "messages": [
            {"role": "user", "content": "Known correct answer: B"},
            {"role": "assistant", "content": "Reasoning: x\nFinal answer: B"},
        ],
    }

    try:
        validate_sft_row(row)
    except AssertionError as exc:
        assert "Gold label leaked" in str(exc)
    else:
        raise AssertionError("Expected leaked gold answer to fail validation")
