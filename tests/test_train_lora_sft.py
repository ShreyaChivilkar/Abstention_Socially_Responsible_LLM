import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from training_scripts.train_lora_sft import (  # noqa: E402
    DEFAULT_RUN_NAME,
    default_output_dir,
    render_example,
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


def test_render_example_preserves_token_type_ids_when_present():
    class FakeTokenizer:
        pad_token_id = 0
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt:
                return "prompt"
            return "full"

        def __call__(self, text, **kwargs):
            if text == "prompt":
                return {"input_ids": [10, 11]}

            return {
                "input_ids": [10, 11, 12, 13],
                "attention_mask": [1, 1, 1, 1],
                "token_type_ids": [0, 0, 1, 1],
            }

    row = {
        "example_id": 7,
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Reasoning: r\nFinal answer: A"},
        ],
    }

    rendered = render_example(FakeTokenizer(), row, max_seq_length=16)
    assert rendered["token_type_ids"] == [0, 0, 1, 1]
    assert rendered["labels"][:2] == [-100, -100]
