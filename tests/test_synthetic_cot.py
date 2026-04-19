import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from training_scripts.generate_synthetic_cot import (  # noqa: E402
    make_records,
    parse_teacher_output,
    teacher_user_prompt,
)


def tiny_row(context_condition="ambig"):
    return {
        "example_id": 123,
        "group_id": 7,
        "unknown_pos": 2,
        "context_condition": context_condition,
        "question_polarity": "neg",
        "category": "Race_x_SES",
        "context": "Alex and Jordan were mentioned, but the story does not say who was late.",
        "question": "Who was late?",
        "ans0": "Alex",
        "ans1": "Jordan",
        "ans2": "Cannot be determined",
        "answer_info": {
            "ans0": ["Alex", "person"],
            "ans1": ["Jordan", "person"],
            "ans2": ["Cannot be determined", "unknown"],
        },
        "additional_metadata": {"subcategory": "x", "source": "unit"},
        "label": 2,
    }


def test_parse_teacher_output_extracts_reasoning_and_final_answer():
    reasoning, final_answer = parse_teacher_output(
        "Reasoning: The context does not identify who was late.\nFinal answer: C"
    )

    assert reasoning == "The context does not identify who was late."
    assert final_answer == "C"


def test_make_records_keeps_teacher_raw_and_sft_student_prompt_clean():
    row = tiny_row()
    teacher_prompt = teacher_user_prompt(row)
    raw_output = "Reasoning: The context does not identify who was late.\nFinal answer: C"

    raw_record, sft_record = make_records(row, "teacher/model", teacher_prompt, raw_output)

    assert raw_record["valid"]
    assert raw_record["teacher_prompt"] == teacher_prompt
    assert raw_record["teacher_raw_output"] == raw_output
    assert sft_record["messages"][1]["content"].endswith("Final answer: C")
    assert "Known correct answer" not in sft_record["messages"][0]["content"]
