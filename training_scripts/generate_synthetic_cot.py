import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


LETTERS = ["A", "B", "C"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--raw_output", required=True)
    parser.add_argument("--sft_output", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_invalid", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def label_letter(row):
    label = row["label"]
    assert label in {0, 1, 2}, f"Bad label for example {row.get('example_id')}: {label}"
    return LETTERS[label]


def unknown_pos_from_metadata(row):
    positions = []
    for i in range(3):
        answer_info = row["answer_info"][f"ans{i}"]
        assert len(answer_info) >= 2, f"Malformed answer_info for example {row.get('example_id')}: {answer_info}"
        if answer_info[1] == "unknown":
            positions.append(i)
    assert len(positions) == 1, f"Expected one unknown option for example {row.get('example_id')}: {positions}"
    return positions[0]


def validate_row(row):
    required = [
        "example_id",
        "group_id",
        "unknown_pos",
        "context_condition",
        "category",
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
    assert row["unknown_pos"] == unknown_pos_from_metadata(row), f"Bad unknown_pos for example {row['example_id']}"


def answer_text(row, idx):
    return row[f"ans{idx}"]


def teacher_user_prompt(row):
    correct = label_letter(row)
    correct_text = answer_text(row, row["label"])
    condition_instruction = (
        "This is an ambiguous example. The reasoning must first say that the context gives no direct evidence "
        "for the asked attribute. Then explain that neither named option is supported, and that race, ethnicity, "
        "SES, gender, or other demographics must not be used to guess. Conclude that the only supported answer "
        f"is the unknown/not-answerable option: {correct}. {correct_text}."
        if row["context_condition"] == "ambig"
        else
        "This is a disambiguated example. The reasoning must first identify the direct evidence in the context "
        "that answers the question. Then explain why that evidence supports the known correct option instead of "
        "abstaining or guessing from demographics."
    )

    return f"""You are creating concise supervised training data for a question-answering model.

The correct answer is already known. Your job is to write a short explanation for why the known correct answer is appropriate.

Rules:
- Do not introduce stereotypes or unsupported assumptions.
- Use only evidence from the context.
- Teach the decision rule: direct evidence supports an answer; missing evidence requires abstention.
- For ambiguous examples, explicitly reject demographic or stereotype-based inference.
- For disambiguated examples, cite the concrete contextual evidence.
- Keep the reasoning to 2-4 concise sentences.
- End with exactly: Final answer: {correct}

{condition_instruction}

Context:
{row["context"]}

Question:
{row["question"]}

Options:
A. {row["ans0"]}
B. {row["ans1"]}
C. {row["ans2"]}

Known correct answer: {correct}. {correct_text}

Write the output in this exact format:
Reasoning: <short explanation>
Final answer: {correct}"""


def student_user_prompt(row):
    return f"""Answer the following multiple choice question.

Context:
{row["context"]}

Question:
{row["question"]}

Options:
A. {row["ans0"]}
B. {row["ans1"]}
C. {row["ans2"]}"""


def apply_chat_template(tokenizer, user_prompt):
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_teacher_output(text):
    match = re.search(r"Final answer:\s*([ABC])\b", text, flags=re.IGNORECASE)
    if not match:
        return None, None

    final_answer = match.group(1).upper()
    reasoning = text[:match.start()].strip()
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    return reasoning, final_answer


def make_records(row, teacher_model, teacher_prompt, raw_output):
    reasoning, final_answer = parse_teacher_output(raw_output)
    expected = label_letter(row)
    valid = bool(reasoning) and final_answer == expected

    raw_record = {
        "example_id": row["example_id"],
        "group_id": row["group_id"],
        "context_condition": row["context_condition"],
        "question_polarity": row["question_polarity"],
        "category": row["category"],
        "additional_metadata": row["additional_metadata"],
        "context": row["context"],
        "question": row["question"],
        "ans0": row["ans0"],
        "ans1": row["ans1"],
        "ans2": row["ans2"],
        "answer_info": row["answer_info"],
        "label": row["label"],
        "label_letter": expected,
        "unknown_pos": row["unknown_pos"],
        "teacher_model": teacher_model,
        "teacher_prompt": teacher_prompt,
        "teacher_raw_output": raw_output,
        "teacher_reasoning": reasoning,
        "teacher_final_answer": final_answer,
        "valid": valid,
    }

    sft_record = None
    if valid:
        sft_record = {
            "messages": [
                {"role": "user", "content": student_user_prompt(row)},
                {"role": "assistant", "content": f"Reasoning: {reasoning}\nFinal answer: {expected}"},
            ],
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "context_condition": row["context_condition"],
            "question_polarity": row["question_polarity"],
            "label": row["label"],
            "label_letter": expected,
            "unknown_pos": row["unknown_pos"],
        }

    return raw_record, sft_record


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args = parse_args()
    input_path = Path(args.input)
    raw_output_path = Path(args.raw_output)
    sft_output_path = Path(args.sft_output)

    assert input_path.exists(), f"Missing input file: {input_path}"
    assert args.overwrite or not raw_output_path.exists(), f"Raw output exists: {raw_output_path}"
    assert args.overwrite or not sft_output_path.exists(), f"SFT output exists: {sft_output_path}"

    rows = read_jsonl(input_path)
    rows = rows if args.num_samples is None else rows[:args.num_samples]
    assert rows, "No input rows selected"
    for row in rows:
        validate_row(row)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm_kwargs = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    raw_records = []
    sft_records = []

    for start in tqdm(range(0, len(rows), args.batch_size)):
        batch_rows = rows[start:start + args.batch_size]
        teacher_prompts = [teacher_user_prompt(row) for row in batch_rows]
        model_prompts = [apply_chat_template(tokenizer, prompt) for prompt in teacher_prompts]
        outputs = llm.generate(model_prompts, sampling_params)

        for row, teacher_prompt, output in zip(batch_rows, teacher_prompts, outputs):
            raw_output = output.outputs[0].text.strip()
            raw_record, sft_record = make_records(row, args.model, teacher_prompt, raw_output)
            raw_records.append(raw_record)
            if sft_record is not None:
                sft_records.append(sft_record)

    invalid_records = [row for row in raw_records if not row["valid"]]

    write_jsonl(raw_output_path, raw_records)
    write_jsonl(sft_output_path, sft_records)

    print("\nSynthetic generation complete")
    print(f"Model: {args.model}")
    print(f"Input rows: {len(rows)}")
    print(f"Valid rows: {len(sft_records)}")
    print(f"Invalid rows: {len(invalid_records)}")
    print(f"Raw output: {raw_output_path}")
    print(f"SFT output: {sft_output_path}")

    if invalid_records[:5]:
        print("\nFirst invalid examples:")
        for row in invalid_records[:5]:
            print({
                "example_id": row["example_id"],
                "label_letter": row["label_letter"],
                "teacher_final_answer": row["teacher_final_answer"],
                "teacher_raw_output": row["teacher_raw_output"][:200],
            })

    assert args.allow_invalid or not invalid_records, f"Invalid teacher outputs: {len(invalid_records)}"


if __name__ == "__main__":
    main()
