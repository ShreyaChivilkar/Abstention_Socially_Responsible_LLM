import argparse
import json
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT_DIR / "outputs" / "synthetic" / "sft_data" / "train__gemma-4-31b-it__full_v2_sft.jsonl"
DEFAULT_RUN_NAME = "gemma-4-31b-v2-lora"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def safe_name(text):
    safe = []
    for char in text.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {".", "-"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def default_output_dir(model_name):
    return ROOT_DIR / "outputs" / "sft" / "checkpoints" / f"{safe_name(model_name)}__{DEFAULT_RUN_NAME}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def validate_sft_row(row):
    assert "messages" in row, f"Missing messages for example {row.get('example_id')}"
    assert len(row["messages"]) == 2, f"Expected 2 messages for example {row.get('example_id')}"
    roles = [message["role"] for message in row["messages"]]
    assert roles == ["user", "assistant"], f"Bad message roles for example {row.get('example_id')}: {roles}"
    assert "Known correct answer" not in row["messages"][0]["content"], f"Gold label leaked in user prompt: {row.get('example_id')}"
    assert "Reasoning:" in row["messages"][1]["content"], f"Missing reasoning for example {row.get('example_id')}"
    assert "Final answer:" in row["messages"][1]["content"], f"Missing final answer for example {row.get('example_id')}"


def load_rows(path, max_samples):
    rows = read_jsonl(path)
    rows = rows if max_samples is None else rows[:max_samples]
    assert rows, f"No rows loaded from {path}"
    for row in rows:
        validate_sft_row(row)
    return rows


def render_example(tokenizer, row, max_seq_length):
    user_message = row["messages"][:1]
    full_messages = row["messages"]

    prompt_text = tokenizer.apply_chat_template(
        user_message,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    assert any(label != -100 for label in labels), (
        f"All assistant tokens were masked or truncated for example {row.get('example_id')}. "
        f"Increase --max_seq_length."
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_id": row.get("example_id"),
    }


def package_versions():
    versions = {}
    for package in ["torch", "transformers", "peft"]:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", None)
        except ImportError:
            versions[package] = None
    return versions


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_file = Path(args.train_file)
    assert train_file.exists(), f"Missing train file: {train_file}"

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    assert args.overwrite or not output_dir.exists(), f"Output dir already exists: {output_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(train_file, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template is not None, f"Tokenizer has no chat template: {args.model}"

    tokenized_rows = [render_example(tokenizer, row, args.max_seq_length) for row in rows]

    class SFTDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    dataloader = DataLoader(
        SFTDataset(tokenized_rows),
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    batches_per_epoch = len(dataloader)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_optimizer_steps = max(1, math.ceil(args.epochs * optimizer_steps_per_epoch))
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    print("\nSFT CONFIG")
    print(f"Model: {args.model}")
    print(f"Train file: {train_file}")
    print(f"Rows: {len(rows)}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.per_device_batch_size}")
    print(f"Grad accumulation: {args.gradient_accumulation_steps}")
    print(f"Optimizer steps: {total_optimizer_steps}")
    print(f"Warmup steps: {warmup_steps}")

    completed_optimizer_steps = 0
    running_loss = 0.0
    losses_since_log = 0
    logged_losses = []
    optimizer.zero_grad(set_to_none=True)

    epoch = 0
    progress = tqdm(total=total_optimizer_steps)
    while completed_optimizer_steps < total_optimizer_steps:
        epoch += 1
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation_steps
            losses_since_log += 1

            should_step = step % args.gradient_accumulation_steps == 0 or step == len(dataloader)
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_optimizer_steps += 1
            progress.update(1)

            if completed_optimizer_steps % args.logging_steps == 0 or completed_optimizer_steps == 1:
                avg_loss = running_loss / losses_since_log
                logged_losses.append({
                    "optimizer_step": completed_optimizer_steps,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                })
                print(logged_losses[-1])
                running_loss = 0.0
                losses_since_log = 0

            if completed_optimizer_steps >= total_optimizer_steps:
                break
    progress.close()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    train_args = vars(args).copy()
    train_args["output_dir"] = str(output_dir)
    train_args["train_file"] = str(train_file)

    with open(output_dir / "training_args.json", "w") as f:
        json.dump({
            "args": train_args,
            "git_commit": get_git_commit(),
            "package_versions": package_versions(),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump({
            "num_train_rows": len(rows),
            "num_optimizer_steps": completed_optimizer_steps,
            "logged_losses": logged_losses,
        }, f, indent=2)

    print(f"\nSaved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
