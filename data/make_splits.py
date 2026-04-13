import json
import random

INPUT_PATH = "data/Race_x_SES.jsonl"

def save_jsonl(path, data):
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

def main():
    # Load local dataset
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    print(f"Original size: {len(data)}")

    #Keep only ambiguous examples
    data = [x for x in data if x["context_condition"] == "ambig"]

    print(f"Ambiguous only: {len(data)}")

    # Shuffle
    random.shuffle(data)

    # Split
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Save
    save_jsonl("data/train.jsonl", train_data)
    save_jsonl("data/test.jsonl", test_data)

    print("Saved train/test splits!")

if __name__ == "__main__":
    main()