import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).parent
INPUT_PATH = DATA_DIR / "Race_x_SES.jsonl"
SEED = 2026
SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}


def save_jsonl(path, data):
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def get_unknown_pos(row):
    positions = []
    for i in range(3):
        answer_info = row["answer_info"][f"ans{i}"]
        assert len(answer_info) >= 2, f"Malformed answer_info for example {row.get('example_id')}: {answer_info}"
        if answer_info[1] == "unknown":
            positions.append(i)

    assert len(positions) == 1, f"Expected exactly one unknown option for example {row.get('example_id')}: {positions}"
    return positions[0]


def get_group_key(row):
    required = ["category", "question_index", "answer_info", "additional_metadata"]
    missing = [key for key in required if key not in row]
    assert not missing, f"Missing required fields for example {row.get('example_id')}: {missing}"

    metadata = row["additional_metadata"]
    assert "subcategory" in metadata, f"Missing subcategory for example {row.get('example_id')}"

    answer_info = row["answer_info"]
    for i in range(3):
        assert f"ans{i}" in answer_info, f"Missing ans{i} metadata for example {row.get('example_id')}"

    return (
        row["category"],
        row["question_index"],
        metadata["subcategory"],
        metadata.get("source"),
        tuple(answer_info["ans0"]),
        tuple(answer_info["ans1"]),
        tuple(answer_info["ans2"]),
    )


def split_group_ids(group_ids, rng):
    group_ids = list(group_ids)
    rng.shuffle(group_ids)

    n_total = len(group_ids)
    n_train = round(SPLIT_RATIOS["train"] * n_total)
    n_dev = round(SPLIT_RATIOS["dev"] * n_total)

    train_ids = group_ids[:n_train]
    dev_ids = group_ids[n_train:n_train + n_dev]
    test_ids = group_ids[n_train + n_dev:]

    return train_ids, dev_ids, test_ids


def print_distribution(name, rows):
    group_ids = {row["group_id"] for row in rows}
    counts = Counter((row["additional_metadata"]["subcategory"], row["unknown_pos"]) for row in rows)

    print(f"\n{name}:")
    print(f"  rows: {len(rows)}")
    print(f"  groups: {len(group_ids)}")
    print("  counts by subcategory and unknown_pos:")
    for key, count in sorted(counts.items()):
        print(f"    {key}: {count}")


def main():
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    print(f"Total rows: {len(data)}")

    raw_group_keys = sorted({get_group_key(row) for row in data})
    group_id_by_key = {key: idx for idx, key in enumerate(raw_group_keys)}

    groups = defaultdict(list)
    group_strata = {}

    for row in data:
        group_key = get_group_key(row)
        group_id = group_id_by_key[group_key]
        unknown_pos = get_unknown_pos(row)
        subcategory = row["additional_metadata"]["subcategory"]
        stratum = (subcategory, unknown_pos)

        if group_id in group_strata:
            assert group_strata[group_id] == stratum, f"Group {group_id} has inconsistent strata"
        else:
            group_strata[group_id] = stratum

        row = dict(row)
        row["group_id"] = group_id
        row["unknown_pos"] = unknown_pos
        groups[group_id].append(row)

    print(f"Total groups: {len(groups)}")

    groups_by_stratum = defaultdict(list)
    for group_id, stratum in group_strata.items():
        groups_by_stratum[stratum].append(group_id)

    rng = random.Random(SEED)
    split_group_ids_by_name = {"train": [], "dev": [], "test": []}

    for stratum, group_ids in sorted(groups_by_stratum.items()):
        train_ids, dev_ids, test_ids = split_group_ids(group_ids, rng)
        split_group_ids_by_name["train"].extend(train_ids)
        split_group_ids_by_name["dev"].extend(dev_ids)
        split_group_ids_by_name["test"].extend(test_ids)

    train_group_ids = set(split_group_ids_by_name["train"])
    dev_group_ids = set(split_group_ids_by_name["dev"])
    test_group_ids = set(split_group_ids_by_name["test"])

    train_dev_overlap = train_group_ids & dev_group_ids
    train_test_overlap = train_group_ids & test_group_ids
    dev_test_overlap = dev_group_ids & test_group_ids
    leakage = bool(train_dev_overlap or train_test_overlap or dev_test_overlap)

    assert not leakage, "Group leakage detected across splits"
    assert train_group_ids | dev_group_ids | test_group_ids == set(groups), "Some groups were not assigned to a split"

    splits = {
        "train": [row for group_id in sorted(train_group_ids) for row in groups[group_id]],
        "dev": [row for group_id in sorted(dev_group_ids) for row in groups[group_id]],
        "test": [row for group_id in sorted(test_group_ids) for row in groups[group_id]],
    }

    assert sum(len(rows) for rows in splits.values()) == len(data), "Row count changed during splitting"

    for split_name, rows in splits.items():
        save_jsonl(DATA_DIR / f"{split_name}.jsonl", rows)

    print(f"Seed: {SEED}")
    print(f"Ratios: {SPLIT_RATIOS}")

    for split_name in ["train", "dev", "test"]:
        print_distribution(split_name, splits[split_name])

    print(f"\nGroup leakage across splits: {leakage}")
    print("Saved train/dev/test splits.")


if __name__ == "__main__":
    main()
