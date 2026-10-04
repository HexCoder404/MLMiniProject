"""Dataset loading, duplicate auditing and reproducible leakage-aware splits."""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text, normalize_content

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = ["Genuine", "Potentially fake"]
SEED = 42


def load_dataset(data_dir=ROOT / "archive"):
    frames = []
    for filename, label in (("true.csv", 0), ("fake.csv", 1)):
        path = Path(data_dir) / filename
        frame = pd.read_csv(path)
        if not {"title", "text"}.issubset(frame.columns):
            raise ValueError(f"{path} must contain title and text columns")
        frame["label"] = label
        frame["source_row"] = [f"{filename}:{i}" for i in range(len(frame))]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    audit = {"raw_rows": len(data), "raw_class_counts": data.label.value_counts().sort_index().to_dict()}
    if "subject" in data.columns:
        audit["subject_counts_by_label"] = {
            LABEL_NAMES[label]: group.subject.fillna("unknown").value_counts().to_dict()
            for label, group in data.groupby("label")
        }
    audit["reuters_marker_fraction_by_label"] = {
        LABEL_NAMES[label]: float(group.text.fillna("").str.contains(r"\breuters\b", case=False, regex=True).mean())
        for label, group in data.groupby("label")
    }
    data["title"] = data.title.fillna("").map(clean_text)
    data["text"] = data.text.fillna("").map(clean_text)
    # Require article content: a headline alone is outside this project's scope.
    empty = data.text.eq("")
    audit["empty_articles_removed"] = int(empty.sum())
    data = data.loc[~empty].copy()
    normalized = data.text.map(normalize_content)
    data["content_hash"] = normalized.map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    conflicts = data.groupby("content_hash").label.nunique()
    conflict_ids = conflicts[conflicts > 1].index
    audit["conflicting_label_rows_removed"] = int(data.content_hash.isin(conflict_ids).sum())
    data = data.loc[~data.content_hash.isin(conflict_ids)].copy()
    before = len(data)
    data = data.drop_duplicates("content_hash")
    audit["exact_duplicates_removed"] = before - len(data)
    # Conservative near-duplicate heuristic: identical normalized first 200 words.
    # Remove repeats before splitting; this is not a complete semantic duplicate detector.
    data["prefix_hash"] = data.text.map(normalize_content).map(
        lambda t: hashlib.sha256(" ".join(t.split()[:200]).encode()).hexdigest()
    )
    conflict_prefix = data.groupby("prefix_hash").label.nunique()
    bad_prefix = conflict_prefix[conflict_prefix > 1].index
    audit["conflicting_prefix_rows_removed"] = int(data.prefix_hash.isin(bad_prefix).sum())
    data = data.loc[~data.prefix_hash.isin(bad_prefix)]
    before = len(data)
    data = data.drop_duplicates("prefix_hash").reset_index(drop=True)
    audit["repeated_prefix_rows_removed"] = before - len(data)
    data["article"] = data.title + "\n" + data.text
    audit["clean_rows"] = len(data)
    audit["clean_class_counts"] = data.label.value_counts().sort_index().to_dict()
    return data, audit


def split_dataset(data):
    train, other = train_test_split(data, test_size=.30, stratify=data.label, random_state=SEED)
    validation, test = train_test_split(other, test_size=.50, stratify=other.label, random_state=SEED)
    for left, right in ((train, validation), (train, test), (validation, test)):
        assert set(left.prefix_hash).isdisjoint(right.prefix_hash)
        assert set(left.content_hash).isdisjoint(right.content_hash)
    return train, validation, test
