"""Run with python -m src.explore; no notebook required."""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import ROOT, LABEL_NAMES, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "archive"))
    args = parser.parse_args()
    data, audit = load_dataset(args.data_dir)
    reports = ROOT / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    audit["word_count_summary"] = {
        LABEL_NAMES[label]: group.text.str.split().str.len().describe().to_dict()
        for label, group in data.groupby("label")
    }
    (reports / "exploration_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data.label.value_counts().sort_index().plot.bar(ax=axes[0], color=["#128f78", "#d66b47"])
    axes[0].set_xticklabels(LABEL_NAMES, rotation=0)
    axes[0].set_title("Clean dataset class counts")
    for label, name in enumerate(LABEL_NAMES):
        lengths = data.loc[data.label.eq(label), "text"].str.split().str.len()
        axes[1].hist(lengths.clip(upper=2000), bins=50, alpha=.5, label=name)
    axes[1].set_title("Article lengths (capped at 2,000 words)")
    axes[1].set_xlabel("Words")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(reports / "figures" / "dataset_overview.png", dpi=160)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
