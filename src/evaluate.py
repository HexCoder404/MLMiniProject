"""Generate classification reports, error samples, metrics and confusion matrices."""
import json
import pickle
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score)

from src.data import ROOT, LABEL_NAMES


def metrics(y, prediction, scores=None):
    result = {
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
        "fake_precision": precision_score(y, prediction, zero_division=0),
        "fake_recall": recall_score(y, prediction, zero_division=0),
        "fake_f1": f1_score(y, prediction, zero_division=0),
    }
    if scores is not None:
        result.update(roc_auc=roc_auc_score(y, scores), average_precision=average_precision_score(y, scores))
    return result


def evaluate_saved():
    reports = ROOT / "reports"
    test = pd.read_csv(reports / "test_articles.csv", keep_default_na=False)
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    (reports / "classification_reports").mkdir(exist_ok=True)
    (reports / "figures").mkdir(exist_ok=True)
    for name, info in manifest["models"].items():
        path = ROOT / "models" / info["file"]
        with path.open("rb") as stream:
            model = pickle.load(stream)
        start = perf_counter()
        predicted = model.predict(test.article)
        seconds = perf_counter() - start
        scores = (model.decision_function(test.article) if hasattr(model, "decision_function")
                  else model.predict_proba(test.article)[:, 1])
        row = {"model": name, **metrics(test.label, predicted, scores),
               "prediction_ms_per_article": 1000 * seconds / len(test),
               "training_seconds": info["training_seconds"], "model_size_mb": path.stat().st_size / 1024**2}
        rows.append(row)
        report = classification_report(test.label, predicted, labels=[0, 1], target_names=LABEL_NAMES,
                                       zero_division=0, output_dict=True)
        folder = reports / "classification_reports"
        (folder / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (folder / f"{name}.txt").write_text(classification_report(
            test.label, predicted, labels=[0, 1], target_names=LABEL_NAMES, digits=4, zero_division=0), encoding="utf-8")
        display = ConfusionMatrixDisplay.from_predictions(test.label, predicted, labels=[0, 1],
                                                         display_labels=LABEL_NAMES, cmap="Blues")
        display.ax_.set_title(name.replace("_", " "))
        display.figure_.tight_layout()
        display.figure_.savefig(reports / "figures" / f"{name}_confusion_matrix.png", dpi=160)
        plt.close(display.figure_)
        output = test[["source_row", "title", "label"]].copy()
        output["prediction"] = predicted
        output["decision_score"] = scores
        output.to_csv(reports / f"{name}_test_predictions.csv", index=False)
        errors = output.loc[output.label.ne(output.prediction)].copy()
        errors["article_excerpt"] = test.loc[errors.index, "article"].str[:1200]
        errors.to_csv(reports / f"{name}_errors.csv", index=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    comparison.set_index("model")[["accuracy", "macro_f1", "fake_precision", "fake_recall"]].plot.bar(ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out test score")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "model_comparison.png", dpi=160)
    plt.close(fig)
    write_analysis(comparison, manifest)
    print(comparison.round(4).to_string(index=False))
    return comparison


def write_analysis(comparison, manifest):
    experiments = pd.read_csv(ROOT / "reports" / "feature_experiments.csv")
    combined = experiments.loc[experiments.linguistic]
    chosen = combined.loc[combined.k.astype(str).eq(str(manifest["selected_k"]))].iloc[0]
    baseline = combined.loc[combined.k.astype(str).eq("all")].iloc[0]
    text_only = experiments.loc[~experiments.linguistic].iloc[0]
    lines = ["# Model performance analysis", "",
             f"The deployment model is **{manifest['best_model']}**, selected using validation macro F1 before test evaluation.", "",
             "## Held-out test results", "", comparison.round(4).to_string(index=False), "",
             "## Why performance differs", ""]
    by_name = comparison.set_index("model")
    for name, explanation in {
        "linear_svm": "A linear boundary can aggregate thousands of weak word and phrase signals. Scaling the linguistic features prevents raw article lengths from dominating the margin.",
        "random_forest": "Each tree searches a random subset of sparse features. Many absent terms and the restricted tree depth can make it harder to combine weak text signals. Trees can capture nonlinear linguistic patterns.",
        "xgboost": "Boosted trees sequentially correct errors and learn interactions. Histogram bins, depth, regularization and the limited boosting budget influence both generalization and cost.",
    }.items():
        lines.append(f"- {name}: test macro F1 = {by_name.loc[name, 'macro_f1']:.4f}. {explanation}")
    lines += ["", "These are plausible explanations, not proof of causation. Model families have different capacity despite equal candidate counts and folds.",
              "", "## Feature experiments (validation only)", "",
              experiments.round(4).to_string(index=False),
              "", f"Chosen term count: {manifest['selected_k']}. Its validation macro F1 differs from the all-terms baseline by {chosen.macro_f1 - baseline.macro_f1:+.6f}. Adding linguistic features at the chosen term count changes validation macro F1 by {chosen.macro_f1 - text_only.macro_f1:+.6f}.",
              "", "Compare selected TF-IDF with all terms for selection effects, and the chosen TF-IDF-only variant with the corresponding linguistic variant for added-feature effects. These experiments use Linear SVM; they do not establish the effect for every algorithm.",
              "", "## Dataset and error limitations", "",
              "Subject and date metadata are excluded. Duplicate bodies and identical 200-word prefixes are removed before splitting. This heuristic cannot detect every paraphrase or syndicated article.",
              "", "The true.csv examples contain Reuters attribution, while the subject categories differ entirely between files. Random article splits can reward source/style recognition. High scores here do not establish accuracy on current news, new publishers, other languages or factual claims. A source-disjoint or chronological external dataset is needed to measure that generalization.",
              "", "The per-model *_errors.csv files include misclassified titles and excerpts for inspection. False positives can flag genuine articles; false negatives can miss potentially fake articles. ROC AUC uses decision scores; these scores are not calibrated confidence probabilities."]
    top_terms = pd.read_csv(ROOT / "reports" / "feature_scores.csv").head(15)
    lines += ["", "## Highest chi-square terms", "", top_terms.round(4).to_string(index=False),
              "", "Chi-square measures association with the labels; it does not establish that a word is evidence of misinformation, and this table does not show the direction of the association."]
    exploration_path = ROOT / "reports" / "exploration_summary.json"
    if exploration_path.exists():
        exploration = json.loads(exploration_path.read_text(encoding="utf-8"))
        if exploration.get("clean_rows") == manifest["dataset"]["clean_rows"]:
            markers = exploration.get("reuters_marker_fraction_by_label", {})
            if markers:
                lines += ["", "## Source-marker diagnostic", "",
                          f"In the raw dataset, Reuters appears in {markers['Genuine']:.2%} of genuine articles and {markers['Potentially fake']:.2%} of fake-labeled articles. This label-associated source cue is a concrete reason to avoid interpreting near-perfect article-split scores as factual verification."]
    (ROOT / "reports" / "performance_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evaluate_saved()
