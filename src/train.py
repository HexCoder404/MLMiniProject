"""Train three models, select features, tune on training folds and evaluate once."""
import argparse
import json
import pickle
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.data import ROOT, SEED, load_dataset, split_dataset
from src.evaluate import metrics, evaluate_saved
from src.features import make_features


def make_pipeline(classifier, k=6000, linguistic=True):
    return Pipeline([("features", make_features(k, linguistic)), ("classifier", classifier)])


def save_pickle(value, path):
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "archive"))
    parser.add_argument("--cv-rows", type=int, default=6000,
                        help="Training-only stratified tuning sample; 0 means full training set")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse-feature-experiments", action="store_true",
                        help="Reuse completed feature experiments after an interrupted run; requires identical splits")
    args = parser.parse_args()
    if args.cv_rows < 0 or (0 < args.cv_rows < 30) or args.threads < 1:
        parser.error("--cv-rows must be 0 or at least 30; --threads must be positive")
    reports, models = ROOT / "reports", ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    data, audit = load_dataset(args.data_dir)
    train, validation, test = split_dataset(data)
    print(f"Clean articles: {len(data)}; train={len(train)}, validation={len(validation)}, test={len(test)}", flush=True)
    audit["split_sizes"] = {"train": len(train), "validation": len(validation), "test": len(test)}
    (reports / "dataset_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    split_rows = []
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        rows = split[["source_row", "content_hash", "prefix_hash", "label"]].copy()
        rows["split"] = name
        split_rows.append(rows)
    split_manifest = pd.concat(split_rows).reset_index(drop=True)
    if args.reuse_feature_experiments:
        previous = pd.read_csv(reports / "split_manifest.csv")
        if not previous.equals(split_manifest):
            raise ValueError("Cannot reuse feature experiments: dataset/splits have changed")
    split_manifest.to_csv(reports / "split_manifest.csv", index=False)
    test[["source_row", "title", "article", "label"]].to_csv(reports / "test_articles.csv", index=False)

    # Validation-only selection/ablation, never fit vocabulary or chi2 on validation.
    experiments = []
    if args.reuse_feature_experiments:
        experiments = pd.read_csv(reports / "feature_experiments.csv").to_dict("records")
        for row in experiments:
            row["k"] = "all" if row["k"] == "all" else int(row["k"])
        print("Reusing completed feature experiments with identical dataset splits", flush=True)
    for k in (() if args.reuse_feature_experiments else (3000, 6000, "all")):
        print(f"Feature experiment: k={k}, linguistic=True", flush=True)
        pipe = make_pipeline(LinearSVC(C=1, max_iter=5000, random_state=SEED), k)
        start = perf_counter()
        pipe.fit(train.article, train.label)
        seconds = perf_counter() - start
        transformed_size = pipe.named_steps["features"].named_steps["union"].transform([train.article.iloc[0]]).shape[1]
        experiments.append({"k": k, "linguistic": True, "feature_count": transformed_size,
                            "training_seconds": seconds, **metrics(validation.label, pipe.predict(validation.article))})
    selected = max((row for row in experiments if row["linguistic"]),
                   key=lambda row: (row["macro_f1"], -row["feature_count"]))
    chosen_k = selected["k"]
    print(f"Chosen k={chosen_k}; {'reusing' if args.reuse_feature_experiments else 'running'} TF-IDF-only ablation", flush=True)
    if not args.reuse_feature_experiments:
        text_only = make_pipeline(LinearSVC(C=1, max_iter=5000, random_state=SEED), chosen_k, False)
        start = perf_counter()
        text_only.fit(train.article, train.label)
        experiments.append({"k": chosen_k, "linguistic": False,
                            "feature_count": text_only.named_steps["features"].transform([train.article.iloc[0]]).shape[1],
                            "training_seconds": perf_counter() - start,
                            **metrics(validation.label, text_only.predict(validation.article))})
    pd.DataFrame(experiments).to_csv(reports / "feature_experiments.csv", index=False)

    tuning = train
    if args.cv_rows and args.cv_rows < len(train):
        tuning, _ = train_test_split(train, train_size=args.cv_rows, stratify=train.label, random_state=SEED)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    candidates = {
        "linear_svm": (LinearSVC(max_iter=5000, random_state=SEED), {"classifier__C": [.5, 2.0]}),
        "random_forest": (RandomForestClassifier(n_estimators=120, max_features="sqrt",
                                                 min_samples_leaf=2, n_jobs=args.threads, random_state=SEED),
                          {"classifier__max_depth": [25, None]}),
        "xgboost": (XGBClassifier(n_estimators=140, learning_rate=.12, tree_method="hist",
                                  colsample_bytree=.8, subsample=.9, n_jobs=args.threads,
                                  eval_metric="logloss", random_state=SEED),
                    {"classifier__max_depth": [3, 6]}),
    }
    manifest = {"best_model": None, "selected_k": chosen_k, "seed": SEED,
                "cv_rows": len(tuning), "cv_folds": 3, "dataset": audit,
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "models": {}}
    validation_rows = []
    for name, (classifier, grid) in candidates.items():
        print(f"Tuning {name}: 2 candidates x 3 folds on {len(tuning)} training articles", flush=True)
        search = GridSearchCV(make_pipeline(classifier, chosen_k), grid, scoring="f1_macro", cv=cv,
                              n_jobs=1, refit=False, error_score="raise")
        start = perf_counter()
        search.fit(tuning.article, tuning.label)
        tuning_seconds = perf_counter() - start
        pd.DataFrame(search.cv_results_).to_csv(reports / f"{name}_cv_results.csv", index=False)
        model = make_pipeline(clone(classifier), chosen_k).set_params(**search.best_params_)
        print(f"Fitting {name} on full training set: {search.best_params_}", flush=True)
        start = perf_counter()
        model.fit(train.article, train.label)
        training_seconds = perf_counter() - start
        result = metrics(validation.label, model.predict(validation.article))
        validation_rows.append({"model": name, **result})
        filename = f"{name}.pkl"
        save_pickle(model, models / filename)
        manifest["models"][name] = {"file": filename, "best_params": search.best_params_,
                                    "cv_macro_f1": float(search.best_score_), "validation": result,
                                    "training_seconds": training_seconds, "tuning_seconds": tuning_seconds}
        print(f"{name} validation macro F1: {result['macro_f1']:.4f}", flush=True)
    best = max(validation_rows, key=lambda row: row["macro_f1"])["model"]
    manifest["best_model"] = best
    pd.DataFrame(validation_rows).to_csv(reports / "validation_comparison.csv", index=False)
    with (models / manifest["models"][best]["file"]).open("rb") as stream:
        best_pipeline = pickle.load(stream)
    save_pickle(best_pipeline, models / "best_pipeline.pkl")
    union = best_pipeline.named_steps["features"].named_steps["union"]
    text_branch = dict(union.transformer_list)["text"]
    vectorizer = text_branch.named_steps["tfidf"]
    selector = text_branch.named_steps["select"].selector_
    names = vectorizer.get_feature_names_out()
    pd.DataFrame({"term": names, "chi2_score": selector.scores_, "selected": selector.get_support()}).sort_values(
        "chi2_score", ascending=False).to_csv(reports / "feature_scores.csv", index=False)
    (models / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Selected deployment model: {best}. Evaluating held-out test set.", flush=True)
    evaluate_saved()


if __name__ == "__main__":
    main()
