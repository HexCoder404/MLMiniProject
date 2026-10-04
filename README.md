# Fake News Detection System

A Python-script ML project that compares **Linear SVM, Random Forest and XGBoost**, performs feature selection and evaluates predictions, with a Flask interface. No notebooks or joblib serialization are used. Pipelines are saved with Python's built-in pickle.

## Run the project

Use Python 3.11–3.13. In PowerShell, from this directory:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.explore
.venv\Scripts\python.exe -m src.train
.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:5000**. If the generated models are already present, you can start the app immediately after installing dependencies. The app loads the saved pipeline without retraining. Restart Flask after retraining to load the new artifact.

With an existing Python environment containing the requirements:

```powershell
python -m src.explore
python -m src.train
python -m src.evaluate
python app.py
```

Training already runs evaluation; the separate evaluate command regenerates reports for the saved models and held-out articles. It does not tune parameters. Training time depends on the machine; tree-model tuning can take several minutes. The default uses a stratified **6,000-row training-only tuning sample**, two candidates per model and three CV folds. Final models fit all training rows. Use `python -m src.train --cv-rows 0` for full-training-set tuning. Training defaults to one thread for compatibility with restricted Windows environments; `--threads 4` enables additional tree-model workers where supported. The CV sample limit makes the comparison practical; it is not an exhaustive search.

If a run is interrupted after feature experiments complete, `python -m src.train --reuse-feature-experiments` reuses those results and reruns model tuning/training. This checks that the dataset splits match. Only use it when feature/preprocessing code has not changed; an ordinary training run recomputes everything.

## Dataset and preparation

- `archive/true.csv`: label 0, genuine.
- `archive/fake.csv`: label 1, potentially fake.
- Both require `title` and `text`; subject/date columns are used only for inspection, never as classifier inputs.
- Remove empty article bodies, normalized duplicate bodies and identical normalized first-200-word prefixes. Remove conflicting-label bodies/prefixes rather than choosing a label arbitrarily.
- Use reproducible stratified 70% training / 15% validation / 15% test splits, seed 42. Split manifests verify that retained duplicate groups do not cross partitions.
- Normalize HTML, entities and whitespace. Preserve punctuation and case for linguistic extraction. TF-IDF lowercases internally.
- Combine optional title with body. A body is required by the web interface.

The prefix rule is a conservative near-duplicate heuristic, not a semantic similarity guarantee. The supplied dataset has different subject categories between classes and genuine examples contain Reuters attribution. An article-level split does **not** establish generalization to unseen publishers, dates or domains. Dataset file names supply the labels; this project does not independently verify them.

## Features and selection

TF-IDF is a representation, not a classification algorithm. We use word unigrams/bigrams, up to 12,000 terms, `min_df=3`, `max_df=0.98` and sublinear term frequency.

Chi-square selection compares **3,000**, **6,000** and **all** available terms with Linear SVM on the validation set. The chosen term count is shared by all three model families. The selected configuration is also tested without linguistic features. Eight linguistic features are word count, sentence count, mean word length, mean sentence length, uppercase ratio, punctuation ratio and exclamation/question-mark rates.

TF-IDF and chi-square are fitted only on training data; feature selection stays within each CV pipeline. Chi-square receives nonnegative TF-IDF values. Linguistic features are separately standardized and appended afterward. The linguistic ablation uses SVM only; its result should not be generalized to every model.

## Models and evaluation

| Model | Tuned candidates | Fixed configuration |
| --- | --- | --- |
| Linear SVM | C = 0.5, 2.0 | LinearSVC, max_iter=5,000 |
| Random Forest | max_depth = 25, unlimited | 120 trees, min leaf size 2, sqrt feature sampling |
| XGBoost | max_depth = 3, 6 | 140 trees, histogram training, learning rate 0.12, row/column subsampling |

Tune using training CV macro F1, fit on all training rows and select the deployment model by **validation macro F1**. The test set is reserved for final reporting. Models are retained as fitted on the training partition, so the exported artifact corresponds directly to the reported test scores.

Reports include accuracy, macro F1, fake-class precision/recall/F1, ROC AUC, average precision, training time, batch prediction latency per article and artifact size. Class 1 is the positive class. Latency measures `predict` over the test batch and excludes loading, score extraction and HTTP overhead. Tuning time is stored separately in the manifest.

Each model gets a classification report, confusion matrix, test predictions and error excerpts. The generated `reports/performance_analysis.md` explains measured differences and limitations. Sparse TF-IDF often suits a linear margin; forests split on subsets of individual features, while boosted trees iteratively learn interactions. These are possible mechanisms, not proven causal explanations for a score difference. No score is promised in advance.

## Generated outputs

```text
models/
  best_pipeline.pkl        # Flask deployment pipeline
  linear_svm.pkl
  random_forest.pkl
  xgboost.pkl
  manifest.json            # Winner, parameters, versions and timings
reports/
  exploration_summary.json
  dataset_summary.json
  split_manifest.csv
  feature_experiments.csv
  feature_scores.csv
  validation_comparison.csv
  model_comparison.csv
  performance_analysis.md
  test_articles.csv
  *_cv_results.csv
  *_test_predictions.csv
  *_errors.csv
  classification_reports/ # JSON and readable text per model
  figures/                # EDA, model comparison and confusion matrices
```

Training overwrites generated model/report files. Keep the dataset unchanged to reproduce the split. Pickle should only load artifacts generated by this project and must be loaded with compatible library versions; never deserialize an untrusted model file. scikit-learn uses joblib internally in places, but this project does not use it to persist models.

## Prediction API

`POST /api/predict`, content type `application/json`:

```json
{"title": "Optional headline", "article": "A complete article containing at least ten words goes here."}
```

The response contains `label`, `class_id`, `model` and a limitations note. Invalid input returns 400, oversized requests 413, and a missing trained model 503. The interface limits titles to 300 characters and bodies to 50,000 characters. No raw SVM score is presented as a confidence probability. Nothing submitted is saved by the app.

## Checks

```powershell
python -m unittest discover -s tests -v
```

Checks cover pipeline pickle round trips, browser/API predictions, input limits, escaped form content, missing artifacts, split integrity and the deployment winner. Artifact checks run after training.

## References

- [scikit-learn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html): chi-square selection and pipelines.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): fit preprocessing only on training data.
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html): sklearn-compatible classifier options.

This is an educational text classifier. It recognizes dataset patterns and cannot establish whether a real-world claim is true. External, source-disjoint evaluation is required before relying on it beyond this dataset.
