# Fake News Detection System - Project Report

> **Preparation note:** This report covers every heading in the supplied table-of-contents images. Replace bracketed student/institution fields before printing. Page numbers must be assigned after final report layout; the image page numbers are not verified pagination for this document.
>
> **Model alignment:** Chapter 4 describes the three algorithms implemented in this project: Linear SVM, Random Forest, and XGBoost. Random Forest is the validation-selected deployment model.

## Certificate

*Template for institutional completion and approval; this is not an issued certificate.*

This is to certify that the mini-project entitled **"Fake News Detection System"** has been carried out by **[Student Name(s)]**, bearing **[Roll Number(s)]**, of **[Department Name]**, **[College/Institute Name]**, under the guidance of **[Guide Name and Designation]**, during the academic year **[Academic Year]**. This report is submitted in partial fulfillment of the requirements for **[Course/Degree Name]** of **[University Name]**.

The project presents dataset preparation, text feature extraction, comparison of machine learning classifiers, evaluation of experimental results, and implementation of a web interface for article classification.

**Project guide:** [Name and signature]

**Head of department:** [Name and signature]

**Examiner:** [Name and signature, if required]

**Place:** [City]

**Date:** [Approval date]

**Institute seal:** [If required]

## Project Report

**Project title:** Fake News Detection System

**Application name:** NewsLens

**Submitted by:** [Student Name(s) and Roll Number(s)]

**Department:** [Department Name]

**Institute:** [College/Institute Name]

**University:** [University Name]

**Submitted to:** [Guide Name and Designation]

**Academic year:** [Academic Year]

**Course:** [Mini-Project/Course Name]

This project develops a supervised text-classification system that accepts an English article and predicts whether its language resembles genuine or fake-labeled examples in the supplied corpus. It combines selected TF-IDF and linguistic features, compares three implemented classifiers, and exposes the selected saved pipeline through a browser interface and JSON API. The report describes its methodology, measured results, practical outcome, and limitations.

## Abstract

This project develops a Fake News Detection System that classifies English news articles as **Genuine (0)** or **Potentially fake (1)** using supervised machine learning. It compares Linear Support Vector Machine (SVM), Random Forest, and XGBoost through a shared text-processing pipeline and provides a Flask interface for predictions. The supplied dataset contains 44,898 articles; removing empty bodies, normalized duplicates, and repeated article prefixes leaves 38,358 articles. Reproducible stratified training, validation, and test partitions contain 26,850, 5,754, and 5,754 articles respectively.

The representation combines 3,000 chi-square-selected TF-IDF word features with eight standardized linguistic features. Model hyperparameters are tuned through three-fold stratified cross-validation on a 6,000-article training-only sample. Random Forest is selected for deployment by validation macro F1. On the held-out test set it achieves **99.7046% accuracy**, **0.997021 macro F1**, and **0.999939 ROC AUC**. Linear SVM and XGBoost each achieve **99.7393% test accuracy**. These results demonstrate strong classification within the supplied dataset, while publisher and stylistic cues limit claims about generalization. The system recognizes textual patterns and does not independently verify factual claims.

**Keywords:** Fake news detection, text classification, TF-IDF, chi-square selection, Linear SVM, Random Forest, XGBoost, Flask.

## Table of Content

The structure follows the supplied images. Complete page numbers after final formatting.

| Sr. No. | Content | Page No. |
| ---: | --- | --- |
| 1 | Certificate | [Assign] |
| 2 | Project Report | [Assign] |
| 3 | Abstract | [Assign] |
| 4 | Table of Content | [Assign] |
| 5 | List of Figures | [Assign] |
| 6 | List of Tables | [Assign] |
| 7 | List of Abbreviation | [Assign] |
| 8 | Chapter 1 - Introduction | [Assign] |
| | 1.1 Introduction | [Assign] |
| | 1.2 Problem Statement | [Assign] |
| | 1.3 Objectives | [Assign] |
| | 1.4 Scope of the Project | [Assign] |
| | 1.5 Significance of the Study | [Assign] |
| 9 | Chapter 2 - Literature Survey | [Assign] |
| | 2.1 Review of Existing Research | [Assign] |
| | 2.2 Research Gap | [Assign] |
| 10 | Chapter 3 - Dataset and Data Preprocessing | [Assign] |
| | 3.1 Dataset Description | [Assign] |
| | 3.2 Dataset Features | [Assign] |
| | 3.3 Target Variable | [Assign] |
| | 3.4 Dataset Class Distribution | [Assign] |
| | 3.5 Data Cleaning | [Assign] |
| | 3.6 Exploratory Data Analysis | [Assign] |
| | 3.7 Feature Processing | [Assign] |
| | 3.8 Feature Scaling and Encoding | [Assign] |
| | 3.9 Train-Test Split | [Assign] |
| 11 | Chapter 4 - Machine Learning Methodology | [Assign] |
| | 4.1 Machine Learning Pipeline | [Assign] |
| | 4.2 Linear Support Vector Machine | [Assign] |
| | 4.3 Random Forest | [Assign] |
| | 4.4 XGBoost | [Assign] |
| | 4.5 Model Comparison | [Assign] |
| | 4.6 Hyperparameter Optimization | [Assign] |
| | 4.7 Final Model | [Assign] |
| 12 | Chapter 5 - System Implementation | [Assign] |
| | 5.1 System Architecture | [Assign] |
| | 5.2 Prediction Input | [Assign] |
| | 5.3 Prediction Process | [Assign] |
| | 5.4 Web Application | [Assign] |
| | 5.5 Deployment | [Assign] |
| 13 | Chapter 6 - Result and Evaluation | [Assign] |
| | 6.1 Final Model Performance | [Assign] |
| | 6.2 Confusion Matrix | [Assign] |
| | 6.3 ROC Curve | [Assign] |
| | 6.4 Feature Importance | [Assign] |
| | 6.5 Prediction Demonstration | [Assign] |
| 14 | Chapter 7 - Discussion | [Assign] |
| | 7.1 Model Performance Analysis | [Assign] |
| | 7.2 Prediction Error Analysis | [Assign] |
| | 7.3 Feature Importance Analysis | [Assign] |
| | 7.4 System Limitations | [Assign] |
| | 7.5 Dataset Limitations | [Assign] |
| 15 | Chapter 8 - Conclusion and Future Scope | [Assign] |
| | 8.1 Conclusion | [Assign] |
| | 8.2 Key Findings | [Assign] |
| | 8.3 Practical Outcome | [Assign] |
| | 8.4 Future Scope | [Assign] |
| 16 | Chapter 9 - References | [Assign] |

## List of Figures

Figures 1-6 are existing images embedded below. Figure 7 is a text diagram that can be redrawn for the printed report.

| Figure No. | Caption | Section | Page No. |
| --- | --- | --- | --- |
| 1 | Review of existing research | 3.6 | [Assign] |
| 2 | Dataset fields and their use | 4.5 | [Assign] |
| 3 | Dataset class distribution | 6.2 | [Assign] |
| 4 | Data-cleaning audit | 6.2 | [Assign] |
| 5 | Linguistic feature definitions | 6.2 | [Assign] |
| 6 | Training, validation, and test partitions | 6.3 | [Assign] |
| 7 | Validation model comparison | 5.1 | [Assign] |

## List of Tables

Table captions and page placeholders are provided for the report layout.

| Table No. | Caption | Section | Page No. |
| --- | --- | --- | --- |
| 1 | Review of existing research | 2.1 | [Assign] |
| 2 | Dataset fields and their use | 3.2 | [Assign] |
| 3 | Dataset class distribution | 3.4 | [Assign] |
| 4 | Data-cleaning audit | 3.5 | [Assign] |
| 5 | Linguistic feature definitions | 3.7 | [Assign] |
| 6 | Training, validation, and test partitions | 3.9 | [Assign] |
| 7 | Validation model comparison | 4.5 | [Assign] |
| 8 | Held-out test model comparison | 4.5 | [Assign] |
| 9 | Hyperparameter tuning results | 4.6 | [Assign] |
| 10 | Software and experiment configuration | 5.5 | [Assign] |
| 11 | Final Random Forest performance | 6.1 | [Assign] |
| 12 | Final-model confusion matrix | 6.2 | [Assign] |
| 13 | Confusion counts for all implemented models | 6.2 | [Assign] |
| 14 | Highest-ranked chi-square terms | 6.4 | [Assign] |
| 15 | Saved test prediction examples | 6.5 | [Assign] |
| 16 | Computational performance | 7.1 | [Assign] |
| 17 | Feature configuration experiments | 7.3 | [Assign] |

## List of Abbreviation

| Abbreviation | Full form / meaning |
| --- | --- |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| TF-IDF | Term Frequency-Inverse Document Frequency |
| SVM | Support Vector Machine |
| RF | Random Forest |
| XGBoost | Extreme Gradient Boosting |
| EDA | Exploratory Data Analysis |
| CV | Cross-Validation |
| ROC | Receiver Operating Characteristic |
| AUC | Area Under the Curve |
| TP / TN | True Positive / True Negative |
| FP / FN | False Positive / False Negative |
| TPR / FPR | True Positive Rate / False Positive Rate |
| API | Application Programming Interface |
| HTTP | Hypertext Transfer Protocol |
| JSON | JavaScript Object Notation |
| CSV | Comma-Separated Values |
| HTML | Hypertext Markup Language |
| CSS | Cascading Style Sheets |
| ISOT | Information Security and Object Technology |
| NFKC | Unicode Normalization Form Compatibility Composition |
| SHA-256 | Secure Hash Algorithm with a 256-bit digest |
| MiB | Mebibyte; 1,048,576 bytes |
| MLOps | Machine Learning Operations |

## Chapter 1 - Introduction

### 1.1 Introduction

Online news can spread rapidly, making manual review of every article impractical. Automated text classification can support initial screening by identifying patterns associated with labeled genuine and fake articles. However, a classifier trained on historical examples may learn publisher identity, writing style, or topic rather than factual accuracy. This project therefore studies both predictive performance and the limitations of the dataset.

The Fake News Detection System, presented through the NewsLens interface, connects text preprocessing, numerical feature extraction, classifier comparison, and browser-based inference. It provides an educational demonstration of an end-to-end supervised learning workflow.

### 1.2 Problem Statement

The task is binary article classification. Given an optional title and a required article body, the system predicts one of the two dataset-derived classes. Labels originate from the input files; the project does not independently adjudicate their truthfulness.

The problem is to build a reproducible classifier that prepares English article data, learns from labeled training examples, compares candidate methods, and predicts a class for a new article. The workflow must limit duplicate overlap and prevent preprocessing from fitting on validation or test data. A further challenge is distinguishing generalizable article patterns from publisher and style cues associated with the dataset labels.

### 1.3 Objectives

1. Audit the dataset and reduce duplicate overlap before splitting.
2. Build a reusable pipeline for cleaning, TF-IDF extraction, feature selection, and linguistic features.
3. Compare Linear SVM, Random Forest, and XGBoost under a common evaluation procedure.
4. Study vocabulary selection and the contribution of linguistic features using validation experiments.
5. Evaluate classification quality, ranking quality, runtime, artifact size, and errors.
6. Serve the selected saved pipeline through a browser interface and JSON API.

### 1.4 Scope of the Project

The implementation covers English article text, local training, saved model artifacts, generated evaluation reports, and Flask inference. Subject and date metadata are inspected but excluded from the classifier. Social-network propagation, image analysis, evidence retrieval, multilingual classification, and production-scale hosting are outside the implemented scope.

The three implemented and evaluated candidates are Linear SVM, Random Forest, and XGBoost. Chapter 4 describes each algorithm and reports their shared experimental setup and measured results.

### 1.5 Significance of the Study

The study demonstrates how an auditable machine learning experiment can become a usable application. Dataset summaries, split manifests, fitted pipelines, parameter-search results, and saved prediction errors make the workflow easier to reproduce and review. A shared pipeline also keeps training and inference transformations consistent.

The project shows why high accuracy needs interpretation. Source-related cues may help distinguish this dataset without establishing factual reliability. Examining those cues alongside performance metrics is useful for designing future publisher-disjoint and external evaluations.

## Chapter 2 - Literature Survey

### 2.1 Review of Existing Research

*Table 1. Review of existing research.*

| Research or solution | Main idea | Relationship to this project |
| --- | --- | --- |
| Joachims (1998), *Text Categorization with Support Vector Machines: Learning with Many Relevant Features* | Studies SVMs for sparse, high-dimensional text classification. | Supports Linear SVM as the text-classification baseline. [Paper](https://www.cs.cornell.edu/People/tj/publications/joachims_98a.pdf). |
| Breiman (2001), *Random Forests* | Combines randomized decision trees into an ensemble. | Motivates comparison with a nonlinear tree ensemble. [Paper](https://doi.org/10.1023/A%3A1010933404324). |
| Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System* | Describes scalable tree boosting, including techniques for sparse data. | Motivates the boosted-tree model used in the experiments. [Paper](https://arxiv.org/abs/1603.02754). |
| Wang (2017), *“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection* | Introduces a benchmark for evaluating fake-news detection. | Provides a relevant external benchmark for future evaluation; LIAR is not used in the reported experiments. [Paper](https://aclanthology.org/P17-2067/). |
| University of Victoria ISOT dataset documentation | Describes a corpus of genuine and fake news, with class and subject counts matching the supplied files. | Provides the dataset-family reference; the repository does not record the exact download history of its local copy. [Dataset documentation](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf). |

The implemented baseline uses TF-IDF with Linear SVM. Random Forest and XGBoost provide alternative decision rules on the same representation. An all-terms SVM configuration and a selected-TF-IDF-only SVM configuration serve as feature baselines. TF-IDF is a representation, rather than a classifier. The scikit-learn documentation describes TF-IDF extraction and chi-square selection used in this pipeline. [Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction), [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html).

The literature motivates the methods; published scores are not treated as directly comparable to this project's results because datasets, cleaning rules, and partitions may differ.

### 2.2 Research Gap

The local question investigated is whether a compact representation combining selected TF-IDF terms and linguistic measurements can support accurate classification, a fair comparison of implemented model families, and a usable web interface. The project addresses this through validation feature experiments, training-only cross-validation, held-out evaluation, and saved deployment artifacts.

A broader gap remains unresolved: successful classification of historical articles does not establish factual verification or performance on unfamiliar publishers. The measured Reuters-label association provides evidence for this concern in the supplied corpus. Source-disjoint tests, chronological tests, and retrieval of independent evidence would be needed to address it. This describes the limits of the present experiment rather than claiming that all prior studies share the same gap.

## Chapter 3 - Dataset and Data Preprocessing

### 3.1 Dataset Description

The project reads [archive/true.csv](archive/true.csv) and [archive/fake.csv](archive/fake.csv). Their row counts and subject distributions match the ISOT corpus documentation cited above. The locally supplied files are the actual experimental inputs.

The two files contain 21,417 genuine and 23,481 fake-labeled records, totaling 44,898 articles. Cleaning leaves 38,358 records. The dataset consists of English news text and associated metadata. All reported experiments use these local files; no live news feed or external benchmark is included.

### 3.2 Dataset Features

*Table 2. Dataset fields and their use.*

| Column | Meaning | Use |
| --- | --- | --- |
| `title` | Article headline | Combined with the body as classifier input; optional during inference. |
| `text` | Article body | Required classifier input. |
| `subject` | Dataset topic category | Exploratory analysis only. |
| `date` | Publication-date metadata | Excluded from classifier inputs. |
| `label` | Added by the loader: 0 for `true.csv`, 1 for `fake.csv` | Target variable. |
| `source_row`, `content_hash`, `prefix_hash` | Generated identifiers | Audit and split-integrity checks only. |

### 3.3 Target Variable

The target variable is binary: **0 = Genuine** and **1 = Potentially fake**. The loader assigns label 0 to records from `true.csv` and label 1 to records from `fake.csv`. Class 1 is the positive class for the reported fake-class precision, recall, F1, and ROC AUC.

Labels are inherited from the dataset files and are not independently verified or reannotated by this project. The interface uses "Potentially fake" to communicate a model classification without claiming conclusive evidence that an article is false.

### 3.4 Dataset Class Distribution

*Table 3. Dataset class distribution.*

| Dataset stage | Genuine (0) | Potentially fake (1) | Total |
| --- | ---: | ---: | ---: |
| Raw input | 21,417 | 23,481 | 44,898 |
| Cleaned dataset | 20,918 | 17,440 | 38,358 |
| Training | 14,642 | 12,208 | 26,850 |
| Validation | 3,138 | 2,616 | 5,754 |
| Test | 3,138 | 2,616 | 5,754 |

After cleaning, genuine articles constitute **54.53%** and potentially fake articles **45.47%**. Stratification preserves approximately these proportions in every partition. No class resampling is implemented.

### 3.5 Data Cleaning

Missing titles and bodies are replaced with empty strings. Empty cleaned bodies are removed, while an absent title is allowed. Cleaning decodes HTML entities, removes HTML tags, applies Unicode NFKC normalization, collapses whitespace, and trims text. Case and punctuation remain available for linguistic extraction; TF-IDF lowercases its input internally. The implemented pipeline does not apply stemming, lemmatization, or a separate stop-word-removal list.

For duplicate auditing, the body is lowercased and represented by normalized word tokens. A SHA-256 hash identifies repeated normalized bodies. A second hash represents the first 200 normalized body words. All rows in a conflicting-label body or prefix group would be removed; other duplicate groups retain one article.

*Table 4. Data-cleaning audit.*

| Cleaning operation | Rows removed |
| --- | ---: |
| Empty article bodies | 631 |
| Conflicting-label body groups | 0 |
| Normalized exact duplicate bodies | 5,899 |
| Conflicting-label prefix groups | 0 |
| Repeated first-200-word prefixes | 10 |
| **Total removed** | **6,540** |

The prefix rule reduces a specific kind of near duplication. It does not detect every paraphrase, revision, or syndicated article.

Article-length distributions are inspected, but no statistical outlier clipping or removal is applied. Nonempty unusually short and long articles remain in training.

### 3.6 Exploratory Data Analysis

Exploratory analysis examines class counts, topic categories, article lengths, and source attribution. These checks help identify imbalance and potential shortcuts before interpreting classifier results.

Genuine subjects are `politicsNews` and `worldnews`. Fake-labeled subjects are `News`, `politics`, `left-news`, `Government News`, `US_News`, and `Middle-east`. These category sets differ between classes. The raw dataset contains a Reuters marker in **99.82% of genuine bodies** and **1.31% of fake-labeled bodies**, creating a strong source-related cue.

In the cleaned dataset, mean body length is approximately 384.74 words for genuine articles and 425.01 for potentially fake articles. The observed ranges are 22–5,172 and 1–8,135 words respectively, according to the exploratory report.

![Dataset overview](reports/figures/dataset_overview.png)

*Figure 1. Existing exploratory dataset visualization.*

Evidence: [dataset summary](reports/dataset_summary.json), [exploration summary](reports/exploration_summary.json), and [split manifest](reports/split_manifest.csv).

### 3.7 Feature Processing

The vectorizer uses word unigrams and bigrams, a maximum vocabulary of **12,000 terms**, `min_df=3`, `max_df=0.98`, and sublinear term frequency. With term count `c(t,d)`, document frequency `df(t)`, and training-document count `N`, the configured weighting can be expressed as:

```text
tf(t,d) = 1 + ln(c(t,d))                 when c(t,d) > 0
idf(t)  = ln((1 + N) / (1 + df(t))) + 1
weight(t,d) = tf(t,d) * idf(t)
x_d = weight_vector_d / ||weight_vector_d||_2
```

Absent terms receive zero weight. Smoothed IDF and L2 normalization follow the vectorizer defaults. [scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction).

Chi-square scores measure association between nonnegative term features and class labels. Conceptually, the statistic compares observed and expected feature mass across classes:

```text
chi2(t) = sum_over_classes ((Observed(t,c) - Expected(t,c))^2 / Expected(t,c))
```

SVM validation experiments compare 3,000, 6,000, and all available terms. The selected count is **3,000**, shared by all model families. Adding eight linguistic features produces **3,008 input features**.

*Table 5. Linguistic feature definitions.*

| Linguistic feature | Implemented definition |
| --- | --- |
| Word count | Number of matches to the word-token regular expression. |
| Sentence count | Number of punctuation groups matching `[.!?]+`, with a minimum of one. |
| Mean word length | Total word-token characters divided by word count. |
| Mean sentence length | Word count divided by the sentence-count approximation. |
| Uppercase ratio | Uppercase characters divided by alphabetic-character count. |
| Punctuation ratio | Non-alphanumeric, non-whitespace characters divided by text length. |
| Exclamations per word | Exclamation marks divided by word count. |
| Questions per word | Question marks divided by word count. |

Denominators are protected against division by zero. Sentence count is a punctuation heuristic rather than a linguistic sentence parser. Scaling uses `z_j = (x_j - mean_train,j) / std_train,j`.

### 3.8 Feature Scaling and Encoding

Text is encoded as sparse TF-IDF vectors; the binary labels are already represented as 0 and 1. Subject and date metadata are excluded rather than one-hot encoded. TF-IDF lowercases internally while the shared cleaner preserves case and punctuation for the linguistic branch.

Article-length distributions are inspected during exploration. No statistical outlier-removal or clipping rule is implemented, so nonempty unusually short or long articles remain in training. The web interface separately requires at least ten body words and limits bodies to 50,000 characters and titles to 300 characters.

Title and body are concatenated for feature extraction. Eight linguistic features describe length, capitalization, and punctuation. `StandardScaler` fits their means and standard deviations on training data. TF-IDF vectors use the vectorizer's default L2 normalization. Linguistic features are appended after chi-square selection so that chi-square operates on nonnegative TF-IDF values.

The linguistic transformation follows `z = (x - mean_train) / std_train`. Its fitted parameters are reused at inference. Chi-square selection is applied before standardized linguistic features are appended, ensuring that the selector receives nonnegative TF-IDF values. No transformation is fitted on the held-out test set.

### 3.9 Train-Test Split

The cleaned dataset uses stratified **70% training / 15% validation / 15% test** partitions with seed **42**, subject to integer rounding. Duplicate removal precedes splitting. The split manifest records identifiers to check that retained body and prefix groups do not cross partitions. TF-IDF, feature selection, and scaling fit only on training data, including within each cross-validation fold. Validation is used for feature configuration and deployment selection; test data is reserved for final reporting.

*Table 6. Training, validation, and test partitions.*

| Partition | Genuine (0) | Potentially fake (1) | Total | Purpose |
| --- | ---: | ---: | ---: | --- |
| Training | 14,642 | 12,208 | 26,850 | Fit features and classifiers. |
| Validation | 3,138 | 2,616 | 5,754 | Select features and deployment model. |
| Test | 3,138 | 2,616 | 5,754 | Report held-out performance. |

## Chapter 4 - Machine Learning Methodology

### 4.1 Machine Learning Pipeline

The experimental workflow loads and labels articles, cleans the text, removes duplicates, and creates stratified partitions. Feature experiments determine the shared representation. Training-only cross-validation selects model parameters, and each selected configuration is fitted on all training articles. Validation macro F1 selects the deployment model before held-out test evaluation. The complete fitted pipeline is serialized for inference.

**Linear SVM:** `LinearSVC` learns a linear decision boundary. For mathematical description, map class labels to `y_i ∈ {-1,+1}`. Its default squared-hinge formulation is represented by:

```text
minimize  0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i * (w^T x_i + b))^2
```

This expression summarizes the main loss and weight penalty; the library also handles the fitted intercept. The selected value is `C=2.0`, with `max_iter=5000`.

**Random Forest:** An ensemble of 120 randomized trees aggregates tree class probabilities and predicts the class with the largest mean probability. The model uses square-root feature sampling and a minimum leaf size of two. Unlimited maximum depth is selected.

**XGBoost:** Additive boosted trees optimize a regularized classification objective, summarized as:

```text
F(x) = sum_k f_k(x)
Objective = sum_i classification_loss(y_i, F(x_i)) + sum_k regularization(f_k)
```

The implementation uses 140 trees, learning rate 0.12, histogram tree construction, row subsampling 0.9, column subsampling 0.8, and selected maximum depth six. Binary classification uses the classifier's logistic objective; `logloss` is configured as its evaluation metric. [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

### 4.2 Linear Support Vector Machine

A Support Vector Machine (SVM) learns a decision boundary that separates examples from different classes. The linear version is suitable for sparse, high-dimensional text representations because it can combine evidence across many word and phrase features. Its regularization parameter `C` controls the trade-off between fitting the training examples and keeping the decision boundary simpler. The classifier produces a decision score; this score is not automatically a calibrated probability.

This project implements `LinearSVC` with `max_iter=5000`. The candidate values for `C` are 0.5 and 2.0, and training-only three-fold cross-validation selects `C=2.0`. The fitted model uses the shared 3,000 selected TF-IDF terms and eight standardized linguistic features. On the held-out test set it achieves 99.7393% accuracy and 0.997371 macro F1. Its saved model is the smallest of the three measured artifacts. See the [Linear SVM paper](https://www.cs.cornell.edu/People/tj/publications/joachims_98a.pdf).

### 4.3 Random Forest

Random Forest is an ensemble of decision trees. Each tree is trained with randomized choices, including a subset of features considered at a split. The ensemble combines the trees' class predictions, reducing dependence on any single tree. It can represent nonlinear feature interactions, although tree splits on sparse text features may behave differently from a linear text boundary. The method is described by Breiman (2001), [Random Forests](https://doi.org/10.1023/A%3A1010933404324).

The project uses 120 trees, `max_features="sqrt"`, `min_samples_leaf=2`, and random seed 42. Cross-validation compares maximum depth 25 with unlimited depth and selects `max_depth=None`. Random Forest achieves 99.7046% test accuracy and 0.997021 macro F1. It has the highest validation macro F1 by a very small margin and is therefore selected for the NewsLens application. Its serialized artifact is larger than the SVM and XGBoost artifacts in this experiment.

### 4.4 XGBoost

XGBoost is a gradient-boosted tree method. It adds trees sequentially so later trees can improve the combined predictions made by earlier trees. A regularized objective and options for row and feature subsampling help control model complexity. Chen and Guestrin (2016) describe the method in [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).

The configured classifier uses 140 trees, learning rate 0.12, histogram-based tree construction, row subsampling 0.9, column subsampling 0.8, and log-loss evaluation. Cross-validation compares maximum depths 3 and 6 and selects depth 6. XGBoost achieves 99.7393% test accuracy and 0.997371 macro F1, matching Linear SVM's accuracy to the reported precision. Its test ROC AUC is 0.999987. It is not the deployed model because deployment selection is based on validation macro F1.

### 4.5 Model Comparison

*Table 7. Validation model comparison.*

| Model | Validation accuracy (%) | Validation macro F1 |
| --- | ---: | ---: |
| Linear SVM | 99.6698 | 0.996669802 |
| **Random Forest** | **99.7046** | **0.997020349** |
| XGBoost | 99.7046 | 0.997020156 |

Random Forest has the highest unrounded validation macro F1 and is therefore deployed. Its advantage over XGBoost is approximately **0.000000192**, despite identical validation accuracy. This very small difference does not establish a meaningful general superiority of Random Forest.

*Table 8. Held-out test model comparison.*

| Model | Accuracy (%) | Macro F1 | Fake precision | Fake recall | Fake F1 | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 99.7393 | 0.997371 | 0.997323 | 0.996942 | 0.997132 | 0.999802 | 0.999794 |
| Random Forest | 99.7046 | 0.997021 | 0.997702 | 0.995795 | 0.996748 | 0.999939 | 0.999929 |
| XGBoost | 99.7393 | 0.997371 | 0.998467 | 0.995795 | 0.997129 | 0.999987 | 0.999985 |

Linear SVM and XGBoost each make 15 test errors, while Random Forest makes 17. SVM has the highest test macro F1 at full precision. Test results do not change the previously selected deployment model. XGBoost's ROC AUC is extremely high but below 1.0; rounding it to four decimal places would conceal this distinction.

![Model comparison](reports/figures/model_comparison.png)

*Figure 2. Existing comparison of held-out classification metrics.*

### 4.6 Hyperparameter Optimization

Hyperparameter optimization selects model settings using only the training partition. The project uses `GridSearchCV`, macro F1, three-fold stratified cross-validation, and a stratified 6,000-article training-only sample. Each model has two candidate settings, for six fold fits per model. The preprocessing pipeline is fitted within each fold, preventing validation-fold text from fitting the vocabulary, selector, or scaler. After tuning, each selected configuration is fitted on all 26,850 training articles. The following table summarizes the searches.

`GridSearchCV` uses macro F1, three-fold `StratifiedKFold`, shuffling, and seed 42. The saved run uses a stratified 6,000-row subset of the training partition. Each model has two candidates, giving six fold fits per model. Preprocessing is fitted inside the pipeline for each fold. Grid search uses one job, and the final model is fitted separately on the full training partition.

*Table 9. Hyperparameter tuning results.*

| Model | Candidate parameter values | Selected value | Best CV macro F1 | Tuning time (s) |
| --- | --- | --- | ---: | ---: |
| Linear SVM | `C`: 0.5, 2.0 | 2.0 | 0.993445 | 55.47 |
| Random Forest | `max_depth`: 25, unlimited | Unlimited | 0.995127 | 66.50 |
| XGBoost | `max_depth`: 3, 6 | 6 | 0.996300 | 92.54 |

The search is intentionally small. Equal candidate and fold counts do not imply equal model capacity or exhaustive optimization. Feature count is chosen using the separate validation set, so these CV scores are not estimates from a fully nested model-selection procedure.

Evidence: [training manifest](models/manifest.json), [requirements](requirements.txt), and the per-model `reports/*_cv_results.csv` files.

### 4.7 Final Model

Feature experiments fit SVM with `C=1` on the training partition and compare validation macro F1. Each model then undergoes training-only cross-validation for its hyperparameters. The chosen configuration is fitted on all 26,850 training articles. Deployment selection compares validation macro F1 across these fitted models.

The final deployed classifier is **Random Forest**, selected by the highest unrounded validation macro F1. It uses 120 trees, unlimited maximum depth, square-root feature sampling, minimum leaf size two, and seed 42, with 3,000 selected TF-IDF terms and eight standardized linguistic features.

Random Forest is saved as [models/best_pipeline.pkl](models/best_pipeline.pkl). The exported models remain fitted on the training partition; validation articles are not added through a final refit. Flask loads the complete saved pipeline at startup and exposes `/` and `POST /api/predict`. It validates inputs and returns a label, class ID, model name, and limitations note. Submitted articles are not persisted, and no confidence percentage is returned.

## Chapter 5 - System Implementation

### 5.1 System Architecture

The system has an offline training workflow and an online prediction workflow. Offline processing produces model artifacts and evaluation reports. The online Flask service loads the selected artifact at startup and uses it for requests without retraining.

```text
Local genuine/fake CSV files
    -> Label assignment, cleaning, and duplicate audit
    -> Stratified train / validation / test partitions
    -> Title + body
    -> Shared text cleaner
       -> TF-IDF -> Chi-square term selection
       -> Eight linguistic features -> Standard scaling
    -> Feature union
    -> Linear SVM / Random Forest / XGBoost
    -> Validation-based deployment selection
    -> Held-out test evaluation
    -> Saved pickle pipeline -> Flask form and prediction API
```

*Figure 7. Training and prediction system architecture.*

Data loading and auditing are implemented in [src/data.py](src/data.py); shared cleaning in [src/preprocessing.py](src/preprocessing.py); feature extraction in [src/features.py](src/features.py); model fitting in [src/train.py](src/train.py); evaluation in [src/evaluate.py](src/evaluate.py); and serving in [app.py](app.py).

### 5.2 Prediction Input

Users submit an optional title and a required article body. The title is limited to 300 characters. The article is limited to 50,000 characters and must contain at least ten word-token matches after cleaning. API values must be strings. The server also caps request size at 256 KiB.

The browser form supplies a headline field and a multiline article field. The API uses the fields `title` and `article`. Headlines alone are unsupported. Training accepts nonempty shorter bodies, so the interface minimum introduces an input-policy difference documented in the limitations.

### 5.3 Prediction Process

1. Receive a browser form or a JSON object.
2. Validate field types, lengths, and minimum body word count.
3. Clean the title and body and combine them.
4. Apply the loaded pipeline's fitted TF-IDF vocabulary and weights, selected terms, linguistic extractor, and scaler.
5. Predict the numeric class with the saved Random Forest.
6. Map 0 to Genuine and 1 to Potentially fake.
7. Return the label, class ID, model name, and limitations note.

Invalid input returns HTTP 400, oversized requests return 413, and an unavailable trained model returns 503. Articles are not persisted, no external evidence source is consulted, and no confidence percentage is displayed.

### 5.4 Web Application

The NewsLens interface uses Flask, an HTML template, and CSS. A GET request to `/` displays the form. A POST request to the same route processes the article and displays its prediction, classifier name, and an explanation of the result. Submitted values remain available in the rendered form. The API endpoint `POST /api/predict` provides the same classification behavior through JSON.

The template is [templates/index.html](templates/index.html), and styling is in [static/style.css](static/style.css). Both routes call the shared prediction function in [app.py](app.py). The interface explains that predictions reflect writing patterns and can be wrong for unfamiliar sources or topics.

The existing [system tests](tests/test_system.py) cover pipeline pickle round trips, browser/API predictions, input limits, escaped form content, missing artifacts, split integrity, and deployment selection. These checks verify software behavior rather than external prediction reliability.

### 5.5 Deployment

The implemented deployment is a local Flask service. With the dependencies installed and saved models present, run:

```powershell
python app.py
```

Open `http://127.0.0.1:5000`. The service binds to localhost with debug disabled. The `PORT` environment variable can change the port. If models need to be generated, use:

```powershell
python -m src.explore
python -m src.train
python app.py
```

Training writes generated model/report files. Restart Flask after retraining to load the updated artifact. The separate `python -m src.evaluate` command regenerates evaluation reports for saved models. Run existing checks using `python -m unittest discover -s tests -v`.

*Table 10. Software and experiment configuration.*

| Component | Recorded version or configuration |
| --- | --- |
| Python | 3.13.3, recorded in the training manifest. |
| scikit-learn | 1.7.2, recorded in the training manifest. |
| XGBoost | 3.4.1, recorded in the training manifest. |
| NumPy | 2.3.3, recorded in the training manifest. |
| pandas | 2.3.2, recorded in the training manifest. |
| Flask | 3.1.2, pinned in `requirements.txt`. |
| Matplotlib | 3.10.6, pinned in `requirements.txt`. |
| Persistence | Python built-in pickle. |
| Execution | Python modules/scripts; local Flask application. |
| Random seed | 42. |
| Hardware | CPU model, RAM, and original training-machine specifications are not recorded. |

The current workspace uses Windows and PowerShell. The manifest does not establish the original training operating system or hardware. Timing results therefore describe the saved run and should not be treated as hardware-independent benchmarks. Tree-model training defaults to one worker; the command line supports a configurable worker count. No GPU training is configured.

No public hosting deployment is recorded. Load only trusted project-generated pickle artifacts with compatible dependencies. Production serving, concurrency tests, and monitoring are future extensions.

## Chapter 6 - Result and Evaluation

### 6.1 Final Model Performance

Class 1 is the positive, potentially fake class. `TP`, `TN`, `FP`, and `FN` denote true positives, true negatives, false positives, and false negatives.

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
Macro F1  = (F1_class0 + F1_class1) / 2
FPR       = FP / (FP + TN)
TPR       = TP / (TP + FN)
```

ROC AUC evaluates score ranking across thresholds. Average precision summarizes precision-recall ranking. SVM uses decision-function scores, while the tree models use positive-class probabilities. These scores have not undergone probability calibration.

**RMSE is not reported:** this is a classification task with discrete class labels, rather than prediction of a continuous target. Classification and ranking metrics are the relevant evaluation measures; adding regression RMSE would not establish classification reliability.

The selected Random Forest is evaluated on 5,754 held-out articles. Its measured performance is summarized below.

*Table 11. Final Random Forest performance.*

| Metric | Test result |
| --- | ---: |
| Accuracy | 99.7046% |
| Macro F1 | 0.997021 |
| Potentially fake precision | 0.997702 |
| Potentially fake recall | 0.995795 |
| Potentially fake F1 | 0.996748 |
| Genuine precision | 0.996500 |
| Genuine recall | 0.998088 |
| Genuine F1 | 0.997293 |
| ROC AUC | 0.999939 |
| Average precision | 0.999929 |
| Correct predictions | 5,737 of 5,754 |
| Errors | 17 of 5,754 |

Evidence: [test metrics](reports/model_comparison.csv) and [Random Forest classification report](reports/classification_reports/random_forest.json). These are saved experimental results; this report preparation does not retrain the models.

### 6.2 Confusion Matrix

For the deployed model, the confusion matrix is:

*Table 12. Final-model confusion matrix.*

| Actual class | Predicted Genuine (0) | Predicted Potentially fake (1) |
| --- | ---: | ---: |
| Genuine (0) | 3,132 (TN) | 6 (FP) |
| Potentially fake (1) | 11 (FN) | 2,605 (TP) |

The model correctly classifies 5,737 articles and misclassifies 17. A false positive flags a genuine article; a false negative misses a fake-labeled article. The comparison below shows the counts for all three implemented models.

The counts below are calculated from the saved prediction CSV files. Rows are actual classes and columns are predicted classes.

*Table 13. Confusion counts for all implemented models.*

| Model | Actual genuine, predicted genuine (TN) | Actual genuine, predicted fake (FP) | Actual fake, predicted genuine (FN) | Actual fake, predicted fake (TP) |
| --- | ---: | ---: | ---: | ---: |
| Linear SVM | 3,131 | 7 | 8 | 2,608 |
| Random Forest | 3,132 | 6 | 11 | 2,605 |
| XGBoost | 3,134 | 4 | 11 | 2,605 |

![Linear SVM confusion matrix](reports/figures/linear_svm_confusion_matrix.png)

*Figure 3. Linear SVM test confusion matrix.*

![Random Forest confusion matrix](reports/figures/random_forest_confusion_matrix.png)

*Figure 4. Random Forest test confusion matrix.*

![XGBoost confusion matrix](reports/figures/xgboost_confusion_matrix.png)

*Figure 5. XGBoost test confusion matrix.*

### 6.3 ROC Curve

The Receiver Operating Characteristic curve plots true positive rate against false positive rate as the threshold varies. `TPR = TP/(TP+FN)` and `FPR = FP/(FP+TN)`. ROC AUC summarizes how well scores rank positive articles above negative ones.

Random Forest obtains AUC 0.999939, Linear SVM 0.999802, and XGBoost 0.999987. XGBoost's value is below 1.0 even though four-decimal rounding would display 1.0000. The ranking scores are not calibrated factual-confidence probabilities.

![Held-out ROC curves](reports/figures/roc_curves.png)

*Figure 6. ROC curves calculated from saved test labels and decision scores. The second panel enlarges the upper-left region because the full curves nearly overlap. The dashed diagonal represents chance-level ranking.*

### 6.4 Feature Importance

*Table 14. Highest-ranked chi-square terms.*

| Term | Training chi-square score |
| --- | ---: |
| `reuters` | 314.9533 |
| `video` | 278.2368 |
| `you` | 226.5156 |
| `via` | 216.9766 |
| `image` | 200.6434 |

These highly ranked terms indicate association with dataset labels. Chi-square scores do not show the direction of association, prove factual unreliability, or provide model-specific feature importance. The strength of `reuters`, together with the measured source-marker distribution, supports concern that source identity contributes to the task. Random Forest impurity importance, SVM coefficient analysis, and SHAP explanations are not included in the existing reports.

The full ranking is available in [reports/feature_scores.csv](reports/feature_scores.csv). These are training chi-square selection scores, not measured Random Forest-specific importance values.

### 6.5 Prediction Demonstration

The table below contains actual saved Random Forest test predictions. Labels are inherited from the dataset files; these examples are not newly generated live predictions.

*Table 15. Saved test prediction examples.*

| Headline | Dataset label | Saved prediction | Outcome |
| --- | --- | --- | --- |
| Factbox: What to watch in negotiations over details of U.S. tax bill | Genuine | Genuine | Correct |
| OINK, OINK! 2015 PIG BOOK SHOWS LAWMAKERS APPROPRIATING GLUTTONOUS SPENDING | Potentially fake | Potentially fake | Correct |
| Clinton says Trump is most divisive candidate 'in our lifetimes' | Genuine | Potentially fake | False positive |

Evidence: [saved predictions](reports/random_forest_test_predictions.csv) and [saved errors](reports/random_forest_errors.csv).

For a report-book demonstration, start NewsLens, paste the complete body and optional headline of a selected article, and submit the form. Record the actual displayed class and model name and capture the result screen for the printed report. A headline by itself is insufficient. No live browser screenshot is claimed here.

The API accepts the following illustrative request; this is not a recorded prediction result:

```json
{"title": "Optional headline", "article": "Paste the complete English article body here with at least ten words."}
```

A successful response contains `label`, `class_id`, `model`, and `note`. Explain during the demonstration that the result expresses similarity to training patterns and does not certify factual truth.

## Chapter 7 - Discussion

### 7.1 Model Performance Analysis

All three models perform strongly on the saved article split. SVM has the highest test macro F1 at full precision, while XGBoost has the highest ROC AUC. Random Forest remains deployed because its validation macro F1 was the selection criterion. The extremely small validation difference does not establish meaningful general superiority.

A linear margin can combine many sparse word signals, Random Forest can represent nonlinear splits, and boosting can successively improve an additive prediction. These are plausible explanations of behavior; the experiment does not isolate their causal contribution. Only one fixed split and a small parameter search are reported. No confidence intervals or statistical significance tests are included.

*Table 16. Computational performance.*

| Model | Final training time (s) | Batch prediction time per article (ms) | Serialized size (MiB) |
| --- | ---: | ---: | ---: |
| Linear SVM | 56.03 | 0.7265 | 0.6012 |
| Random Forest | 67.06 | 0.7918 | 15.3605 |
| XGBoost | 52.04 | 0.7642 | 0.8082 |

Training times exclude hyperparameter tuning and feature experiments. Prediction timing measures `predict` over the complete test batch, divided by its article count. It excludes model loading, separate score extraction, and HTTP overhead. Artifact sizes use bytes divided by `1024^2`. SVM provides the smallest artifact and lowest measured batch prediction latency in this run.

Evidence: [test comparison](reports/model_comparison.csv), [validation comparison](reports/validation_comparison.csv), [feature experiments](reports/feature_experiments.csv), and [classification reports](reports/classification_reports/).

### 7.2 Prediction Error Analysis

False positives flag genuine articles as potentially fake, while false negatives classify fake-labeled articles as genuine. Random Forest produces six false positives and eleven false negatives. SVM reduces false negatives to eight, while XGBoost reduces false positives to four. The choice of metric therefore matters when assessing different kinds of error.

Misclassified titles and article excerpts are retained in [SVM errors](reports/linear_svm_errors.csv), [Random Forest errors](reports/random_forest_errors.csv), and [XGBoost errors](reports/xgboost_errors.csv). These support manual investigation. The reports do not establish verified causes for individual errors, so specific claims about satire, quotations, or ambiguous reporting require further review.

A linear model can combine many sparse text signals, while trees represent nonlinear splits and interactions. These mechanisms offer plausible explanations for performance differences, but the experiments do not isolate a causal explanation. Only one fixed article split and a small tuning grid are reported; confidence intervals and statistical significance tests are not provided.

The Clinton headline shown in Section 6.5 is an observed false positive. Its exact cause has not been established. A follow-up could group errors by publisher, topic, length, attribution, and formatting, then investigate hypotheses under controlled experiments rather than inferring causes from headlines alone.

### 7.3 Feature Importance Analysis

The high chi-square score for `reuters` agrees with the measured source-marker imbalance. Terms such as `video`, `image`, and `featured image` can also reflect presentation conventions. Their association with labels does not show causal importance or which class each term favors. Masking source markers and measuring the change in performance would test their influence more directly.

These experiments use Linear SVM with `C=1` and evaluate the validation partition.

*Table 17. Feature configuration experiments.*

| Selected terms | Linguistic features | Total features | Validation accuracy (%) | Validation macro F1 |
| --- | --- | ---: | ---: | ---: |
| 3,000 | Yes | 3,008 | 99.7393 | 0.997371 |
| 6,000 | Yes | 6,008 | 99.7219 | 0.997196 |
| All (12,000) | Yes | 12,008 | 99.6698 | 0.996670 |
| 3,000 | No | 3,000 | 99.6003 | 0.995968 |

Selecting 3,000 terms instead of all terms improves validation macro F1 by approximately **0.000701**. Adding linguistic features at this term count improves it by approximately **0.001403**. These are observed SVM validation differences, not established effects for every model. The feature-experiment SVM uses a different `C` from the final tuned SVM, explaining why its validation score can differ.

### 7.4 System Limitations

The application classifies textual patterns without consulting independent evidence. It does not support headline-only input, other languages, images, videos, or propagation analysis. It does not show calibrated confidence or abstain on uncertain inputs. Unfamiliar vocabulary, capitalization, punctuation, quotations, and topics may affect predictions; these cases have not been separately benchmarked.

The server is a local demonstration with no recorded production workload evaluation or public hosting. It loads the model at startup, so retraining requires a restart. The interface minimum of ten body words differs from the nonempty-body training rule. No individual prediction explanation is generated.

### 7.5 Dataset Limitations

The supplied corpus has class-specific topic categories and a strong association between Reuters attribution and genuine labels. Excluding subject/date columns does not remove source or topic cues embedded in article text. An article-level random split therefore does not establish performance on unseen publishers, later news, other languages, or unrelated domains.

Labels are inherited from input files rather than independently verified by this project. Hashes and matching 200-word prefixes cannot detect every paraphrase or syndicated revision. Feature configuration and deployment selection reuse the validation partition, which may introduce selection optimism, although the test set remains separate.

The complete documented limitations are:

- **Source and topic bias:** Label-specific subject categories and Reuters attribution can reward source/style recognition despite exclusion of metadata columns.
- **Generalization:** A random article split does not test unseen publishers, chronological shifts, current news, other languages, or unrelated domains.
- **Duplicate coverage:** Hashes and the 200-word-prefix heuristic cannot identify every semantic duplicate.
- **Validation reuse:** Feature choices and deployment selection use the same validation partition, introducing possible selection optimism. The test partition remains separate.
- **Input mismatch:** Training retains nonempty articles shorter than ten words; the web interface rejects such submissions. Headlines alone are unsupported.
- **Language and formatting:** Unusual capitalization, punctuation, quotations, or unseen vocabulary may alter predictions; these edge cases have not been separately benchmarked.
- **Truth verification:** No external evidence or knowledge source is consulted. Fluent misinformation and genuine sensational writing may be difficult to distinguish through text patterns alone.
- **Confidence:** Scores are not calibrated, and the application does not display a probability of factual truth.

## Chapter 8 - Conclusion and Future Scope

### 8.1 Conclusion

The project implements a complete local workflow from dataset auditing and duplicate reduction to feature extraction, model comparison, held-out evaluation, and Flask inference. A combination of 3,000 selected TF-IDF terms and eight linguistic features provides a compact shared representation. Validation experiments favor this representation over the tested all-terms and text-only SVM baselines.

Random Forest is deployed according to the predefined validation macro-F1 criterion and achieves 99.7046% held-out accuracy. Linear SVM and XGBoost each achieve 99.7393%; SVM also offers the smallest saved artifact. The close results support comparing quality and computational cost together, while the tiny validation selection margin should be interpreted cautiously.

### 8.2 Key Findings

1. Cleaning reduces 44,898 raw records to 38,358 usable articles.
2. Stratified partitions contain 26,850 training, 5,754 validation, and 5,754 test articles.
3. The 3,000-term TF-IDF representation with eight linguistic features gives the best tested SVM feature-configuration validation score.
4. Random Forest wins deployment selection by a very small validation macro-F1 margin.
5. Random Forest makes 17 test errors; SVM and XGBoost each make 15.
6. SVM provides the smallest artifact and lowest measured batch prediction latency in the saved run.
7. Reuters attribution is a strong label-associated cue, limiting claims about publisher-independent generalization.
8. The system supports local browser and API predictions using the complete saved pipeline.

### 8.3 Practical Outcome

The practical outcome is NewsLens, a local Flask application that accepts an English article and displays the saved model's classification. A JSON API provides the same functionality for programmatic use. Keeping preprocessing and classification in one artifact makes the inference workflow consistent with training.

The project also supplies dataset audits, split manifests, feature experiments, CV results, model comparisons, classification reports, confusion matrices, ROC curves, and saved prediction errors. These artifacts support review and demonstration.

The main contributions are a reproducible comparison of three classifiers, training-only fitted preprocessing, recorded feature experiments, auditable predictions and errors, and integration of the selected pipeline with an application. The practical takeaway is that high performance on this dataset establishes recognition of its label-associated text patterns. Broader fake-news screening requires evaluation that separates factual signals from publisher and stylistic cues.

### 8.4 Future Scope

1. Evaluate publisher-disjoint and chronological partitions, together with external corpora. Adapt label mappings explicitly when using datasets such as LIAR.
2. Mask source markers such as Reuters in controlled experiments to quantify dependence on attribution and publisher style.
3. Improve duplicate detection using similarity methods while keeping related article groups within one partition.
4. Extend training-only tuning, repeat experiments across seeds, and report uncertainty intervals. Use nested selection when estimating the full model-selection procedure.
5. Compare additional text baselines, contextual language models, and transformer architectures under the same split controls.
6. Add evidence retrieval and claim-level reasoning so that textual screening can be supplemented by factual evidence.
7. Study probability calibration, threshold selection, and an abstention option for uncertain or unfamiliar inputs.
8. Add model-specific explanations and structured manual review of errors, including quotations, satire, and short articles.
9. Establish multilingual benchmarks and robust preprocessing for languages beyond the present English corpus.
10. For deployment, introduce versioned datasets and artifacts, reproducible environments, API load testing, drift monitoring, documented retraining criteria, and privacy-conscious operational logging. These are proposed MLOps improvements rather than existing features.

Further work can expand the tuning search, repeat experiments across random seeds, and compare additional text representations and contextual language models under the same split controls. These are proposed extensions rather than existing features.

## Chapter 9 - Reference

[1] T. Joachims, "Text Categorization with Support Vector Machines: Learning with Many Relevant Features," in *Proceedings of the 10th European Conference on Machine Learning (ECML)*, pp. 137-142, 1998. [Online]. Available: https://www.cs.cornell.edu/People/tj/publications/joachims_98a.pdf.

[2] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.

[3] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785-794, 2016, doi: 10.1145/2939672.2939785.

[4] W. Y. Wang, "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection," in *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pp. 422-426, 2017, doi: 10.18653/v1/P17-2067.

[5] University of Victoria, ISOT Research Lab, "ISOT Fake News Dataset ReadMe." [Online]. Available: https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf. [Accessed: Oct. 9, 2026]. The supplied local CSV counts are consistent with this documentation; the precise download provenance of the local copies is not recorded.

[6] Scikit-learn Developers, "Text Feature Extraction." [Online]. Available: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction. [Accessed: Oct. 9, 2026].

[7] Scikit-learn Developers, "Feature Selection." [Online]. Available: https://scikit-learn.org/stable/modules/feature_selection.html. [Accessed: Oct. 9, 2026].

[8] XGBoost Developers, "XGBoost Python API Reference." [Online]. Available: https://xgboost.readthedocs.io/en/stable/python/python_api.html. [Accessed: Oct. 9, 2026].

[9] Pallets Projects, "Flask Documentation." [Online]. Available: https://flask.palletsprojects.com/en/stable/. [Accessed: Oct. 9, 2026].

[10] [Project author(s)], *Fake News Detection System: Source Code and Experimental Reports*. Local project files: README.md, src/data.py, src/preprocessing.py, src/features.py, src/train.py, src/evaluate.py, and app.py, 2026.

[11] [Project author(s)], *Fake News Detection System: Experimental Results*. Local project artifacts: models/manifest.json, reports/dataset_summary.json, reports/model_comparison.csv, reports/validation_comparison.csv, reports/feature_experiments.csv, and reports/feature_scores.csv, 2026.

The linked software documentation may describe versions newer than those used in this experiment. The project version records appear in Section 5.5. No external application-code repository is identified as a borrowed source.
