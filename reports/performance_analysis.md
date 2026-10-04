# Model performance analysis

The deployment model is **random_forest**, selected using validation macro F1 before test evaluation.

## Held-out test results

        model  accuracy  macro_f1  fake_precision  fake_recall  fake_f1  roc_auc  average_precision  prediction_ms_per_article  training_seconds  model_size_mb
   linear_svm    0.9974    0.9974          0.9973       0.9969   0.9971   0.9998             0.9998                     0.7265           56.0292         0.6012
random_forest    0.9970    0.9970          0.9977       0.9958   0.9967   0.9999             0.9999                     0.7918           67.0599        15.3605
      xgboost    0.9974    0.9974          0.9985       0.9958   0.9971   1.0000             1.0000                     0.7642           52.0425         0.8082

## Why performance differs

- linear_svm: test macro F1 = 0.9974. A linear boundary can aggregate thousands of weak word and phrase signals. Scaling the linguistic features prevents raw article lengths from dominating the margin.
- random_forest: test macro F1 = 0.9970. Each tree searches a random subset of sparse features. Many absent terms and the restricted tree depth can make it harder to combine weak text signals. Trees can capture nonlinear linguistic patterns.
- xgboost: test macro F1 = 0.9974. Boosted trees sequentially correct errors and learn interactions. Histogram bins, depth, regularization and the limited boosting budget influence both generalization and cost.

These are plausible explanations, not proof of causation. Model families have different capacity despite equal candidate counts and folds.

## Feature experiments (validation only)

   k  linguistic  feature_count  training_seconds  accuracy  macro_f1  fake_precision  fake_recall  fake_f1
3000        True           3008           52.3118    0.9974    0.9974          0.9985       0.9958   0.9971
6000        True           6008           54.3735    0.9972    0.9972          0.9985       0.9954   0.9969
 all        True          12008           42.8643    0.9967    0.9967          0.9977       0.9950   0.9964
3000       False           3000           19.4895    0.9960    0.9960          0.9981       0.9931   0.9956

Chosen term count: 3000. Its validation macro F1 differs from the all-terms baseline by +0.000701. Adding linguistic features at the chosen term count changes validation macro F1 by +0.001403.

Compare selected TF-IDF with all terms for selection effects, and the chosen TF-IDF-only variant with the corresponding linguistic variant for added-feature effects. These experiments use Linear SVM; they do not establish the effect for every algorithm.

## Dataset and error limitations

Subject and date metadata are excluded. Duplicate bodies and identical 200-word prefixes are removed before splitting. This heuristic cannot detect every paraphrase or syndicated article.

The true.csv examples contain Reuters attribution, while the subject categories differ entirely between files. Random article splits can reward source/style recognition. High scores here do not establish accuracy on current news, new publishers, other languages or factual claims. A source-disjoint or chronological external dataset is needed to measure that generalization.

The per-model *_errors.csv files include misclassified titles and excerpts for inspection. False positives can flag genuine articles; false negatives can miss potentially fake articles. ROC AUC uses decision scores; these scores are not calibrated confidence probabilities.

## Highest chi-square terms

              term  chi2_score  selected
           reuters    314.9533      True
             video    278.2368      True
               you    226.5156      True
               via    216.9766      True
             image    200.6434      True
          featured    196.4421      True
    featured image    195.4330      True
           said on    183.0725      True
         image via    175.7334      True
               com    166.3597      True
             watch    160.8608      True
washington reuters    157.2173      True
       twitter com    151.6384      True
               pic    147.4057      True
       pic twitter    145.7382      True

Chi-square measures association with the labels; it does not establish that a word is evidence of misinformation, and this table does not show the direction of the association.

## Source-marker diagnostic

In the raw dataset, Reuters appears in 99.82% of genuine articles and 1.31% of fake-labeled articles. This label-associated source cue is a concrete reason to avoid interpreting near-perfect article-split scores as factual verification.