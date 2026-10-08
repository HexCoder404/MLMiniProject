# Fake News Detection System: Viva Questions and Answers

This guide covers likely viva questions about the actual NewsLens project. Answers use the current source code and saved experimental reports. Read the short answer first, then explain the reasoning when asked a follow-up. The list is broad; a professor can also ask questions specific to your own contribution or demonstration.

## Quick revision facts

| Item | Answer to remember |
| --- | --- |
| Task | Supervised binary classification of English article text |
| Classes | 0 = Genuine; 1 = Potentially fake |
| Dataset files | `archive/true.csv` and `archive/fake.csv` |
| Raw data | 44,898: 21,417 genuine and 23,481 fake-labeled |
| Clean data | 38,358: 20,918 genuine and 17,440 fake-labeled |
| Removed | 631 empty bodies, 5,899 normalized duplicates, 10 repeated prefixes |
| Partitions | Training 26,850; validation 5,754; test 5,754 |
| Split | Stratified approximately 70/15/15, seed 42 |
| Representation | 3,000 selected TF-IDF terms plus eight standardized linguistic features |
| Algorithms | Linear SVM, Random Forest, XGBoost |
| Tuning | Three-fold stratified CV on a 6,000-row training-only sample |
| Deployment selection | Highest unrounded validation macro F1 |
| Deployed model | Random Forest, 120 trees, unlimited maximum depth |
| Final test accuracy | RF 99.7046%; SVM and XGBoost each 99.7393% |
| Main limitation | Source/style recognition can produce high scores without factual verification |
| Interface | Flask browser form and `POST /api/predict` |
| Saved artifact | `models/best_pipeline.pkl`, a complete fitted pipeline |

## 1. Introduction and project purpose

### Q1. What is your project about?

It classifies an English news article as Genuine or Potentially fake using learned text patterns. It compares three classifiers and serves the validation-selected pipeline through NewsLens.

### Q2. Give a one-minute project introduction.

"We prepared 44,898 labeled articles and retained 38,358 after cleaning and duplicate removal. We used stratified training, validation, and test splits. Articles are represented by 3,000 selected TF-IDF terms and eight linguistic features. We compared Linear SVM, Random Forest, and XGBoost using training-only cross-validation. Random Forest was selected by validation macro F1 and achieved 99.7046% held-out test accuracy. Flask provides article predictions. The main limitation is that dataset source and style cues may contribute to these high scores, so the system does not independently verify facts."

### Q3. What problem does the system solve?

It automates preliminary classification of articles using a labeled corpus. It demonstrates an auditable screening workflow that can help prioritize material for further review.

### Q4. Why is this a machine learning problem?

The decision rule is learned from labeled examples rather than manually specifying every suspicious phrase. The fitted model applies the learned rule to new inputs.

### Q5. Is it supervised, unsupervised, or reinforcement learning?

Supervised learning: every training article has a file-derived class label. There is no reward-based interaction or unlabeled clustering stage.

### Q6. Is it classification or regression?

Binary classification, because the output is one of two discrete labels. We do not predict a continuous quantity.

### Q7. What are the input and output?

The input is a required article body and optional headline. The output includes the label, class ID, model name, and a limitations note.

### Q8. What does NLP mean in this project?

Natural Language Processing is the handling of written language as machine-processable data. Our implementation normalizes text and extracts word, phrase, and linguistic features.

### Q9. Does the model determine whether an article is factually true?

It predicts resemblance to dataset classes. It does not retrieve supporting evidence, check claims against sources, or independently establish truth.

### Q10. Why display "Potentially fake" rather than "Definitely fake"?

The model can be wrong, especially on unfamiliar sources. The wording reflects uncertainty and the difference between pattern classification and factual verification.

### Q11. What are your project's main contributions?

Dataset auditing, duplicate reduction, a shared feature pipeline, reproducible model comparison, saved predictions and errors, and browser/API integration of the selected artifact.

### Q12. Is the algorithm itself new?

No. The contribution is the implemented and evaluated workflow using established algorithms. We should not claim invention of SVM, Random Forest, or XGBoost.

## 2. Dataset and labels

### Q13. Which dataset did you use?

The actual inputs are the local `true.csv` and `fake.csv` files. Their counts and schema match the ISOT dataset documentation; exact local download provenance is not recorded.

### Q14. How many articles are there before and after cleaning?

There are 44,898 raw records and 38,358 retained articles. Cleaning removes 6,540 records.

### Q15. What columns does the original dataset contain?

`title`, `text`, `subject`, and `date`. The loader adds labels, identifiers, duplicate hashes, and the combined article text.

### Q16. What is the target variable?

`label`: 0 for articles from `true.csv`, and 1 for articles from `fake.csv`. It is derived from file membership.

### Q17. Which class is positive in your evaluation?

Class 1, Potentially fake. Fake precision and recall therefore refer to detection of fake-labeled articles.

### Q18. Did you independently verify the labels?

No. We inherit the supplied dataset labels. Label quality is a limitation, and the model cannot repair incorrect labels automatically.

### Q19. Why exclude subject and date from the classifier?

Subject categories differ between the two files and could reveal the class directly. Date is also excluded to focus the experiment on article text.

### Q20. Does excluding metadata remove all source bias?

No. Publisher names, attribution, topic words, and formatting remain in the text. Excluding metadata prevents one shortcut but does not eliminate others.

### Q21. Is the cleaned dataset balanced?

It contains approximately 54.53% genuine and 45.47% fake-labeled articles. It is moderately uneven, and stratification preserves these proportions.

### Q22. Did you use SMOTE, oversampling, undersampling, or class weights?

No such balancing intervention is configured in the implemented experiments. Macro F1 gives both class F1 values equal weight during selection.

### Q23. What is the majority-class accuracy baseline?

Always predicting Genuine would achieve approximately 54.53% on the cleaned corpus. This is a calculated reference, not a separately trained baseline experiment.

### Q24. What does the source-marker audit show?

Reuters appears in about 99.82% of raw genuine bodies and 1.31% of raw fake-labeled bodies. This is a strong label-associated publisher cue.

### Q25. What are the subject categories?

Genuine subjects are `politicsNews` and `worldnews`. Fake-labeled subjects include `News`, `politics`, `left-news`, `Government News`, `US_News`, and `Middle-east`.

### Q26. What did article-length analysis show?

Cleaned genuine bodies average about 384.74 words and fake-labeled bodies about 425.01. Length alone is not proof of truth or falsehood.

### Q27. Where can you prove the dataset numbers?

Use [dataset_summary.json](reports/dataset_summary.json) for cleaning counts and [exploration_summary.json](reports/exploration_summary.json) for source, subject, and length analysis.

### Q28. Can this dataset establish performance on current news?

No. Historical random article splits do not test future news, unseen publishers, or different domains. Those require additional evaluation data.

## 3. Cleaning, duplicates, and exploratory analysis

### Q29. What does data preprocessing mean?

Preparing raw records for learning through cleaning, auditing, feature construction, and appropriate partitioning. In this project it also includes duplicate reduction.

### Q30. How do you handle missing text?

Missing titles and bodies become empty strings. Empty cleaned bodies are removed; an absent title is permitted.

### Q31. What exactly does `clean_text` do?

It decodes HTML entities, removes HTML tags, applies Unicode NFKC normalization, collapses whitespace, and trims the text. It preserves case and punctuation.

### Q32. Why preserve case and punctuation?

The linguistic branch measures uppercase and punctuation patterns. TF-IDF separately lowercases its input internally.

### Q33. Did you use stemming or lemmatization?

No. Neither is implemented. Adding them would be a new experiment whose benefit should be measured rather than assumed.

### Q34. Did you remove stop words?

No explicit stop-word list is configured. Document-frequency filtering removes sufficiently common terms, but this is different from a stop-word list.

### Q35. How are duplicates detected?

We lowercase and tokenize normalized body text, then compute a SHA-256 hash. Matching hashes identify repeated normalized bodies.

### Q36. Why detect duplicates before splitting?

Copies in both training and test data could make evaluation unrealistically easy. Removing them before splitting reduces that overlap.

### Q37. What is the 200-word-prefix rule?

We hash the first 200 normalized body words and remove repeated prefixes. It detects one kind of near duplication, not every paraphrase.

### Q38. How do you handle duplicate content with conflicting labels?

The loader removes all records in a conflicting body or prefix group. No such conflicts were recorded in the saved dataset audit.

### Q39. Give the cleaning removal counts.

631 empty bodies, 5,899 normalized body duplicates, and 10 repeated prefixes were removed. Conflicting-label body and prefix removals were both zero.

### Q40. Are duplicate hashes classifier features?

No. They are identifiers for auditing and split-integrity checks. Feeding them into the classifier would be inappropriate for article understanding.

### Q41. What is EDA and what did you inspect?

Exploratory Data Analysis examines data before interpreting models. We inspect class counts, topics, lengths, and publisher markers, with a dataset overview figure.

### Q42. Did you remove unusually long or short articles as outliers?

No statistical length clipping or outlier-removal rule is implemented. Nonempty bodies remain in training, even if short or unusually long.

### Q43. Why is preprocessing before splitting not always leakage?

Fixed text normalization and duplicate auditing do not learn a predictive vocabulary. TF-IDF, chi-square selection, and scaling learn from data and must fit only on training partitions.

## 4. Features, TF-IDF, selection, and scaling

### Q44. Why convert text to numbers?

The classifiers operate on numerical vectors. Our representation combines word/phrase weights with measurements of length, capitalization, and punctuation.

### Q45. What is TF-IDF?

Term Frequency-Inverse Document Frequency weights terms using their occurrence in an article and rarity across training articles. See the [vectorizer documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

### Q46. What weighting formula does your configuration use?

For positive count `c`, `tf = 1 + ln(c)` and `idf = ln((1+N)/(1+df)) + 1`. Their product is L2-normalized per document.

### Q47. What are unigrams and bigrams?

Unigrams are individual tokens; bigrams are adjacent token pairs. The vectorizer uses `ngram_range=(1,2)` to represent both words and short phrases.

### Q48. Why use bigrams?

They can preserve short phrase information that individual words miss. Their inclusion is a representation choice, not proof of full semantic understanding.

### Q49. What does `max_features=12000` mean?

The vectorizer caps its vocabulary at 12,000 terms before chi-square selection. It is a maximum, not a guarantee for every fold or small dataset.

### Q50. What do `min_df=3` and `max_df=0.98` mean?

Terms must occur in at least three fitted training documents and no more than 98% of them. These filters limit very rare and extremely common terms.

### Q51. What is sublinear term frequency?

It uses `1 + ln(count)` for present terms. Repetition therefore increases the weight more slowly than raw counts.

### Q52. Why use sparse matrices?

Most vocabulary terms are absent from a particular article. Sparse storage represents mostly the nonzero entries, reducing memory usage.

### Q53. What is chi-square feature selection?

It ranks nonnegative features by association with labels and retains the requested number. Our `SelectKBest(chi2)` branch selects terms from training TF-IDF vectors.

### Q54. What does a high chi-square score mean?

The feature is associated with class labels in training data. It does not prove misinformation, show association direction, or establish causal importance.

### Q55. Why select 3,000 terms?

Validation experiments compared 3,000, 6,000, and all available terms using SVM. The 3,000-term configuration with linguistic features achieved the highest validation macro F1.

### Q56. Did you independently optimize vocabulary size for each model?

No. SVM experiments selected the shared term count. This improves comparability but does not guarantee the best vocabulary size for each tree ensemble.

### Q57. What are the eight linguistic features?

Word count, approximate sentence count, mean word length, mean sentence length, uppercase ratio, punctuation ratio, exclamations per word, and questions per word.

### Q58. How do you count sentences?

We count punctuation groups matching `[.!?]+`, with a minimum of one. It is a heuristic that can miscount abbreviations or unconventional punctuation.

### Q59. How do you avoid division by zero?

The extractor uses minimum-one denominators for word count, sentence count, letters, and text length where needed.

### Q60. Why standardize linguistic features?

Lengths and ratios have different scales. Standardization makes their scale more comparable for the shared pipeline, particularly the linear model.

### Q61. What is the scaling formula?

`z = (x - mean_train) / std_train`. The training means and scales are saved and reused for validation, test, and new submissions.

### Q62. Do tree models require standardization?

Usually tree splits do not require it. We retain the same scaled representation for all candidates, while the linear SVM is more sensitive to feature scale.

### Q63. Why select terms before appending standardized features?

Chi-square requires nonnegative inputs, while standardized linguistic values can be negative. It receives only the nonnegative TF-IDF branch.

### Q64. Why are there 3,008 features?

The saved full-training representation contains 3,000 selected term features plus eight linguistic measurements. It is not a vocabulary of 3,008 words.

### Q65. What is `SafeChi2Selector`?

A custom wrapper that limits requested `k` to the available vocabulary size. This prevents failure when a small fitted corpus has fewer terms.

### Q66. What are `Pipeline` and `FeatureUnion`?

`Pipeline` applies sequential transformations and classification. `FeatureUnion` concatenates the outputs of the text and linguistic branches.

### Q67. What happens to an unseen word during prediction?

It contributes no new TF-IDF vocabulary coordinate. Known terms still contribute, and linguistic features are extracted from the submitted text.

### Q68. Can the model predict if all words are outside its vocabulary?

The linguistic branch still supplies features, so prediction is possible. Its reliability for such an unfamiliar input is not established.

### Q69. Why not use embeddings or BERT?

They are possible extensions. The current experiment provides an interpretable classical feature workflow with manageable computation; it does not establish superiority over transformers.

### Q70. What is an ablation experiment?

It removes a component to measure its contribution. We compare selected TF-IDF alone against the same selected terms with linguistic features, using SVM.

### Q71. What did the feature experiments show?

The 3,000-term-plus-linguistic variant scored 0.997371 validation macro F1; the text-only variant scored 0.995968. The observed improvement is approximately 0.001403.

### Q72. Does a smaller vocabulary always train faster?

No. The saved all-terms SVM feature experiment trained faster than some selected-term variants. Selection adds work; timing and predictive quality must be measured separately.

## 5. Splits, leakage, cross-validation, and tuning

### Q73. Why have training, validation, and test partitions?

Training learns parameters, validation supports feature/deployment selection, and test provides final held-out reporting. Reusing the test set for selection would bias that report.

### Q74. What are your split proportions and counts?

Approximately 70/15/15: 26,850 training, 5,754 validation, and 5,754 test articles. Integer rounding explains minor differences from exact percentages.

### Q75. What is stratification?

Partitioning while preserving approximately the same class proportions. Both split stages use `stratify=label` and seed 42.

### Q76. Why random seed 42?

It makes the chosen randomized procedure reproducible under compatible conditions. The number is conventional and has no special predictive property.

### Q77. What is data leakage?

Information from evaluation data influences training or model selection in a way that makes evaluation optimistic. Duplicate overlap and globally fitted vocabulary are examples.

### Q78. How do you limit leakage?

Remove audited duplicates before splitting, check disjoint hashes, fit learned preprocessing only on training data, and keep test results out of the deployment choice.

### Q79. What is cross-validation?

Repeatedly train on some folds and score on the remaining fold. We use three stratified folds with shuffling on the training-only tuning sample.

### Q80. Why tune on 6,000 rows instead of all training data?

It reduces tuning cost. After parameter selection, each model is fitted on all 26,850 training articles. Full-training tuning is available with `--cv-rows 0`.

### Q81. How many rows are used in a tuning fold?

With the saved 6,000-row sample and three folds, approximately 4,000 are used for fitting and 2,000 for fold evaluation per fit.

### Q82. Is preprocessing refitted within each CV fold?

Yes. `GridSearchCV` receives the whole pipeline. Vocabulary, selector, and scaler fit only on that fold's training portion.

### Q83. What is `GridSearchCV` optimizing?

Macro F1 over the specified parameter candidates. It does not search every possible model configuration or optimize test accuracy.

### Q84. What parameters were tuned?

SVM `C` in {0.5, 2.0}; Random Forest depth in {25, unlimited}; XGBoost depth in {3, 6}.

### Q85. How many CV fits were performed?

Two candidates times three folds gives six fits per model and 18 across three families, excluding feature experiments and final full-training fits.

### Q86. What is the difference between a parameter and a hyperparameter?

Parameters are learned, such as SVM weights. Hyperparameters are settings supplied to the learning procedure, such as `C`, tree count, and depth.

### Q87. Why `refit=False` in grid search?

The search selects settings on the smaller sample. Code then constructs and fits a pipeline separately on the entire training partition.

### Q88. Is this nested cross-validation?

No. Term count uses the separate validation set, while model parameters use training CV. It is not a fully nested estimate of the whole selection procedure.

### Q89. Does validation reuse introduce any concern?

Yes. Feature choices and deployment choice reuse validation data, which can introduce selection optimism. The untouched test partition remains separate.

### Q90. What is overfitting and how would you investigate it?

A model fits training-specific patterns poorly transferable to new data. Investigate learning curves, training-versus-validation performance, repeated seeds, and external source-disjoint evaluation.

### Q91. Can you claim no overfitting because accuracy is high?

No. Source cues can occur in both sides of a random split. We have not established publisher-independent or chronological generalization.

### Q92. Is the deployment artifact refitted on training plus validation?

No. It remains fitted on training data only. This preserves correspondence between the saved artifact and the reported test results.

## 6. Linear SVM

### Q93. Define Linear SVM.

A supervised classifier that learns a linear separating boundary with a margin-based loss. Our implementation is `LinearSVC`. See [official documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.LinearSVC.html).

### Q94. What is a hyperplane?

The boundary `w^T x + b = 0`. Its sign separates the two classes in a linear binary classifier.

### Q95. What is the margin?

The separation around the decision boundary. SVM training balances a larger margin against penalties for violations.

### Q96. What does `C` control?

The strength of the loss penalty relative to regularization. Larger `C` places more emphasis on fitting training examples.

### Q97. What loss does this SVM use?

The default squared hinge loss: `max(0, 1 - y*f(x))^2`, with mathematical labels -1/+1. Dataset labels remain 0/1.

### Q98. Why use Linear SVM for text?

It is a useful sparse-text baseline. In our experiment it provides high macro F1 with the smallest saved artifact.

### Q99. Why not use an RBF SVM?

Only the linear variant was evaluated. An RBF comparison would require additional tuning and measured computational cost.

### Q100. Which SVM settings were selected?

`C=2.0`, `max_iter=5000`, seed 42. Its best tuning macro F1 is 0.993445.

### Q101. Does LinearSVC provide probabilities?

It provides decision-function scores rather than native `predict_proba`. Calibration would be a separate procedure.

### Q102. Is LinearSVC identical to `SVC(kernel="linear")`?

No. Their solvers and defaults differ; do not substitute their names when describing the implementation.

## 7. Random Forest

### Q103. Define Random Forest.

An ensemble of randomized decision trees that combines their outputs. See the [classifier documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

### Q104. What is bagging?

Learning multiple models using sampled datasets and aggregating their predictions. Our forest uses the default bootstrap sampling.

### Q105. What does bootstrap mean?

Sampling training records with replacement. A sample can repeat some records and omit others.

### Q106. Why randomly sample features?

It encourages diversity between trees and reduces their dependence on identical split candidates.

### Q107. How does this forest predict?

The sklearn classifier averages tree class probabilities and selects the class with the largest average.

### Q108. What is Gini impurity?

`1 - sum(p_class^2)`. Zero means a pure node. Our forest uses the default Gini split criterion.

### Q109. What does `max_features="sqrt"` mean?

About the square root of the input feature count is considered per split; roughly 54 for 3,008 features.

### Q110. What does `min_samples_leaf=2` do?

It requires at least two training samples in each resulting leaf, limiting very small leaves.

### Q111. What does unlimited depth mean?

No explicit depth cap; other stopping conditions still apply. The search selected it over depth 25.

### Q112. Why use 120 trees?

It is a fixed experimental configuration, not an independently optimized tree count. More trees need evaluation against runtime and size.

### Q113. Does Random Forest prevent all overfitting?

No. It can still learn source-specific patterns. Ensemble averaging is not a guarantee of external generalization.

## 8. XGBoost and ensemble comparisons

### Q114. Define XGBoost.

A gradient-boosted tree implementation that optimizes an additive prediction with regularization. See [official API documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier).

### Q115. What is boosting?

Adding models sequentially to improve the combined objective, rather than training independent members for averaging.

### Q116. How does Random Forest differ from XGBoost?

Forest trees use randomized bagging; boosted trees contribute sequentially to an additive predictor. Both can represent nonlinear patterns.

### Q117. What are the XGBoost settings?

140 trees, depth 6, learning rate 0.12, row sampling 0.9, column sampling 0.8, histogram method, seed 42.

### Q118. What does learning rate mean?

It scales each boosting update. We fixed it at 0.12 and did not search learning-rate values.

### Q119. What are `subsample` and `colsample_bytree`?

Row and column sampling fractions. The configured values are 0.9 and 0.8, respectively.

### Q120. What does `tree_method="hist"` mean?

Histogram-based split construction. Our run uses this method without a GPU configuration.

### Q121. What are objective, evaluation metric, and selection metric?

Binary logistic is the classifier objective, `logloss` is configured for evaluation, and grid-search selection uses macro F1.

### Q122. Did you use early stopping?

No early-stopping configuration or validation `eval_set` is provided in the current training code.

### Q123. Why compare these three algorithm families?

They compare a linear sparse-text boundary, randomized tree ensembles, and boosted trees using the same representation and partitions.

### Q124. Did you train Logistic Regression, KNN, or a standalone Decision Tree?

No. The implemented candidates are Linear SVM, Random Forest, and XGBoost. Trees occur inside the ensembles, not as a separately evaluated classifier.

## 9. Metrics and interpretation

### Q125. Define accuracy.

`(TP+TN)/(TP+TN+FP+FN)`: the fraction of correctly classified articles. It alone can hide class imbalance or error trade-offs.

### Q126. Define precision.

For the fake class, `TP/(TP+FP)`: how many articles predicted fake are actually fake-labeled in the dataset.

### Q127. Define recall.

For the fake class, `TP/(TP+FN)`: how many fake-labeled articles are detected.

### Q128. Define F1.

`2*precision*recall/(precision+recall)`. It balances precision and recall through their harmonic mean.

### Q129. Why macro F1 for selection?

It averages the two class F1 values equally, so each class contributes equally despite different supports.

### Q130. How does weighted F1 differ from macro F1?

Weighted F1 averages class F1 using class supports; macro F1 gives each class equal weight. Neither is simply overall accuracy.

### Q131. What does support mean in a classification report?

The number of actual examples of a class. Test supports are 3,138 genuine and 2,616 fake-labeled articles.

### Q132. What are TP, TN, FP, and FN here?

TP = fake detected; TN = genuine recognized; FP = genuine flagged fake; FN = fake missed and predicted genuine.

### Q133. Which mistake is more serious?

It depends on use. FP can unfairly flag genuine reporting; FN can miss misleading material. A deployment policy should specify these costs.

### Q134. Define ROC and ROC AUC.

ROC plots `TPR=TP/(TP+FN)` against `FPR=FP/(FP+TN)` across thresholds. AUC summarizes score ranking; it is not accuracy at one threshold.

### Q135. How can AUC be nearly perfect while classification still makes errors?

Ranking across thresholds can be excellent while the model's default decision rule misclassifies a few examples. The metrics measure different properties.

### Q136. What does AUC 0.5 mean?

Chance-level ranking under the usual binary interpretation. It does not necessarily mean exactly 50% classification accuracy at a chosen threshold.

### Q137. Why also report average precision?

It summarizes precision-recall ranking and complements ROC AUC. It uses prediction scores rather than only hard class labels.

### Q138. Why not report RMSE?

Our primary task is classification. Accuracy, class metrics, macro F1, ROC AUC, and average precision directly describe the evaluated classification behavior.

### Q139. Are the scores calibrated confidence probabilities?

No calibration study is recorded. SVM margins are not probabilities, and tree probability outputs do not automatically establish empirical calibration or factual confidence.

### Q140. Why do the printed reports show 0.9974 instead of 99.74?

The reports print fractions rounded to four decimals. Multiplying by 100 converts a fraction into a percentage.

## 10. Each model's results and the deployment decision

### Q141. Summarize Linear SVM's report.

Test accuracy is 99.7393%, macro F1 0.997371, fake precision 0.997323, and fake recall 0.996942. It makes 15 errors: seven FP and eight FN.

### Q142. Summarize Random Forest's report.

Test accuracy is 99.7046%, macro F1 0.997021, fake precision 0.997702, and fake recall 0.995795. It makes 17 errors: six FP and eleven FN.

### Q143. Summarize XGBoost's report.

Test accuracy is 99.7393%, macro F1 0.997371, fake precision 0.998467, and fake recall 0.995795. It makes 15 errors: four FP and eleven FN.

### Q144. Which model has the highest test accuracy?

Linear SVM and XGBoost tie at 99.7393%. Their confusion matrices differ despite equal total errors.

### Q145. Which model has the highest test macro F1?

Linear SVM at full precision. Its 0.997371401 is slightly higher than XGBoost's 0.997370896, though both round to 0.997371.

### Q146. Which model has the highest fake recall and precision?

SVM has the highest fake recall, about 0.996942. XGBoost has the highest fake precision, about 0.998467.

### Q147. Which model has the highest ROC AUC?

XGBoost, approximately 0.999987. It is below 1.0, although rounding to four decimal places can conceal that distinction.

### Q148. Why was Random Forest chosen for the web application?

The predefined code criterion is highest validation macro F1. RF scored 0.9970203488261405, slightly above XGBoost's 0.9970201563947237, and was saved as `best_pipeline.pkl`.

### Q149. Why not choose SVM or XGBoost after seeing their higher test accuracy?

The test set is reserved for reporting. Selecting on its scores would make it part of model selection and require another independent evaluation set.

### Q150. Is RF's validation lead meaningful?

Its lead over XGBoost is only about 0.000000192 macro F1. This does not establish statistically meaningful superiority; it merely determines the winner under the implemented rule.

### Q151. Was Random Forest chosen because it is faster or smaller?

No. The saved measurements show SVM is smaller and has lower batch latency. RF was selected by validation macro F1, not resource efficiency.

### Q152. Could SVM be a reasonable deployment alternative?

Yes, if a predefined policy includes resource cost and validation quality. It would require documenting that new selection policy and independent assessment as appropriate.

### Q153. Why does XGBoost win CV but not deployment?

CV selects settings within each family on a training-only sample. Deployment compares full-training fitted candidates on a different validation partition.

### Q154. Why do feature-experiment and final SVM validation scores differ?

Feature experiments use `C=1`, whereas the tuned final SVM uses `C=2`. They are different fitted configurations.

### Q155. Does 99.7% accuracy prove real-world reliability?

No. Source/style shortcuts and a random article split limit the conclusion. External, publisher-disjoint, and chronological evaluation are needed.

### Q156. What are the actual wrong predictions?

The per-model `*_errors.csv` files identify them. RF wrongly flags the genuine-labeled Clinton headline shown in the report; the error's exact cause is unverified.

### Q157. Why not claim that satire or sensational wording caused every error?

The reports do not prove those causes. We should inspect articles and conduct controlled tests before giving a causal explanation.

### Q158. Can the models be compared fairly?

They share splits, representation, metric, candidate count, and folds. Model capacity differs, and the small searches are not exhaustive or equally optimal for every family.

## 11. Web application, architecture, and deployment

### Q159. What is Flask and why use it?

It is the Python web framework used to expose the saved classifier through routes, templates, and JSON responses. It integrates directly with the Python pipeline.

### Q160. Explain the online prediction flow.

Validate input, clean and combine title/body, transform using fitted pipeline steps, predict with RF, then return the class and explanatory response.

### Q161. Is preprocessing retrained for each submission?

No. The fitted vocabulary, selected columns, scaler parameters, and classifier are reused. The submitted article only goes through transformation and prediction.

### Q162. Why save the complete pipeline rather than only the classifier?

The classifier expects the same feature coordinates used during training. Saving preprocessing with it preserves vocabulary, selected terms, scaling, and transformation order.

### Q163. What is pickle and what should you know about loading it?

Python serialization used for model persistence here. Loading untrusted pickle can execute code, so only trusted project artifacts should be deserialized.

### Q164. What are the endpoints?

`GET /` displays the browser form; `POST /` predicts and renders a page; `POST /api/predict` predicts from JSON.

### Q165. What are the input limits?

Optional title at most 300 characters; required body at most 50,000 characters and at least ten cleaned word-token matches. Total request limit is 256 KiB.

### Q166. Why reject headline-only input?

The supported workflow expects an article body. The interface requires ten body words; headline-only performance was not evaluated separately.

### Q167. Is the ten-word minimum scientifically optimized?

No. It is an interface policy, not an experimentally selected threshold. Training retains any nonempty cleaned body, creating a documented mismatch.

### Q168. What HTTP errors does the application return?

400 for invalid input, 413 for requests exceeding the total request-size limit, and 503 for valid prediction input when the model is unavailable.

### Q169. Does exceeding 50,000 article characters always return 413?

No. A parsed request exceeding the field limit returns 400. The separate overall request-size cap triggers 413.

### Q170. What fields are returned by the API?

`label`, `class_id`, `model`, and `note`. No confidence percentage is returned.

### Q171. Are submitted articles stored in a database?

No. The application does not persist submissions and has no implemented database storage workflow.

### Q172. Can the app automatically switch between models for each article?

No. It loads one selected pipeline at startup. There is no model selector or per-article ensemble of the three candidates.

### Q173. What happens after retraining?

Generated artifacts and reports are overwritten. Restart Flask to load the new saved pipeline; the current in-memory model does not update automatically.

### Q174. Is it publicly hosted or production-ready?

The recorded deployment is localhost with debug disabled. Public hosting, load testing, monitoring, and production serving are not implemented outcomes.

### Q175. How would you demonstrate the project in viva?

Start Flask, show a full-article submission, explain the result and limitations, show an invalid-input case, and open the saved metrics and selected-model manifest.

### Q176. What if a live example is classified incorrectly?

Explain that the system recognizes learned corpus patterns. Inspect the input and compare against known limitations; do not claim the prediction establishes truth.

## 12. Code, reports, reproducibility, and checks

### Q177. What does each source file do?

`data.py` loads/audits/splits; `preprocessing.py` cleans; `features.py` transforms; `train.py` tunes/saves; `evaluate.py` reports; `app.py` serves predictions.

### Q178. What is `models/manifest.json`?

A record of the selected model, term count, seed, versions, dataset counts, parameters, CV scores, validation metrics, and timings.

### Q179. What is `reports/split_manifest.csv`?

It records original row identifiers, body/prefix hashes, labels, and partition membership. It supports checks that audited duplicate groups do not cross partitions.

### Q180. Which files contain each model's report?

`reports/classification_reports/<model>.txt` and `.json`; `model_comparison.csv`; `*_test_predictions.csv`; `*_errors.csv`; and model confusion-matrix images.

### Q181. How can someone reproduce the workflow?

Use compatible dependencies and the same dataset, seed, code, and commands. Hash auditing and split manifests help compare runs; hardware affects timing.

### Q182. What versions were recorded?

Python 3.13.3, sklearn 1.7.2, XGBoost 3.4.1, NumPy 2.3.3, and pandas 2.3.2 in the manifest. Flask 3.1.2 is pinned in requirements.

### Q183. Was a GPU used and how many workers were configured?

No GPU training is configured. Tree models default to one worker, with `--threads` allowing a different worker count.

### Q184. What did the software tests check?

Existing tests cover serialization parity, browser/API behavior, input limits, escaped form content, missing artifacts, split integrity, and deployment consistency. They are not external accuracy tests.

### Q185. Why do some tests use a small SVM rather than the deployed RF?

They test generic pipeline and Flask behavior quickly using synthetic examples. Separate artifact checks inspect the actual deployment winner and prediction consistency.

### Q186. Does rerunning evaluation retrain models?

No. It loads saved models and held-out articles and regenerates evaluation outputs. Training is a separate module.

### Q187. What does the prediction latency measurement include?

Batch `predict` over test articles divided by their count. It excludes model loading, separate score extraction, and HTTP overhead; it is not measured browser latency.

### Q188. Are original training hardware details available?

CPU model and RAM are not recorded. Give recorded timing but avoid claims of a hardware-independent speed benchmark.

### Q189. What is `--reuse-feature-experiments`?

It reuses completed feature results after an interrupted run if split manifests match. It should not be used after changing feature/preprocessing code.

## 13. Limitations, ethics, and future scope

### Q190. What is the biggest weakness?

Labels are strongly associated with source and style. High random-split performance may reflect those shortcuts instead of generalizable factual signals.

### Q191. How would you test dependence on Reuters?

Mask attribution consistently, rerun controlled training/evaluation, and compare results. Simply deleting it from one submitted article does not quantify general dependence.

### Q192. What is a publisher-disjoint split?

Publishers in evaluation are absent from training. It tests transfer beyond shared publisher conventions and requires reliable publisher/group annotations.

### Q193. What is a chronological split?

Train on earlier articles and evaluate on later ones. It tests temporal transfer; date parsing and corpus coverage need careful auditing first.

### Q194. What are data drift and concept drift?

Data drift changes input distributions; concept drift changes the relationship between inputs and labels. Both can reduce performance after deployment.

### Q195. How would you improve duplicate detection?

Use similarity or embeddings to identify related articles and keep groups in one partition. The current exact/prefix hashes cannot detect all paraphrases.

### Q196. Can it handle Hindi, other languages, images, or video?

These are outside the evaluated English text scope. Processing such input does not establish reliable multilingual or multimodal classification.

### Q197. Could genuine sensational news or fluent misinformation confuse it?

Yes. Text style is not factual evidence. These cases require explicit evaluation and ideally evidence-based verification.

### Q198. What would evidence retrieval add?

It could locate documents supporting or contradicting extracted claims. This is future work, not an existing step in the prediction pipeline.

### Q199. What is calibration and what is abstention?

Calibration tests whether predicted probabilities match observed frequencies. Abstention lets a system withhold a label when uncertainty or unfamiliarity is high. Neither is implemented here.

### Q200. What ethical concern should you mention?

Incorrect labels can damage trust or suppress legitimate reporting. Present results as screening signals and support review with sources and evidence.

### Q201. How would you strengthen your conclusions?

Use independent datasets, publisher/chronological splits, repeated seeds, wider tuning, uncertainty intervals, and source-masking experiments. Keep new evaluation data separate from selection.

### Q202. What would you say about statistical significance?

The current work reports point estimates without confidence intervals or significance tests. Tiny score differences should not be presented as established superiority.

### Q203. What did you personally contribute?

Describe only your real work, such as preprocessing, experiments, interface integration, analysis, or documentation. This guide cannot establish your individual contribution or team roles.

### Q204. What would you improve first?

Prioritize publisher-disjoint external evaluation and source-marker ablation, because they address the main uncertainty behind the near-perfect corpus scores.

## 14. Formula and worked-example revision

For this project, class 1 is positive:

```text
Accuracy       = (TP + TN) / N
Fake precision = TP / (TP + FP)
Fake recall    = TP / (TP + FN)
Fake F1        = 2TP / (2TP + FP + FN)
Macro F1       = (Genuine F1 + Fake F1) / 2
FPR            = FP / (FP + TN)
TPR            = TP / (TP + FN)
Standardize    = (x - training mean) / training standard deviation
Gini impurity  = 1 - sum(class_probability^2)
```

For Random Forest, `TN=3132`, `FP=6`, `FN=11`, `TP=2605`:

```text
Accuracy       = 5737 / 5754          = 0.99704553 = 99.7046%
Fake precision = 2605 / 2611          = 0.99770203
Fake recall    = 2605 / 2616          = 0.99579511
Fake F1        = 5210 / 5227          = 0.99674766
Genuine F1     = 6264 / (6264 + 17)   = 0.99729342
Macro F1       = average of class F1 = 0.99702054
```

Be ready to explain the difference between a fraction and a percentage. An increase of 0.001 in F1 is not automatically a 0.1% relative improvement; identify whether you mean an absolute metric difference or a relative percentage.

## 15. Answers to rehearse carefully

**Why Random Forest for the web application?** "We selected the saved pipeline by validation macro F1 before test evaluation. RF won that criterion by a very small margin. Its test accuracy is 99.7046%. SVM and XGBoost performed slightly better on test accuracy, but using test scores to revise the winner would turn the test set into selection data. RF was not selected because it was the fastest or smallest model."

**Why is accuracy so high?** "The corpus has strong label-associated source and style signals, including Reuters attribution. Our duplicate controls improve split integrity, but the random split still shares publisher conventions. These scores demonstrate strong classification of this corpus, and external evaluation is needed for broader claims."

**What is the model actually learning?** "It learns relationships between labels and selected word/phrase weights plus eight linguistic measurements. It may learn source identity and formatting as well as other language patterns. It has no independent factual evidence."

**What is your strongest result?** "A reproducible complete workflow with recorded feature experiments, three-model evaluation, auditable errors, and a working saved-pipeline application. The measured predictive results are strong within the defined dataset scope."

**What claims should you avoid?** Do not claim perfect accuracy, proven truth verification, calibrated confidence, exhaustive tuning, deployment on unseen publishers, transformer comparisons, or algorithms that were not implemented. Describe proposed extensions as future work.

## 16. Commands and evidence for the viva

Run these from the project directory with compatible dependencies installed:

```powershell
python app.py
python -m src.explore
python -m src.train
python -m src.evaluate
python -m unittest discover -s tests -v
```

The first command starts the application at `http://127.0.0.1:5000` when artifacts exist. Exploration, training, and evaluation generate or overwrite reports; only run them when you intend to regenerate those outputs. Training can take several minutes. The application is not started merely by creating this guide.

| Professor asks for evidence | Open this file |
| --- | --- |
| Dataset cleaning counts | [dataset_summary.json](reports/dataset_summary.json) |
| Source-marker or article-length audit | [exploration_summary.json](reports/exploration_summary.json) |
| Split integrity | [split_manifest.csv](reports/split_manifest.csv) |
| Winner, settings, and versions | [manifest.json](models/manifest.json) |
| Deployment decision | [validation_comparison.csv](reports/validation_comparison.csv) |
| Test comparison and timing | [model_comparison.csv](reports/model_comparison.csv) |
| Vocabulary and linguistic ablation | [feature_experiments.csv](reports/feature_experiments.csv) |
| Term association | [feature_scores.csv](reports/feature_scores.csv) |
| RF mistakes | [random_forest_errors.csv](reports/random_forest_errors.csv) |
| Application validation and routes | [app.py](app.py) |
| Actual model construction and tuning | [src/train.py](src/train.py) |

## 17. Full model reports and comparison tables

The following appendix is populated from the saved reports so you can revise each model's actual class metrics, confusion counts, selection scores, and measured costs together.

### Validation selection and training-only CV

| Model | Best CV macro F1 | Validation accuracy (%) | Validation macro F1 (full saved precision) | Selected tuned setting |
| --- | ---: | ---: | ---: | --- |
| Linear SVM | 0.993445 | 99.6698 | 0.9966698016292159 | `C=2.0` |
| Random Forest | 0.995127 | 99.7046 | 0.9970203488261405 | `max_depth=None` |
| XGBoost | 0.996300 | 99.7046 | 0.9970201563947237 | `max_depth=6` |

Here `None` means unlimited maximum depth. CV scores use the 6,000-row tuning sample, whereas validation scores come from models fitted on the full training partition.

### Held-out ranking and resource comparison

| Model | Test ROC AUC | Average precision | Final fit (s) | Tuning (s) | Batch predict/article (ms) | Artifact (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 0.999802 | 0.999794 | 56.03 | 55.47 | 0.7265 | 0.6012 |
| Random Forest | 0.999939 | 0.999929 | 67.06 | 66.50 | 0.7918 | 15.3605 |
| XGBoost | 0.999987 | 0.999985 | 52.04 | 92.54 | 0.7642 | 0.8082 |

Timing excludes feature experiments from final fit time. Batch prediction measurements are not HTTP response times, and original hardware specifications are not recorded. Artifact size is bytes divided by `1024^2`.

### Linear SVM: full saved classification report

Test accuracy: **99.7393%**. Test macro F1: **0.997371401**.

```text
precision    recall  f1-score   support

         Genuine     0.9975    0.9978    0.9976      3138
Potentially fake     0.9973    0.9969    0.9971      2616

        accuracy                         0.9974      5754
       macro avg     0.9974    0.9974    0.9974      5754
    weighted avg     0.9974    0.9974    0.9974      5754
```

| Actual / predicted | Genuine (0) | Potentially fake (1) |
| --- | ---: | ---: |
| Genuine (0) | 3131 TN | 7 FP |
| Potentially fake (1) | 8 FN | 2608 TP |

Errors: **15**. Source: [classification report](reports/classification_reports/linear_svm.txt), [predictions](reports/linear_svm_test_predictions.csv), and [errors](reports/linear_svm_errors.csv).

### Random Forest: full saved classification report

Test accuracy: **99.7046%**. Test macro F1: **0.997020541**.

```text
precision    recall  f1-score   support

         Genuine     0.9965    0.9981    0.9973      3138
Potentially fake     0.9977    0.9958    0.9967      2616

        accuracy                         0.9970      5754
       macro avg     0.9971    0.9969    0.9970      5754
    weighted avg     0.9970    0.9970    0.9970      5754
```

| Actual / predicted | Genuine (0) | Potentially fake (1) |
| --- | ---: | ---: |
| Genuine (0) | 3132 TN | 6 FP |
| Potentially fake (1) | 11 FN | 2605 TP |

Errors: **17**. Source: [classification report](reports/classification_reports/random_forest.txt), [predictions](reports/random_forest_test_predictions.csv), and [errors](reports/random_forest_errors.csv).

### XGBoost: full saved classification report

Test accuracy: **99.7393%**. Test macro F1: **0.997370896**.

```text
precision    recall  f1-score   support

         Genuine     0.9965    0.9987    0.9976      3138
Potentially fake     0.9985    0.9958    0.9971      2616

        accuracy                         0.9974      5754
       macro avg     0.9975    0.9973    0.9974      5754
    weighted avg     0.9974    0.9974    0.9974      5754
```

| Actual / predicted | Genuine (0) | Potentially fake (1) |
| --- | ---: | ---: |
| Genuine (0) | 3134 TN | 4 FP |
| Potentially fake (1) | 11 FN | 2605 TP |

Errors: **15**. Source: [classification report](reports/classification_reports/xgboost.txt), [predictions](reports/xgboost_test_predictions.csv), and [errors](reports/xgboost_errors.csv).

### Feature experiments: validation only

| Selected terms | Linguistic branch | Total features | Validation accuracy (%) | Validation macro F1 |
| --- | --- | ---: | ---: | ---: |
| 3000 | True | 3008 | 99.7393 | 0.997371 |
| 6000 | True | 6008 | 99.7219 | 0.997196 |
| all | True | 12008 | 99.6698 | 0.996670 |
| 3000 | False | 3000 | 99.6003 | 0.995968 |

These experiments use SVM with `C=1`. The final tuned SVM uses `C=2`, so their validation results should not be conflated.

### Source notes

Project-specific answers use the source files and saved artifacts linked above. The algorithm definitions and default behavior can be reviewed in the official [LinearSVC](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.LinearSVC.html), [RandomForestClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [TF-IDF vectorizer](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), and [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) documentation. The XGBoost stable documentation may describe a newer release than the recorded 3.4.1 experiment.
