# Problem definition — Adult income prediction

**Task.** Binary classification: predict whether a person's yearly
income is above $50K from one row of the 1994 US Census extract (UCI
Adult). This is a classic benchmark. In this lab its purpose is to test
the data pipeline more than to build the best possible model.

**Inputs.** 14 features per person:

- 6 numeric: age, fnlwgt, education-num, capital-gain, capital-loss,
  hours-per-week.
- 8 categorical: workclass, education, marital-status, occupation,
  relationship, race, sex, native-country.

Missing values are coded as `'?'` and appear in workclass, occupation
and native-country.

**Output.** `income`, with the two classes `<=50K` and `>50K`. The
positive class is `>50K`, which is about 24% of the rows, so the
classes are imbalanced.

**Success metrics.** F1 on the positive class is the main metric.
Because only 24% of people are `>50K`, a model that always predicts
`<=50K` already gets about 76% accuracy, so accuracy alone can look
good while missing every positive. Accuracy is still reported because
it is easy to compare with other work. Both metrics are measured once
on a held-out 30% test set, and F1 is also measured with 5-fold
cross-validation on the training set to show how much it varies.

**Constraints.**

- *No leakage.* The test set is split off before any cleaning. Every
  statistic used to clean or transform data (imputation values, clip
  limits, scaling, category lists) is learned from the training set
  only. Duplicate removal is applied to train only.
- *Reproducible.* One command (`python lab04/src/pipeline.py`) goes
  from download to a processed dataset. Seeds are fixed, the fitted
  preprocessor is saved, and the processed data gets a SHA256 hash
  stored with the configuration, so anyone can check they got the same
  bytes.
- *Raw data is never edited.* Validation reports problems but never
  drops or fixes rows.

**Out of scope.** Tuning the model and mitigating fairness problems.
The dataset's known biases are documented in the data card and will be
looked at again in Lab 11.
