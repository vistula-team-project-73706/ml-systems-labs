# Data card — UCI Adult (Census Income), processed for Lab 4

## Source

- **Original dataset:** UCI Machine Learning Repository, "Adult"
  (also called "Census Income"), donated in 1996 by Ronny Kohavi and
  Barry Becker. It was extracted from the 1994 US Census Bureau
  database.
- **How it was obtained here:** `fetch_openml(name='adult', version=2,
  as_frame=True)`, which is OpenML dataset 1590. OpenML's version 2
  combines the original `adult.data` and `adult.test` files.
- **Raw snapshot:** `data/raw/adult_raw.csv`. Missing values are
  written back as `'?'`, as in the original UCI file.
- **Raw file SHA256:** stored in `results/config.yaml` under
  `hashes_sha256.raw_file`.

## License

The UCI repository lists Adult under the Creative Commons Attribution
4.0 International (CC BY 4.0) license, and OpenML marks it as public.
It can be used and shared as long as the source is credited (Becker &
Kohavi, 1996, UCI Machine Learning Repository).

## Size and structure

| Item | Value |
|---|---|
| Rows (raw) | 48,842 |
| Columns (raw) | 15 (14 features + `income`) |
| Raw CSV size | about 5.0 MB |
| Positive class (`>50K`) | 23.93% |
| Rows with at least one `'?'` | 3,620 (7.4%) |
| Missing `'?'` per column | workclass 2,799 · occupation 2,809 · native-country 857 |
| Exact duplicate rows | 52 |
| Train / test after split | 34,189 / 14,653 (70/30 stratified, seed 42) |
| Train rows after dropping duplicates | 34,165 (24 dropped, train only) |
| Processed columns | 108 (8 scaled numeric, 99 one-hot, 1 label) |

## Processing applied

1. The data is split 70/30, stratified on `income`, before anything
   else.
2. `'?'` is replaced with NaN. Missing categorical values are filled
   with the training mode and missing numeric values with the training
   median.
3. Numeric columns are clipped to the training set's 1st and 99th
   percentiles, then standardized. Categorical columns are one-hot
   encoded, with categories not seen in training ignored.
4. Two derived features are added: `capital_net = capital-gain −
   capital-loss` and `hours_per_age = hours-per-week / (age − 16)`.
5. Exact duplicate rows are dropped from train only. The test set is
   not changed.

The fitted preprocessor is `results/preprocessor.joblib`. The values
learned from the training set are in `results/fitted_statistics.json`.
Hashes of the processed train and test files are in
`results/config.yaml`.

## Known biases and limitations

- **Old data.** The incomes are from 1994. $50K in 1994 is worth about
  $100K today, so what a `>50K` label means has changed, and so have
  occupations and wages. The data should not be used to make claims
  about people today.
- **Sex imbalance.** The data is 66.8% male and 33.2% female. The
  `>50K` rate is 30.4% for men and 10.9% for women. A model trained on
  this data learns that gap.
- **Race imbalance.** 85.5% of rows are White, and only 1.0% are
  Amer-Indian-Eskimo and 0.8% are Other. Results for small groups rest
  on very few examples. The positive rate is 25.4% for White and 12.1%
  for Black respondents.
- **Country.** 89.7% of rows are from the United States, and many
  countries have fewer than 100 rows.
- **Proxy features.** `relationship` (Husband/Wife) mostly encodes sex
  again, and `marital-status` is strongly linked to it. Removing `sex`
  from the inputs would not make a model blind to sex.
- **Missing values are not random.** Rows with `'?'` in workclass and
  occupation are often older or retired people, so filling them with
  the mode (`Private`, `Prof-specialty`) adds some bias.
- **Capped values.** `capital-gain = 99999` appears 244 times. It is a
  top-code (a placeholder for "very large"), not a real amount. Our
  clipping to the 99th percentile (15,024) treats these rows the same
  as other large gains.
- **Label noise.** 5 groups (10 rows) have identical 14 features but
  different labels. 4 rows code relationship and sex inconsistently
  with the 1994 census coding.
- **fnlwgt is not a personal attribute.** It is a census sampling
  weight. It is kept as a feature because the lab sheet asks for it,
  but it has little meaning for an individual.

## Privacy concerns

- The records are anonymized: there are no names, addresses or IDs,
  and the data has been public since 1996.
- However, it contains sensitive attributes (race, sex, marital status,
  native country and income). Combining rare values, for example a
  small country with an uncommon occupation and an exact age, can make
  a row close to unique. That is a re-identification risk if it were
  joined with other data.
- Nothing in this lab stores more than the public data, and the
  processed files stay in the git-ignored `data/` folder.

## Intended use

Teaching and benchmarking data pipelines and classification models. It
should not be used to make decisions about real people, such as credit,
hiring or pay.
