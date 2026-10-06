# Lab 01 — Environment, Baseline Models, and Deployment Constraints

## Goal

The goal of this lab is to establish the machine-learning development environment, train baseline classification models on the Breast Cancer Wisconsin dataset, measure their performance and resource usage, and determine which deployment targets they can satisfy.

The two baseline models are:

- Logistic Regression
- Random Forest

The measurements include training time, single-sample inference latency, model size, and peak memory usage.

---

## Method

The Breast Cancer Wisconsin dataset was loaded using `load_breast_cancer()` from scikit-learn.

The dataset was split into:

- 70% training data
- 30% test data

The split was stratified and used `random_state=42`.

Two models were trained:

- Logistic Regression with `max_iter=1000` and `random_state=42`
- Random Forest with 100 estimators and `random_state=42`

Model accuracy was measured on the test set.

For timing measurements, one warm-up run was performed before the measured runs. Training time was measured over five runs and the median was reported.

For inference latency, a single test sample was used. After one warm-up inference, 100 inference runs were measured and the median latency was reported.

Model size was measured from the serialized Joblib model files.

Peak memory was measured during training and inference using process memory monitoring.

---

## Baseline Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.9415 |
| Random Forest | 0.9357 |

Logistic Regression achieved slightly higher test accuracy than Random Forest.

---

## Performance and Resource Measurements

| Model | Training Median (s) | Inference Median (ms) | Model Size (KB) | Training Peak Memory (MB) | Inference Peak Memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1056 | 0.0874 | 1.08 | 113.75 | 113.77 |
| Random Forest | 0.0707 | 1.0553 | 284.09 | 114.70 | 114.92 |

### Observations

Logistic Regression has the smaller serialized model and significantly lower inference latency.

Random Forest has slightly faster training in the measured experiment, but its serialized model is substantially larger.

Both models have similar peak process memory usage, around 114 MB.

---

## Deployment Budget Analysis

The deployment budgets are:

| Target | Memory | Latency | Model Size |
|---|---:|---:|---:|
| Cloud | >= 1 GB | <= 100 ms | <= 500 MB |
| Edge | 256–1024 MB | <= 50 ms | <= 50 MB |
| Mobile | 64–256 MB | <= 20 ms | <= 10 MB |
| TinyML | <= 256 KB | <= 10 ms | <= 100 KB |

### Logistic Regression

| Constraint | Measurement | Result |
|---|---:|---|
| Cloud memory | 113.77 MB | Pass |
| Cloud latency | 0.0874 ms | Pass |
| Cloud size | 1.08 KB | Pass |
| Edge memory | 113.77 MB | Does not meet stated 256 MB lower bound |
| Edge latency | 0.0874 ms | Pass |
| Edge size | 1.08 KB | Pass |
| Mobile memory | 113.77 MB | Pass |
| Mobile latency | 0.0874 ms | Pass |
| Mobile size | 1.08 KB | Pass |
| TinyML memory | 113.77 MB | Fail |
| TinyML latency | 0.0874 ms | Pass |
| TinyML size | 1.08 KB | Pass |

### Random Forest

| Constraint | Measurement | Result |
|---|---:|---|
| Cloud memory | 114.92 MB | Pass |
| Cloud latency | 1.0553 ms | Pass |
| Cloud size | 284.09 KB | Pass |
| Edge memory | 114.92 MB | Does not meet stated 256 MB lower bound |
| Edge latency | 1.0553 ms | Pass |
| Edge size | 284.09 KB | Pass |
| Mobile memory | 114.92 MB | Pass |
| Mobile latency | 1.0553 ms | Pass |
| Mobile size | 284.09 KB | Pass |
| TinyML memory | 114.92 MB | Fail |
| TinyML latency | 1.0553 ms | Pass |
| TinyML size | 284.09 KB | Fail |

---

## Conclusions

1. **Logistic Regression achieved the highest accuracy and the smallest model size.** Its test accuracy was 0.9415 and its serialized model was only 1.08 KB.

2. **Random Forest trained slightly faster, but required a substantially larger model.** Its median training time was 0.0707 seconds compared with 0.1056 seconds for Logistic Regression, while its model size was 284.09 KB compared with 1.08 KB.

3. **Both models are suitable for mobile deployment based on the measured memory, latency, and model-size constraints, but neither is suitable for TinyML.** The measured process memory of approximately 114 MB is far above the TinyML memory limit of 256 KB. Random Forest also exceeds the TinyML model-size limit of 100 KB.
