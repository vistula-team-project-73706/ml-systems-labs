# Individual Project Proposal

**Student:** Aron Palamoottil Biju  
**Student ID:** _fill in_  
**Course:** Machine Learning Systems

## 1. Problem

Classify the type of waste (cardboard, glass, metal, paper, plastic, other
trash) from one phone-camera photo taken at a recycling bin. The user is a
person sorting rubbish who is not sure which bin to use. The model has to
run offline on the device, because bins are often outdoors or underground
with no reliable network, and sending photos to a server adds delay and
cost.

## 2. Dataset

- **TrashNet** (Thung and Yang, Stanford), a public research dataset of
  2,527 RGB photos in 6 classes: cardboard, glass, metal, paper, plastic
  and trash.
- The photos are taken against a plain background, so I will check how the
  model handles real cluttered photos with a small set of my own pictures
  used only for testing.
- Split 70 / 15 / 15 (train / validation / test), stratified by class, with
  a fixed seed. The test split is used once, at the end.
- Images resized to 224 × 224. Training augmentation: random crop, flip
  and colour jitter.

## 3. Success metric

| Metric | Target |
|---|---|
| Macro F1 on the test set (the classes are unbalanced, and "trash" is small) | ≥ 0.85 |
| Latency per image, batch 1, CPU | ≤ 50 ms |
| Model file size | ≤ 5 MB |

Accuracy and the confusion matrix are reported as well.

## 4. Deployment constraints

- Latency ≤ 50 ms per prediction, measured as the median of 100 runs after
  10 warm-up runs, on a laptop CPU limited to 1 thread
  (`torch.set_num_threads(1)`) as a stand-in for a phone.
- Model size ≤ 5 MB on disk.
- Fully offline: no network calls during prediction.
- Peak memory during prediction ≤ 200 MB.

## 5. Baseline and planned optimisation

- **Baseline model:** MobileNetV3-Small pre-trained on ImageNet and
  fine-tuned on TrashNet. In float32 it is about 6 MB, so it does not meet
  the size limit as it is.
- **Planned optimisation: post-training int8 quantization.** Storing
  weights in 8 bits instead of 32 makes the model about 4× smaller, which
  takes it under 5 MB, and int8 kernels usually lower CPU latency too.
- **Fallback:** if macro F1 drops by more than 2 points after quantization,
  use quantization-aware training instead. If the model is still too slow,
  try structured pruning of channels.

## 6. Plan

| Week | Work |
|---|---|
| 1 | Data pipeline and baseline training with the Lab 6 YAML + MLflow setup |
| 2 | Measure accuracy, macro F1, latency and size of the baseline |
| 3 | Apply int8 quantization and compare against the baseline |
| 4 | Offline demo (photo in, class out) and report |

## 7. Risks

| Risk | Mitigation |
|---|---|
| Dataset is small (2,527 images), so the model may overfit | Pre-trained weights, augmentation, early stopping on validation F1 |
| Plain backgrounds do not match real photos | Test on my own photos; add background augmentation if needed |
| Accuracy drops after quantization | Quantization-aware training |
| The "trash" class is small and has the worst F1 | Class-weighted loss; report per-class F1 |
