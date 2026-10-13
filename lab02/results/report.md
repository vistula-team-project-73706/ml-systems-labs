# Lab 2 — Neural Network From Scratch

## Dataset and preprocessing

- Dataset: scikit-learn handwritten digits.
- Input features: 64 pixels per sample.
- Pixel values scaled from 0–16 to 0–1 by dividing by 16.
- Labels: one-hot encoded into 10 classes.
- Split: stratified 70% training and 30% testing.
- Random state: 42.
- Training samples: 1257.
- Testing samples: 540.

## Architecture

The network was implemented from scratch using NumPy:

`64 → W → ReLU → W → ReLU → 10 → Softmax`

The hidden widths tested were 16 and 64. Weights were initialized
from a normal distribution with mean 0 and standard deviation 0.01.
Biases were initialized to zero.

The loss function was categorical cross-entropy. Parameters were
updated using mini-batch gradient descent.

## Gradient checking

Numerical gradients were calculated using central finite differences
with epsilon = 1e-5. The numerical and analytical gradients were
compared using relative error, with a pass threshold of 1e-6.

ReLU activation patterns can change when parameters are perturbed
near zero. Candidate perturbations that changed the ReLU activation
patterns were rejected and resampled because the central difference
would cross a nondifferentiable point.

- Width 16: 9/10 checks passed.
- Width 64: 10/10 checks passed.

See `gradient_check_results.csv` for individual measurements.

## Training configuration

- Epochs: 50
- Mini-batch size: 32
- Learning rates: [0.001, 0.01, 0.1]
- Hidden widths: [16, 64]

## Experiment results

| Hidden width | Learning rate | Final training loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.302546 | 0.1019 |
| 16 | 0.01 | 2.302482 | 0.1019 |
| 16 | 0.1 | 1.639741 | 0.4759 |
| 64 | 0.001 | 2.302520 | 0.1019 |
| 64 | 0.01 | 2.302290 | 0.1019 |
| 64 | 0.1 | 0.133598 | 0.9444 |

Best measured test accuracy: **0.9444**
with hidden width 64 and learning rate
0.1.

See `experiment_results.csv` for the complete results and
`loss_curves.png` for the training loss curves.

## Timing

Forward and backward passes were measured separately using a fixed
mini-batch. Each timing is the median of 100 repetitions.

- Width 16: forward median 0.018916 ms; backward median 0.018709 ms (batch size 32, 100 repetitions).
- Width 16: forward median 0.018833 ms; backward median 0.018625 ms (batch size 32, 100 repetitions).
- Width 16: forward median 0.018875 ms; backward median 0.018583 ms (batch size 32, 100 repetitions).
- Width 64: forward median 0.023708 ms; backward median 0.028271 ms (batch size 32, 100 repetitions).
- Width 64: forward median 0.023667 ms; backward median 0.028417 ms (batch size 32, 100 repetitions).
- Width 64: forward median 0.023792 ms; backward median 0.028334 ms (batch size 32, 100 repetitions).

Forward propagation performs matrix multiplications to calculate
layer activations. Backpropagation also performs matrix multiplications
to propagate gradients and calculate weight gradients. Which stage
takes longer depends on the hidden width, batch size, implementation,
and hardware. The measured medians above should be used to describe
the observed result rather than assuming one stage is always slower.

## Outputs

- `loss_curves.png`
- `experiment_results.csv`
- `gradient_check_results.csv`
- `timing_results.csv`
- `report.md`
