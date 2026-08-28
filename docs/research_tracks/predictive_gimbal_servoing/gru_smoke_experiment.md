# Causal GRU Smoke Experiment

## Purpose

This experiment checks whether the privileged dataset can train a compact,
strictly causal temporal target-state model. It is a pipeline and mechanism
test, not a generalization claim.

The model consumes one declared deployment profile and predicts a Gaussian
distribution over body-relative bearing and rate at 0.0, 0.1, 0.2, and 0.3
seconds. Its unidirectional GRU has no access to future features. Circular
bearing residuals are used in both training and evaluation.

The checkpoint can be wrapped by `GRUTargetStateEstimator`, which implements the
same target-state interface as the analytical estimator. The existing rate and
position controller adapters therefore remain unchanged.

## Initial configuration

- observation profile: O1 servo-aware;
- model: 48-dimensional feature embedding and 48-dimensional GRU state;
- trainable parameters: 20,752;
- optimizer: AdamW, learning rate `1e-3`, weight decay `1e-5`;
- training: 30 epochs on CPU;
- episodes: 108 train, 36 validation, 36 test;
- data: six fixed development scenarios and three collection behaviors;
- deterministic seeds: train `1000–1005`, validation `2000–2001`, test
  `3000–3001`.

## Smoke results

The table compares both methods only where the causal constant-velocity
estimator has a valid estimate. This avoids giving either method credit for a
different set of frames.

| Horizon | GRU bearing RMSE | Analytical bearing RMSE | GRU rate RMSE | Analytical rate RMSE |
|---:|---:|---:|---:|---:|
| 0.0 s | 4.34° | **2.52°** | 22.24°/s | **22.16°/s** |
| 0.1 s | 5.88° | **5.00°** | **24.22°/s** | 29.86°/s |
| 0.2 s | **7.40°** | 8.12° | **25.25°/s** | 36.12°/s |
| 0.3 s | **9.03°** | 11.64° | **26.61°/s** | 40.50°/s |
| Aggregate | **6.87°** | 7.60° | **24.62°/s** | 32.83°/s |

The GRU is worse at reconstructing the immediate bearing, where explicit camera
geometry is a very strong inductive bias. It becomes better as the requested
prediction horizon grows, especially for angular rate. That is the behavior the
research track needs to investigate rather than a blanket “neural is better”
result.

The analytical estimator is valid on 89.9% of eligible labels. The GRU emits on
100%, although its streaming deployment adapter still applies a configurable
staleness watchdog. On analytical support, GRU uncertainty coverage is 72.3%
and 94.99% for bearing at one and two standard deviations, and 73.8% and 95.01%
for rate. The two-standard-deviation calibration is close to the nominal 95.45%
Gaussian value.

## Important limitation

Different seeds in this smoke run alter detector randomness, but the six
development motion traces themselves are fixed. Consequently, the validation
and test blocks do not demonstrate unseen-motion generalization. The numbers
only establish that:

1. the privileged labels and causal observations train end to end;
2. online and batched GRU execution agree;
3. the learned uncertainty head is usable;
4. longer-horizon prediction can improve on constant-velocity extrapolation.

The next scientifically meaningful dataset must independently randomize target
and body motion amplitude, frequency, phase, maneuver onset, and actuator/camera
parameters before observation-profile ablations or control claims are made.

## Reproduce

Install the optional learning dependency and generate disjoint datasets as
described in the [dataset specification](privileged_dataset.md). Then run:

```bash
aol-train-gimbal-gru \
  --train-data artifacts/gimbal_gru_train.npz \
  --validation-data artifacts/gimbal_gru_validation.npz \
  --test-data artifacts/gimbal_gru_test.npz \
  --profile o1_servo_aware \
  --epochs 30 \
  --hidden-dim 48 \
  --embedding-dim 48 \
  --checkpoint artifacts/gimbal_gru_o1.pt \
  --output artifacts/gimbal_gru_o1_results.json
```

The result JSON stores both full-availability GRU metrics and a matched-support
comparison against the analytical estimator. The checkpoint stores the model
configuration, feature schema, target schema, dataset hashes, training
configuration, and validation/test metrics.
