# O2 GRU Uncertainty Calibration Experiment

## Question

Can the disturbance-aware O2 GRU's predicted bearing/rate uncertainty be
calibrated on the existing validation split, then improve uncertainty quality
on the untouched test split and provide a more trustworthy signal to the
loss-of-view recovery manager?

This experiment changes only the predicted standard deviations. It does not
retrain the GRU, alter its bearing/rate means, select another forecast horizon,
or tune a controller threshold on test results.

## Protocol

The fixed O2 checkpoint predicts a diagonal Gaussian at 0.0, 0.1, 0.2, and
0.3-second horizons. For each horizon and each output dimension, a positive
scale is fit from validation residuals with the closed-form Gaussian
maximum-likelihood solution:

```text
scale = sqrt(mean((residual / predicted_std)^2))
```

Bearing uses a wrapped angular residual. Scales are bounded by configurable
minimum/maximum values, frozen, and then applied to the test standard
deviations. The implementation verifies the checkpoint checksum, feature
schema, prediction horizons, dataset configuration hashes, and disjoint
validation/test seed blocks before fitting.

The four horizons contain 34,350, 33,906, 33,486, and 33,066 valid validation
labels. The final validation and test summaries cover 134,808 and 130,590
horizon labels, respectively.

## Fit result

All fitted values are modest expansions:

| Horizon | Bearing std scale | Rate std scale |
|---:|---:|---:|
| 0 ms | 1.048 | 1.041 |
| 100 ms | 1.007 | 1.062 |
| 200 ms | 1.046 | 1.073 |
| 300 ms | 1.040 | 1.055 |

The model was therefore slightly too confident in aggregate, particularly for
rate at the longer horizons.

## Validation and untouched-test results

Nominal Gaussian one- and two-standard-deviation central intervals correspond
to 68.27% and 95.45% coverage. Lower negative log likelihood (NLL) is better.
RMSE is identical before and after calibration because the predicted means are
unchanged.

| Split / signal | RMSE | NLL before | NLL after | 1σ before | 1σ after | 2σ before | 2σ after |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation bearing | 21.06° | -1.5033 | **-1.5047** | 69.96% | 71.52% | 94.41% | 95.08% |
| Validation rate | 26.58°/s | -0.3258 | **-0.3291** | 69.62% | 72.43% | 94.90% | 95.74% |
| Test bearing | 23.29° | -1.3813 | **-1.3823** | 69.46% | 71.01% | 94.86% | **95.46%** |
| Test rate | 29.90°/s | -0.2268 | **-0.2330** | 70.27% | 72.69% | 93.74% | **94.68%** |

The held-out result supports the narrow claim that the scale correction
improves Gaussian likelihood and tail coverage. Test bearing 2σ coverage is
almost exactly nominal; rate 2σ coverage improves by 0.94 percentage points
but remains 0.77 points below nominal.

## Reliability limitation

The result is not full distribution calibration. Mean absolute calibration
error (MACE) across the 50%, 68.27%, 80%, 90%, 95.45%, and 99% central coverage
levels gets worse:

| Split / signal | MACE before | MACE after |
|---|---:|---:|
| Validation bearing | 1.10% | 1.70% |
| Validation rate | 1.14% | 2.53% |
| Test bearing | 0.72% | 1.40% |
| Test rate | 1.73% | 2.49% |

The single Gaussian scale improves the proper score it is fit for and the
high-coverage tail, but over-expands several lower central intervals. This is
evidence of non-Gaussian residuals and conditional miscalibration, not a reason
to hide the favorable 2σ result or to call the entire predictive distribution
calibrated.

## Measurement and dropout regimes

The test stratification makes the conditional effect explicit:

| Regime | Labels | Bearing 2σ before → after | Rate 2σ before → after |
|---|---:|---:|---:|
| Fresh valid detection | 75,204 | 94.33% → 94.96% | 92.68% → 93.68% |
| Between valid frames | 17,194 | 93.64% → 94.46% | 93.99% → 95.01% |
| Invalid detector output | 38,192 | 96.43% → 96.90% | 95.73% → 96.51% |
| Detection gap 150–650 ms | 6,131 | 94.68% → 95.65% | 92.11% → 93.74% |
| Detection gap ≥650 ms | 11,571 | 97.97% → 98.35% | 97.34% → 97.74% |
| Target out of view | 29,166 | 95.75% → 96.41% | 95.59% → 96.35% |

Fresh and short-gap predictions remain under-covered after scaling, while long
dropouts and out-of-view samples were already over-dispersed. A single
per-horizon factor cannot correct both regimes. A future conditional calibrator
should use deployable context such as measurement age, frame validity, and time
since the last valid detection.

## Recovery-policy result

The frozen calibration was replayed through the complete 24-variant
hold/blind/belief recovery suite. Every O2 aggregate—mean error, P95 error,
loss-of-view fraction, control cost, unrecovered-event count, and event-weighted
recovery time—was numerically unchanged. The modest 1.01–1.07× expansions did
not cross a recovery state threshold on any tested trajectory.

This is a useful null result. Calibration now improves reported uncertainty and
provides a versioned deployment artifact, but it does not justify a control
performance claim. Recovery thresholds still require development-seed tuning
followed by evaluation on a fresh held-out recovery block.

## Reproduce

Fit on validation and evaluate the untouched test split:

```bash
aol-calibrate-gimbal-uncertainty \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --output artifacts/gimbal_o2_uncertainty_calibration.json \
  --batch-size 24
```

Inspect the reliability curves and horizon coverage interactively:

```bash
aol-visualize-gimbal --demo calibration
```

Apply the frozen artifact in the recovery replay:

```bash
aol-evaluate-gimbal-recovery \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --uncertainty-calibration artifacts/gimbal_o2_uncertainty_calibration.json \
  --output artifacts/gimbal_calibrated_belief_recovery_comparison.json
```

The calibration JSON records method, fit/evaluation splits, bounds, per-horizon
scales, checkpoint checksum, dataset hashes, aggregate metrics, complete
reliability curves, and visibility/measurement-age/dropout strata.
