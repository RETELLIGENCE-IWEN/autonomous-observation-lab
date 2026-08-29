# Contextual Calibration and Recovery Development/Test Protocol

## Questions

This phase asks two deliberately separate questions:

1. Does a lightweight context-aware variance model generalize better than the
   global per-horizon O2 calibration?
2. Can recovery thresholds selected only on a development seed block improve
   O2 rate recovery on a fresh test block without sacrificing recoverability?

The contextual model is evaluated on the existing prediction test split. The
recovery policy uses a new split: seeds 42000–42007 for development and
43000–43007 for one frozen test evaluation. The earlier 41000-series authored
recovery results are not reused as the new test claim.

## Deployable contextual calibrator

The calibrator uses only state available in the streaming estimator:

- whether the detector produced a frame update;
- whether the current detection is valid;
- measurement age, divided at 160 ms; and
- time since the last valid detector arrival, divided at 150 and 650 ms.

These signals define eight mutually exclusive contexts: fresh-young,
fresh-old, between-frame-young, between-frame-old, invalid-short-gap,
invalid-medium-gap, invalid-long-gap, and no valid detection history. Bearing
and rate scales are fit separately at each prediction horizon. Each contextual
variance estimate uses 512 validation samples of shrinkage toward the global
per-horizon variance, so sparse contexts cannot produce unconstrained scales.

The implementation has an exact streaming contract: the GRU retains its
causal hidden state, tracks the last valid detector arrival, classifies the
current deployable context, and changes standard deviations only. Predicted
means and control gains remain unchanged.

## Contextual calibration result

The model fits validation better but fails to generalize. Lower NLL is better;
nominal 2σ coverage is 95.45%.

| Split / method | Bearing NLL | Rate NLL | Bearing 2σ | Rate 2σ | Bearing MACE | Rate MACE |
|---|---:|---:|---:|---:|---:|---:|
| Validation uncalibrated | -1.5033 | -0.3258 | 94.41% | 94.90% | 1.10% | 1.14% |
| Validation global | -1.5047 | -0.3291 | 95.08% | 95.74% | 1.70% | 2.53% |
| Validation contextual | **-1.5268** | **-0.3382** | 95.38% | 95.76% | **1.32%** | **2.39%** |
| Test uncalibrated | -1.3813 | -0.2268 | 94.86% | 93.74% | **0.72%** | **1.73%** |
| Test global | **-1.3823** | **-0.2330** | **95.46%** | **94.68%** | 1.40% | 2.49% |
| Test contextual | -1.3561 | -0.2142 | 93.88% | 94.40% | 1.06% | 2.52% |

The contextual table's validation advantage reverses on test. Bearing NLL is
worse than even the uncalibrated model, and bearing 2σ coverage falls by almost
one percentage point. The likely cause is that the residual distribution still
depends on scenario and collection behavior inside each coarse detector
context. More conditional bins would increase, not solve, that instability.

The contextual artifact is therefore retained as a reproducible negative
experiment. It is not selected for recovery or deployment. The simpler global
calibration remains the uncertainty source for the recovery protocol.

## Recovery development selection

The development grid varies the two thresholds that decide when `COAST` becomes
`SEARCH`:

- maximum coast duration: 0.45, 0.65, or 0.85 s;
- maximum coast bearing standard deviation: 12°, 18°, or 24°.

All other recovery, camera, servo, and domain-randomization values remain in
the serialized configuration. The nine candidates see the same 24 development
world/plant variants. Selection is lexicographic: minimize unrecovered events,
then mean control cost, search while the target is visible, and P95 error.

Every candidate has nine unrecovered development events. The selected candidate
is therefore the lowest-cost one: 0.65 s and 12°. Relative to the original
0.65 s / 18° configuration on development, it changes:

| Metric | Original threshold | Selected threshold |
|---|---:|---:|
| Mean error | 21.29° | **21.10°** |
| Mean episode P95 | 42.31° | **42.10°** |
| Loss of view | 42.10% | **41.40%** |
| Control cost | 1.805 | **1.786** |
| Search while target visible | **5.98%** | 6.55% |
| Unrecovered events | 9 | 9 |

The full configuration is frozen before test access.

## Fresh recovery test

The table averages the 24 fresh test variants, including eight intentionally
unreachable terminal cases.

| O2 rate strategy | Mean error | Mean episode P95 | Loss of view | Cost | Unrecovered |
|---|---:|---:|---:|---:|---:|
| Hold | 23.32° | 45.57° | 43.79% | 2.105 | **8** |
| Blind sweep | 34.24° | 77.72° | 48.03% | 3.994 | 12 |
| Development-selected belief | **21.97°** | **44.00°** | **43.31%** | **1.887** | 10 |

Against hold, the selected belief policy improves mean error by 1.35°, P95 by
1.57°, loss-of-view time by 0.48 percentage points, event-weighted recovery
time by 0.23 s, and control cost by 0.218. It wins paired cost on 17/24
variants. Blind sweep remains decisively unsafe.

However, belief recovery leaves two recoverable detector-burst events
unrecovered; hold leaves only the eight deliberately impossible terminal
events. The average improvements therefore fail a safety/non-inferiority gate.
The selected belief policy must remain experimental, and native hold remains
the O2 rate deployment default.

This failure was not visible on the eight-seed development block because every
threshold candidate had the same unrecovered-event count there. More candidate
search on the now-observed test block would be test leakage and is not done.

## Reproduce

Fit and evaluate the contextual calibration experiment:

```bash
aol-calibrate-gimbal-contextual-uncertainty \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --output artifacts/gimbal_o2_contextual_uncertainty_calibration.json \
  --batch-size 24
```

Run development selection and one fresh test evaluation with the accepted
global calibration:

```bash
aol-evaluate-gimbal-recovery-protocol \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --uncertainty-calibration artifacts/gimbal_o2_uncertainty_calibration.json \
  --output artifacts/gimbal_recovery_development_test_protocol.json \
  --test-output artifacts/gimbal_recovery_fresh_test.json
```

Inspect the contextual reliability result and the exact fresh recovery replay:

```bash
aol-visualize-gimbal --demo calibration \
  --uncertainty-calibration \
  artifacts/gimbal_o2_contextual_uncertainty_calibration.json

aol-visualize-gimbal --demo recovery \
  --recovery-results artifacts/gimbal_recovery_fresh_test.json \
  --seed 43000
```

The protocol artifact records every candidate, selection key, disjoint seed
blocks, selected configuration, calibration/checkpoint provenance, and complete
fresh-test result. The test result is also written separately for exact replay.

The follow-up [expanded recovery robustness experiment](recovery_robustness_experiment.md)
adds four threshold- and direction-sensitive scenarios plus a per-scenario
native-hold safety gate. It rejects every current belief threshold on
development and confirms the rejection on a new 45000-series test block.
