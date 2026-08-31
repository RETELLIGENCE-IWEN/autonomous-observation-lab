# Edge-Conditioned Recovery Experiment

## Question

Can a deployable image-edge gate retain the useful parts of directed recovery
without following stale target motion through centered detector outages and
target reversals?

## Controller change

Legacy belief recovery enters directed search when coast time or bearing
uncertainty crosses a threshold. The new optional gate additionally requires:

1. the last valid bbox-center error magnitude to exceed a configured fraction
   of camera half-FOV; and
2. the bbox error to be moving outward faster than a configured normalized
   image speed.

Both signals are part of the deployable observation. The bbox error is already
normalized by the configured camera half-FOV, so no camera specification is
hardcoded. When the gate is not supported, the wrapper defers exactly to the
native controller instead of applying stale belief projection. The feature is
disabled by default, preserving the earlier controller and old artifacts.

## Protocol

The expanded seven-scenario suite and 44000-series development seeds are reused
for engineering. The previously observed 45000-series test block is not reused.
Twelve candidates combine:

- last-seen edge fraction: 0.45, 0.60, 0.75, or 0.90 of half-FOV; and
- minimum outward bbox speed: 0.0, 0.25, or 0.50 normalized half-FOV/s.

Coast time is fixed at 0.35 s and coast bearing standard deviation at 10°.
Each candidate must match or improve native hold's unrecovered events in
aggregate and inside every scenario family. The selected candidate is frozen
before a single evaluation on new seeds `46000–46007`.

Fresh-test acceptance is deliberately strict: recoverability must remain
non-inferior, control cost must fall, and mean error, P95 error, and loss of view
must not worsen.

## Development result

All 12 edge-conditioned candidates pass development recoverability. The
selected 0.45 / 0.25 gate has the lowest control cost. On the exact same 56
development variants, it also repairs the failure of ungated belief recovery:

| Metric | Native hold | Ungated belief | Edge-conditioned |
|---|---:|---:|---:|
| Mean error | 18.25° | 21.32° | **17.87°** |
| Mean episode P95 | **39.88°** | 42.62° | 39.88° |
| Loss of view | 34.23% | 41.83% | **34.23%** |
| Control cost | 1.480 | 1.785 | **1.405** |
| Unrecovered events | **16** | 24 | **16** |
| Event-weighted recovery | 1.48 s | 1.34 s | **1.26 s** |

The edge gate removes all eight additional development failures from the
ungated controller: seven target-reversal failures and one body-maneuver
failure. This is a real architectural improvement, not merely threshold
movement.

## Fresh 46000-series result

The selected configuration does not pass the frozen deployment gate:

| O2 rate strategy | Mean error | P95 | Loss of view | Cost | Unrecovered |
|---|---:|---:|---:|---:|---:|
| Native hold | 18.55° | **40.13°** | 33.79% | 1.499 | 15 |
| Blind sweep | 22.20° | 58.53° | **31.26%** | 2.055 | **14** |
| Edge-conditioned belief | **18.53°** | 41.12° | 34.92% | **1.459** | 16 |

Relative to hold, edge conditioning changes mean error by -0.02°, P95 by
+0.99°, loss of view by +1.13 percentage points, cost by -0.040, and
unrecovered events by +1. The extra event occurs in detector-burst seed 46002.
Target reversal remains fixed: both hold and edge recovery finish all eight
variants without an unrecovered event.

Blind sweep again demonstrates why aggregate event counts are insufficient. It
recovers all seven body-maneuver failures but introduces six detector-burst
failures and greatly worsens error and cost, so it fails the per-scenario gate.

## Decision

Edge-conditioned belief recovery is retained as an experimental improvement
but is not deployment-eligible. Native hold remains the default.

The result narrows the failure mechanism. Edge/outward motion prevents stale
search during centered reversal, but ordinary tracked motion can also place a
bbox near the image edge immediately before detector dropout. Image evidence
alone therefore produces false physical-exit hypotheses. Further hand-tuning on
the now-observed 46000-series block would be test leakage and is not done.

Rather than add another synthetic heuristic immediately, the next useful step
is to define the deployment telemetry contract and collect recorded body,
gimbal, detector-validity, and bbox traces. Those data can determine whether a
mechanical-travel cue, short continuation policy, or learned loss classifier is
actually identifiable on the custom platform.

## Reproduce and inspect

```bash
aol-evaluate-gimbal-edge-recovery \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --uncertainty-calibration artifacts/gimbal_o2_uncertainty_calibration.json \
  --output artifacts/gimbal_edge_recovery_protocol.json \
  --test-output artifacts/gimbal_edge_recovery_fresh_test.json
```

Replay the detector-burst failure with edge-evidence telemetry:

```bash
aol-visualize-gimbal --demo recovery \
  --recovery-results artifacts/gimbal_edge_recovery_fresh_test.json \
  --recovery-scenario detector_burst_recovery \
  --seed 46002
```

For mouse-only access, launch `scripts/open_gimbal_edge_recovery_dashboard.sh`.
