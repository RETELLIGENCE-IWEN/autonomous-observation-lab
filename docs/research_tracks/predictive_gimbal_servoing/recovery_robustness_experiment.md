# Expanded Recovery Robustness Experiment

## Question

Can the belief-guided recovery thresholds be selected safely when development
contains enough outage, direction, and motion diversity to expose failures that
the original three-scenario suite missed?

## Expanded suite

The `gimbal_expanded_recovery_suite_v1` suite retains the original detector
burst, positive travel-limit re-entry, and physically unreachable cases and
adds four constructed stresses:

- `detector_micro_bursts`: four 0.25–0.65 s interruptions spanning candidate
  coast thresholds;
- `target_reversal_outage`: a reachable target reverses direction during a
  2.1 s detector outage;
- `negative_travel_limit_reentry`: a mirror of positive-side re-entry; and
- `body_maneuver_outage`: a stationary target while the body reverses rotation
  during a 1.7 s outage.

Every scenario still uses the configurable camera, servo, timing, objective,
and hardware-domain-randomization records. Eight development seeds
(`44000–44007`) produce 56 paired variants. A disjoint, unopened block
(`45000–45007`) supplies 56 fresh test variants.

## Selection protocol

The grid contains 16 combinations:

- maximum coast duration: 0.35, 0.50, 0.65, or 0.80 s;
- maximum coast bearing standard deviation: 10°, 14°, 18°, or 24°.

Native O2 hold is evaluated once on the same development variants. A belief
candidate must have no more unrecovered events than hold both in aggregate and
inside every scenario family. Passing candidates are ranked by recoverability,
control cost, false search, and P95 error. If none passes, deployment remains
hold; the least-bad belief configuration is frozen only as a diagnostic test
candidate. No threshold is changed after fresh-test access.

## Development result

Native hold produces 16 unrecovered events. Every belief threshold produces
24, so all 16 candidates fail the recoverability gate. The least-bad candidate
set contains all four 0.35 s coast candidates with identical metrics; the
declared grid order deterministically freezes 10° for diagnostic replay. Its
development comparison is:

| Metric | Native hold | Belief 0.35 s / 10° | Delta |
|---|---:|---:|---:|
| Mean error | **18.25°** | 21.32° | +3.07° |
| Mean episode P95 | **39.88°** | 42.62° | +2.74° |
| Loss of view | **34.23%** | 41.83% | +7.60 pp |
| Control cost | **1.480** | 1.785 | +0.305 |
| Unrecovered events | **16** | 24 | +8 |

Seven added failures occur in target reversal and one in the body-maneuver
outage. The standard-deviation threshold cannot repair this: once active
search follows an incorrect projected direction, all candidates share the same
terminal failures. Deployment is therefore frozen to native hold before test.

## Single fresh-test access

The preselected diagnostic belief threshold and declared baselines are replayed
once on the 56 fresh variants:

| O2 rate strategy | Mean error | P95 | Loss of view | Cost | Unrecovered |
|---|---:|---:|---:|---:|---:|
| Native hold | **17.51°** | **39.08°** | **30.53%** | **1.348** | 13 |
| Blind sweep | 22.13° | 57.22° | 31.03% | 1.965 | **10** |
| Belief 0.35 s / 10° | 21.57° | 42.48° | 42.46% | 1.755 | 25 |

Belief recovery adds 12 unrecovered events, 4.06° mean error, 3.40° P95,
11.93 percentage points loss-of-view time, and 0.407 control cost relative to
hold. The target-reversal case is decisive: hold recovers all eight variants,
while belief recovery loses all eight; loss-of-view time rises from 9.63% to
63.48%.

Blind sweep has fewer total unrecovered events because it recovers all five
body-maneuver failures left by hold. It nevertheless fails per-scenario safety:
it introduces two unrecovered detector-burst cases, and its P95 and cost are
substantially worse. Aggregate event counts alone would therefore have selected
an unsafe trade.

## Decision and engineering implication

Native hold remains the deployment default. This is no longer merely caution
from a small test: the broadened development suite rejects the current
belief-projection/search mechanism before fresh-test access, and fresh test
confirms the rejection.

The next recovery design should not search merely because a stale projected
bearing crosses a time or variance threshold. It needs deployable evidence
that distinguishes likely physical boundary loss from a detector outage while
the target was centered. A promising next change is edge-conditioned search:
retain hold after centered detector loss, and permit directed search only when
the last valid image error and travel-boundary geometry support an exit
hypothesis.

## Reproduce and inspect

```bash
aol-evaluate-gimbal-recovery-robustness \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --uncertainty-calibration artifacts/gimbal_o2_uncertainty_calibration.json \
  --output artifacts/gimbal_recovery_robustness_protocol.json \
  --test-output artifacts/gimbal_recovery_robustness_fresh_test.json
```

Replay the target-reversal failure with:

```bash
aol-visualize-gimbal --demo recovery \
  --recovery-results artifacts/gimbal_recovery_robustness_fresh_test.json \
  --recovery-scenario target_reversal_outage \
  --seed 45000
```

For mouse-only access, launch
`scripts/open_gimbal_robust_recovery_dashboard.sh`.

The follow-up [edge-conditioned recovery experiment](edge_conditioned_recovery_experiment.md)
uses last-valid bbox edge and outward-motion evidence to suppress stale search.
It repairs the development failures but narrowly fails a new 46000-series
deployment gate, so native hold remains selected.
