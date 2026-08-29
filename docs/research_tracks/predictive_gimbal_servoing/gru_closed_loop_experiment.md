# GRU Closed-Loop Control Experiment

## Question

Does the causal target-state GRU improve actual gimbal tracking—not just
open-loop prediction—when connected to the existing desired-rate and absolute
position adapters?

The comparison uses exact randomized scenarios recorded in dataset manifests.
Every controller sees the same 24 untouched test world/plant variants: four
seeds across six motion, sensing, actuator, and recovery families. Exogenous
target/body motion and environment seeds are paired across controllers.

## Mixed-command training support

The first GRU checkpoints had been collected only under rate commands. Rather
than treat position control as an unsupported extrapolation, this experiment
regenerates the same seed-disjoint splits with six collection behaviors:

- proportional, analytical-predictive, and privileged-oracle rate control;
- proportional, analytical-predictive, and privileged-oracle position control.

This produces 432 train, 144 validation, and 144 test episodes while retaining
72/24/24 distinct train/validation/test world-and-plant variants. O1 and O2 are
retrained for 50 epochs with the same 36,240-parameter GRU, initialization seed,
optimizer, and calibrated 27-feature schema.

## Controller protocol

The rate adapter uses predicted target angular rate as feed-forward plus a
configured `2.5 s^-1` bearing-error gain. The position adapter converts predicted
bearing directly to the configured asymmetric travel range; zero remains body
forward. Both retain the simulator's configurable latency, lag, deadband,
quantization, rate, acceleration, and travel limits.

Forecast horizon is selected separately for each profile and command mode on
the validation split by the configured mean tracking cost. The untouched test
split is evaluated once. Selected horizons are:

| Profile | Rate adapter | Position adapter |
|---|---:|---:|
| O1 servo-aware | 0.1 s | 0.0 s |
| O2 disturbance-aware | 0.0 s | 0.2 s |

The zero-horizon O2 rate choice is not evidence that prediction is useless. Its
current-state output already includes angular-rate feed-forward, which supplies
anticipation without leading the bearing target. The position loop benefits
from the 0.2-second bearing forecast.

## Untouched test results

Each value is the mean of the 24 paired test episodes. Control cost mirrors the
configured environment objective: squared normalized tracking error,
loss-of-view penalty, command effort, and command-change penalty.

### Desired-rate control

| Controller | Mean error | Mean episode P95 | Loss of view | Rate saturation | Command variation/s | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Proportional | 26.65° | 54.52° | 33.45% | 4.08% | 2.016 | 2.181 |
| Analytical constant velocity | 22.84° | 47.10° | 27.06% | 13.62% | 2.301 | 1.742 |
| O1 GRU | 21.19° | 43.46° | 22.61% | 18.31% | 4.458 | 1.533 |
| O2 GRU | **18.27°** | **37.47°** | **19.88%** | **13.21%** | 2.587 | **1.279** |

Against analytical rate control, O2 reduces mean error by 4.56°, mean episode
P95 by 9.63°, loss-of-view time by 7.18 percentage points, and cost by 0.462.
It wins the paired cost on 19/24 variants and mean error on 19/24. Rate
saturation is slightly lower in aggregate, while command variation increases
by 0.286/s.

### Absolute-position control

| Controller | Mean error | Mean episode P95 | Loss of view | Rate saturation | Command variation/s | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Proportional | 24.94° | 49.92° | 29.35% | 0.00% | 1.007 | 1.960 |
| Analytical constant velocity | 18.90° | 39.12° | 21.28% | 0.00% | 1.032 | 1.275 |
| O1 GRU | 18.14° | 37.50° | 18.71% | 0.00% | **0.998** | 1.185 |
| O2 GRU | **17.20°** | **36.22°** | **17.69%** | 0.00% | 1.187 | **1.136** |

Against analytical position control, O2 reduces mean error by 1.71°, mean
episode P95 by 2.90°, loss-of-view time by 3.59 percentage points, and cost by
0.140. It wins paired mean error on 21/24 variants and cost on 19/24.

## Where the advantage occurs

O2 rate control improves mean error, P95 error, and loss-of-view time in the
nominal, high-latency, dropout/noise, slow-servo, and aggressive-motion
families. The travel-limit case is unchanged: both analytical and learned rate
controllers spend 76.24% of the episode out of view because the target leaves
the reachable mechanical/FOV envelope. Position control shows the same physical
ceiling at roughly 75.6% loss of view.

This separation is useful. The GRU helps where inference and disturbance
awareness can change the outcome; it does not pretend to overcome unreachable
geometry.

## Recovery fallback result

A declared travel-envelope sweep was evaluated after the estimator's
configurable 0.5-second staleness watchdog expired. It is not enabled by
default. For O2 rate control, the sweep changes loss-of-view time only from
19.88% to 19.80%, but worsens mean error from 18.27° to 20.01°, P95 from 37.47°
to 41.86°, cost from 1.279 to 1.680, and unrecovered terminal events from seven
to nine. Position search is worse again.

The failure is concentrated in travel-limit recovery: a blind sweep can move
away from the returning target and drives P95 error beyond 110°. The proper
next recovery policy needs a belief over last-seen direction, boundary crossing,
and likely re-entry—not a direction-agnostic scan.

The learned controllers reduce how often the target is lost, but do not yet
reduce recovery duration once a loss occurs. Event-weighted recovery time is
1.58 s for O2 rate versus 1.07 s analytically, and 1.38 s for O2 position versus
0.98 s analytically. Prevention and recovery remain separate research problems.

## Interpretation limits

This experiment originally used one training seed over six authored motion
families and 24 test plant realizations. The follow-up
[multi-seed replication](gru_multi_seed_replication.md) repeats O2 training with
three independent initializations and reproduces the core rate/position gains
for all three. It also weakens the horizon-specific interpretation: position
selection varies between `0.1 s` and `0.2 s`. Recorded flight motion and broader
test distributions are still required before a robust deployment claim.
Adapter gains are fixed rather than jointly tuned per controller.

## Reproduce

Generate train/validation/test splits with `--domain-randomization` and all six
rate/position behavior names. Train matched O1/O2 checkpoints with:

```bash
aol-compare-gimbal-gru-profiles \
  --train-data artifacts/gimbal_mixed_train.npz \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint-directory artifacts/gimbal_mixed_checkpoints \
  --output artifacts/gimbal_mixed_gru_profile_comparison.json \
  --profile o1_servo_aware \
  --profile o2_disturbance_aware \
  --epochs 50 --batch-size 24 --seed 17
```

Then run the closed-loop comparison:

```bash
aol-evaluate-gimbal-gru-control \
  --train-data artifacts/gimbal_mixed_train.npz \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --o1-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o1.pt \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --output artifacts/gimbal_mixed_gru_closed_loop_comparison.json
```
