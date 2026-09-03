# Conventional Champion v1: a fair learned-control baseline

## Research question

Can the disturbance-aware GRU outperform a credible deployable conventional
controller when estimator outputs—not downstream control implementations—are
the independent variable?

The earlier challenge arena could not answer this cleanly. Its gain-0.85
reactive controller was unstable under long delay, and its classical predictor
used a simpler command adapter than the GRU. That comparison demonstrated the
value of prediction, but overstated the value of learning.

## Controllers

The revised protocol separates three levels:

1. **Practical feedback.** Delayed bbox error updates an absolute position
   command. Gain is scheduled as a bounded inverse of the configured camera,
   servo-command, servo-rate, and position-response delay. No simulator-only
   state is used.
2. **Conventional Champion v1.** Delayed detections reconstruct body-relative
   target bearing at capture time from gimbal history. Trapezoidal integration
   of causal IMU body-rate history separates vehicle rotation from target
   motion. A filtered constant-velocity target state is projected at 0, 100,
   200, and 300 ms with explicit process uncertainty.
3. **Dream-to-Center.** The accepted disturbance-aware O2 GRU predicts the
   same bearing, rate, and uncertainty representation at the same horizons.

The conventional champion and GRU both feed the exact accepted V2.1 adapter:
hardware-derived arrival-time preview, relative uncertainty trust,
visibility-risk preview, native invalid-estimate hold, position/travel
clipping, and jerk-limited setpoint shaping. Both receive deployable O2
telemetry. This makes estimator family the principal difference.

All camera and servo quantities come from `GimbalServoingConfig`; no fixed
hardware specification is embedded in the policy. Zero gimbal angle remains
body-forward.

## Selection protocol

Only the established 81000–81007 development worlds select conventional
settings. They cover all six scenario families and randomized camera, servo,
timing, noise, latency, and motion parameters.

For practical feedback, five delay-gain products from 0.05 to 0.13 s were
screened. Selection chose the smoothest schedule within 0.05 aggregate
control cost and 0.01 high-latency control cost of their respective minima.
This plateau rule rejected a needlessly aggressive fixed-gain optimum. The
selected schedule has a 0.09 s delay-gain product, gain bounds [0.05, 0.40],
and a configurable 0.15 position-response fraction. On the default frozen
world it realizes gain 0.182 rather than the old 0.85.

For the classical estimator, relative-rate CV and five IMU-compensated filter
coefficients were screened. Minimum development control cost selected
`imu_cv_a070`: velocity update coefficient 0.70, uncertainty update
coefficient 0.20, and 80 deg/s² process-acceleration standard deviation.

The original 82000–82007 V2.1 confirmation block was then replayed without
changing either selected setting. Because that block was already open before
this experiment, these results are controlled historical evidence rather than
a new untouched confirmation claim.

## Aggregate result

The replay contains 48 world/scenario combinations. Dream-to-Center is
aggregated over training seeds 17, 29, and 43; event counts are normalized per
episode before comparison.

| Controller | Mean error | P95 error | Lost view | Variation/s | Control cost | Unrecovered/episode |
|---|---:|---:|---:|---:|---:|---:|
| Practical feedback | 19.68° | 40.95° | 19.46% | **0.351** | 1.428 | 0.292 |
| Conventional Champion v1 | **13.30°** | 30.91° | 11.13% | 0.954 | 0.803 | 0.167 |
| Dream-to-Center, 3 seeds | 13.31° | **29.89°** | **11.09%** | 1.039 | **0.800** | 0.167 |

Relative to the champion, the GRU changes:

- mean error by +0.007°;
- P95 error by −1.014°;
- loss of view by −0.034 percentage points;
- normalized command variation by +0.086/s, or approximately +9.0%; and
- mean control cost by −0.0029.

The current learned controller therefore has a modest tail-error advantage,
not a broad aggregate tracking advantage.

## Scenario decomposition

| Scenario | Champion mean / P95 | GRU mean / P95 | Champion / GRU lost view | Champion / GRU variation |
|---|---:|---:|---:|---:|
| Nominal combined | 7.32° / 19.27° | **6.94° / 17.83°** | 3.91% / **3.78%** | 0.774 / **0.641** |
| High latency | **8.09° / 20.20°** | 9.22° / 21.23° | **0.68%** / 0.83% | **1.010** / 1.739 |
| Dropout + noise | 7.11° / 18.72° | **6.60° / 16.94°** | 0.24% / 0.24% | 1.163 / **0.994** |
| Slow servo | 10.55° / 25.10° | **10.12° / 24.19°** | 3.77% / **3.39%** | **0.781** / 0.805 |
| Aggressive motion | 14.76° / 33.12° | **14.52° / 30.09°** | 12.11% / **11.68%** | **1.547** / 1.666 |
| Travel-limit recovery | **32.00° / 69.04°** | 32.46° / 69.07° | **46.05%** / 46.63% | 0.447 / **0.391** |

The learned benefit is strongest in nominal, noisy/dropout, slow-servo, and
aggressive-motion tails. The conventional model is clearly stronger in the
high-latency family and slightly stronger in travel-limit recovery. The latter
is dominated by reachability and recovery behavior rather than prediction
alone.

## What AI must overcome next

Conventional Champion v1 gives the next learned model a precise target. A
useful V17 should improve states where constant velocity is structurally
wrong—not merely reproduce performance that IMU compensation and delay-aware
preview already provide. The failure atlas should concentrate on:

- angular-rate reversals and non-constant acceleration;
- detector outage coincident with target or body manoeuvres;
- latency jitter that invalidates one deterministic preview time;
- servo-model mismatch and near-capacity motion; and
- tail error and reacquisition without increasing command variation.

Promotion should require a fresh untouched world block, lower P95 and
loss-of-view metrics than the champion, no mean-error regression, equal or
lower unrecovered-event rate, and an explicit smoothness budget. The shared
V2.1 adapter should remain fixed during this estimator study.

## Reproduction and visualization

Generate the ignored detailed result artifact:

```bash
aol-evaluate-gimbal-conventional-champion \
  --visibility-risk-results artifacts/gimbal_adaptive_position_v21.json \
  --output artifacts/gimbal_conventional_champion_v1.json
```

Open the corrected three-controller arena with:

```bash
scripts/open_gimbal_challenge_arena.sh
```

Use `--arena-naive-reactive` only to show the historical unstable P-controller
ablation. It is intentionally excluded from performance claims.
