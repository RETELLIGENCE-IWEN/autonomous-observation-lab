# Predictive Gimbal Challenge Arena v0.1

## Purpose

The challenge arena is the presentation and visual-diagnostics track for
predictive gimbal servoing. It is intentionally separate from the numbered
research experiments: V17 can continue to ask whether safe residual authority
is causally inferable, while the arena makes the already-established value of
prediction visible.

Every column receives the same target motion, body disturbance, detector
randomness, camera, servo, initial state, and random seed. The controllers are:

1. **Practical feedback** — proportional bbox-error position control whose
   gain is scheduled from the configured camera and servo delay;
2. **Conventional Champion v1** — a causal, IMU-compensated
   constant-velocity estimator feeding the exact V2.1 hardware-aware command
   adapter used by the learned controller; and
3. **Dream-to-Center** — the deployable disturbance-aware GRU with the accepted
   V2.1 visibility-risk position adapter.

This supersedes the original showcase comparison. The old gain-0.85 reactive
controller is retained only as a named `Naive reactive P` teaching ablation;
it is not a credible performance baseline.

The privileged V16 authority oracle is not shown as a controller. It remains a
research ceiling, not a deployable result.

## Default showcase

The default is the frozen `high_latency` confirmation case at world seed 82000
and GRU training seed 17. This seed was already part of the established
confirmation block. Its randomized hardware is read directly from the
scenario configuration: 309 ms detector latency, 37.6 ms latency jitter,
99.9 ms servo command latency, 76.8 deg/s servo rate limit, and a 53.2 deg
camera FOV.

| Controller | Mean error | P95 error | Lost view | Command variation/s |
|---|---:|---:|---:|---:|
| Practical feedback | 12.61° | 25.91° | **3.28%** | **0.267** |
| Conventional Champion v1 | **6.66°** | **21.95°** | **0.00%** | 1.166 |
| Dream-to-Center | 9.34° | 21.96° | **0.00%** | 2.795 |

This single world is a diagnostic showcase, not an aggregate performance
claim. Here the conventional champion is better than the learned controller
on mean error and smoothness, while their P95 and visibility are effectively
tied. The dashboard deliberately makes that result visible.

Across the full 48-world historical confirmation replay, Conventional
Champion v1 and the three-seed learned controller are nearly tied:

| Controller | Mean error | P95 error | Lost view | Command variation/s |
|---|---:|---:|---:|---:|
| Practical feedback | 19.68° | 40.95° | 19.46% | **0.351** |
| Conventional Champion v1 | **13.30°** | 30.91° | 11.13% | 0.954 |
| Dream-to-Center (3 seeds) | 13.31° | **29.89°** | **11.09%** | 1.039 |

The learned margin is therefore concentrated in tail error: 1.01° lower P95
with essentially unchanged mean error, visibility, and unrecovered-event rate,
at the cost of 0.086 more normalized command variation per second. These are
honest limits of the current model, not evidence of broad dominance.

## What the viewer shows

Each controller column contains:

- a live HUD with target-visible/lost state, FOV margin, body yaw rate,
  measurement age, configured servo rate, and causal forecast state;
- the existing 3D body/gimbal/FOV geometry;
- the normalized camera frame with true and delayed detector boxes; and
- causal +100, +200, and +300 ms forecast ghost boxes for predictive
  controllers.

Shared plots show target/gimbal angle, tracking error, requested commands,
visibility, FOV margin, body rate, forecast trust, visibility risk, and
effective prediction horizon.

The ghost boxes are model forecasts in the current camera coordinates. They
make anticipatory braking and interception visible without presenting future
truth to the controller.

## Launch

Open the default arena by double-clicking the executable script in a file
manager, or run:

```bash
scripts/open_gimbal_challenge_arena.sh
```

Useful established scenario presets are:

```bash
# Predictable oscillation
scripts/open_gimbal_challenge_arena.sh \
  --arena-scenario nominal_combined --arena-world-seed 82001

# High-frequency motion and repeated relative-rate reversals
scripts/open_gimbal_challenge_arena.sh \
  --arena-scenario aggressive_motion --arena-world-seed 82001

# Slow actuator
scripts/open_gimbal_challenge_arena.sh \
  --arena-scenario slow_servo --arena-world-seed 82002

# Detector dropout and noise
scripts/open_gimbal_challenge_arena.sh \
  --arena-scenario dropout_noise --arena-world-seed 82001
```

The practical feedback gain can still be overridden for diagnostics, and the
ghost horizons remain configurable:

```bash
scripts/open_gimbal_challenge_arena.sh \
  --arena-reactive-gain 0.20 \
  --arena-ghost-horizons-ms 100 200 300
```

To reproduce the original unstable teaching case explicitly:

```bash
scripts/open_gimbal_challenge_arena.sh --arena-naive-reactive
```

Create a portable recording without opening a window:

```bash
aol-visualize-gimbal --demo challenge-arena \
  --output artifacts/gimbal_challenge_arena_v01.rrd
```

## Research interpretation

The corrected arena now asks a useful question: where does the learned
estimator beat a strong conventional predictor when both issue commands
through the same control stack? Current evidence points to a modest tail-error
advantage, not a general tracking advantage. The next model work should target
the champion's known assumption failures—motion reversals, non-constant
acceleration, dropout coincident with manoeuvres, and plant mismatch—while
holding the shared adapter fixed.

The 82000-series block had already been opened for V2.1. Replaying it after
development-only baseline tuning is valid for a controlled historical
comparison, but it is not fresh confirmatory evidence. A future claim that a
new learned model beats this champion requires a new untouched seed block.

## Scope and next visual increment

v0.1 deliberately reuses the validated one-axis research simulator. Its 3D
view provides body/gimbal/FOV geometry, while its camera panel remains a
normalized 2D projection. The next demo-only increment is a perspective
yaw/pitch scene that applies the one-axis controller independently to both
axes. That change should not redefine the 1D research claim or imply that
cross-axis coupling has already been solved.
