# Predictive Gimbal Challenge Arena v0.1

## Purpose

The challenge arena is the presentation and visual-diagnostics track for
predictive gimbal servoing. It is intentionally separate from the numbered
research experiments: V17 can continue to ask whether safe residual authority
is causally inferable, while the arena makes the already-established value of
prediction visible.

Every column receives the same target motion, body disturbance, detector
randomness, camera, servo, initial state, and random seed. The controllers are:

1. **Reactive position** — proportional bbox-error feedback;
2. **Classical predictive** — an explicit constant-velocity target-state
   estimator with plant-delay preview; and
3. **Dream-to-Center** — the deployable disturbance-aware GRU with the accepted
   V2.1 visibility-risk position adapter.

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
| Reactive position | 42.55° | 58.99° | **87.70%** | 0.210 |
| Classical predictive | 10.84° | 26.41° | **4.10%** | 1.152 |
| Dream-to-Center | **9.34°** | **21.96°** | **0.00%** | 2.795 |

This is a showcase case, not a new aggregate performance claim. The dashboard
also exposes command variation so the cost of the stronger learned response is
visible rather than hidden.

## What the viewer shows

Each controller column contains:

- a live HUD with target-visible/lost state, FOV margin, body yaw rate,
  measurement age, configured servo rate, and causal forecast state;
- the existing 3D body/gimbal/FOV geometry;
- the normalized camera frame with true and delayed detector boxes; and
- causal +100, +200, and +300 ms forecast ghost boxes for predictive
  controllers.

Shared plots show target/gimbal angle, tracking error, requested commands,
visibility, FOV margin, body rate, Dream-to-Center forecast trust and
visibility risk, and effective prediction horizon.

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

The reactive gain and ghost horizons remain configurable:

```bash
scripts/open_gimbal_challenge_arena.sh \
  --arena-reactive-gain 0.85 \
  --arena-ghost-horizons-ms 100 200 300
```

Create a portable recording without opening a window:

```bash
aol-visualize-gimbal --demo challenge-arena \
  --output artifacts/gimbal_challenge_arena_v01.rrd
```

## Scope and next visual increment

v0.1 deliberately reuses the validated one-axis research simulator. Its 3D
view provides body/gimbal/FOV geometry, while its camera panel remains a
normalized 2D projection. The next demo-only increment is a perspective
yaw/pitch scene that applies the one-axis controller independently to both
axes. That change should not redefine the 1D research claim or imply that
cross-axis coupling has already been solved.
