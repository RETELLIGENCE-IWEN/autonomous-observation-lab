# Adaptive Predictive Position V2 Experiment

## Question

The first learned position controller chooses one fixed GRU prediction horizon
per training initialization and sends the corresponding bearing directly to
the servo. It tracks better than the analytical controller, but its position
commands are less smooth. This experiment asks whether a deployment-oriented
adapter can preserve that tracking performance while reducing command
activity.

The comparison in this report is deliberately narrower than the earlier
baseline study: **adaptive V2 versus the already successful fixed-horizon O2
GRU position controller**. Passing this gate would justify replacing the
fixed-horizon learned position adapter, not deployment on real hardware.

## V2 controller

The estimator now exposes all trained prediction heads in one causal forward
step. The adapter then performs four operations:

1. It derives a requested forecast time from configurable servo command
   latency, rate time constant, position-loop response, and optional preview.
2. It interpolates bearing, rate, and uncertainty between the trained GRU
   horizons. Bearing interpolation is circular across the ±180° boundary.
3. It compares forecast bearing uncertainty with current-state uncertainty and
   shortens the effective horizon when the forecast is relatively uncertain.
4. It passes the target through configurable rate-, acceleration-, and
   jerk-limited setpoint shaping. The simulated servo remains the independent
   physical safety boundary.

Detection age is already an O2 GRU input and is therefore not added a second
time to the arrival horizon. Invalid estimates retain the native position-hold
behavior. Servo and camera quantities remain outside the model logic and come
from configuration; the adapter's scales and trust thresholds are configurable
as well.

```text
camera/detector + body/servo telemetry
                  │
                  ▼
         causal multi-horizon GRU
          0, 0.1, 0.2, 0.4 s
                  │
        ┌─────────┴──────────┐
        │ servo arrival time │
        │ relative forecast  │
        │ uncertainty        │
        └─────────┬──────────┘
                  ▼
      interpolated, trust-weighted bearing
                  │
                  ▼
       jerk-limited position setpoint
                  │
                  ▼
       hardware-configured servo plant
```

## Development failure that changed the design

The first V2 implementation limited the outer setpoint trajectory to the same
rate and acceleration as the physical servo. That duplicated the actuator
dynamics in cascade: the outer command lagged, and the inner plant lagged the
already delayed command again. A development comparison made commands 60.3%
smoother but regressed mean error by 5.374°, P95 by 9.180°, and loss of view by
8.15 percentage points. It was rejected.

The correction was architectural, not a looser tracking threshold. The outer
trajectory scales may exceed one because they constrain a requested setpoint;
the independently configured servo model still enforces its actual rate,
acceleration, latency, travel, deadband, and quantization limits. The evaluator
was also changed so a candidate that fails validation cannot open the fresh
test block.

## Protocol

The experiment uses all three independently trained O2 models (training seeds
17, 29, and 43). One controller configuration is selected across every model
on the recorded validation worlds. Candidates must first remain within the
declared validation bounds for tracking, visibility, control cost, and
unrecovered events. Eligible candidates are ranked by command variation and
then control cost.

The selected configuration is `light_smoothing_calibrated`: 0.85 times the
configured actuator-arrival delay, relative-uncertainty trust from ratios 1 to
4 with a 0.5 minimum weight, setpoint rate and acceleration scales of 6 and 12,
and a 15 ms jerk rise time. These scales shape the requested setpoint; they are
not physical actuator limits.

Only after selection was frozen did the evaluator generate four disjoint world
seeds, 80000–80003, across all six scenario families. That gives 24 fresh
scenario variants per model and 72 paired fixed/V2 rollouts in total. No gate
threshold was changed after inspecting this block.

The test gate requires:

- no more than 0.25° aggregate mean-error regression, 0.50° aggregate P95
  regression, or 0.5 percentage points additional loss of view;
- at least 5% lower command variation and no additional unrecovered event;
- the same tracking/visibility bounds for each training seed; and
- no more than 2° P95 or 2 percentage points loss-of-view regression in any
  scenario family.

## Fresh result

| Metric | Fixed horizon | Adaptive V2 | V2 − fixed |
|---|---:|---:|---:|
| Mean absolute error | 12.680° | **12.649°** | **−0.031°** |
| P95 absolute error | 30.059° | **29.480°** | **−0.579°** |
| Loss of view | 9.746% | **9.716%** | **−0.030 pp** |
| Command variation/s | 1.138 | **0.984** | **−0.154 (−13.6%)** |
| Command RMS | 0.538 | **0.528** | **−0.009** |
| Actuator acceleration RMS | 0.611 | **0.596** | **−0.015** |
| Mean control cost | 0.744 | **0.737** | **−0.007** |
| Unrecovered loss events | **8** | 9 | **+1** |

All three training-seed core checks and all six scenario tail/visibility checks
pass. Aggregate tracking, visibility, smoothness, and cost also improve. The
sole failed acceptance condition is the additional unrecovered event.

The weakness is localized to aggressive motion. In that family V2 changes
mean error by +0.274°, P95 by −0.155°, loss of view by +0.419 percentage
points, and command variation by −0.155/s. Those tail and visibility changes
remain within their declared scenario bounds, but one episode ends without
recovery before the simulation stops. Because terminal loss is safety-relevant,
the event-count gate is intentionally stricter than the aggregate averages.

At runtime the adapter requested a mean horizon of 133 ms and uncertainty
reduced it to 122 ms. Mean prediction weight was 0.923. The rate, acceleration,
and jerk shapers were active on 0.2%, 34.4%, and 72.3% of valid steps,
respectively.

## Verdict

V2 demonstrates the intended trade: it removes 13.6% of position-command
variation without sacrificing aggregate tracking; in fact, every aggregate
metric improves slightly. It is therefore a credible experimental successor,
not merely a smoother but worse controller.

It does **not** replace the fixed-horizon controller yet. The frozen safety gate
rejects it because 9 unrecovered losses are worse than 8. The fixed-horizon O2
position controller remains the accepted synthetic default, and this fresh
test block must not be used for another tuning cycle. The next controller
iteration should address terminal aggressive-motion recovery on new
development worlds, followed by another untouched confirmation block.

## Reproduce and inspect

Run the validation-selection and fresh-world protocol with:

```bash
python3 -m autonomous_observation_lab.gimbal_servoing.adaptive_position \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint 17=artifacts/gimbal_o2_replication_checkpoints/gimbal_gru_o2_seed_17.pt \
  --checkpoint 29=artifacts/gimbal_o2_replication_checkpoints/gimbal_gru_o2_seed_29.pt \
  --checkpoint 43=artifacts/gimbal_o2_replication_checkpoints/gimbal_gru_o2_seed_43.pt \
  --fresh-test-seed 80000 --fresh-test-seed 80001 \
  --fresh-test-seed 80002 --fresh-test-seed 80003 \
  --output artifacts/gimbal_adaptive_position_v2_fresh.json
```

Open the comparison dashboard with one double-click or one terminal command:

```bash
scripts/open_gimbal_adaptive_position_dashboard.sh
```

The dashboard contains the aggregate result and gate verdict, scenario deltas,
a representative fixed/V2 tracking replay, raw and shaped commands, requested
and effective horizons, and uncertainty trust. A portable recording can be
created with:

```bash
aol-visualize-gimbal --demo adaptive-position \
  --output artifacts/gimbal_adaptive_position_v2.rrd
```
