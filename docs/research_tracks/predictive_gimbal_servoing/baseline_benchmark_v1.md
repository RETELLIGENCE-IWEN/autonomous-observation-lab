# Baseline Benchmark v1: Robust PID and Identified MPC Development

## Purpose

Baseline Benchmark v1 turns the early “compare against about ten baselines”
idea into a versioned protocol. Its purpose is not to manufacture a large
leaderboard. It is to make each comparison answer a distinct scientific
question and to prevent later methods from being tuned on the final test.

This increment implements and development-locks C2, the strong conventional
feedback baseline, and C4, the explicit-model predictive baseline. It does
**not** claim that the ten-method benchmark is complete.

## Concept-locked registry

| ID | Method | Scientific role | Status |
|---|---|---|---|
| C0 | Hold / zero action | Sanity lower bound | Ready |
| C1 | Latency-scheduled proportional visual servo | Minimum credible feedback | Ready |
| C2 | Filtered PID with anti-windup and IMU feed-forward | Strong conventional feedback | Ready |
| C3 | Estimator plus constrained position adapter | Is simple prediction sufficient? | Ready |
| C4 | Identified DMC / MPC | Strong explicit-model prediction | Ready |
| L0 | Feed-forward SAC or TD3 | Is instantaneous nonlinear learning sufficient? | Missing |
| L1 | Recurrent SAC or TD3 | Does model-free learned memory help? | Missing |
| L2 | Supervised neural inverse controller | Is reinforcement learning necessary? | Partial |
| P | Recurrent predictive Dream-to-Center controller | Proposed deployable method | Ready |
| UB | Privileged constrained sequence oracle | Non-deployable ceiling | Ready |

The current implementation count is therefore **7 ready, 1 partial, and 2
missing**. “Ready” means the method exists in the repository; it does not mean
that all ten methods have yet been frozen into one final evaluation artifact.

## C2 controller

C2 is a timestamp-aware feedback controller around normalized image error. It
supports both desired-rate and absolute-position servo interfaces and includes:

- filtered image-error differentiation;
- optional integral action with leak and saturation anti-windup;
- IMU body-rate feed-forward;
- optional inferred target-rate feed-forward;
- latency-dependent gain scheduling;
- detector-gap watchdog behavior; and
- command rate, acceleration, jerk, and travel shaping derived from the
  configured camera and servo.

No camera FOV, latency, frame rate, servo travel, rate, acceleration, or plant
time constant is hard-coded into its runtime logic. The gimbal angle convention
remains zero degrees equals body forward.

## C4 controller

C4 combines the same causal, IMU-compensated constant-velocity estimator
family used by C3 with a receding-horizon dynamic-matrix controller. At every
control update it:

1. reconstructs current and future body-relative target bearing;
2. builds a linear prediction model from the configured/identified servo
   latency, rate time constant, position gain, command polarity, and limits;
3. optimizes a future command sequence against tracking, terminal error,
   target-rate matching, visibility risk, command change, and effort costs;
4. projects every iteration onto command-travel and hardware-relative slew
   constraints; and
5. issues only the first command before replanning.

It supports both rate and position commands and retains issued-command history
to model commands still waiting in the latency queue. Model-parameter scale
factors remain configurable for identification and mismatch studies. The
current simulator evaluation supplies the randomized configured plant values;
real deployment will require those values to be measured by system
identification.

## Development and test separation

The protocol declares two disjoint blocks:

| Block | World seeds | State |
|---|---|---|
| Development | 120000–120007 | Opened for C2/C4 selection |
| Final test | 121000–121007 | Sealed and unopened |

Each development seed instantiates the six established randomized scenarios.
Five scenarios—nominal combined motion, high latency, dropout/noise, slow
servo, and aggressive motion—select the controller. Travel-limit recovery is
reported but excluded from tuning so that it remains a diagnostic challenge
rather than silently determining ordinary tracking gains.

The selection score is

```text
aggregate control cost
+ 0.25 × worst-scenario control cost
+ 2.0 × max(0, command variation/s − 1.25)².
```

The versioned protocol hash is
`473dc5fc3991941d2468bb0cf3cf935b8e070c9726a7ee135e96e7c35c3b3f0d`.
Changing the registry, candidate set, seed blocks, selection rule, or domain
randomization changes this hash.

## C2 development result

An initial coarse sweep reached its low-gain boundary, so a second
development-only refinement was declared before the test seal was touched.
The resulting twelve P, PD, PID, and target-rate-feed-forward candidates were
evaluated independently for the two command interfaces. Both modes selected
`imu_p_low_050`: proportional gain 0.50/s, full IMU body-rate feed-forward,
and zero integral, derivative, and inferred target-rate terms.

This is a useful negative result. In this randomized development distribution,
extra PID terms did not justify their command activity. C2 remains a robust
PID-family implementation, but its selected specialization is P+IMU rather
than a nominally more complicated PID.

### Five selection scenarios

| C2 mode | Mean error | P95 error | Lost view | Variation/s | Selection score |
|---|---:|---:|---:|---:|---:|
| Position | 11.77° | 25.77° | 3.63% | **0.361** | 0.442 |
| Rate | **10.98°** | **24.03°** | **2.51%** | 1.317 | **0.375** |

Rate control is the C2 development winner under the declared score. Position
control is substantially smoother. These are development results, not final
test claims.

## C4 development result

An exploratory development sweep identified command variation as the main MPC
failure mode. Before touching the final block, the C4 family was frozen to six
candidates spanning 200/300/400 ms horizons, two command-change penalties, and
two hardware-relative slew limits. Estimator filtering was held at the
development-selected C3 value so that this comparison isolates the predictive
command optimizer.

| C4 mode | Selected candidate | Mean error | P95 error | Lost view | Variation/s | Score |
|---|---|---:|---:|---:|---:|---:|
| Position | `mpc_h020_w5_s075` | **9.09°** | **21.56°** | **2.91%** | **0.756** | **0.381** |
| Rate | `mpc_h030_w10_s100` | 12.62° | 28.48° | 6.19% | 1.141 | 0.706 |

Position MPC is the locked C4 controller. It uses a 200 ms nominal horizon,
14 projected-gradient iterations, command-change weight 5.0, and a per-step
setpoint slew limit equal to 0.75 times the configured physical rate limit.
Rate MPC remains available as an ablation but is not competitive.

### All six development scenarios

| Controller | Mean error | P95 error | Lost view | Variation/s | Control cost |
|---|---:|---:|---:|---:|---:|
| C1 practical feedback | 17.79° | 36.07° | 15.95% | 0.399 | 1.053 |
| C2 robust PID, position | 17.78° | 34.66° | 14.32% | **0.344** | 1.069 |
| C2 robust PID, rate | 17.05° | 33.02° | 13.22% | 1.174 | 1.014 |
| C3 conventional champion | 13.41° | 28.94° | 11.60% | 1.039 | 0.767 |
| C4 identified MPC, position | 13.27° | 29.04° | 11.27% | 0.670 | 0.751 |
| C4 identified MPC, rate | 16.37° | 34.87° | 14.01% | 0.992 | 0.925 |
| Dream-to-Center, three seeds | **13.06°** | **28.24°** | **10.96%** | 1.048 | **0.735** |

C2 improves ordinary tracking over C1, especially in rate mode, but does not
beat C3 after travel-limit recovery is included. The recovery scenario alone
produces 47.41° mean error and 66.73% loss of view for C2 rate. C2 has a
detector watchdog but no belief or search/re-entry state machine; this is the
main reason its aggregate result collapses.

C4 position narrowly improves over C3 in mean error, visibility, smoothness,
and aggregate control cost. Its P95 error is 0.10° worse. The gain is not
uniform: C3 remains better in nominal and dropout/noise tracking, while C4 is
better under high latency, a slow servo, aggressive motion, and mean
travel-limit recovery. This is development evidence, not a final test claim.

The frozen three-seed Dream-to-Center reference modestly improves over C4
position: 0.21° lower mean error, 0.80° lower P95, 0.32 percentage points less
loss of view, and 2.2% lower control cost. C4 uses 36.1% less command variation.
The learned advantage is concentrated in nominal, dropout/noise, and aggressive
motion; C4 is better in high latency and travel-limit recovery. Thus the current
learned controller has a real but small development-set tracking advantage over
explicit MPC, with a significant smoothness deficit.

## Visual comparison

The Challenge Arena now defaults to C4 identified position MPC, C3 Conventional
Champion v1, and the learned Dream-to-Center controller on the exact same
world. On its default historical high-latency replay:

| Controller | Mean error | P95 error | Lost view | Variation/s |
|---|---:|---:|---:|---:|
| C4 identified MPC | 7.01° | 22.28° | 0.00% | **0.833** |
| C3 conventional champion | **6.66°** | **21.95°** | 0.00% | 1.166 |
| Dream-to-Center | 9.34° | 21.96° | 0.00% | 2.795 |

This single replay is diagnostic, not confirmatory. It shows that the learned
the learned controller does not beat either strong conventional predictor and
uses substantially more command variation. This is the intended honest
comparison, not evidence that C4 is universally superior.

## Reproduction

Generate the ignored development artifact with:

```bash
aol-develop-gimbal-baseline-benchmark
```

Open the visual comparison with:

```bash
scripts/open_gimbal_challenge_arena.sh
```

Use `--arena-feedback-controller robust-pid` to restore C2 in column one, or
`--arena-feedback-controller practical` to restore C1. The historical unstable
teaching ablation remains available through `--arena-naive-reactive`.

## What comes next

The next benchmark work should complete L2 as a supervised non-RL learned
baseline, followed by L0 and L1 under matched observation and action
interfaces. Only after all ten entries are implemented, development-tuned,
and hash-locked should the 121000-series test block be opened once.
