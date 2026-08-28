# Benchmark Specification: Disturbance-Aware 1D Visual Servoing

## 1. Purpose

The benchmark tests continuous learned gimbal control under hidden platform motion and imperfect sensing. It must make a reactive controller adequate in simple conditions but insufficient when delay, oscillatory disturbance, actuator lag, or parameter shift requires temporal inference and prediction.

The benchmark is feature-level. It begins from a valid target bounding box so detector quality does not dominate the control question.

---

## 2. Episode model

Each episode contains:

- one designated target;
- one rotational gimbal axis;
- an exogenous quadcopter angular-motion process;
- an optional independently moving target;
- camera sampling, delay, jitter, noise, and dropout;
- bounded gimbal dynamics with latency and saturation;
- a fixed-duration tracking objective.

The relative target bearing determines the image coordinate. For small angles, a linear projection is sufficient for the first gate; later gates should use perspective projection and configurable FOV.

---

## 3. Dynamics families

### Platform motion

Training mixtures:

- single sinusoid with randomized amplitude, frequency, phase, and bias;
- multi-sine oscillation;
- smooth attitude steps and turns;
- chirp segments;
- band-limited stochastic angular-rate disturbance.

Held-out tests:

- excluded frequency bands and amplitude combinations;
- nonstationary frequency or amplitude;
- sharp maneuver onset;
- recorded or higher-fidelity quadcopter attitude traces;
- impulses, labeled as reactive rather than anticipatory tests.

### Target motion

- stationary world bearing;
- constant angular velocity;
- piecewise acceleration;
- independent sinusoidal motion;
- maneuver onset not synchronized with platform motion.

Target and body motion must be independently randomized so the learner cannot use a fixed phase or trajectory shortcut.

### Actuator

The initial plant should support first- and second-order rate dynamics with:

- rate and acceleration limits;
- command latency and queueing;
- damping and time-constant variation;
- deadband, friction, backlash, and quantization profiles;
- hard angular limits.

Disabled imperfections must not consume random-number state, preserving deterministic replay across compatible configurations.

### Sensor

- configurable image resolution and FOV;
- 15–60 Hz update rate;
- timestamp jitter;
- fixed and variable detection latency;
- center and size noise;
- missed detection bursts;
- stale observations and confidence miscalibration.

---

## 4. Observation and action interface

The canonical observation stores:

\[
o_t=(e_t,w_t,h_t,c_t,m_t,\Delta t_t,q_t,\dot q_t,
\omega_t^{body},u_{t-1}),
\]

with validity masks for every optional field. O0, O1, and O2 profiles select fields through capability masks; unavailable values are never encoded as apparently valid zeros. The primary profile contains every signal available through the real deployment interface. Artificial sensor deprivation is evaluated only as an ablation.

The predictive interface emits a timestamped body-relative target state

\[
\hat s_t=(\hat q_t^{target/body},\hat{\dot q}_t^{target/body},
\sigma_{q,t},\sigma_{\dot q,t},m_t),
\]

including explicit validity, source-measurement time, and prediction horizon. A configured adapter converts this state into exactly one normalized logical command:

- desired rate \(u_t^{rate}\in[-1,1]\), using target-rate feed-forward plus bearing-error feedback; or
- absolute body-relative position \(u_t^{position}\in[-1,1]\), preserving zero as body-forward even with asymmetric travel.

The environment applies command delay, physical limits, and the selected actuator model. Every result must identify its adapter; rate and position commands are never silently mixed.

---

## 5. Leakage controls

The actor must not receive:

- true target or body bearing in O0/O1;
- future disturbance phase or maneuver schedule;
- nominal plant parameters when they are declared unknown;
- simulator-only delay queue state;
- an episode identifier correlated with dynamics family.

Use statistical probes to test whether nuisance fields predict hidden disturbance family, held-out split, or future maneuver onset beyond legitimate history.

The implemented dataset boundary keeps deployable features, behavior actions,
and privileged labels in separate arrays. All O0/O1/O2 views are encoded from
one rollout, while bearing/rate labels have no observation-profile dimension.
Absolute simulator time remains alignment metadata rather than a model input.

---

## 6. Evaluation blocks

Use non-overlapping deterministic seed blocks for:

1. training;
2. validation and checkpoint selection;
3. development evaluation;
4. untouched in-distribution test;
5. untouched shift tests.

Required shift blocks:

- maneuver spectrum;
- observation delay and frame rate;
- actuator time constant and saturation;
- FOV or pixel resolution;
- noise and dropout;
- combined moderate shifts.

The current deterministic development suite provides named nominal-combined, high-latency, dropout/noise, slow-servo, aggressive-motion, and travel-limit/recovery cases. These are development probes, not substitutes for the untouched randomized seed blocks required for final claims.

Use identical trajectories and initial states across controllers. Store generator version, configuration hash, controller observation profile, and model checkpoint hash with every result.

The current dataset manifest records schema version, full scenario and hardware
configuration, seed block, profiles, behaviors, horizons, array shapes, and a
SHA-256 configuration hash. A validation helper rejects overlapping seed blocks.
The domain-randomized generator now independently varies target/body motion,
maneuver pulses, sensing, actuation, timing, and initial state from the split
seed, and records each realized scenario for exact replay. Replaying the fixed
six-case development suite without that option remains invalid as a train/test
separation.

---

## 7. Acceptance gates

### Gate 0 — Identifiability

- a constructed delayed-response case separates feed-forward and history-aware policies;
- a predictable oscillation case rewards anticipation;
- an unpredictable impulse case confirms that bbox-only anticipation is impossible before observation;
- no target or disturbance leakage is detected.

### Gate 1 — Environment and conventional control

- deterministic replay;
- zero-action and sign-error controllers fail as expected;
- tuned PID controls the nominal plant well;
- predictive control improves at least one delayed or transient case;
- frequency sweeps expose declared limits rather than numerical instability.

### Gate 2 — Learned dynamics

- recurrent prediction beats latest-frame and constant-velocity predictors;
- action-conditioned prediction beats action-agnostic prediction;
- multi-step error remains useful over the controller horizon;
- privileged predictive distillation improves at least one declared generalization axis over imitation-only training;
- improvements survive held-out maneuver and plant parameters.

### Gate 3 — Learned policy

- the proposed policy improves P95/P99 error or loss-of-view rate over tuned PID and recurrent model-free RL at matched control effort;
- the advantage occurs specifically where history and prediction are needed;
- the deterministic actor meets the declared embedded inference-rate and tail-latency budget;
- nominal performance does not materially regress;
- results survive multiple training seeds and untouched test blocks.

### Gate 4 — Transfer

- performance survives higher-fidelity simulation or recorded flight motion;
- inference fits the deployment rate and compute budget;
- real hardware operates inside a declared safety envelope;
- failure detection or fallback is tested rather than assumed.

---

## 8. Valuable negative results

The study remains informative if:

- a well-tuned predictive conventional controller matches the learned model;
- recurrence helps but explicit world-model learning does not;
- IMU input is necessary for maneuver classes that bbox history cannot identify;
- model prediction improves but planning exploits residual model error;
- sim-to-real loss is attributable to delay, actuator, or perception mismatch.

Each result narrows where AI is justified in the visual servo loop.
