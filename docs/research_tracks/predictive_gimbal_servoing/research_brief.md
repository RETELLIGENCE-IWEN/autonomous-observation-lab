# Research Brief: Learned Predictive 1D Gimbal Servoing

## At a Glance

### Task

Continuously command a one-axis camera gimbal so that a designated object's bounding-box center remains at the image center while a quadcopter undergoes oscillations and maneuvers.

### Research proposition

A reactive mapping from current image error to gimbal command cannot distinguish target motion, platform rotation, actuator lag, and delayed perception. A compact recurrent controller, trained with privileged predictive supervision but deployed only with available telemetry, may infer the hidden temporal regime and act before the current error becomes large.

### Intended result

The intended result is not merely a controller that works in one simulator. It is evidence that learned predictive state provides a specific advantage under partial observability and distribution shift, measured against strong conventional and learned baselines.

---

## 1. Scope and responsibility boundary

The mini-project owns the mid-level visual servo:

\[
\text{target observation history} \longrightarrow \text{continuous 1D gimbal command}.
\]

It does not initially own:

- object detection or identity selection;
- two-axis coordination, zoom, search, or reacquisition;
- quadcopter flight control;
- motor current commutation or the safety-critical inner servo loop;
- mission-level decisions about which object should be observed.

The learned policy outputs a desired angular-rate command while an ordinary embedded actuator loop enforces electrical and mechanical limits. This places the visual continuous-control decision in the learned policy without pretending that motor firmware is part of the AI contribution.

The controlled axis is abstract. It may represent pan/yaw or tilt/pitch, but one axis must be fixed for an experiment rather than mixing both geometries silently.

---

## 2. Formal problem

Let the selected image axis be normalized to \([-1,1]\), with zero at the image center. From a bounding box

\[
b_t=(c_{x,t},c_{y,t},w_t,h_t),
\]

the controller uses the relevant center coordinate as image error \(e_t\). Width and height provide imperfect scale and motion cues.

The hidden state includes

\[
s_t=(\theta_t^{target},\dot\theta_t^{target},
\theta_t^{body},\dot\theta_t^{body},
q_t,\dot q_t,\psi,\ell_t),
\]

where \(q_t\) is gimbal state, \(\psi\) contains fixed but unknown camera and actuator parameters, and \(\ell_t\) contains latency or command-queue state. Several hidden causes can produce the same instantaneous \(e_t\), so the deployment problem is a POMDP when only image observations are available.

### Observation profiles

| Profile | Inputs | Role |
|---|---|---|
| O0: Vision-only | bbox/center-size, validity, confidence, previous action, time delta | Hard partial-observation condition |
| O1: Servo-aware | O0 plus gimbal angle and angular rate | Recommended primary deployable condition |
| O2: Disturbance-aware | O1 plus quadcopter attitude/body rate | Information-rich ablation or teacher |
| OP: Privileged | true LOS, target/body motion, plant and delay state | Oracle or training teacher only |

The primary condition uses every signal genuinely available to the final payload. O1 is primary when the payload is isolated from vehicle telemetry; O2 is primary when body state is part of the deployable interface. O0 remains a required ablation and becomes primary only if bbox-only operation is a real constraint. O0 tests whether the policy can identify control direction and lag purely from intervention history.

### Continuous action

The initial action is a normalized desired angular-rate command:

\[
u_t\in[-1,1], \qquad \dot q_t^{cmd}=u_t\dot q_{max}.
\]

Rate command is preferred over raw torque for the first study because it isolates visual servo intelligence from motor-current stabilization and makes simulation-to-hardware transfer safer. Angle increment and torque command can be later action-profile ablations.

### Objective

Define the per-step cost

\[
C_t=
\lambda_e\rho(e_t)
+\lambda_b\mathbf 1[|e_t|>e_{warn}]
+\lambda_l\mathbf 1[\text{target lost}]
+\lambda_u u_t^2
+\lambda_{\Delta u}(u_t-u_{t-1})^2.
\]

The policy minimizes discounted expected cost. The robust error \(\rho\) should preserve sensitivity near the center without allowing a few catastrophic losses to dominate every learning signal. Loss of view, saturation, and mechanical-limit violations are reported separately rather than hidden inside a single return.

Privileged body or target state must not enter the reward in a way that makes an undeployable shortcut available to the actor.

---

## 3. Research question and hypotheses

### Central question

> Does learned predictive state create a measurable visual-servo advantage when the instantaneous bounding box is insufficient to infer platform disturbance, target motion, latency, and actuator response?

### H1 — Temporal state

A recurrent policy will outperform a feed-forward policy when observation delay, actuator lag, or oscillatory base motion makes the optimal action history-dependent.

**Falsified if:** a feed-forward policy with an equal observation window matches the recurrent policy on held-out temporal regimes.

### H2 — Predictive model

An action-conditioned recurrent world model will improve high-percentile centering error and loss-of-view rate relative to recurrent model-free RL by predicting the consequences of candidate continuous actions.

**Falsified if:** world-model prediction accuracy does not translate into action advantage, or recurrent SAC/TD3 matches it at equal interaction and parameter budgets.

### H3 — Dynamics generalization

Latent online system identification will retain more performance than a tuned nominal controller when FOV, latency, actuator time constant, mounting sign, and maneuver spectrum change within declared bounds.

**Falsified if:** a gain-scheduled or robust conventional controller matches it across the same shifts.

### H4 — Privileged predictive transfer

Distilling a teacher that observes body motion, true LOS, actuator state, and delay state into a deployable student will improve disturbance inference without requiring simulator-only inputs at deployment.

**Falsified if:** action-only imitation, ordinary recurrent RL, or direct inclusion of a short history matches the distilled student.

---

## 4. Proposed method

### 4.1 Learned information state

A compact recurrent state consumes observation and intervention history:

\[
h_t=f_\phi(h_{t-1},o_t,u_{t-1}).
\]

The explicit inclusion of the previous command is essential. It allows the model to distinguish externally induced image motion from the observed consequence of its own action.

### 4.2 Prediction heads

The world model predicts distributions over:

- future image error and error rate;
- bbox scale or size change;
- target visibility and FOV-exit probability;
- gimbal response and saturation margin;
- observation validity under delay or dropout;
- accumulated tracking cost.

Predicting control-relevant variables is preferred to reconstructing pixels.

### 4.3 Locked controller

The initial deployed controller is a small recurrent actor with auxiliary predictive heads. Training proceeds through:

1. imitation warm-start from a competent PID or MPC teacher;
2. privileged predictive distillation from simulator-only state;
3. SAC fine-tuning under randomized but explicitly bounded dynamics.

The deployed actor is deterministic and emits the full desired-rate command. Online latent MPC is reserved for a later diagnostic or ablation because the first controller must fit a predictable embedded inference budget.

### 4.4 Uncertainty boundary

Periodic disturbances can be inferred and anticipated from history. An unannounced impulsive body rotation is not predictable from bbox history before it occurs. The method must distinguish predictable disturbance, reaction after unpredictable disturbance, and information available only through IMU input.

An uncertainty or ensemble signal may trigger conservative rate limits or an OOD fallback. It must not be presented as a formal stability guarantee without one.

---

## 5. Baselines

| ID | Baseline | Scientific purpose |
|---|---|---|
| C0 | Hold/zero action | Sanity lower bound |
| C1 | Tuned P or PD visual servo | Minimum credible controller |
| C2 | PID with filtering, anti-windup, and rate limits | Strong conventional baseline |
| C3 | Alpha-beta/Kalman prediction plus PID | Tests whether simple state estimation is enough |
| C4 | Identified DMC/MPC | Strong predictive conventional baseline |
| L0 | Feed-forward SAC or TD3 | Tests whether instantaneous nonlinear policy is enough |
| L1 | Recurrent SAC or TD3 | Tests learned memory without an explicit world model |
| L2 | Supervised neural inverse controller | Tests whether RL is needed |
| P | Teacher-warm-started, privileged-distilled recurrent predictive actor | Proposed mechanism |
| UB | Privileged-state constrained MPC | Approximate upper bound |

All baselines must receive the same declared observation profile. Controller tuning budget, training interactions, action rate, and actuator limits must be reported.

---

## 6. Evaluation priorities

Mean pixel or normalized error is insufficient. Report:

- RMS, median, P95, and P99 absolute centering error;
- fraction of time inside a tight center band;
- FOV exits and time to recovery;
- overshoot and settling time after maneuver onset;
- control effort, command variation, saturation, and limit hits;
- error versus disturbance frequency and amplitude;
- inference latency and deployed policy size;
- paired-seed differences and multi-training-seed variation.

The primary result should be a tracking-error versus control-effort frontier, supplemented by tail failures. A policy that centers well by chattering or remaining saturated is not superior.

---

## 7. Minimum publishable shape

The smallest credible contribution is:

1. a deterministic 1D gimbal benchmark with controlled maneuver, delay, sensing, and actuator shifts;
2. a recurrent action-conditioned image-plane dynamics model;
3. a teacher-warm-started and privileged-distilled continuous recurrent policy;
4. comparison with properly tuned PID, predictive control, and recurrent model-free RL;
5. held-out frequency, latency, and actuator evaluations;
6. evidence that prediction quality is causally connected to anticipatory action and lower tail error.

The contribution is weakened substantially if it is only “DDPG centers the box in Unity” or if it compares against an untuned PID controller.

---

## 8. Immediate decisions

Before implementation, record the following hardware parameters. These instantiate the locked concept rather than reopening it:

- physical axis: pan/yaw or tilt/pitch;
- available deployment telemetry: bbox only, gimbal encoder, or vehicle IMU;
- command interface: angle, rate, PWM, torque, or current;
- camera update rate, detection latency, and jitter;
- gimbal range, rate, acceleration, and low-level loop behavior;
- whether the target moves independently of the quadcopter;
- simulator or logged flight source available for training;
- real-hardware validation expectation and safety envelope.

Until these are resolved, the benchmark should expose them as configuration profiles rather than embedding one assumed hardware design.
