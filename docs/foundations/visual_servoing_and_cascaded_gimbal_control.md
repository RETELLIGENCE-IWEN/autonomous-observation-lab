# Visual Servoing and Cascaded Gimbal Control

## At a glance

**Visual servoing** closes a feedback loop around image measurements. Instead of first reconstructing a complete three-dimensional world and then planning camera motion, it uses visual feature error directly to command motion. For a one-axis tracking gimbal, the central feature is usually the target's horizontal image position; bounding-box scale and confidence supply useful context about distance, visibility, and measurement quality.

This note covers the durable control structure underneath both classical and learned gimbal trackers: image geometry, image-based feedback, cascaded outer and inner loops, disturbance rejection, delay, saturation, and loss of view. It does not prescribe one controller. PID, disturbance-observer control, model predictive control, and learned policies can all occupy the outer loop while sharing the same plant and safety envelope.

The key distinction is:

- the **outer visual loop** decides how the camera should move to reduce future image error;
- the **inner actuator loop** makes the motor realize that motion quickly and safely.

Keeping that boundary explicit makes a learned controller easier to train, compare, deploy, and replace.

## 1. The problem begins in the image plane

Let a detector report a bounding box

\[
b_k=(c_{x,k},c_{y,k},w_k,h_k,q_k),
\]

where \((c_x,c_y)\) is its center, \((w,h)\) its size, and \(q\) an optional confidence or validity signal at time \(t_k\). If the desired horizontal location is \(c_x^*\), a focal-length-normalized horizontal error is

\[
e_k = \frac{c_{x,k}-c_x^*}{f_x}.
\]

Normalization makes the feature approximately angular for modest fields of view and makes results less tied to pixel resolution. When camera intrinsics are unavailable, width-normalized pixel error is still useful, but it should not be mistaken for a calibrated angle.

The controller's goal is not merely to make the current \(e_k\) small. It must keep future error small while respecting field of view, angular-rate, acceleration, travel, and electrical limits. That matters on a quadcopter because the measurement reflects several coupled effects:

- vehicle rotation and translation;
- target motion;
- gimbal motion and actuator lag;
- detection, transport, scheduling, and inference delay;
- detector noise, dropouts, and bounding-box deformation.

The same observed error can therefore demand different actions. A target ten pixels right and moving left is not equivalent to a target ten pixels right and accelerating out of frame. This ambiguity is the reason estimation, prediction, or memory enters the problem.

## 2. The geometric core: the interaction matrix

Image-based visual servoing describes how camera velocity changes image features. For a point with normalized image coordinates \(s=(x,y)\), depth \(Z\), and camera twist

\[
v_c=(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z)^\top,
\]

one common camera-velocity convention gives

\[
\begin{bmatrix}
\dot{x}\\
\dot{y}
\end{bmatrix}
=
L_s v_c,
\]

with

\[
L_s=
\begin{bmatrix}
-1/Z & 0 & x/Z & xy & -(1+x^2) & y\\
0 & -1/Z & y/Z & 1+y^2 & -xy & -x
\end{bmatrix}.
\]

The matrix \(L_s\), called the **interaction matrix** or image Jacobian, maps physical camera motion into instantaneous image motion. Its entries show why visual control is not only a pixel-space problem: translation depends on depth, rotational axes couple, and the sensitivity changes across the image. Signs vary with coordinate and twist conventions, so an implementation must document and test its convention rather than copy signs blindly.

For feature error \(e=s-s^*\), the idealized image-based law

\[
v_c=-\lambda \widehat{L}_s^{+}e
\]

uses an estimated pseudoinverse of the interaction matrix to produce exponentially decreasing error under favorable local assumptions. Here \(\lambda>0\) sets the convergence rate. The hat matters: depth, calibration, target geometry, and motion are rarely exact.

For a one-axis gimbal, much of this geometry can be reduced to a local scalar model:

\[
\dot e(t)=d(t)-k(t)\,\omega_g(t),
\]

where \(\omega_g\) is achieved gimbal rate, \(k(t)>0\) is the local image-motion gain, and \(d(t)\) collects target and carrier motion that would occur with a stationary gimbal. Sampled at interval \(\Delta t_k\),

\[
e_{k+1}\approx e_k+\Delta t_k\left(d_k-k_k\omega_{g,k}\right).
\]

This compact equation exposes the real task. A good outer loop must infer the disturbance \(d_k\), understand the effective gain and actuator response, and act early enough that delay does not turn correction into oscillation.

## 3. Cascaded control is an architectural contract

A practical gimbal normally uses nested loops:

\[
\text{detections and telemetry}
\rightarrow
\boxed{\text{visual outer loop}}
\xrightarrow{\omega_g^*}
\boxed{\text{motor/rate inner loop}}
\rightarrow
\text{camera motion}
\rightarrow
\text{next image}.
\]

The outer loop may run at the detector or policy rate, often tens of hertz. It emits a desired angular rate \(\omega_g^*\), or less commonly a desired angle. The inner loop runs much faster, often hundreds of hertz or more, using encoder and inertial measurements to track the command and enforce current, speed, acceleration, and travel limits.

Rate command is a useful learned-control interface because it:

- removes motor commutation and much of the electrical dynamics from the learning problem;
- preserves a stable, testable actuator layer;
- allows the policy to express anticipatory motion rather than only static pointing;
- makes classical and learned outer loops interchangeable at the same boundary.

This division does not make the actuator disappear. The outer controller still experiences lag, saturation, deadband, backlash, rate-dependent gain, and command transport delay. Those effects belong in its observation history, training distribution, or internal model.

An angle-command interface can work when the embedded gimbal already provides a well-characterized position servo. It also hides rate and acceleration decisions, however, and may create an opaque cascade of integrators. Direct torque or current control gives maximum authority but turns a compact visual-servo project into a motor-control and safety project. The interface is therefore part of the scientific claim, not an incidental API choice.

## 4. Classical outer loops and what they reveal

With the sign convention above, a proportional-integral-derivative outer loop can be written

\[
\omega_g^* = K_p e + K_i\sum_j e_j\Delta t_j + K_d\widehat{\dot e}.
\]

The proportional term corrects displacement, the derivative term reacts to image motion, and the integral term removes persistent bias. This controller is strong when sampling, gain, and delay are stable. Its weaknesses are also diagnostic:

- noisy detections corrupt the derivative term;
- delay reduces phase margin and produces overshoot;
- saturation causes integral windup;
- a fixed gain trades slow small-error response against aggressive high-error response;
- purely reactive action cannot cancel a disturbance before error appears.

A state estimator can turn a sequence of detections into estimates of error, error rate, bias, or sinusoidal disturbance state. Feedforward from body rate can cancel predictable carrier rotation before it enters the image, when time alignment and frame transforms are accurate.

A disturbance observer treats the unmodeled term in the scalar dynamics as something to estimate. Informally,

\[
\widehat d_k \approx \widehat{\dot e}_k + \widehat k_k\omega_{g,k}.
\]

The estimate can then be canceled in the command. Robust visual-servo work for inertially stabilized platforms shows why this is a serious baseline: target motion, uncertain depth, angular rate, tracking error, and camera parameters can be grouped as disturbances and addressed with observer-based predictive control.

Model predictive control instead rolls a model forward over a horizon and chooses a command sequence that minimizes an objective such as

\[
\sum_{i=1}^{H}
q_e e_{k+i}^2
+q_u(\omega^*_{k+i})^2
+q_{\Delta u}(\Delta\omega^*_{k+i})^2,
\]

subject to rate, acceleration, travel, and field-of-view constraints. The terms respectively penalize tracking error, excessive motion, and command chatter. MPC makes prediction and constraints explicit, but depends on model quality and sufficient compute. It is an especially important comparator for any learned controller advertised as predictive.

## 5. Delay, irregular time, and actuator dynamics

If the command applied during the next image interval was issued \(d\) steps earlier, a more honest model is

\[
e_{k+1}=e_k+\Delta t_k\left(d_k-k_k\omega_{g,k-d}ight)+\nu_k,
\]

where \(\nu_k\) represents measurement and model error. Treating \(d=0\) when delay is material makes a controller respond to an old world. Treating every interval as the same duration makes velocity estimates and recurrent state inconsistent under jitter.

Useful operational practices are:

- timestamp at capture, inference completion, command issue, and actuator application when available;
- expose \(\Delta t_k\), measurement age, and recent applied commands to an estimator or learned policy;
- simulate command queues and dropped or repeated frames, not only additive noise;
- measure the inner-loop step and frequency response rather than assuming instantaneous rate tracking;
- use anti-windup and rate/acceleration limiting outside any learned component.

Latency is not merely another scalar to randomize. It changes which action caused which observation, so the controller needs enough history or explicit queued-action state to preserve that causal relationship.

## 6. Field of view changes the objective

Mean squared centering error does not fully represent tracking utility. Once a target leaves the frame, the detector may provide no gradient indicating where it went. Near the boundary, a controller should value margin and predicted visibility, not only current centering.

With half image width \(W/2\), define normalized margin

\[
m_k = 1-\frac{|c_{x,k}-c_x^*|+w_k/2}{W/2}.
\]

Positive \(m_k\) indicates that the horizontal extent of the box remains inside the image; negative margin indicates clipping or loss. A predictive controller can estimate future margin or time to boundary and act before the center error becomes large. Bounding-box width is also useful because a nearby or rapidly growing target consumes the field of view faster, although box size is only an imperfect depth cue.

This reframes the task from regulation alone to **constrained visibility maintenance**.

## 7. Capabilities and non-guarantees

The visual-servo formulation provides a meaningful error signal, a local motion model, and a way to compare controllers at a common interface. Cascading isolates the high-bandwidth safety-critical actuator loop. Observer and predictive designs can reject structured disturbances and account for constraints.

These facts do not guarantee global stability, permanent target visibility, correct association, or safe learned behavior. The interaction matrix is local and uncertain; target switching can make the feedback signal discontinuous; saturation can remove control authority; delay can destabilize an otherwise sensible law; and no one-axis gimbal can compensate image motion outside its controllable axis or mechanical range.

Any stability statement is conditional on the modeled plant, delay bounds, feature validity, controllability, and controller assumptions. Empirical success under randomized simulation is evidence of robustness over that tested distribution, not a proof outside it.

## 8. Failure modes and diagnostics

| Failure | Likely cause | Observable symptom | Diagnostic | Mitigation |
|---|---|---|---|---|
| Sustained oscillation | excessive gain, delay, or actuator resonance | alternating error with growing or fixed amplitude | frequency response and error-command phase plot | reduce bandwidth, model delay, notch/filter, predictive compensation |
| Slow lag behind maneuvers | reactive law or insufficient rate authority | error follows body-rate peaks | align body rate, command, and image-error traces | feedforward, state estimation, prediction, more authority if safe |
| Limit cycle near center | quantization, deadband, noisy derivative, reward shaping | persistent small command reversals | command histogram and high-frequency power | deadband compensation, hysteresis, smoothing, action-change penalty |
| Windup after saturation | integral state continues accumulating | long recovery after rate/travel limit | plot integrator and saturation flags | anti-windup, constrained control, reset logic |
| Unexpected loss of view | optimizing center error without future margin | good mean error but abrupt clipping events | conditional error near frame boundary | visibility-risk term, horizon prediction, recovery mode |
| Performance changes with frame rate | fixed-step estimator or policy | bias or instability under jitter | replay identical motion with altered timestamps | timestamp-aware state update and timing randomization |
| Wrong-way correction | frame/sign/calibration mismatch | error initially increases under a step | low-rate signed step test | explicit frame convention and automated polarity test |
| False stability in simulation | ideal actuator or zero-delay sensor | real hardware oscillates despite simulated success | hardware-in-the-loop swept-sine and delay injection | measured actuator/timing model and held-out stress tests |

## 9. Neighboring concepts

| Concept | Relationship | Important difference |
|---|---|---|
| Image-based visual servoing (IBVS) | uses image features directly in feedback | may use an explicit interaction matrix rather than a learned recurrent policy |
| Position-based visual servoing (PBVS) | regulates a reconstructed 3-D pose | requires sufficient geometry and pose estimation |
| Image stabilization | rejects camera motion to keep the whole image steady | target tracking regulates one selected object's location and may intentionally move the background |
| Gaze or active perception | chooses camera motion to improve information | centering is one utility; information gathering may deliberately move off center |
| Disturbance-observer control | estimates and cancels lumped unknown motion | depends on observer bandwidth and nominal model |
| Model predictive control | predicts constrained future behavior | uses an explicit online model and optimizer |
| Learned visual servoing | learns state, prediction, control, or all three | requires distributional validation and does not inherit stability automatically |

## 10. Implications for predictive 1D gimbal servoing

The mini-project can be stated cleanly in this language:

- **Observation:** timestamped bbox center, size, confidence, recent applied commands, and all telemetry truly available on the payload.
- **Hidden state:** line-of-sight rate, carrier disturbance, target motion, actuator state, effective delay, and local image gain.
- **Action:** bounded desired gimbal angular rate at the outer-loop rate.
- **Prediction:** short-horizon future image error, box scale or margin, and actuator response.
- **Utility:** low tail error and loss-of-view rate with smooth commands and limited saturation.
- **Safety boundary:** conventional high-rate inner loop, mechanical/electrical constraints, watchdog, and fallback controller.

The most informative baselines are not a weak proportional law. They are a well-tuned PID with anti-windup, a state-estimator or body-rate-feedforward controller, a disturbance-observer/predictive controller, and an MPC if computationally plausible. These reveal whether learning adds useful inference and adaptation or only replaces standard filtering and prediction.

Useful ablations remove recurrence, timestamps, body telemetry, privileged training, auxiliary prediction, and dynamics randomization one at a time. Metrics should include median and high-percentile absolute image error, time outside tolerance, loss-of-view probability, recovery time, command variation, saturation time, and performance under held-out delay and actuator dynamics.

## Durable takeaways

1. A bbox is a visual feature, not a complete control state; its history and timing carry the motion information.
2. A one-axis gimbal can be modeled locally as error motion minus achieved corrective rate, with unknown disturbance, gain, lag, and delay.
3. The outer visual loop and inner motor/rate loop solve different problems and should remain an explicit interface.
4. Prediction is valuable when it cancels disturbance before image error grows and when it preserves field-of-view margin.
5. Strong observer-based and predictive controllers are necessary baselines for a learned predictive servo.
6. Tail error, target loss, saturation, smoothness, and recovery matter more than average centering error alone.

## Primary sources

- François Chaumette and Seth Hutchinson, [“Visual Servo Control, Part I: Basic Approaches”](https://doi.org/10.1109/MRA.2006.250573), *IEEE Robotics & Automation Magazine*, 2006.
- Seth Hutchinson, Gregory D. Hager, and Peter I. Corke, [“A Tutorial on Visual Servo Control”](https://doi.org/10.1109/70.538972), *IEEE Transactions on Robotics and Automation*, 1996.
- Xiangyang Liu et al., [“Robust Predictive Visual Servoing Control for an Inertially Stabilized Platform with Uncertain Kinematics”](https://doi.org/10.1016/j.isatra.2020.12.039), *ISA Transactions*, 2021.
- Jun Yang et al., [“Sampled-Data Robust Visual Servoing Control for Moving Target Tracking of an Inertially Stabilized Platform with a Measurement Delay”](https://doi.org/10.1016/j.automatica.2021.110105), *Automatica*, 2022.
