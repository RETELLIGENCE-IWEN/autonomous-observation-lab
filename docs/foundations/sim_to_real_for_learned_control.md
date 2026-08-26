# Sim-to-Real for Learned Control

## At a glance

**Sim-to-real transfer** asks whether a controller trained with a simulator remains useful on the physical system. The gap is not one thing. It includes mechanics, actuators, sensing, perception, timing, computation, interfaces, resets, and the distribution of tasks and disturbances.

The main tools are system identification, dynamics and observation randomization, learned actuator models, online adaptation, and hybrid or residual control. None guarantees transfer. They are ways to make the training distribution cover the causal variations that matter and to test which variations the policy can tolerate.

For vision-based gimbal control, timing and actuator response deserve the same status as mass or friction. A controller that sees delayed bounding boxes and commands a filtered motor is acting in a different dynamical system from one with instantaneous frames and ideal rate tracking.

## 1. The reality gap is a set of mismatches

A simulator defines transition and observation processes

\[
x_{k+1}\sim p_{\text{sim}}(x_{k+1}\mid x_k,u_k;\xi),
\qquad
o_k\sim p_{\text{sim}}(o_k\mid x_k;\xi),
\]

parameterized by \(\xi\). The real system follows unknown processes \(p_{\text{real}}\). Transfer fails when the deployed policy relies on a simulated relationship that does not hold physically.

Useful gap categories are:

1. **Dynamics:** inertia, balance, coupling, damping, friction, backlash, flex, vibration, disturbance spectra, and unmodeled axes.
2. **Actuation:** gain, bandwidth, deadband, saturation, acceleration limits, current limiting, rate-loop tuning, and command quantization.
3. **Sensing and perception:** intrinsics, exposure, blur, rolling shutter, detector bias, box jitter, missed frames, false association, confidence calibration, and field-of-view geometry.
4. **Timing and software:** camera exposure time, transport, inference time, scheduler jitter, command queues, packet loss, clock offset, and variable control intervals.
5. **Task distribution:** target motion, vehicle maneuvers, initial state, occlusion, environmental conditions, and episode/reset logic.
6. **Interface semantics:** sign, coordinate frame, physical units, timestamp meaning, whether an action is requested or achieved, and what constitutes a terminal event.

Photorealism is only one part of the observation gap. If the controller consumes detector outputs rather than raw pixels, realistic box error, dropout, latency, and target association may matter more than rendered texture.

## 2. System identification and calibration

System identification estimates a model or parameters from physical input-output data. For a rate-controlled gimbal, useful experiments include:

- signed low-amplitude steps to verify polarity and local gain;
- step responses across command amplitudes to expose deadband and saturation;
- chirps or swept sines to estimate bandwidth, delay, and resonance;
- reversals to expose backlash and asymmetric friction;
- repeated tests across voltage, temperature, payload, and orientation;
- timestamped end-to-end tests from image capture to measured gimbal response.

A point estimate \(\widehat\xi\) supports a calibrated nominal simulator. It is rarely sufficient because parameters vary and the model class omits effects. The residuals of identification experiments are therefore as important as the fitted values. They help define plausible randomization, noise structure, and held-out conditions.

Calibration should prioritize **closed-loop relevant fidelity**. A model that matches motor current in detail but misses forty milliseconds of transport delay may be less useful to the outer visual controller than a simple first-order actuator model with correct delay and saturation.

## 3. Domain and dynamics randomization

Domain randomization trains across a distribution of simulator parameters:

\[
\max_\theta
\mathbb E_{\xi\sim p_{\text{train}}(\xi)}
\left[
J\left(\pi_\theta;p_{\text{sim},\xi}\right)
\right].
\]

The hope is that the real system lies inside a region where the learned policy already succeeds. Visual domain randomization varies appearance and sensing; dynamics randomization varies the transition and actuator process. A gimbal task may randomize:

- inner-loop time constant, delay, gain, damping, deadband, rate limit, and acceleration limit;
- vehicle angular-motion spectra, impulses, and cross-axis coupling;
- target line-of-sight motion and distance/scale evolution;
- camera interval, capture-to-detection latency, command delay, and packet dropout;
- box-center bias, correlated jitter, scale error, confidence, clipping, and temporary loss;
- calibration, field of view, mounting offset, and encoder/IMU bias where relevant.

The distribution \(p_{\text{train}}\) is a model of uncertainty. Wide independent uniform ranges are convenient but often physically implausible. Some variables are correlated: heavier payload can change inertia and actuator bandwidth; lower light increases exposure time and blur; compute load can affect both inference delay and frame drops. Preserving plausible dependence can make training both harder and more realistic.

Randomization has two opposing failure modes:

- **too narrow:** the policy specializes and fails outside nominal conditions;
- **too broad or incoherent:** the task becomes unnecessarily hard, the policy learns a conservative compromise, or it adapts to impossible combinations.

Measured nominal values and variation should define the center and scale when possible. Unmeasured dimensions should be separated into defensible assumptions and tested by sensitivity sweeps.

## 4. Timing is part of the state

In a delayed loop, an observation corresponds to an earlier physical state and the most recent command may not yet have affected the plant. If latency is variable, a fixed-delay policy cannot know the correct action-observation alignment without time information.

A useful observation history contains

\[
(o_k,t_k^{\text{capture}},t_k^{\text{available}},
u_k^{\text{requested}},u_k^{\text{applied}})
\]

or a feasible subset from which measurement age, interval, and queue state can be derived. When clocks cannot be synchronized, relative monotonic timestamps and measured transport durations are still valuable.

Training should reproduce the causal sequence:

1. the physical state evolves during exposure and processing;
2. a delayed or dropped detection becomes available;
3. the policy runs for a variable duration;
4. its command enters a queue or communication path;
5. the inner loop filters and applies it;
6. the next image reflects the achieved motion.

Adding random noise to a current observation does not reproduce this process. Timing randomization must change which past state is observed and which past command acts. Research on real-time robot learning has shown that variable computation and sensing delays can materially change behavior and that exposing elapsed time can improve robustness.

## 5. Learned actuator models

Some systems are difficult to model analytically at the fidelity useful for policy training. A learned actuator model can map command and recent actuator state to achieved torque, rate, or position change:

\[
\widehat y_{k+1}=f_\psi(y_{k-L:k},u_{k-L:k},c_k),
\]

where \(c_k\) may include voltage, temperature, or configuration. Hwangbo and colleagues demonstrated the importance of learned actuator dynamics for transferring agile legged behavior, while other sim-to-real work emphasizes accurate latency and motor models.

The model should be validated on held-out command sequences, particularly around reversals, saturation, and frequencies that the policy uses. A low average one-step error can hide phase error that destabilizes a feedback loop. Frequency-conditioned and rollout validation are therefore important.

A learned actuator model is still a model. It can extrapolate badly, erase stochastic variation, or encourage the policy to exploit prediction artifacts. Randomizing around model residuals and retaining hard actuator constraints can reduce those risks.

## 6. Online adaptation

When real dynamics vary between deployments or during operation, a policy can condition on an estimated latent:

\[
\widehat z_k=g_\psi(h_k),
\qquad
u_k=\pi_\theta(o_k,\widehat z_k).
\]

The latent may capture payload, actuator response, delay, or disturbance regime. Training can use true simulated parameters as privileged supervision and teach the adaptation module to infer their task-relevant embedding from history.

Adaptation requires excitation. A perfectly centered, nearly stationary gimbal may not reveal its full bandwidth or deadband. Aggressive identification actions may be unsafe or visually undesirable. The controller must therefore adapt from naturally occurring commands, use a safe calibration routine, or accept residual uncertainty.

Adaptation also operates on a timescale. Payload inertia may remain constant for a flight; motor temperature drifts slowly; packet delay changes frame to frame. A single latent update rule can blur these processes. Explicitly separating static context, slow state, and fast timing variables often produces a clearer design.

## 7. Hybrid and residual strategies

Full policy transfer is not the only option. A hybrid controller can keep a nominal feedback law and let learning estimate disturbance, tune gains, select modes, or add a bounded residual:

\[
u_k=
u_{\text{nominal}}(o_k)
+\alpha(o_k)\,u_{\text{residual}}(h_k),
\]

with explicit clipping or gating. This preserves useful structure and can reduce the learned action space. It also narrows the novelty claim: the learned component improves a known controller rather than replacing the entire outer loop.

For the present gimbal concept, the intended balance is different: a learned outer-loop policy owns the desired-rate decision, while the conventional inner loop and supervisor own actuator stabilization and safety. PID/MPC imitation is a training tool and fallback, not the deployed nominal decision. This boundary should remain locked unless evidence forces a change.

## 8. A transfer and validation ladder

Transfer should be staged so each failure localizes a gap:

1. **Nominal simulation:** verify the task, signs, metrics, and controller implementation.
2. **Randomized simulation:** train across plausible variations.
3. **Held-out simulation:** test unseen parameter combinations, maneuvers, delays, and disturbance spectra; do not train on this set.
4. **Recorded-data replay:** run perception and policy timing against hardware or flight logs where closed-loop action is not required.
5. **Software- or processor-in-the-loop:** measure real inference, scheduling, serialization, and command timing.
6. **Hardware-in-the-loop or bench gimbal:** drive the real actuator with safe synthetic/recorded target motion and characterize the full command-response path.
7. **Constrained flight:** tethered, guarded, or otherwise risk-limited trials with fallback and logging.
8. **Representative free flight:** test predeclared scenarios and failure recovery.

At every level, compare controllers on matched trajectories or paired random seeds when possible. Report distributions and tails, not a best demonstration video.

For gimbal tracking, useful transfer metrics include:

- median, 95th, and 99th percentile absolute image error;
- probability and duration of target loss;
- time to first loss and recovery time;
- command total variation and high-frequency energy;
- rate, acceleration, current, and travel-limit use;
- end-to-end observation and action latency distributions;
- compute load, missed deadlines, and watchdog/fallback events.

## 9. Capabilities and non-guarantees

System identification improves nominal fidelity. Randomization exposes the controller to variation. Learned actuator models capture difficult dynamics. Adaptation can infer persistent hidden context. Staged testing catches mismatches before the highest-risk trial.

None proves that the physical system lies inside the training support or that a neural policy is safe outside it. Average robustness over \(p_{\text{train}}\) does not imply worst-case robustness. Hardware wear, a new detector, an unseen vibration resonance, or correlated timing failure can invalidate prior evidence.

Operational safety should include hard actuator limits, stale-data detection, a control-rate watchdog, policy-output validity checks, an out-of-distribution signal where it has demonstrated value, a tested fallback controller, and logs sufficient to reconstruct timing and causality. A fallback is useful only if the switch condition and transition have themselves been tested.

## 10. Failure modes and diagnostics

| Failure | Likely cause | Observable symptom | Diagnostic | Mitigation |
|---|---|---|---|---|
| Simulation specialist | narrow or exploitable training model | sharp drop on hardware or minor parameter shift | one-factor and combined held-out sweeps | measured randomization, remove shortcuts, validate support |
| Over-conservative policy | randomization too wide or implausible | slow response even on nominal plant | compare performance across randomization widths | use calibrated/correlated distributions and curriculum |
| Hardware oscillation | missed delay, phase lag, or resonance | simulation stable; bench loop oscillates | end-to-end frequency/phase measurement | timing-aware model, actuator model, bandwidth limit |
| Wrong temporal credit | delay simulated as noise instead of a queue | action seems correlated with wrong observation | causal timestamp trace and impulse test | explicit capture/compute/command queue simulation |
| Parameter shortcut | randomized parameter correlated with visual/context cue | adaptation fails after correlation swap | counterfactual swaps | independent data generation and held-out combinations |
| One-step model looks good, rollout fails | accumulated bias or phase error | long-horizon drift/instability | multi-step and frequency-conditioned validation | recurrent/history model, residual randomization, shorter trusted horizon |
| Adaptation cannot converge | insufficient excitation or rapidly changing latent | unstable or average latent estimates | identifiability test under actual command history | safe calibration, uncertainty, separate timescales |
| Fallback makes event worse | unsafe handover or stale controller state | command jump at mode switch | forced-switch bench tests | bumpless transfer, state synchronization, transition limits |
| Detector reality gap | idealized boxes and dropouts | control fails despite accurate mechanics | inject logged detector errors into simulation | learned/error model, confidence/validity input, dropout recovery |

## 11. Implications for predictive 1D gimbal servoing

The initial simulator should be deliberately modest but causally complete. It needs one-axis camera geometry, target and carrier-induced line-of-sight motion, an inner-loop actuator model, command and measurement queues, variable timestamps, bounds, field-of-view loss, and a detector-output error process. Full photorealistic rendering is optional while bbox features are the policy input.

Randomization should be divided into three sets:

- **calibration distribution:** ranges supported by measured hardware or supplier evidence;
- **training expansion:** a defensible margin beyond measured variation;
- **held-out challenges:** unseen combinations, temporal spectra, and failure modes used only for evaluation.

The experiment should include nominal and randomized training, with and without explicit timestamps, with fixed and variable latency, and with nominal versus learned/measured actuator dynamics. A policy that succeeds only after indiscriminately wide randomization is less informative than one whose gains can be traced to the correct timing and actuator factors.

Before flight, the exported deterministic actor should run at its target compute budget through the real serialization and scheduler path. Its observations must conform to the deployment contract, its requested and applied commands must both be logged, and watchdog/fallback transitions must be forced deliberately on the bench.

## Durable takeaways

1. The reality gap includes dynamics, actuation, perception, timing, software, tasks, and interface semantics.
2. System identification should define a nominal model and uncertainty evidence, not only one best-fit parameter set.
3. Domain randomization is training over a chosen uncertainty distribution; its coverage and correlations are scientific assumptions.
4. Delay and jitter alter causal action-observation alignment and must be simulated as timing processes, not merely as noise.
5. Learned actuator models and online adaptation are useful when their predictions are validated on control-relevant trajectories and timescales.
6. A staged transfer ladder, hard safety envelope, and held-out tail metrics provide evidence; none constitutes a universal transfer guarantee.

## Primary sources

- Josh Tobin et al., [“Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World”](https://arxiv.org/abs/1703.06907), *IROS*, 2017.
- Xue Bin Peng et al., [“Sim-to-Real Transfer of Robotic Control with Dynamics Randomization”](https://arxiv.org/abs/1710.06537), *ICRA*, 2018.
- Jie Tan et al., [“Sim-to-Real: Learning Agile Locomotion for Quadruped Robots”](https://arxiv.org/abs/1804.10332), *Robotics: Science and Systems*, 2018.
- Jemin Hwangbo et al., [“Learning Agile and Dynamic Motor Skills for Legged Robots”](https://doi.org/10.1126/scirobotics.aau5872), *Science Robotics*, 2019.
- Ashish Kumar et al., [“RMA: Rapid Motor Adaptation for Legged Robots”](https://arxiv.org/abs/2107.04034), *Robotics: Science and Systems*, 2021.
- Sandeep Singh Sandha et al., [“Sim2Real Transfer for Deep Reinforcement Learning with Stochastic State Transition Delays”](https://proceedings.mlr.press/v155/sandha21a.html), *CoRL 2020*, 2021 proceedings.
