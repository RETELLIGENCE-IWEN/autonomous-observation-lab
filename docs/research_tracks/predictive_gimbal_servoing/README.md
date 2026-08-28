# Predictive 1D Gimbal Servoing

## Core idea

A one-axis gimbal should continuously keep a detected object centered while a quadcopter introduces oscillatory and maneuver-induced image motion. The controller receives a bounding box or its center and size and emits a continuous gimbal command.

The locked concept is a **learned predictive outer-loop visual servo with a conventional actuator and safety envelope**. A compact recurrent model estimates short-horizon body-relative target bearing, angular rate, and uncertainty. A hardware-configured adapter converts that common target state into either desired gimbal rate or absolute position. The embedded inner loop realizes the selected command and enforces electrical and mechanical limits.

The research question is not whether reinforcement learning can imitate a proportional controller. It is whether privileged predictive training can produce a deployable recurrent controller that internalizes disturbance and actuation dynamics, anticipates their future image-plane effect, and generalizes across maneuvers, delays, and payload parameters that were not seen during training.

## Primary research question

> Can a compact recurrent visual servo, warm-started by conventional control and distilled from privileged platform state, reduce tail tracking error and loss of view under unseen quadcopter motion, latency, and actuator dynamics relative to tuned PID, predictive control, and model-free recurrent RL?

## Working contribution

The proposed contribution is a compact learned predictive visual servo, tentatively called **Dream-to-Center**, together with a disturbance-oriented benchmark. The runtime model uses all telemetry genuinely available to the deployed payload and predicts short-horizon target state, uncertainty, image error, and visibility risk. Rate and position adapters are evaluated separately so conclusions about prediction are not entangled with a specific future servo interface. Bbox-only control is an ablation unless it is the real interface constraint.

This is a candidate contribution, not a novelty claim. The claim must be narrowed after a systematic literature and patent review.

## Documents

- [Locked Concept](concept_lock.md)
- [Research Brief](research_brief.md)
- [Benchmark Specification](benchmark_specification.md)
- [Privileged Target-State Dataset](privileged_dataset.md)
- [Causal GRU Smoke Experiment](gru_smoke_experiment.md)
- [Prior Work and Novelty Boundary](prior_work_and_novelty.md)

## Foundation reading path

The project-specific reading path is maintained in the [Foundation Notes index](../../foundations/README.md#recommended-reading-path-for-predictive-gimbal-servoing). Its new core notes are:

- [Visual Servoing and Cascaded Gimbal Control](../../foundations/visual_servoing_and_cascaded_gimbal_control.md)
- [Continuous-Control Actor-Critic Learning](../../foundations/continuous_control_actor_critic_learning.md)
- [Privileged Learning and Policy Distillation](../../foundations/privileged_learning_and_policy_distillation.md)
- [Sim-to-Real for Learned Control](../../foundations/sim_to_real_for_learned_control.md)

## Status

Concept locked on 2026-08-26. A configurable deterministic one-axis environment, analytical target-state estimator, rate/position adapters, Rerun diagnostics, six-case stress matrix, privileged target-state oracle, deterministic dataset format, and first causal GRU predictor are implemented. The GRU smoke result is promising at longer horizons but uses fixed development motions and is not a generalization or control claim. Hardware parameters and the final deployment telemetry contract remain open.
