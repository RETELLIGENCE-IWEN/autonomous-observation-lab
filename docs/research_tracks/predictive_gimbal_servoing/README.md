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
- [Paper-Style Research Journey](predictive_gimbal_servoing_journey_paper.md)
- [Paper-Style Research Journey — PDF](predictive_gimbal_servoing_journey_paper.pdf)
- [Benchmark Specification](benchmark_specification.md)
- [Privileged Target-State Dataset](privileged_dataset.md)
- [Causal GRU Smoke Experiment](gru_smoke_experiment.md)
- [Randomized Observation-Profile Experiment](observation_profile_experiment.md)
- [GRU Closed-Loop Control Experiment](gru_closed_loop_experiment.md)
- [O2 GRU Multi-Seed Replication](gru_multi_seed_replication.md)
- [Baseline vs Learned Performance Verification](performance_verification.md)
- [Adaptive Predictive Position V2 Experiment](adaptive_position_v2_experiment.md)
- [Visibility-Risk Position V2.1 Experiment](visibility_risk_position_v21_experiment.md)
- [Position Performance Contract and Failure Atlas](position_performance_contract_and_failure_atlas.md)
- [Constrained Predictive Position V3/V3.1](constrained_predictive_position_v3_experiment.md)
- [Control-Aware Predictor V4/V5 Experiment](control_aware_predictor_v4_experiment.md)
- [Critical Curriculum and Adapter Objective V6/V6.1](adaptive_curriculum_v6_experiment.md)
- [Hard Midpoint Dynamics and Adapter Objective V7](midpoint_adapter_v7_experiment.md)
- [Counterfactual Servo-Plant Objective V8--V8.7](counterfactual_plant_v8_experiment.md)
- [Belief-Guided Recovery Experiment](belief_recovery_experiment.md)
- [O2 GRU Uncertainty Calibration Experiment](uncertainty_calibration_experiment.md)
- [Contextual Calibration and Recovery Development/Test Protocol](contextual_calibration_and_recovery_protocol.md)
- [Expanded Recovery Robustness Experiment](recovery_robustness_experiment.md)
- [Edge-Conditioned Recovery Experiment](edge_conditioned_recovery_experiment.md)
- [Prior Work and Novelty Boundary](prior_work_and_novelty.md)

## Foundation reading path

The project-specific reading path is maintained in the [Foundation Notes index](../../foundations/README.md#recommended-reading-path-for-predictive-gimbal-servoing). Its new core notes are:

- [Visual Servoing and Cascaded Gimbal Control](../../foundations/visual_servoing_and_cascaded_gimbal_control.md)
- [Continuous-Control Actor-Critic Learning](../../foundations/continuous_control_actor_critic_learning.md)
- [Privileged Learning and Policy Distillation](../../foundations/privileged_learning_and_policy_distillation.md)
- [Sim-to-Real for Learned Control](../../foundations/sim_to_real_for_learned_control.md)

## Status

Concept locked on 2026-08-26. The configurable simulator, analytical estimator, rate/position adapters, diagnostics, stress matrix, privileged oracle, domain-randomized dataset, causal GRU predictor, configurable belief-recovery state machine, validation-fit O2 variance calibration, contextual-calibration ablation, disjoint recovery development/test protocol, three-initialization O2 replication, expanded seven-scenario recovery robustness protocol, deployable edge-conditioned recovery ablation, adaptive multi-horizon position V2, visibility-risk position V2.1, absolute performance/failure atlas, constrained predictive position V3/V3.1, and control-aware predictor V4/V5 studies are implemented. The V4 consistency objective passes three-seed replication and a fresh open-loop test, then passes the fresh rate-controller gate with 4.25% lower mean error and 0.69 percentage points less avoidable loss than its expanded-data baseline. Position averages improve slightly and become 5.6% smoother, but only one of three initializations improves mean tracking, so the primary position promotion gate fails. V8--V8.7 now adds an exact differentiable servo rollout and a one-sided counterfactual-regret objective. Its seed-17 candidate passes every development guard with better exact tracking, visibility, smoothness, and saturation, but remains unpromoted until seed-matched replication resolves its narrow average-rate margin. Sim-to-real qualification remains deferred.
