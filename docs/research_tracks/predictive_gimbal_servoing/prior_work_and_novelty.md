# Prior Work and Novelty Boundary: Learned Gimbal Servoing

## At a Glance

Target centering with a camera in the feedback loop is established visual servoing. Predictive pan–tilt control, reinforcement-learned PTZ control, recurrent sim-to-real visual servoing, and RL-based UAV gimbal tracking have all been demonstrated. Therefore neither “AI controls a gimbal” nor “RL centers a bounding box” is a defensible novelty claim.

The candidate research gap is a carefully controlled study of **latent online dynamics inference and continuous predictive control under maneuver-induced image motion**, with rigorous temporal, spectral, actuator, and latency generalization tests.

This is an initial landscape scan, not a systematic review or patent search.

---

## 1. Closest prior directions

### Classical image-based visual servoing

Visual servoing has long formalized the use of image features in a robotic feedback loop. A bbox-center error is therefore a standard image-based control signal, not a new task formulation.

- F. Chaumette and S. Hutchinson, [Visual Servo Control, Part I: Basic Approaches](https://doi.org/10.1109/MRA.2006.250573), IEEE Robotics & Automation Magazine, 2006.

### Predictive pan–tilt control

Predictive control has been applied to pan–tilt tracking using explicit second-order joint models and predicted target trajectories. The work reports a real robot validation and demonstrates why a predictive conventional baseline is mandatory.

- P. D. Domański et al., [Predictive tracking of an object by a pan–tilt camera of a robot](https://doi.org/10.1007/s11071-023-08295-z), Nonlinear Dynamics, 2023.

### Active UAV gimbal tracking

Active gimbal orientation using joint and vision information has been evaluated in UAV-oriented Gazebo environments. Centering improves pose estimation by reducing peripheral distortion and motion-related degradation, but the work is not a learned continuous-control study.

- J. G. Hansen and R. P. de Figueiredo, [Active Object Detection and Tracking Using Gimbal Mechanisms for Autonomous Drone Applications](https://doi.org/10.3390/drones8020055), Drones, 2024.

### End-to-end learned PTZ control

Eagle learns an image-to-PTZ policy with deep RL, introduces a photorealistic simulator, uses domain randomization, and reports embedded inference and sim-to-real transfer. Its principal framing is removal of the detector/control pipeline and lightweight end-to-end operation.

- S. S. Sandha et al., [Eagle: End-to-end Deep Reinforcement Learning based Autonomous Control of PTZ Cameras](https://arxiv.org/abs/2304.04356), IoTDI/arXiv, 2023.

### RL gimbal positioning under disturbance

A Unity-based study applies DDPG to gimbal positioning for UAV object detection and includes simulated environmental interference. Its reported action design uses directional actions, so continuous one-axis predictive dynamics and rigorous held-out system identification remain distinct questions.

- [Manipulating Camera Gimbal Positioning by Deep Deterministic Policy Gradient Reinforcement Learning for Drone Object Detection](https://doi.org/10.3390/drones8050174), Drones, 2024.

### RL PTZ drone tracking benchmark

A simulated PTZ tracking study tests basic, dynamic, and obstacle cases, reports limitations and reward pathologies, and explicitly identifies model-based world models, algorithm comparison, repeatability, and sim-to-real as open directions.

- M. Ward et al., [Towards Fully Autonomous Drone Tracking by a Reinforcement Learning Agent Controlling a Pan–Tilt–Zoom Camera](https://doi.org/10.3390/drones8060235), Drones, 2024.

### Recurrent and model-based learned visual servoing

Recurrent learned control has been used to infer action effects under unknown viewpoints and transfer from simulation. Earlier learned visual-servo work also combined deep features, learned predictive dynamics, and fitted Q-iteration, demonstrating that learned dynamics are not novel by themselves.

- F. Sadeghi et al., [Sim2Real Viewpoint Invariant Visual Servoing by Recurrent Control](https://openaccess.thecvf.com/content_cvpr_2018/html/Sadeghi_Sim2Real_Viewpoint_Invariant_CVPR_2018_paper.html), CVPR, 2018.
- A. X. Lee, S. Levine, and P. Abbeel, [Learning Visual Servoing with Deep Features and Fitted Q-Iteration](https://arxiv.org/abs/1703.11000), 2017.

### Continuous-control learning foundations

SAC and TD3 are standard continuous-action baselines, not contributions of this project. A world-model controller must show why explicit prediction improves over them.

- T. Haarnoja et al., [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), 2018.
- S. Fujimoto, H. van Hoof, and D. Meger, [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html), ICML, 2018.

---

## 2. Claims that are not safe

Do not claim:

- the first AI-controlled or RL-controlled gimbal;
- the first learned PTZ target tracker;
- the first bbox-to-control policy;
- the first recurrent visual servo;
- the first predictive controller for pan–tilt tracking;
- superiority over conventional control without expert tuning and equal information;
- robustness based only on random training disturbance;
- sim-to-real from domain randomization without hardware evidence.

---

## 3. Candidate novelty dimensions

### 3.1 Hidden-disturbance identification from interventions

The policy observes the combined image effect of target motion, body motion, gimbal response, and delay. Conditioning recurrent inference on previous actions lets it identify how its interventions affect the image and separate some endogenous from exogenous motion.

Candidate claim:

> A recurrent action-conditioned latent state performs online visual system identification and enables anticipatory centering without direct vehicle-attitude input.

This claim requires an action-agnostic ablation and tests where histories are genuinely ambiguous.

### 3.2 Spectral and temporal generalization

Many demonstrations randomize disturbance during training but do not make generalization across excluded frequency bands, nonstationary oscillations, sensor rates, and latency the central scientific object.

Candidate claim:

> A learned predictive servo generalizes across held-out maneuver spectra and delay/actuator regimes better than nominal and learned reactive controllers.

A robust or gain-scheduled conventional baseline can falsify this claim.

### 3.3 Prediction-to-control causality

It is not enough for a world model to predict well or for a policy to score well. The study should show that removing action conditioning, shortening prediction below the delay horizon, or corrupting predicted dynamics removes the anticipatory behavior.

Candidate claim:

> Multi-step image-plane prediction is the mechanism responsible for lower transient and tail error.

### 3.4 Observability-aware privileged distillation

A privileged teacher observes quadcopter motion, true LOS, actuator state, and delay state while the deployed student receives only signals available through the real payload interface. The goal is to transfer useful predictive belief without implying that unobservable shocks can be anticipated. If vehicle telemetry is available at deployment, it is included rather than artificially withheld.

Candidate claim:

> Privileged disturbance-state distillation improves predictable maneuver compensation while calibrated uncertainty preserves the boundary of bbox-only observability.

This is part of the locked method and connects the mini-project to the broader Epistemic Distillation track. Its incremental value over imitation-only warm-starting must be isolated experimentally.

---

## 4. Recommended contribution statement

The strongest initial wording is:

> We study continuous one-axis visual servoing as learned predictive adaptation rather than direct bbox-error regulation. A compact recurrent controller is warm-started by conventional control and distilled from privileged platform, actuator, and delay state, but deploys only with signals available through the real payload interface. A controlled benchmark tests whether its predictive state improves tail tracking error and loss of view across unseen maneuver spectra and plant parameters relative to tuned PID, explicit-model predictive control, and recurrent model-free RL.

The wording should be revised after systematic searches across IEEE Xplore, Scopus/Web of Science, Google Scholar citation graphs, patents, theses, and relevant autonomous-sensor-management programs.

---

## 5. Practical first research sequence

1. Lock the physical input/output contract and axis.
2. Build the deterministic feature-level plant and maneuver generator.
3. Tune PID and identified DMC/MPC before training AI.
4. Establish a constructed temporal case in which memory is necessary.
5. Train feed-forward and recurrent SAC/TD3 baselines.
6. Train an action-conditioned recurrent dynamics model.
7. Validate multi-step prediction before policy optimization.
8. Train actor or latent MPC and perform causal ablations.
9. Run untouched maneuver, delay, actuator, and FOV shifts.
10. Only then connect real detector output and hardware.
