# Model-Based Reinforcement Learning and Latent Imagination

## At a Glance

> Model-Based Reinforcement Learning (MBRL) learns or uses a model of environment dynamics to improve decisions; latent imagination rolls that model forward in a compact learned state space so policies, values, or planners can learn from predicted futures without executing every candidate action in reality.

### Why this concept matters

Dream-to-Look should evaluate future consequences of sensing actions: whether a wide scan will reveal candidates, whether zoom will resolve identity, whether a target will exit the field of regard, and whether a flight request will create a superior viewpoint. Latent imagination offers a way to evaluate these possibilities before committing scarce sensing time or platform cooperation.

### This note covers

- what makes RL model-based;
- model learning, planning, Dyna, MPC, and imagined policy learning;
- latent versus observation-space models;
- Dreamer-style actor–critic learning in imagination;
- model bias, rollout horizon, uncertainty, and grounding;
- a Dream-to-Look research program.

### Reading map

| Perspective | Main question |
|---|---|
| World model | What future-relevant quantities are predicted? |
| Use of model | Planning, synthetic data, policy learning, or representation? |
| Latent imagination | Why predict compact states rather than pixels? |
| Dreamer | How are actor and value learned from imagined trajectories? |
| Reliability | When does imagined improvement fail in reality? |

---

## 1. What makes reinforcement learning model-based?

An MDP has transition (T(s'\mid s,a)) and reward (R(s,a)). Model-free RL learns a policy or value without explicitly using a predictive model for decision improvement. MBRL uses a known or learned approximation

\[
\hat T_\theta(s'\mid s,a),
\qquad
\hat R_\theta(s,a)
\]

to plan, generate learning targets, synthesize experience, or train a policy.

The boundary is functional rather than architectural. A network is a **world model** for control when its predictions are used to reason about consequences of actions. A representation trained with an auxiliary next-state loss is not necessarily model-based RL if decision learning never uses its rollouts or predictions.

### 1.1 Main families

| Family | How the model is used | Representative idea |
|---|---|---|
| Planning with known model | Search or dynamic programming at decision time | classical planning/control |
| Learned-model MPC | Optimize action sequences through short predicted rollouts | PETS, TD-MPC |
| Dyna-style learning | Mix real and model-generated transitions in updates | Dyna |
| Imagination-trained policy/value | Train behavior from latent rollouts | Dreamer |
| Search with learned abstract model | Predict reward/value/policy-relevant quantities for tree search | MuZero |

These are not mutually exclusive.

---

## 2. Learning a world model

### 2.1 Observation-space dynamics

A direct model predicts future observations:

\[
p_\theta(o_{t+1}\mid o_{\leq t},a_{\leq t}).
\]

Pixel prediction is interpretable but spends capacity on texture and other details irrelevant to decisions. Small pixel errors can dominate the loss while task-critical object identity or visibility is poorly modeled.

### 2.2 Latent dynamics

Encode observations into latent state and predict there:

\[
z_t=E_\theta(o_{\leq t},a_{<t}),
\qquad
p_\theta(z_{t+1}\mid z_t,a_t).
\]

Additional heads predict quantities needed by control:

\[
\hat r_t=r_\theta(z_t,a_t),
\qquad
\hat c_t=c_\theta(z_t),
\qquad
\hat o_t\sim p_\theta(o_t\mid z_t).
\]

Latent models are compact and can ignore irrelevant detail, but their states are harder to inspect and may omit information needed by a changed downstream task.

### 2.3 Observation-equivalent versus value-equivalent models

An observation-predictive model attempts to reproduce future sensory data. A value-equivalent model need only preserve predictions relevant to rewards, values, and policies. MuZero exemplifies the latter: its learned dynamics need not reconstruct the environment observation.

Dream-to-Look likely needs a hybrid. It need not synthesize photorealistic EO/IR frames, but should predict evidence-relevant outcomes such as visibility, detection quality, object posterior change, and task value.

---

## 3. Planning with a learned model

### 3.1 Open-loop action-sequence optimization

Given current state estimate (z_t), model predictive control chooses

\[
a_{t:t+H-1}^*
=\arg\max_{a_{t:t+H-1}}
\mathbb E_{\hat T}
\left[
\sum_{k=0}^{H-1}\gamma^k\hat r_{t+k}
+\gamma^H\hat V(z_{t+H})
\right].
\]

Only the first action is executed; the system observes reality, updates state, and replans. Candidate sequences can be optimized through random shooting, cross-entropy method, gradients, tree search, or a learned proposal policy.

### 3.2 Strength of receding-horizon planning

Replanning limits the time model error can accumulate and incorporates new evidence. It also naturally adapts to changing constraints. Its cost is repeated online optimization and possible short-horizon behavior.

### 3.3 Dyna

Dyna alternates between:

- learning from real transitions;
- updating the model;
- generating simulated transitions;
- improving value or policy from both real and simulated experience.

The model increases data reuse. If synthetic transitions are biased, it also increases reuse of error.

---

## 4. Latent imagination

### 4.1 Filtering reality, imagining futures

An RSSM uses observations to infer the current posterior state. Future imagination then uses the latent prior because future observations are unavailable:

\[
z_t\sim q_\theta(z_t\mid h_t,o_t),
\]

\[
h_{t+k+1}=f_\theta(h_{t+k},z_{t+k},a_{t+k}),
\]

\[
z_{t+k+1}\sim p_\theta(z_{t+k+1}\mid h_{t+k+1}).
\]

Reward and continuation heads turn imagined states into learning signals.

### 4.2 Dreamer-style behavior learning

Dreamer learns:

- a world model from replayed real experience;
- an actor (pi_\phi(a\mid s));
- a value model (V_\psi(s)).

Starting from posterior states inferred from real sequences, it rolls the actor and world model forward in latent space. A truncated λ-return can be written

\[
G_t^\lambda
=\hat r_t
+\gamma\hat c_t
\left[
(1-\lambda)V_\psi(s_{t+1})
+\lambda G_{t+1}^\lambda
\right].
\]

The value model regresses toward imagined returns:

\[
\mathcal L_V
=\mathbb E\left[(V_\psi(s_t)-\operatorname{sg}(G_t^\lambda))^2\right],
\]

where (operatorname{sg}) stops gradients through the target.

The actor maximizes imagined return, often with an entropy term:

\[
J_\pi
=\mathbb E
\left[
G_t^\lambda
+\eta\mathcal H(\pi_\phi(\cdot\mid s_t))
\right].
\]

Depending on the variant, gradients reach the actor through differentiable dynamics, likelihood-ratio estimators, or a mixture. The enduring principle is that behavior learning consumes predicted latent trajectories rather than only real environment transitions.

### 4.3 Imagination is conditional prediction, not fantasy

Useful imagination begins from a state grounded in real history, conditions on candidate actions, and predicts task-relevant consequences. It should not be interpreted as unconstrained generative creativity.

---

## 5. Model bias and rollout horizon

Let true return be (J(\pi)) and model-estimated return be (hat J(\pi)). Policy optimization selects actions with high (hat J), so even small systematic errors can be amplified:

\[
\arg\max_\pi \hat J(\pi)
\]

may deliberately seek regions where (hat J-J) is most positive. This is the optimizer's curse in learned-model control.

### 5.1 Horizon trade-off

- Short rollouts reduce compounding error but provide limited long-term credit.
- Long rollouts expose delayed effects but become less grounded.
- Value bootstrapping shortens the required model horizon but transfers error to the value model.

A practical design combines short-to-medium imagination, terminal value, uncertainty awareness, and frequent real-state anchoring.

### 5.2 Distribution shift

The policy changes the state–action distribution used to train the model. New behavior can visit regions unsupported by replay. Iterative data collection, uncertainty-aware exploration, and balanced replay are therefore part of MBRL, not peripheral engineering.

### 5.3 Uncertainty-aware imagination

Possible mechanisms include:

- probabilistic latent trajectories;
- model ensembles;
- penalties for disagreement or unsupported states;
- pessimistic reward/value estimates;
- adaptive rollout truncation;
- branching only over decision-relevant uncertainties.

Uncertainty should not always be penalized. During authorized exploration it can identify valuable evidence. During mission execution it may trigger shorter planning horizons, conservative actions, or a request for observation.

---

## 6. Planning versus amortized policy

| Property | Online planning/MPC | Imagination-trained policy |
|---|---|---|
| Runtime compute | Higher | Low, fixed inference |
| Adaptation to constraints | Natural | Requires conditioning/training |
| Dependence on model at runtime | Direct | Indirect through learned actor/value |
| Model exploitation | During each optimization | Can be baked into policy |
| Deployment simplicity | Lower | Higher |
| Replanning from new evidence | Explicit | Through recurrent policy update |

A hybrid is attractive: an imagination-trained proposal policy supplies good candidates, while short-horizon MPC refines or verifies them. For edge deployment, the final choice depends on compute, latency, and the number of candidate payload actions.

---

## 7. Failure modes and diagnostics

| Failure | Symptom | Diagnostic or mitigation |
|---|---|---|
| One-step-good, rollout-bad | Validation loss low but plans fail | Multi-step open-loop and closed-loop error by horizon |
| Reward-model exploitation | Imagined high reward absent in reality | Privileged-state audit and adversarial planning tests |
| Representation omission | Latent predicts pixels/reward but loses needed evidence variable | Probe future visibility, identity, and belief-update targets |
| Posterior–prior gap | Good reconstruction, poor imagination | Compare posterior and open-loop prior rollouts |
| Policy-model collusion | Actor exploits shared learned error | Independent evaluator and ensemble disagreement |
| Excessive horizon | Unrealistic long imagined sequences | Adaptive horizon and real-state anchoring |
| Too-short horizon | Misses staged sensing value | Terminal value and sequence-specific benchmark |
| Mode collapse | One predicted future suppresses alternatives | Stochastic/multimodal dynamics and coverage tests |
| Offline support violation | Planner leaves data support | Conservative objectives and support constraints |
| Compute mismatch | Research planner cannot deploy on edge hardware | Runtime-budgeted comparisons and distilled policy |

World-model quality should be evaluated both **open loop** and **decision closed loop**. Low prediction error does not guarantee good control, and high pixel error does not necessarily imply poor task decisions.

---

## 8. Implications for Dream-to-Look

### 8.1 What should be imagined?

The model need not predict full-resolution imagery. A task-oriented imagination state can predict:

- object existence, identity, pose, and motion distributions;
- visibility and field-of-view occupancy;
- detection/classification quality under viewpoint and sensor settings;
- occlusion and reacquisition probability;
- belief change or expected evidence;
- time, motion, coverage, and flight-request costs.

Candidate payload actions then produce imagined **evidence trajectories**, not merely camera trajectories.

### 8.2 Dream-to-Look loop

1. Infer an object-centric posterior from real observation history.
2. Generate candidate pan/tilt/zoom/modality/dwell actions and optional flight requests.
3. Roll them through the latent prior.
4. Predict future evidence, uncertainty, task value, and cost.
5. Select an action by actor, MPC, or a hybrid.
6. Execute one action, observe reality, correct the posterior, and repeat.

### 8.3 Core research hypotheses

1. Latent evidence imagination will outperform reactive and recurrent model-free policies on delayed-evidence tasks.
2. Object-centric imagination will generalize better to changed object counts, arrangements, and occlusion patterns.
3. Uncertainty-aware rollout truncation will reduce catastrophic model exploitation.
4. Predicting task-relevant evidence outcomes will outperform pure pixel reconstruction under equal compute.
5. A hybrid actor-plus-short-MPC design will retain most planning value within edge latency limits.

### 8.4 Baselines and evaluation

Compare heuristic next-look, reactive policy, recurrent model-free RL, latent MPC, Dreamer-style actor, and hybrid actor–MPC. Include oracle dynamics and privileged-belief upper bounds.

Measure real task return, sensing cost, sample efficiency, imagination-to-reality return gap, rollout error by horizon, posterior–prior divergence, calibration, planning latency, robustness to detector/backend swaps, and generalization to unseen evidence sequences.

The most revealing benchmark should contain **staged information value**: the best first look is not immediately rewarding but enables a later decisive observation. This distinguishes genuine imagination from reactive confidence maximization.

---

## 9. Durable takeaways

1. MBRL is defined by using predictive models for decision improvement, not merely by training a dynamics auxiliary loss.
2. Latent imagination trades visual fidelity for compact task-relevant prediction.
3. Planning, Dyna, and imagination-trained actors are different ways to consume a world model.
4. Dreamer trains actor and value from trajectories rolled forward in learned latent dynamics.
5. Policy optimization amplifies favorable model errors; uncertainty and real-data grounding are essential.
6. Rollout horizon trades delayed credit against compounding error.
7. Dream-to-Look should imagine future evidence and decision quality, not necessarily future pixels.

---

## 10. Primary references

- Sutton, R. S. (1991). [Dyna, an Integrated Architecture for Learning, Planning, and Reacting](https://doi.org/10.1145/122344.122377).
- Ha, D., & Schmidhuber, J. (2018). [World Models](https://arxiv.org/abs/1803.10122).
- Hafner, D. et al. (2019). [Learning Latent Dynamics for Planning from Pixels (PlaNet)](https://arxiv.org/abs/1811.04551).
- Hafner, D. et al. (2020). [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603).
- Hafner, D. et al. (2023). [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104).
- Schrittwieser, J. et al. (2020). [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- Chua, K. et al. (2018). [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)](https://arxiv.org/abs/1805.12114).
- Hansen, N. et al. (2024). [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828).

