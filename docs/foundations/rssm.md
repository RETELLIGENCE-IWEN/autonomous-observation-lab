# Recurrent State-Space Model (RSSM)

## At a Glance

> An RSSM is a latent sequential model that combines a deterministic recurrent memory with stochastic latent states to infer the hidden state of a partially observed system and predict how it may evolve under actions.

### Why this concept matters

An agent rarely observes the true Markov state of the world. It receives incomplete and noisy measurements: images, bounding boxes, delayed detections, and intermittent tracks. An RSSM converts the history of observations and actions into a compact latent belief-like state, updates that state when evidence arrives, and rolls it forward without new observations to imagine possible futures.

### This note covers

- the problem RSSMs are designed to solve;
- deterministic and stochastic latent states;
- prior, posterior, transition, observation, reward, and continuation models;
- the variational training objective;
- filtering with observations and imagination without observations;
- RSSM's role in PlaNet and Dreamer;
- strengths, limitations, and common failure modes;
- implications for an object-centric Dream-to-Look model.

### This note does not cover

- a complete Dreamer implementation;
- library-specific code or hyperparameter recipes;
- every recurrent state-space model variant;
- detailed actor–critic derivations.

### Reading map

| Perspective | Main question |
|---|---|
| Core principle | Why combine recurrent memory and stochastic state? |
| Mathematical definition | What distributions and transitions are learned? |
| Inference and imagination | How does the model update from reality and predict without it? |
| Learning | What forces the latent state to become predictive? |
| Value | Why is this useful for planning and reinforcement learning? |
| Limitations | When can the learned latent world become misleading? |
| Dream-to-Look | How could RSSM predict future visibility and evidence? |

---

## 1. The problem RSSM addresses

### 1.1 The environment state is hidden

In a fully observed Markov decision process, the current state \(x_t\) is assumed to contain everything required to predict the next state:

\[
p(x_{t+1}\mid x_{1:t},a_{1:t})=p(x_{t+1}\mid x_t,a_t).
\]

In visual and sensing problems, the agent does not receive \(x_t\). It receives an observation \(o_t\) generated from an unknown state:

\[
o_t\sim p(o_t\mid x_t).
\]

A single image or bounding box is generally not Markov. The same bounding box can arise from different ranges, target motions, zoom settings, and platform motions. A missed detection can mean occlusion, detector failure, target departure, or a false previous track.

The agent therefore needs an internal state summarizing history:

\[
b_t=f(o_{1:t},a_{1:t-1}).
\]

Ideally, \(b_t\) behaves like a belief state: a sufficient summary of what the agent currently knows about the hidden world.

### 1.2 Why not only frame stacking?

Frame stacking provides a short fixed window. It can expose local motion, but it does not naturally:

- remember events beyond the window;
- bridge long occlusions;
- represent uncertainty over hidden causes;
- predict forward for an arbitrary horizon;
- update a prior prediction when delayed evidence arrives.

### 1.3 Why not only a deterministic RNN?

A deterministic RNN can compress history into a hidden vector, but it maps the same history to one hidden state. When several futures or explanations are plausible, uncertainty is only implicit in a point representation.

A stochastic latent variable makes uncertainty and multimodality representable in principle. It also creates a probabilistic prior that can be matched to a posterior inferred from new evidence.

### 1.4 Why not only a stochastic state-space model?

If every piece of history must pass through a sampled stochastic state, long-term information can be difficult to preserve and optimization may become noisy. A deterministic recurrent path can carry stable information across time, while stochastic variables represent uncertain or varying aspects of the current state.

This complementary division is the central RSSM idea.

---

## 2. Core intuition

An RSSM maintains two forms of latent state.

### Deterministic recurrent state \(h_t\)

The deterministic state is a learned memory of prior latent states and actions:

\[
h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1}).
\]

It is commonly implemented by a GRU or a related recurrent transition. It is useful for stable temporal context: action history, persistent scene context, and information that should flow across many steps.

### Stochastic state \(z_t\)

The stochastic state represents uncertain information about the current hidden state. It is sampled either from:

- a **prior**, predicted from history before seeing the current observation; or
- a **posterior**, inferred after incorporating the current observation.

The complete model state is typically the pair:

\[
s_t=(h_t,z_t).
\]

The deterministic state should not be interpreted as “known truth,” nor the stochastic state as a complete Bayesian belief. Both are learned representations. Their intended roles emerge from architecture and training losses.

### The central rhythm

The simplest way to remember an RSSM is:

> **Predict with the prior, correct with the posterior, and remember through the recurrent state.**

---

## 3. Mathematical formulation

Notation differs across papers and implementations. This note uses:

- \(o_t\): observation;
- \(a_t\): action;
- \(h_t\): deterministic recurrent state;
- \(z_t\): stochastic latent state;
- \(r_t\): reward or task signal;
- \(c_t\): continuation/nonterminal indicator.

### 3.1 Observation encoder

High-dimensional observations are encoded into features:

\[
e_t=\operatorname{Enc}_\theta(o_t).
\]

For images, this may be a convolutional encoder. For object-centric sensing, it may be a set or graph encoder over detections, tracks, and payload state.

### 3.2 Deterministic transition

The recurrent dynamics consume the previous deterministic state, stochastic state, and action:

\[
h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1}).
\]

This answers: given what the model previously believed and what the agent did, what temporal context should be carried into the next step?

### 3.3 Prior dynamics

Before seeing \(o_t\), the model predicts a distribution over the next stochastic state:

\[
p_\theta(z_t\mid h_t).
\]

This is the dynamics prior. It is the distribution used during imagination, because imagined future observations are unavailable.

For continuous latents, the prior may be a diagonal Gaussian:

\[
p_\theta(z_t\mid h_t)=\mathcal{N}(\mu^p_t,(\sigma^p_t)^2).
\]

DreamerV2 and DreamerV3 use discrete categorical latent representations rather than the continuous Gaussian representation used in earlier versions.

### 3.4 Posterior or representation model

After seeing the observation feature \(e_t\), the model infers:

\[
q_\theta(z_t\mid h_t,e_t).
\]

For a continuous latent:

\[
q_\theta(z_t\mid h_t,e_t)=\mathcal{N}(\mu^q_t,(\sigma^q_t)^2).
\]

This posterior is not a posterior over the true physical state in a strict analytical model. It is an amortized variational posterior over the learned latent state.

### 3.5 Observation model

The latent state predicts or reconstructs the observation:

\[
p_\theta(o_t\mid h_t,z_t).
\]

This forces the latent state to retain information about what was observed. Depending on the research goal, the prediction target can be:

- raw pixels;
- image embeddings;
- object states;
- bounding boxes and visibility;
- semantic or task-relevant features.

### 3.6 Reward and continuation models

In model-based RL, the world model commonly predicts reward and episode continuation:

\[
p_\theta(r_t\mid h_t,z_t),
\]

\[
p_\theta(c_t\mid h_t,z_t).
\]

These heads allow imagined trajectories to produce predicted returns without decoding full observations at every step.

### 3.7 Generative factorization

A simplified action-conditioned RSSM generative model can be written as:

\[
p(o_{1:T},r_{1:T},c_{1:T},z_{1:T}\mid a_{1:T-1})
=\prod_{t=1}^{T}
p(z_t\mid h_t)
p(o_t\mid h_t,z_t)
p(r_t\mid h_t,z_t)
p(c_t\mid h_t,z_t),
\]

with:

\[
h_t=f(h_{t-1},z_{t-1},a_{t-1}).
\]

The inference model replaces the unknown latent state with samples from \(q(z_t\mid h_t,e_t)\) during training and filtering.

---

## 4. Filtering and imagination

RSSM has two operational modes that must be distinguished clearly.

### 4.1 Observation-conditioned filtering

When a real observation is available:

1. update \(h_t\) using the previous latent state and action;
2. encode \(o_t\) into \(e_t\);
3. infer posterior \(q(z_t\mid h_t,e_t)\);
4. sample or select \(z_t\);
5. use \((h_t,z_t)\) as the current filtered state.

The posterior corrects the prediction using evidence.

### 4.2 Open-loop imagination

For a candidate future action sequence, observations do not yet exist:

1. choose action \(a_t\);
2. update the recurrent state;
3. sample \(z_{t+1}\) from the prior;
4. predict reward, continuation, visibility, or other quantities;
5. repeat for the imagination horizon.

The model therefore rolls forward using:

\[
z_{t+1}\sim p(z_{t+1}\mid h_{t+1}).
\]

This separation gives RSSM its practical value: the same learned dynamics supports both online state estimation and counterfactual future simulation.

### 4.3 Missing observations

If a detector drops out or a target becomes occluded, the model can continue with its prior. When evidence returns, the posterior can correct accumulated error.

This does not guarantee accurate tracking through arbitrary occlusion. The prior may drift, become overconfident, or fail to represent a new mode. The quality of missing-observation behavior must be measured explicitly.

---

## 5. Learning objective

The RSSM is typically trained with a variational objective containing prediction terms and a KL regularizer.

### 5.1 Prediction losses

A generic negative log-likelihood objective is:

\[
\mathcal{L}_{pred}
=-\sum_t\left[
\log p(o_t\mid h_t,z_t)
+\log p(r_t\mid h_t,z_t)
+\log p(c_t\mid h_t,z_t)
\right].
\]

These terms make the latent state informative about observations and task-relevant outcomes.

### 5.2 Dynamics KL

The posterior has access to current evidence; the prior does not. To make the prior useful for imagination, it is trained toward the posterior:

\[
\mathcal{L}_{KL}
=\sum_t D_{KL}\left(
q(z_t\mid h_t,e_t)
\;\|\;
p(z_t\mid h_t)
\right).
\]

The combined loss is conceptually:

\[
\mathcal{L}_{RSSM}=\mathcal{L}_{pred}+\beta\mathcal{L}_{KL}.
\]

### 5.3 Interpretation of the KL trade-off

- If KL pressure is too weak, the posterior may encode details that the prior cannot predict. Reconstruction may look good while imagined rollouts fail.
- If KL pressure is too strong, the posterior may ignore the observation and collapse toward the prior, losing information required for state estimation.

Practical Dreamer variants use techniques such as KL balancing and free bits/free nats to stabilize this trade-off. DreamerV3 also adds robust normalization and transformations for cross-domain stability.

### 5.4 Latent overshooting

PlaNet introduced latent overshooting to encourage multi-step consistency rather than only one-step prior-to-posterior matching. The idea is to roll dynamics forward for multiple steps and constrain the predicted latent distributions toward later inferred posteriors.

This targets a central weakness of learned world models: small one-step errors can compound over long imagined trajectories.

### 5.5 Reconstruction is a means, not the objective

Pixel reconstruction is commonly used because it provides dense supervision. However, a model can spend capacity reproducing texture and background details irrelevant to decisions.

For Dream-to-Look, task-oriented prediction may be more suitable:

- target visibility;
- bounding-box or LOS distribution;
- occlusion probability;
- FOV and gimbal-limit margin;
- detection/identification probability;
- track-loss risk;
- future epistemic or mission value.

Whether reconstruction-free or task-oriented training produces better decisions is an empirical question, not a foregone conclusion.

---

## 6. Why deterministic and stochastic states coexist

### 6.1 Deterministic path: continuity and memory

The recurrent state can efficiently preserve:

- action history;
- persistent scene context;
- long-term temporal dependencies;
- information not expected to change abruptly.

### 6.2 Stochastic path: uncertainty and unpredictable variation

The stochastic state can represent:

- partial observability;
- uncertain current state;
- multiple possible futures;
- effects not deterministically inferable from prior history.

### 6.3 Neither role is guaranteed

The architecture encourages this division but does not enforce a semantic decomposition. The deterministic state may hide uncertainty-relevant information, and the stochastic state may be ignored. Diagnostics should inspect:

- KL magnitude and usage;
- posterior–prior gap;
- latent sensitivity to observations;
- calibration of predictive distributions;
- performance when observations are masked.

---

## 7. RSSM in PlaNet and Dreamer

### 7.1 PlaNet

PlaNet introduced RSSM for latent planning from pixels. It learned deterministic and stochastic latent dynamics, then used online planning in latent space to select actions. Latent overshooting encouraged multi-step predictive consistency.

The important separation is:

- RSSM learns the latent environment model;
- a planner searches action sequences using that model.

### 7.2 Dreamer

Dreamer retained the world-model idea but replaced expensive online action search with an actor–critic trained from latent imagined trajectories.

The loop is:

1. collect real experience;
2. train the RSSM and prediction heads;
3. start from posterior states sampled from replay;
4. imagine futures with the prior and actor;
5. predict rewards, continuation, and values;
6. update actor and critic using imagined trajectories;
7. execute the actor in the real environment and repeat.

RSSM is the learned world in which policy learning occurs; it is not itself the actor or critic.

### 7.3 DreamerV2

DreamerV2 adopted discrete categorical stochastic latents and demonstrated that behavior learned inside a separately trained world model could reach human-level aggregate performance on Atari while also supporting continuous control.

### 7.4 DreamerV3

DreamerV3 retained categorical latent world modeling and emphasized robustness across more than 150 tasks using one configuration. Its contributions include normalization, balancing, and transformations that make the broader Dreamer system stable across reward scales and domains.

For conceptual work, distinguish enduring RSSM principles from version-specific Dreamer engineering.

---

## 8. Meaning and value

### 8.1 Compact state estimation

RSSM turns a history of high-dimensional observations and actions into a compact state suitable for prediction and control.

### 8.2 Learning from imagined experience

Once the dynamics are learned, many policy updates or candidate evaluations can occur in latent space without additional environment interaction. This can improve sample efficiency when real or high-fidelity simulation experience is expensive.

### 8.3 Counterfactual evaluation

The model can compare futures under actions that were not actually executed:

- What if the sensor zooms out now?
- What if it keeps tracking?
- What if it looks at the predicted exit?
- What if it switches to IR?

### 8.4 Bridging observation gaps

The prior supplies a prediction during occlusion, latency, or detector dropout; later observations update the posterior.

### 8.5 Representation focused on controllable dynamics

Because action is an input to transition, the model can learn how sensor and platform actions affect future observations. This is essential for active perception: the observation process is action-dependent.

---

## 9. What RSSM does not automatically provide

### 9.1 A physically correct world model

RSSM learns predictive latent variables, not guaranteed physical quantities. A latent coordinate need not correspond to range, identity, visibility, or any human-interpretable state unless training and architecture encourage it.

### 9.2 Calibrated epistemic uncertainty

A stochastic latent does not automatically distinguish:

- aleatoric randomness;
- epistemic model uncertainty;
- ambiguity from missing observations.

The model may sample diverse futures while remaining unaware that its dynamics are wrong. Ensembles, explicit uncertainty heads, OOD detection, or Bayesian approximations may be required.

### 9.3 Reliable long-horizon prediction

Prior rollouts are open loop. Errors compound, rare events may disappear, and generated trajectories can enter unsupported regions.

### 9.4 Protection from model exploitation

An actor or planner can discover trajectories that look valuable under model errors but fail in the real environment. Shorter horizons, uncertainty penalties, real-data grounding, ensembles, and conservative objectives can mitigate this.

### 9.5 Object permanence and identity

A global latent may reconstruct a scene while swapping object identity or losing individual uncertainty. Object-centric structure may be necessary for tracking and inquiry.

---

## 10. Common failure modes and diagnostics

| Failure | Meaning | Useful diagnostics |
|---|---|---|
| Posterior collapse | observation adds little beyond prior | KL usage, masked-observation response |
| Unpredictable posterior | reconstruction works but prior cannot follow | posterior–prior gap, open-loop rollout |
| Compounding error | multi-step predictions drift | error vs horizon, latent overshooting ablation |
| Model exploitation | policy prefers unrealistic imagined states | real vs imagined return, uncertainty correlation |
| Background fixation | model spends capacity on irrelevant pixels | task-head performance, saliency, decoder ablation |
| Identity mixing | objects are reconstructed but associations fail | ID-switch rate, per-object latent probes |
| False certainty | narrow predictions are wrong | NLL, Brier score, coverage, ensemble disagreement |
| Latent underuse | stochastic variables carry little information | KL per dimension/category, posterior diversity |

World-model quality should not be judged by reconstruction alone. Decision performance and calibrated prediction of task-relevant quantities are the final criteria.

---

## 11. Related and alternative approaches

### Deterministic RNN state estimator

Simpler and often easier to train. It may be sufficient when dynamics are nearly deterministic and uncertainty is not central. It is a necessary baseline.

### Kalman and particle filters

They provide explicit probabilistic state estimation under specified dynamics and observation models. They are interpretable and often data-efficient. They become difficult when state, observation, and association models are highly nonlinear or learned from rich perception.

### Transformer sequence models

Transformers can model long histories and variable object sets without recurrent compression. They may require larger context and compute and do not inherently define a compact predictive belief or calibrated uncertainty.

### Modern state-space sequence models

These can provide efficient long-context sequence processing. A sequence-model state is not automatically a generative probabilistic state-space model; the objective and inference structure still matter.

### Video prediction and diffusion world models

These can represent rich multimodal visual futures but are often expensive and may model details unnecessary for action selection. They become attractive when realistic visual consequences or strongly multimodal futures are essential.

### Explicit object trackers and scene graphs

They preserve identity and interpretable geometry but depend on detector and association quality. A promising direction is to combine explicit object structure with learned stochastic dynamics.

---

## 12. Implications for Dream-to-Look

### 12.1 Proposed model state

Dream-to-Look should consider both global context and per-object belief:

\[
s_t=\left(h^{scene}_t,\{z^i_t\}_{i=1}^{N_t}\right).
\]

Possible object-state content:

- image position and apparent size;
- LOS and relative motion;
- identity/existence belief;
- visibility and occlusion;
- observation age;
- uncertainty and competing hypotheses.

The set of objects changes over time, so association, slot birth/death, and permutation handling are first-class problems.

### 12.2 Action conditioning

Transition should condition on all actions that alter future observations:

- pan/tilt or look-point;
- zoom/FOV;
- EO/IR selection;
- ROI and processing mode;
- relevant platform motion or predicted motion command.

Without platform state/action, the model may confuse target motion with camera-induced image motion.

### 12.3 Task-oriented prediction heads

High-priority heads include:

- future target visibility;
- bbox/LOS distribution;
- FOV and gimbal-limit margin;
- detection and identification probability;
- track-loss probability;
- observation sufficiency;
- mission-value or epistemic-value estimate.

### 12.4 Candidate-action imagination

Starting from the current posterior, the model can compare candidate observation futures:

\[
a^*_{t:t+H}=\arg\max_{a_{t:t+H}}
\mathbb{E}_{p_{RSSM}}
\left[
\sum_{\tau=t}^{t+H}
R_{obs}(s_\tau,a_\tau)
\right].
\]

The observation reward may combine:

- mission-relevant evidence gain;
- target visibility;
- identity continuity;
- future reacquisition probability;
- sensor motion and switching cost;
- model uncertainty penalty.

### 12.5 Nightfall as a controlled benchmark

Project Nightfall offers a useful first test because it provides:

- partial bbox observations;
- exact simulator gate pose and geometry for evaluation;
- narrow-FOV and high-image-motion failures;
- controllable detector dropout and latency;
- independent platform motion from GP-G-;
- privileged targets for representation learning and distillation.

A first RSSM study should ask a narrow question:

> Does an object-centric RSSM predict future visibility and FOV loss well enough to produce anticipatory gaze or FOV actions that a recurrent model-free baseline does not?

### 12.6 Required baselines

- constant-velocity geometric predictor;
- frame stack;
- deterministic GRU predictor;
- recurrent model-free RL;
- stochastic RSSM without imagination-based policy learning;
- RSSM with latent actor or planner;
- privileged/oracle dynamics upper bound.

### 12.7 Required ablations

- deterministic-only vs stochastic latent;
- global vs object-centric state;
- observation reconstruction vs task-oriented heads;
- with and without platform motion;
- single model vs ensemble uncertainty;
- imagination horizon;
- posterior/prior training balance.

---

## 13. Takeaways

1. RSSM exists because current observations are not sufficient Markov states.
2. Its state combines deterministic recurrent memory \(h_t\) and stochastic latent state \(z_t\).
3. The posterior uses current evidence; the prior predicts without it.
4. The prior must learn to approximate useful posterior states so imagined futures remain grounded.
5. Observation, reward, continuation, or task-specific predictions train the latent state to retain useful information.
6. RSSM enables both filtering from real evidence and open-loop imagination under candidate actions.
7. Dreamer uses RSSM as its world model, then trains actor and critic inside imagined latent trajectories.
8. A stochastic latent does not automatically provide calibrated epistemic uncertainty.
9. Long-horizon rollout error and policy exploitation are central risks.
10. For Dream-to-Look, object identity, visibility, FOV, sensor action, and platform motion matter more than photorealistic reconstruction.

The shortest durable summary is:

> **An RSSM learns a compact, action-conditioned latent belief: it remembers through recurrence, represents uncertainty stochastically, corrects itself with observations, and imagines futures through its prior.**

---

## 14. Primary references

1. Hafner, D. et al. **Learning Latent Dynamics for Planning from Pixels (PlaNet).** ICML 2019.  
   <https://arxiv.org/abs/1811.04551>

2. Hafner, D. et al. **Dream to Control: Learning Behaviors by Latent Imagination.** ICLR 2020.  
   <https://arxiv.org/abs/1912.01603>

3. Hafner, D. et al. **Mastering Atari with Discrete World Models (DreamerV2).** ICLR 2021.  
   <https://arxiv.org/abs/2010.02193>

4. Hafner, D. et al. **Mastering Diverse Domains through World Models (DreamerV3).**  
   <https://arxiv.org/abs/2301.04104>

5. PlaNet reference implementation.  
   <https://github.com/google-research/planet>

6. DreamerV3 reference implementation.  
   <https://github.com/danijar/dreamerv3>

