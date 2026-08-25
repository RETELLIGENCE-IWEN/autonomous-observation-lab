# Research Brief: Decision-Aware Dream-to-Look

## At a Glance

### Working title

**Decision-Aware Active Observation through Object-Centric Latent Imagination**

### Objective

Develop an autonomous observation agent that maintains uncertain hypotheses about multiple objects, imagines the evidence consequences of candidate gaze actions, and allocates a limited sensing budget to identify, retain, and reacquire the mission-relevant target.

### Central claim

> A policy that plans over an object-centric predictive belief will acquire more decision-relevant evidence per sensing cost than reactive, uncertainty-greedy, or recurrent model-free observation policies—especially when identity evidence is incomplete, observations are interrupted, and several objects compete for attention.

### First-cycle scope

- object-feature observations rather than pixels;
- staged identification, observation allocation, occlusion, and reacquisition;
- controllable detector noise and missing observations;
- object-centric RSSM world model;
- policy/value learning through latent imagination;
- no dependency on a 3D fusion map or particular flight backend.

### Non-goals

- improving the upstream detector;
- photorealistic image generation;
- continuous low-level gimbal stabilization;
- multi-UAV role or target allocation;
- full Unreal or real-aircraft integration in the first cycle.

---

## 1. Research question

> Can an object-centric RSSM predict how candidate observation actions will change mission-relevant evidence well enough for a latent-imagination policy to outperform policies that react only to current detections, confidence, or recurrent memory?

This requires three capabilities:

1. **Evidence seeking:** select which candidate should be inspected to resolve target identity.
2. **Persistent belief:** retain object hypotheses through occlusion, missed detection, and FOV exit.
3. **Resource allocation:** decide which evidence is worth acquiring under limited time and look budget.

The contribution is not autonomous tracking alone. The agent must decide what deserves observation, why, and when enough evidence has been obtained.

---

## 2. Novelty sought

### 2.1 From saliency to decision relevance

The agent should not simply look at the most salient, uncertain, or high-confidence object. It should estimate whether an observation can change a downstream mission decision.

### 2.2 From tracking state to evidence belief

An object state should contain more than geometry. It should represent competing identity hypotheses, evidence quality, visibility, persistence, relevance, and uncertainty.

### 2.3 From future visibility to future evidence

The world model should predict not only whether an object will be visible, but whether the resulting observation is likely to be discriminative enough to alter the belief.

### 2.4 From payload control to observation intelligence

The intended output is a reusable observation principle. Gimbal configuration, detector, map source, and flight backend are replaceable interfaces rather than defining assumptions.

---

## 3. Formal problem

The benchmark is a POMDP

\[
\mathcal P=\langle\mathcal S,\mathcal A,\mathcal O,T,Z,R,\gamma,b_0\rangle.
\]

The hidden state contains object identities, kinematics, occlusion processes, target designation, and sensor/world variables. The observation contains only noisy object features for currently detected entities and payload state.

The policy uses a learned information state:

\[
\pi_\phi(a_t\mid \tilde b_t),
\qquad
\tilde b_t=
\left(h_t^g,\{h_t^{(k)},z_t^{(k)}\}_{k=1}^{K}\right).
\]

Its objective is terminal decision utility minus sensing cost:

\[
J(\pi)=\mathbb E_\pi
\left[U(d_\tau,s_\tau)-\sum_{t=0}^{\tau-1}C(a_t)\right],
\]

where (d_\tau) is the final target decision and (	au) is the stopping time.

---

## 4. Hypotheses and falsifiers

### H1 — Decision-aware acquisition

Under irrelevant uncertainty, a decision-aware policy will achieve higher terminal utility than an entropy-greedy policy at equal sensing cost.

**Falsified if:** entropy-greedy matches it when uncertain nuisance objects are systematically varied.

### H2 — Persistent object belief

Object-centric stochastic state will improve reacquisition and identity retention after occlusion relative to reactive and deterministic recurrent baselines.

**Falsified if:** gains disappear when observation history and model capacity are controlled.

### H3 — Latent imagination

Multi-step evidence imagination will outperform one-step acquisition and recurrent model-free RL when an initially weak look enables a later decisive observation.

**Falsified if:** gains occur only in immediate-evidence cases or are reproduced by greedy confidence selection.

### H4 — Object-centric generalization

Object-centric dynamics will generalize better than a monolithic RSSM to unseen object counts, arrangements, and attribute combinations.

**Falsified if:** the monolithic model matches held-out performance under comparable capacity and data.

### H5 — Cost-aware stopping

Explicit commit/abstain actions and sensing cost will reduce redundant looks while preserving calibrated decision accuracy.

**Falsified if:** the policy observes until timeout or commits prematurely without a superior accuracy–cost frontier.

---

## 5. Proposed model

### Observation encoder

A permutation-equivariant set encoder receives variable-length detections and payload state. It never receives privileged object identity.

### Persistent object memory

Detections bind to latent slots capable of representing birth, disappearance, missed detection, uncertain association, identity evidence, and time since meaningful observation. A noisy observed track handle may be used as a hint, but evaluation must include handle loss and switching.

### Object-centric RSSM

Each slot has deterministic memory and stochastic state; a global state represents sensor and scene context. Relational dynamics model interactions and shared camera geometry.

Prediction heads cover:

- object existence and next state;
- visibility and detection probability;
- appearance evidence and quality;
- reward/decision utility;
- continuation and terminal conditions.

Raw pixel reconstruction is excluded.

### Imagination actor and value

Posterior states inferred from real trajectories seed latent prior rollouts. The actor selects observation and commit actions; the value predicts cost-adjusted terminal utility. The imagined horizon must cover at least one staged evidence sequence.

An optional later variant can use short-horizon latent MPC to rerank actor-proposed actions.

---

## 6. Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Random valid look | Lower bound and sanity check |
| B1 | Fixed scan schedule | Tests whether adaptivity matters |
| B2 | Detection-centering heuristic | Conventional smart tracking |
| B3 | Confidence/entropy greedy | Generic uncertainty seeking |
| B4 | Oracle one-step VoI | Validates benchmark decision structure |
| B5 | Reactive model-free RL | Tests value of memory |
| B6 | Deterministic recurrent RL | Tests stochastic belief/world model value |
| B7 | Monolithic RSSM Dreamer | Tests object-centric factorization |
| B8 | Object-centric RSSM, one-step | Tests multi-step imagination |
| B9 | Object-centric RSSM imagination | Proposed method |
| UB | Privileged-state planner | Approximate upper bound |

All learned baselines receive identical observable information and comparable parameter and interaction budgets.

---

## 7. Required ablations

- remove object-centric factorization;
- replace stochastic object state with deterministic state;
- remove relational dynamics;
- remove explicit uncertainty outputs;
- remove sensing cost and stopping;
- replace decision utility with entropy reduction;
- shorten imagination below the staged-evidence horizon;
- remove association corruption or occlusion;
- evaluate with and without detector confidence.

An ablation must isolate the claimed mechanism rather than merely changing parameter count.

---

## 8. Success criteria

### Minimum scientific success

1. Decision-aware acquisition beats entropy-greedy under irrelevant uncertainty.
2. Multi-step imagination beats one-step/reactive policies under staged evidence.

### Model success

The object-centric RSSM predicts visibility, evidence quality, and identity-belief consequences well enough that imagined action ranking correlates with realized ranking.

### Generalization success

The proposed policy retains an advantage on at least two held-out axes: object count, occlusion duration, attribute combination, detector-noise regime, sensing budget, or payload FOV/zoom configuration.

### Valuable negative result

If imagination fails, the experiment should localize the cause to world-model error, object binding, value learning, or policy optimization. A benchmark that enables this diagnosis remains valuable.

---

## 9. Research phases

### Phase A — Benchmark and oracle validation

Implement the generative environment and privileged belief/VoI oracles. Demonstrate separation among fixed scan, entropy-greedy, and decision-aware acquisition.

### Phase B — Representation and world model

Train the object-centric RSSM on policy-diverse trajectories. Validate filtering, open-loop prediction, persistence, evidence prediction, and calibration.

### Phase C — Latent-imagination policy

Train actor and value from imagined trajectories seeded by replay. Compare against recurrent model-free and monolithic world-model baselines.

### Phase D — Robustness and transfer

Test held-out observation models, object compositions, budgets, and payload configurations before connecting rendered or real detector output.

---

## 10. Decision log

| Decision | Initial choice | Reason |
|---|---|---|
| Perception input | Object features | Isolate observation intelligence from detector quality |
| Scenario | Identification + allocation + reacquisition | Require all three capabilities in one causal sequence |
| Cycle scope | Through latent imagination | Test the full Dream-to-Look proposition |
| Representation | Object-centric plus global context | Support persistent hypotheses and variable object sets |
| Flight coupling | Excluded initially | Preserve backend independence |
| Image generation | Excluded initially | Predict evidence rather than pixels |

---

## 11. Immediate deliverables

1. executable environment and seeded scenario generator;
2. scripted and oracle policies;
3. deterministic evaluation suite;
4. object-centric RSSM training pipeline;
5. latent-imagination actor–critic;
6. baseline and ablation matrix;
7. reproducible configs, artifacts, and reports.

