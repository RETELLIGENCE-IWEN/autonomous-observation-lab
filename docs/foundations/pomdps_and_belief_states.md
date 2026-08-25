# POMDPs and Belief States

## At a Glance

> A Partially Observable Markov Decision Process (POMDP) models sequential decisions when the environment has a Markov state but the agent cannot observe that state directly; the agent must act using a belief state inferred from its observation–action history.

### Why this concept matters

Autonomous observation is not merely control with an incomplete input vector. The agent's actions change both the world and what can be learned about the world. Looking toward a suspected target, changing zoom, switching EO/IR, dwelling, or revisiting may have little immediate task reward yet improve later decisions by changing the agent's information state.

POMDPs provide the cleanest formal language for this coupling:

- the world has a hidden state;
- sensors produce incomplete and noisy evidence;
- history induces a belief over possible states;
- actions affect future state, future evidence, or both;
- a policy should optimize task return while accounting for information gathering.

### This note covers

- the POMDP tuple and generative process;
- histories, belief states, and Bayesian filtering;
- belief as a sufficient statistic and the belief-MDP reduction;
- the Bellman equation over beliefs;
- the distinction between physical reward and information value;
- concise comparisons with MDPs, Markov Games, POSGs, and Dec-POMDPs;
- implications for Dream-to-Look and learned belief representations.

### This note does not cover

- a survey of exact and approximate POMDP solvers;
- detailed proofs of piecewise-linear convex value functions;
- implementation recipes for a particular RL library;
- a full treatment of Bayesian filtering, active sensing, or multi-agent planning.

### Reading map

| Perspective | Main question |
|---|---|
| Formal model | What exactly is hidden, observed, controlled, and rewarded? |
| Belief | What should summarize the entire history? |
| Decision-making | How is value defined over uncertain world states? |
| Information | Why can looking be useful before it produces task reward? |
| Neighboring models | When is the problem an MDP, MG, POSG, or Dec-POMDP instead? |
| Dream-to-Look | What does the formalism imply for an autonomous observation agent? |

---

## 1. The problem POMDPs address

### 1.1 The world may be Markov even when the observation is not

Let (s_t) denote the complete environment state. The Markov property states that, conditioned on the present state and action, older history provides no additional information about the next state:

\[
p(s_{t+1}\mid s_{0:t},a_{0:t})
=p(s_{t+1}\mid s_t,a_t).
\]

The agent, however, receives an observation (o_t), not (s_t). In general,

\[
p(o_{t+1}\mid o_{0:t},a_{0:t})
\neq p(o_{t+1}\mid o_t,a_t).
\]

A bounding box does not uniquely determine range, target motion, occlusion state, detector reliability, camera geometry, or future visibility. The physical world can therefore be Markov while the stream exposed to the policy is non-Markov.

### 1.2 Partial observability creates two coupled problems

The agent must simultaneously:

1. **infer** what state the world may be in; and
2. **control** what to do given that uncertainty.

In active sensing, control also changes future inference. A gimbal command changes the next field of view; zoom changes resolution and coverage; sensor selection changes the observation noise model; dwell time changes evidence quality. This is why a POMDP is more than an MDP preceded by a passive state estimator.

---

## 2. Formal definition

A finite POMDP is commonly written as

\[
\mathcal{P}=
\langle \mathcal{S},\mathcal{A},\mathcal{O},T,Z,R,\gamma,b_0\rangle,
\]

where:

- \(\mathcal{S}\): hidden environment states;
- \(\mathcal{A}\): agent actions;
- \(\mathcal{O}\): observations;
- \(T(s'\mid s,a)\): transition model;
- \(Z(o'\mid s',a)\): observation model;
- \(R(s,a)\), or more generally \(R(s,a,s')\): reward model;
- \(\gamma\in[0,1)\): discount factor;
- \(b_0(s)\): initial belief over states.

This note uses the following timing convention:

1. the agent holds belief (b_t);
2. it selects (a_t);
3. the world transitions from (s_t) to (s_{t+1});
4. observation (o_{t+1}) is generated from (s_{t+1}) and (a_t);
5. the belief is updated to (b_{t+1}).

The corresponding generative process is

\[
s_{t+1}\sim T(\cdot\mid s_t,a_t),
\qquad
o_{t+1}\sim Z(\cdot\mid s_{t+1},a_t).
\]

Other texts attach (o_t) to (s_t) before action (a_t). The formulations are equivalent after consistent re-indexing; mixing timing conventions is a common source of incorrect filters.

### 2.1 History and policy

The information available before choosing (a_t) is the history

\[
h_t=(o_0,a_0,o_1,a_1,\ldots,a_{t-1},o_t).
\]

A general history-dependent policy is

\[
\pi(a_t\mid h_t).
\]

The objective is the expected discounted return

\[
J(\pi)
=\mathbb{E}_{\pi,T,Z,b_0}
\left[\sum_{t=0}^{\infty}\gamma^t R(s_t,a_t)\right].
\]

The expectation matters: the agent is optimizing over uncertainty in the initial state, transitions, observations, and possibly its own stochastic policy.

---

## 3. Belief state

### 3.1 Definition

The exact belief state is the posterior distribution over the current hidden state given all available history:

\[
b_t(s)
\triangleq
P(s_t=s\mid h_t).
\]

For a finite state space, (b_t) lies on the probability simplex:

\[
b_t(s)\geq 0,
\qquad
\sum_{s\in\mathcal{S}}b_t(s)=1.
\]

The belief is not merely an estimate of the most likely state. It preserves competing hypotheses and their probabilities. A point estimate such as

\[
\hat{s}_t=\arg\max_s b_t(s)
\]

throws away uncertainty that may change the optimal action.

### 3.2 Bayesian belief update

After action (a_t) and observation (o_{t+1}), the belief update has two conceptual stages.

**Prediction:**

\[
\bar b_{t+1}(s')
=\sum_{s\in\mathcal{S}}
T(s'\mid s,a_t)b_t(s).
\]

This propagates the old belief through the dynamics before seeing new evidence.

**Correction:**

\[
b_{t+1}(s')
=\eta\,
Z(o_{t+1}\mid s',a_t)\bar b_{t+1}(s'),
\]

where the normalizer is

\[
\eta^{-1}
=P(o_{t+1}\mid b_t,a_t)
=\sum_{\tilde s}
Z(o_{t+1}\mid \tilde s,a_t)
\bar b_{t+1}(\tilde s).
\]

Combining both stages gives

\[
b_{t+1}(s')
=
\frac{
Z(o_{t+1}\mid s',a_t)
\sum_s T(s'\mid s,a_t)b_t(s)
}{
\sum_{\tilde s}Z(o_{t+1}\mid \tilde s,a_t)
\sum_s T(\tilde s\mid s,a_t)b_t(s)
}.
\]

We can abbreviate the deterministic update as

\[
b_{t+1}=\tau(b_t,a_t,o_{t+1}).
\]

### 3.3 Why the belief is sufficient

When (T), (Z), and (b_0) are known and the belief is exact, (b_t) is a sufficient statistic of the history for predicting future states, observations, rewards, and returns. Thus an optimal policy can be written as

\[
\pi(a_t\mid b_t)
\]

without retaining the full raw history.

This statement is narrower than it sometimes sounds:

- it assumes the model and filter are correct;
- the belief may be infinite- or high-dimensional in continuous problems;
- sufficient does not mean computationally convenient;
- an approximate learned memory is not automatically a sufficient statistic.

---

## 4. The belief-MDP

A POMDP can be transformed into a fully observable MDP whose state is the belief. This is the conceptual bridge between partial observability and standard dynamic programming.

### 4.1 Belief-space reward

The expected immediate reward at belief (b) is

\[
r_B(b,a)
=\mathbb{E}_{s\sim b}[R(s,a)]
=\sum_s b(s)R(s,a).
\]

### 4.2 Observation probability

The probability of receiving observation (o') after taking action (a) from belief (b) is

\[
P(o'\mid b,a)
=\sum_{s'}Z(o'\mid s',a)
\sum_s T(s'\mid s,a)b(s).
\]

The next belief is then (	au(b,a,o')). The belief transition is stochastic because the future observation is stochastic, even though the update is deterministic once (o') is known.

### 4.3 Bellman optimality equation

The optimal value over beliefs satisfies

\[
V^*(b)
=\max_{a\in\mathcal A}
\left[
r_B(b,a)
+\gamma
\sum_{o'\in\mathcal O}
P(o'\mid b,a)
V^*\!\left(\tau(b,a,o')\right)
\right].
\]

The associated action value is

\[
Q^*(b,a)
=r_B(b,a)
+\gamma
\sum_{o'}P(o'\mid b,a)
V^*\!\left(\tau(b,a,o')\right).
\]

This equation exposes the essential POMDP idea: an action is valuable not only because of its immediate expected reward, but because of the distribution of observations it may produce and the improved decisions enabled by the resulting beliefs.

### 4.4 What the reduction does—and does not—do

The belief-MDP is fully observable because the agent knows its own belief. It does **not** make the physical state observable or the problem easy. It replaces a finite hidden-state problem with a continuous-state control problem over a probability simplex.

For finite-horizon finite POMDPs, the optimal value function is piecewise-linear and convex in belief and can be represented by α-vectors:

\[
V_t(b)=\max_{\alpha\in\Gamma_t}\alpha^\top b.
\]

This elegant structure motivates classical exact methods, but the number of relevant conditional plans can grow rapidly. Modern large-scale applications therefore rely on approximation, sampling, learned representations, or restricted policy classes.

---

## 5. Information value

### 5.1 Information gathering need not be a separate reward

In the exact POMDP objective, an observation action may be selected even when it has no immediate reward. Its value can arise instrumentally through future belief-dependent decisions.

For example, zooming onto an ambiguous object may delay area coverage now but reduce the probability of a later classification or tracking error. The Bellman backup credits that action if the resulting observation branches improve future return.

This is sometimes called **implicit information value** or **dual control** behavior: an action both affects the system and probes it.

### 5.2 Information gain is not identical to task value

A common proxy is expected entropy reduction:

\[
\operatorname{IG}(b,a)
=H(b)-
\mathbb E_{o'\sim P(\cdot\mid b,a)}
\left[H(\tau(b,a,o'))\right].
\]

This measures how much the action is expected to reduce belief uncertainty. It is not generally equal to the value of information for the task.

An observation can greatly reduce entropy about an irrelevant variable and have no decision value. Conversely, a small probability shift near a critical decision boundary can have high task value with little global entropy reduction. Dream-to-Look should therefore distinguish:

- **informativeness**: how much the belief changes;
- **decision relevance**: whether that change can improve mission decisions;
- **cost**: time, motion, coverage loss, blur, energy, or exposure incurred to obtain it.

Active sensing and value of information deserve their own foundation note because they refine precisely this distinction.

---

## 6. Relationship to neighboring decision models

### 6.1 Compact comparison

| Model | Agents | State visibility to decision-maker(s) | Reward structure | Policy input | Central difficulty |
|---|---:|---|---|---|---|
| MDP | 1 | Full Markov state | Single reward | (s_t) | Sequential control |
| POMDP | 1 | Partial/noisy observation | Single reward | history or belief | Inference plus control |
| Markov Game (MG) | 2+ | Usually Markov state is available under the basic formulation | Individual rewards (R_i) | state and/or local input | Strategic interaction and non-stationarity |
| Partially Observable Stochastic Game (POSG) | 2+ | Each agent may receive private observations | Individual rewards | private history/belief | Partial observability plus strategic interaction |
| Dec-POMDP | 2+ cooperative | Private observations; no single agent necessarily sees the joint information state | Shared team reward | each agent's local history | Decentralized inference and coordination |

Terminology varies across communities. **Stochastic Game** and **Markov Game** are often used interchangeably; POSG is its partially observable extension.

### 6.2 MDP versus POMDP

An MDP is commonly written

\[
\mathcal M=\langle\mathcal S,\mathcal A,T,R,\gamma\rangle.
\]

The agent observes (s_t), and the policy may be Markov:

\[
\pi(a_t\mid s_t).
\]

A POMDP adds an observation space and observation model because (s_t) is hidden. The relevant policy state becomes history or belief:

\[
\pi(a_t\mid h_t)
\quad\text{or}\quad
\pi(a_t\mid b_t).
\]

Giving an RL policy a vector called `state` does not make the environment an MDP. The test is whether that vector is sufficient for predicting future outcomes under actions.

### 6.3 POMDP versus Markov Game

An (N)-agent Markov Game may be written

\[
\mathcal G=
\langle
\mathcal S,
\{\mathcal A_i\}_{i=1}^N,
T,
\{R_i\}_{i=1}^N,
\gamma
\rangle,
\]

with joint action

\[
\mathbf a_t=(a_t^1,\ldots,a_t^N),
\]

transition

\[
T(s'\mid s,\mathbf a),
\]

and agent-specific rewards

\[
R_i(s,\mathbf a).
\]

The defining change is not merely multiple vehicles. It is that multiple decision-making agents jointly affect transitions and returns. Other agents' changing policies can make the learning problem non-stationary from any one agent's perspective.

A payload agent interacting with a fixed flight controller is not necessarily in a Markov Game. If the flight stack is part of stationary environment dynamics, a single-agent POMDP may be sufficient. It becomes game-like when another adaptive decision-maker has its own policy and possibly a distinct objective that materially shapes the transition process.

### 6.4 POSG and Dec-POMDP

A POSG augments a Markov Game with private observation functions. Each agent acts from its own action–observation history and may have a different reward.

A Dec-POMDP is the cooperative special case in which agents share a team reward but make decisions from decentralized information. A standard finite-horizon form is

\[
\langle
I,\mathcal S,
\{\mathcal A_i\},
T,
R,
\{\mathcal O_i\},
Z,
b_0,
H
\rangle.
\]

The joint policy is composed of local policies:

\[
\boldsymbol\pi=(\pi_1,\ldots,\pi_N),
\qquad
\pi_i(a_t^i\mid h_t^i).
\]

No agent necessarily has access to the joint history. This makes decentralized coordination fundamentally harder than solving a centralized POMDP over the joint observations.

For the current Payload Intelligence scope, role allocation and spatial coverage assignment are supplied by another model. Unless Payload Intelligence itself must jointly coordinate private beliefs and actions across vehicles, modeling the local observer as a POMDP is cleaner than prematurely adopting a Dec-POMDP.

---

## 7. Belief states in learned systems

### 7.1 Exact belief versus learned belief-like state

Classical belief filtering assumes access to (T), (Z), and a tractable state representation. Learned agents often replace the exact posterior with a recurrent or latent state:

\[
z_t=f_\theta(z_{t-1},a_{t-1},o_t).
\]

Examples include RNN hidden states, Bayesian filters with learned components, particle representations, transformers over histories, and RSSMs.

These representations may be useful **belief-like states**, but a vector (z_t) is not automatically a probability distribution and need not be calibrated, sufficient, identifiable, or uncertainty-aware.

### 7.2 RSSM as an approximate belief mechanism

An RSSM separates recurrent memory from stochastic latent variables and learns a prior and observation-conditioned posterior. In Dream-to-Look, it can approximate the repeated POMDP operations:

| POMDP concept | RSSM-style analogue |
|---|---|
| Predict belief through dynamics | latent prior rollout |
| Incorporate new evidence | observation-conditioned posterior |
| Information state | deterministic plus stochastic latent state |
| Predict observation consequences | decoder or task-relevant prediction heads |
| Evaluate future actions | latent imagination and value model |

The analogy is useful, not exact. The RSSM posterior is over a learned latent variable, not necessarily the true physical state, and its uncertainty may reflect the chosen variational family and training objective more than calibrated epistemic belief.

### 7.3 What should be tested

A learned memory should be evaluated by what it enables, not by its name. Useful tests include:

- **history dependence:** performance under occlusion, intermittent detections, and delayed evidence;
- **predictive sufficiency:** whether the latent state predicts relevant future observations and task outcomes;
- **ambiguity preservation:** whether mutually plausible hypotheses survive instead of collapsing prematurely;
- **belief correction:** whether contradictory evidence appropriately changes predictions;
- **calibration:** whether stated confidence matches empirical frequency;
- **policy relevance:** whether improved state inference actually improves observation decisions.

---

## 8. Common misconceptions and failure modes

| Misconception or failure | Why it is wrong | Diagnostic |
|---|---|---|
| “The simulator exposes state, so the policy problem is an MDP.” | The relevant question is what the deployed policy observes. | Compare simulator state, actor input, and critic-only privileged input. |
| “An RNN solves the POMDP.” | Memory capacity does not guarantee correct inference or sufficient statistics. | Test long occlusions, aliasing, and counterfactual histories with the same latest observation. |
| “Belief means the most likely state.” | MAP estimation discards alternate hypotheses and their decision consequences. | Construct two beliefs with the same MAP state but different optimal actions. |
| “Entropy reduction is always useful.” | Information about irrelevant variables may not change any decision. | Compare information gain with improvement in downstream return or risk. |
| “Belief-MDP makes the problem fully observed and therefore easy.” | The belief space is continuous and often high-dimensional. | Measure approximation error and planning cost as state/horizon grows. |
| “A stochastic latent is calibrated uncertainty.” | Latent randomness can encode nuisance variation or optimization artifacts. | Reliability diagrams, ensemble disagreement, OOD tests, and posterior predictive checks. |
| “Multiple UAVs imply a Dec-POMDP.” | Multiple physical systems are not necessarily multiple decentralized decision-makers. | Identify each adaptive policy, its information, action authority, and reward. |
| Model mismatch | Incorrect (T) or (Z) produces systematically wrong beliefs. | Innovation/residual tests and performance under sensor/dynamics shift. |
| Belief collapse | Approximation commits too early to one hypothesis. | Track multimodality through ambiguous and re-observation phases. |
| Unobservable task variables | No available action produces evidence that distinguishes critical states. | Observability analysis or paired hidden states with identical observation distributions. |

The last failure is especially important: no memory architecture can recover information that the sensing process never makes observable. Active observation helps only when the action space contains looks that can generate discriminating evidence.

---

## 9. Implications for Dream-to-Look

### 9.1 A minimal POMDP interpretation

Dream-to-Look can be posed as a POMDP without assuming a particular detector, map, or flight intelligence stack.

| Element | Possible Dream-to-Look meaning |
|---|---|
| Hidden state (s_t) | object existence, class, pose, motion, occlusion, relevance, terrain/scene state, platform state, sensor condition |
| Observation (o_t) | images or features, detections, tracks, confidence, payload telemetry, optional map/fusion information |
| Belief (b_t) | uncertainty over object states, scene hypotheses, visibility, and mission-relevant facts |
| Payload action (a_t^P) | pan/tilt, angular rate, zoom/FOV, EO/IR choice, dwell, track/revisit/search mode |
| Flight request (a_t^F) | requested heading, orbit, standoff, altitude, or viewpoint constraint when payload-only action is insufficient |
| Transition (T) | object/world dynamics plus consequences of platform and payload motion |
| Observation model (Z) | geometry-, modality-, resolution-, blur-, weather-, and detector-dependent evidence process |
| Reward (R) | mission evidence value minus time, motion, risk, coverage, and request costs |

The flight controller need not be Flight Intelligence. Its realized motion and constraints can enter as exogenous dynamics and observations. A cooperative request interface is needed only when payload actions alone cannot reach a valuable observation state.

### 9.2 Observation actions are epistemic actions

A useful distinction is:

- **world-directed actions**, intended mainly to change physical task state;
- **information-directed actions**, intended mainly to change what the agent can infer;
- **dual-purpose actions**, which do both.

Most gimbal and sensor commands are information-directed, while flight requests are often dual-purpose. Their value should be evaluated through the evidence and future decisions they enable—not only through centering error or instantaneous detection confidence.

### 9.3 Object-centric belief factorization

A practical object-centric approximation might factor belief into persistent object slots plus global context:

\[
\tilde b_t
=\left(g_t,\{z_t^{(k)}\}_{k=1}^{K}\right),
\]

where (z_t^{(k)}) represents a hypothesized object's identity, kinematics, visibility, relevance, and uncertainty, and (g_t) represents global scene and sensor context.

This is a design hypothesis, not a consequence of POMDP theory. Objects may be dependent through occlusion, interaction, data association, and shared camera geometry. A graph or attention mechanism may therefore be needed to represent relations rather than assuming independent slots.

### 9.4 The key research hypothesis

> An autonomous observation policy that maintains a predictive, uncertainty-aware object-centric information state and evaluates candidate looks by their expected downstream evidence value will outperform reactive view control, especially under ambiguity, occlusion, intermittent detection, and limited sensing budgets.

This hypothesis separates the proposed research value from “better tracking.” The claim is that the agent chooses observations to resolve mission-relevant uncertainty before failure becomes visible in immediate perception metrics.

### 9.5 Suggested baselines and ablations

| Category | Candidate |
|---|---|
| Reactive baseline | center highest-priority detection; fixed zoom schedule |
| Memory baseline | deterministic RNN policy using identical observations |
| Geometric baseline | uncertainty-agnostic scan/revisit heuristic |
| Belief ablation | point estimate instead of distributional/object-hypothesis state |
| Value ablation | immediate confidence reward without future decision value |
| Action ablation | payload-only versus payload plus priced flight requests |
| Representation ablation | monolithic latent versus object-centric latent |
| Model ablation | model-free recurrent policy versus RSSM imagination |

### 9.6 Suggested evaluation axes

- mission-relevant uncertainty reduction, not only global entropy;
- decision accuracy after ambiguous observation histories;
- recovery after occlusion and missed detections;
- time and action cost required to obtain sufficient evidence;
- calibration of object existence, identity, and motion hypotheses;
- robustness when map/fusion inputs are absent or degraded;
- transfer across detectors, payload configurations, and flight backends;
- frequency and value of flight requests;
- task return under fixed sensing budgets.

---

## 10. Durable takeaways

1. A POMDP is an MDP with hidden state and an explicit observation process—not simply an MDP with a noisy vector.
2. The exact belief (b_t=P(s_t\mid h_t)) is a distribution and, under the assumed model, a sufficient statistic of history.
3. Bayesian filtering alternates prediction through (T) and correction through (Z).
4. A POMDP becomes an MDP over beliefs, but this trades hidden finite state for continuous information state rather than making the problem easy.
5. The Bellman equation values observations through the future decisions enabled by their resulting beliefs.
6. Information gain and task value are related but not equivalent.
7. RNNs and RSSMs can implement useful belief-like states, but sufficiency and calibrated uncertainty must be demonstrated empirically.
8. Multiple vehicles do not by themselves imply a Markov Game or Dec-POMDP; the number of adaptive decision-makers, their objectives, and their information structure do.
9. Dream-to-Look is naturally a POMDP because looking changes both evidence and future decision quality.

---

## 11. Primary references

- Åström, K. J. (1965). [Optimal Control of Markov Processes with Incomplete State Information](https://doi.org/10.1016/0022-247X(65)90154-X). An early formal treatment of control under incomplete state information and the information-state transformation.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). [Planning and Acting in Partially Observable Stochastic Domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf). The canonical AI introduction to beliefs, value functions, and POMDP planning.
- Smallwood, R. D., & Sondik, E. J. (1973). [The Optimal Control of Partially Observable Markov Processes over a Finite Horizon](https://doi.org/10.1287/opre.21.5.1071). Foundational finite-horizon structure, including piecewise-linear value functions over beliefs.
- Shapley, L. S. (1953). [Stochastic Games](https://www.pnas.org/doi/10.1073/pnas.39.10.1095). The foundational stochastic-game formulation underlying Markov Games.
- Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. (2002). [The Complexity of Decentralized Control of Markov Decision Processes](https://doi.org/10.1287/moor.27.4.819.297). A foundational complexity treatment of decentralized partially observable decision-making.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1994). [Acting Optimally in Partially Observable Stochastic Domains](https://cs.brown.edu/research/pubs/techreports/reports/CS-94-20.html). Earlier technical report leading to the canonical 1998 treatment.

