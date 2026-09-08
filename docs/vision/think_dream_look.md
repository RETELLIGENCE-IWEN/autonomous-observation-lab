# Think–Dream–Look (TDL)
## A Conceptual Architecture for Autonomous Observation Intelligence

## 1. Motivation

Autonomous observation should not be reduced to a mapping from the latest detection to a payload command.

A useful observation agent must decide not only **where to look**, but also:

- what it currently knows;
- what remains uncertain, contradictory, or unobserved;
- whether the current evidence can be improved by further internal reasoning;
- what future evidence different sensing actions are likely to produce;
- when additional observation is worth its time, compute, and sensing cost;
- when enough evidence has been collected to stop observing and move attention elsewhere.

Think–Dream–Look (TDL) frames this as a resource-bounded epistemic decision problem:

> **Given limited compute, time, and sensing resources, should the agent refine its current belief, imagine possible futures, or query the physical world for new evidence?**

TDL is the overarching conceptual architecture for the Autonomous Observation Lab. Dream-to-Look (D2L) is a concrete research track inside this architecture, focused on using a learned world model to imagine future observation consequences and convert them into sensing actions.

---

## 2. Core architecture

TDL separates autonomous observation into three distinct but coupled forms of computation.

### Think — refine the present belief

Think operates at the same physical time step. It performs additional internal computation over the evidence already available.

$$
h_t^{(k+1)} = F_\theta(h_t^{(k)}, o_t, m_t)
$$

- $o_t$: currently available observations;
- $m_t$: mission context, prospective memory, and unresolved questions;
- $h_t^{(k)}$: latent reasoning state after the $k$-th internal refinement step.

The purpose of Think is not to generate language. It is to improve the internal information state before acting.

Useful quantities that may emerge or be explicitly represented include:

- object identity belief;
- visibility and persistence belief;
- mission relevance;
- unresolved ambiguity;
- contradiction between evidence sources;
- expected reappearance or occlusion state;
- decision-relevant uncertainty;
- observation priority;
- evidence sufficiency.

The central question is:

> **What do I know now, and what exactly do I still not know?**

This component is conceptually related to recurrent computation, recurrent depth, latent reasoning, adaptive pondering, and fixed-point refinement studied in modern sequence models. The transferable idea is not a particular LLM architecture, but the separation of **physical time** from **internal computational depth**.

---

### Dream — imagine possible futures

Dream advances an internal model through hypothetical future states or future observation outcomes without yet moving the real sensor.

$$
z_t \rightarrow \hat z_{t+1} \rightarrow \hat z_{t+2} \rightarrow \cdots \rightarrow \hat z_{t+H}
$$

For a candidate sensing action $a_t^{look}$, the world model may estimate

$$
p(\hat o_{t+1}, \hat z_{t+1} \mid z_t, a_t^{look}).
$$

Dream asks questions such as:

- If I keep tracking this object, what evidence will I lose elsewhere?
- If I zoom out now, will I preserve future reacquisition probability?
- Where is an occluded object likely to reappear?
- Which gaze direction is likely to produce discriminative evidence?
- Will EO, IR, zoom, or a different aspect reduce the uncertainty that matters to the mission?
- Is a weak observation now valuable because it enables a decisive observation later?

The central question is:

> **If I observe in this way, what am I likely to learn next?**

D2L is primarily concerned with this **Dream → Look** link: learning a predictive belief/world model whose imagined futures are useful enough to change real sensing behavior.

---

### Look — acquire new evidence from the world

Look performs an actual sensing action.

A payload action may include

$$
a_t^{look} = [\text{azimuth},\text{elevation},\text{zoom},\text{modality},\text{dwell},\ldots].
$$

Depending on the embodiment, some of these dimensions may be absent or delegated to lower-level controllers.

The goal is not merely to center the most confident target. The agent should select observations according to expected mission value and evidence value:

$$
a_t^* = \arg\max_a \; \mathbb E[\text{Mission Value} + \lambda\,\text{Information Value} - C_{sense}(a)].
$$

The central question is:

> **Given what I know and what I expect, where should I actually look?**

---

## 3. Two different kinds of internal imagination

A key distinction in TDL is between **computational refinement** and **temporal imagination**.

### Computational depth — Think

$$
h_t^{(0)} \rightarrow h_t^{(1)} \rightarrow h_t^{(2)} \rightarrow \cdots
$$

Physical time remains fixed at $t$. The agent spends more computation interpreting the current evidence.

### Temporal depth — Dream

$$
z_t \rightarrow z_{t+1} \rightarrow z_{t+2} \rightarrow \cdots
$$

The model advances through hypothetical future time to predict consequences.

These axes should not be conflated. A system may think deeply without predicting far into the future, or imagine long futures from a poorly refined current belief. TDL treats them as distinct resources that can be composed.

A useful high-level architecture is therefore

$$
\boxed{
\text{Perception}
\rightarrow
\text{Recurrent Belief Refinement}
\rightarrow
\text{Temporal Imagination}
\rightarrow
\text{Active Observation}
}
$$

---

## 4. Closed-loop epistemic agent

TDL is not a one-way pipeline. The actual process is a closed loop:

$$
\boxed{
\text{Observe}
\rightarrow
\text{Think}
\rightarrow
\text{Dream}
\rightarrow
\text{Look}
\rightarrow
\text{Observe}
\rightarrow \cdots
}
$$

A Look action produces new evidence. New evidence updates the belief. The updated belief may invalidate previous hypotheses, create new uncertainty, confirm an expected event, or reveal that an expected event failed to occur.

The agent therefore behaves less like a camera controller and more like an **epistemic agent that actively interrogates the world**.

---

## 5. Think, Dream, or Look?

The three stages do not need to execute with fixed depth on every decision.

An easy observation may require almost no extra reasoning:

$$
\text{Observe} \rightarrow \text{Look}.
$$

An ambiguous situation may justify additional internal processing:

$$
\text{Observe}
\rightarrow \text{Think}^K
\rightarrow \text{Dream}^H
\rightarrow \text{Look}.
$$

This introduces a second-order decision problem: **how much computation and sensing should be spent before acting?**

Conceptually, the objective can include both computational and sensing cost:

$$
J =
R_{mission}
+ \lambda I_{gain}
- \beta C_{compute}
- \gamma C_{sense}.
$$

This allows the agent to learn that more computation is useful only when it changes the eventual observation decision or mission outcome.

---

## 6. Think versus Look: two ways to reduce uncertainty

One of the most important TDL research questions is that uncertainty can be reduced in two fundamentally different ways.

### Internal reduction

The agent can perform another latent refinement step:

$$
h_t^{(k)} \rightarrow h_t^{(k+1)}.
$$

This consumes compute but does not obtain new physical evidence.

### External reduction

The agent can perform a sensing action and receive a new observation:

$$
b_t \xrightarrow{a_t^{look}} b_{t+1}.
$$

This consumes time, sensor authority, and possibly attention that could have been spent elsewhere.

A useful abstraction is therefore

$$
V_{think} \approx \frac{\Delta U_{internal}}{C_{compute}},
\qquad
V_{look} \approx \frac{\mathbb E[\Delta U_{observation}]}{C_{sense}},
$$

where $\Delta U$ should be tied to decision-relevant uncertainty or expected mission utility rather than generic entropy alone.

This yields the question:

> **Can an embodied observation agent learn when the current evidence only needs more computation, and when no amount of internal reasoning can replace a new physical observation?**

This is a central point of contact between adaptive test-time computation and active sensing.

---

## 7. Object-centric instantiation

The initial TDL research does not require raw image reasoning. Upstream perception may provide object features such as

$$
O_i = [bbox_i, confidence_i, appearance_i, visibility_i, age_i, motion_i, \ldots].
$$

The agent maintains persistent object-centric latent states and a global scene/context state.

During Think, these states may be repeatedly refined:

$$
O_i^{(k+1)} =
F_\theta
\left(
O_i^{(k)},
\{O_j^{(k)}\},
s_{platform},
m_{mission}
\right).
$$

During Dream, the refined belief seeds imagined future states and evidence outcomes. During Look, the policy selects the next real sensor action.

This factorization is especially useful for autonomous observation because the important hidden variables are naturally object-specific:

- identity;
- visibility;
- association;
- evidence quality;
- persistence;
- uncertainty;
- relevance;
- predicted reappearance;
- expected information value.

---

## 8. Relationship to Dream-to-Look

TDL and D2L are not competing names for the same idea.

Their relationship is intentionally hierarchical:

$$
\boxed{\text{Think–Dream–Look} \supset \text{Dream-to-Look}}
$$

### Think–Dream–Look

The overarching architecture for autonomous observation intelligence. It covers:

- belief formation and refinement;
- adaptive internal computation;
- latent/world-model imagination;
- active information acquisition;
- compute/sensing resource allocation;
- stopping and evidence sufficiency.

### Dream-to-Look

A concrete research track that tests whether learned latent prediction can improve the selection of future observations.

Its first-cycle question remains deliberately narrower:

> Can an object-centric RSSM predict the evidence consequences of candidate observation actions well enough for latent-imagination policies to outperform reactive, greedy, and recurrent model-free baselines?

This makes D2L an important experimental route toward the broader TDL hypothesis rather than requiring the current D2L benchmark to solve the entire TDL problem at once.

---

## 9. Core research hypotheses

### TDL-H1 — Recurrent belief refinement

Additional latent computation should improve observation decisions in ambiguous situations even when no new sensor evidence is supplied.

**Falsifier:** matched-capacity feed-forward or single-pass recurrent policies achieve the same decision frontier.

### TDL-H2 — Adaptive computation

A policy that can vary internal compute should allocate more computation to genuinely ambiguous cases and less to easy cases while preserving or improving mission utility.

**Falsifier:** compute depth is uncorrelated with task difficulty or fixed-depth policies dominate the utility–compute frontier.

### TDL-H3 — Temporal imagination

Multi-step latent imagination should improve decisions when the value of a sensing action depends on its future evidence consequences rather than immediate confidence gain.

**Falsifier:** one-step or reactive policies reproduce the gains under controlled information and capacity budgets.

### TDL-H4 — Think versus Look

The agent should learn situations where internal computation is sufficient and situations where only new physical evidence can resolve the decision.

**Falsifier:** the learned policy collapses to always computing more or always observing more without a superior cost–utility frontier.

### TDL-H5 — Closed-loop epistemic behavior

Joint Think–Dream–Look reasoning should produce purposeful information-seeking behavior such as counterevidence search, preemptive reacquisition, selective modality switching, and evidence-aware stopping.

**Falsifier:** observed gains reduce to smoother tracking, target centering, or detector-confidence heuristics.

---

## 10. Suggested research progression

TDL should be built progressively rather than introduced as a monolithic architecture.

### Stage 0 — Reactive Look

Establish conventional baselines:

- fixed scan;
- center/track heuristic;
- confidence or entropy greedy;
- reactive model-free policy.

### Stage 1 — Think-to-Look

Introduce persistent belief and recurrent latent refinement without a learned future model.

Question:

> Does additional internal computation improve the next observation decision?

### Stage 2 — Dream-to-Look

Introduce the object-centric RSSM and latent imagination.

Question:

> Does explicit prediction of future evidence improve active observation?

This is the current committed D2L research track.

### Stage 3 — Adaptive Think-versus-Look

Allow the agent to trade internal computation against external sensing cost.

Question:

> Can it learn when to reason longer and when to acquire new evidence?

### Stage 4 — Integrated Think–Dream–Look

Jointly allocate recurrent computation depth, imagination depth, and sensing actions under mission-level utility.

Question:

> Can the agent learn an efficient closed-loop epistemic strategy rather than a fixed perception-planning-control pipeline?

---

## 11. What success should look like

The strongest evidence for TDL is behavioral rather than architectural.

A successful agent should demonstrate behaviors such as:

- immediately acting when the evidence is already sufficient;
- spending more internal computation when several hypotheses remain plausible;
- recognizing that further internal reasoning cannot resolve missing evidence;
- looking away from the currently tracked target to resolve a more important uncertainty;
- anticipating an occluded object's reappearance and orienting before it becomes visible;
- choosing an observation that is weak immediately but enables a later decisive observation;
- switching EO/IR or zoom because the predicted evidence quality differs by modality;
- remembering an unresolved target and revisiting it later;
- terminating observation when additional evidence has low expected decision value.

The intended reaction is not:

> “The gimbal tracks automatically.”

It is:

> **“The system appears to know what it needs to find out next.”**

---

## 12. Long-term view

TDL treats a sensor not as a passive input channel, but as an **active interface through which intelligence queries the physical world**.

The long-term system should be able to:

1. maintain a calibrated belief about mission-relevant entities and hypotheses;
2. recognize ignorance, contradiction, and insufficient evidence;
3. spend internal computation only when it is useful;
4. imagine the future evidence consequences of candidate observations;
5. choose gaze, zoom, modality, dwell, or observation requests according to expected mission value;
6. stop observing when the evidence is sufficient;
7. repeat this process as a closed-loop epistemic agent.

In compact form:

> **Think** — What do I know, and what do I still need to resolve?  
> **Dream** — What would I learn if I observed this way?  
> **Look** — Which real observation should I acquire now?

That is the conceptual target of autonomous observation intelligence in this lab.
