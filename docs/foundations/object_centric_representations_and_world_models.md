# Object-Centric Representations and World Models

## At a Glance

> An object-centric model represents a scene as a structured collection of entity-like latent variables and their relations, enabling dynamics and decisions to operate on persistent things rather than a single undifferentiated scene vector.

### Why this concept matters

Dream-to-Look must reason about objects that enter and leave the field of view, become occluded, compete for attention, and remain mission-relevant while unseen. A monolithic latent state can encode this information, but does not explicitly encourage persistence, separability, relational prediction, or per-object uncertainty. Object-centric representation is therefore a promising inductive bias for an autonomous observer.

### This note covers

- objects as an inductive bias rather than a guaranteed ontology;
- slot-based scene decomposition and competitive binding;
- temporal identity, relational dynamics, and object-centric world models;
- supervised, detection-seeded, and unsupervised variants;
- major failure modes;
- an object-centric RSSM hypothesis for Dream-to-Look.

### Reading map

| Perspective | Main question |
|---|---|
| Representation | What makes a latent variable object-centric? |
| Binding | How are visual features assigned to slots? |
| Dynamics | How should entities and relations evolve? |
| Persistence | How is identity maintained through time and occlusion? |
| Dream-to-Look | Which object properties must an observation policy remember? |

---

## 1. Why object-centric representation?

Natural scenes are compositional: entities persist, move, interact, occlude one another, and can recur in new combinations. An object-centric representation aims to encode a scene as

\[
Z_t=\{z_t^{(1)},\ldots,z_t^{(K)}\},
\]

possibly with global context (g_t), where each slot describes an entity or coherent component.

Compared with a single vector (z_t), this bias may support:

- variable numbers of relevant entities;
- permutation-aware set processing;
- per-object state and uncertainty;
- relational dynamics and interactions;
- selective attention and action assignment;
- compositional generalization to new object combinations.

These are desired properties, not automatic consequences. A slot can represent a fragment, texture, background region, or mixture of objects.

---

## 2. What counts as an object?

There is no universally correct object decomposition. “Object” can mean:

- a physical instance;
- a track maintained by a detector/tracker;
- a functional unit relevant to action;
- a coherent region under a generative model;
- a persistent latent cause of observations.

For autonomous observation, the most useful definition is operational:

> An object is a persistent, separately actionable hypothesis whose properties and uncertainty matter to future evidence acquisition or mission decisions.

This may include a suspected but unconfirmed object, a group, an occluder, or a region of interest. The representation should serve the research objective rather than imitate human instance segmentation for its own sake.

---

## 3. Slot-based decomposition

### 3.1 Encoder, slots, and decoder

A common architecture encodes an observation into features

\[
X=\{x_1,\ldots,x_N\},
\]

then maps them into (K) exchangeable slots:

\[
Z=\operatorname{Bind}_\theta(X)
=\{z^{(1)},\ldots,z^{(K)}\}.
\]

A compositional decoder predicts per-slot appearance (hat o^{(k)}) and mask (m^{(k)}), then combines them:

\[
\hat o=\sum_{k=1}^{K}m^{(k)}\odot\hat o^{(k)},
\qquad
\sum_k m^{(k)}=1.
\]

Reconstruction encourages the slots collectively to explain the scene; competition encourages specialization.

### 3.2 Slot Attention

Slot Attention uses iterative attention from slots to visual features. In simplified form,

\[
q_k=W_q z_k,
\qquad
k_i=W_k x_i,
\qquad
v_i=W_v x_i,
\]

with assignment logits

\[
\ell_{ik}=\frac{k_i^\top q_k}{\sqrt d}.
\]

The crucial normalization is competitive across slots for each input feature:

\[
\alpha_{ik}=\operatorname{softmax}_k(\ell_{ik}).
\]

Each slot aggregates its assigned evidence and is iteratively updated, commonly with a GRU and MLP. The slots are permutation-equivariant: exchanging their initialization order exchanges their output order rather than changing the represented set.

### 3.3 Variational and sequential predecessors

MONet decomposes scenes through recurrent attention masks and a component VAE. IODINE performs iterative amortized inference over multiple latent components. Slot Attention provides a simpler differentiable binding module. SAVi extends slot-based binding across video by initializing current slots from previous slots.

The enduring idea is not a specific module: **multiple latent components compete to explain observations and persist as reusable entity representations.**

---

## 4. Object-centric dynamics

### 4.1 Independent transition is usually insufficient

A naive dynamics model predicts every slot independently:

\[
z_{t+1}^{(k)}=f_\theta(z_t^{(k)},a_t).
\]

This misses collisions, occlusion, coordinated motion, and shared camera transformations. Relational dynamics instead aggregate interactions:

\[
m_t^{(j\rightarrow k)}
=\phi_\theta(z_t^{(j)},z_t^{(k)}),
\]

\[
z_{t+1}^{(k)}
=f_\theta\left(
z_t^{(k)},a_t,
\sum_{j\neq k}m_t^{(j\rightarrow k)},g_t
\right).
\]

Graph neural networks and transformers are natural implementations because they operate on sets and model pairwise or higher-order relations.

### 4.2 Camera and world motion must be separated

For a gimballed airborne sensor, apparent object motion combines:

- object dynamics;
- aircraft motion;
- gimbal motion;
- zoom/FOV change;
- projective geometry;
- detection and association noise.

An object-centric world model should condition on payload and platform actions, or explicitly transform slots between camera-centric and world-centric frames. Otherwise it may learn spurious “object dynamics” that are actually ego-motion.

### 4.3 Object-centric world model

A general controlled model is

\[
p_\theta(Z_{t+1},g_{t+1}\mid Z_t,g_t,a_t),
\]

with an observation model

\[
p_\theta(o_t\mid Z_t,g_t).
\]

Task heads may predict object existence, identity, pose, visibility, future observation quality, or mission relevance rather than reconstructing every pixel.

G-SWM introduced structured object-centric world modeling for visual environments; SlotFormer applies transformer dynamics to slot sequences. These works illustrate the promise, but neither establishes that unsupervised slots will correspond to operational targets in unconstrained EO/IR data.

---

## 5. Temporal binding and persistence

### 5.1 Slot identity is not guaranteed

Slots are exchangeable. Slot index (k) has no inherent identity, so the same physical object can switch slots between frames. Temporal consistency requires one or more of:

- previous-slot initialization;
- explicit data association;
- transition prediction followed by observation matching;
- contrastive identity objectives;
- persistent memory with birth, death, and occlusion states.

### 5.2 Birth, death, and absence

An observation agent needs to distinguish:

- object never observed;
- currently visible;
- temporarily occluded or outside FOV;
- disappeared from the world;
- detector missed it;
- duplicate or false track.

A practical slot may contain

\[
z_t^{(k)}=
(e_t^{(k)},c_t^{(k)},x_t^{(k)},v_t^{(k)},q_t^{(k)},u_t^{(k)}),
\]

where (e) is existence, (c) class/identity, (x,v) geometry and motion, (q) visibility/quality, and (u) uncertainty. This factorization is conceptual; the learned state may be continuous and distributed.

### 5.3 Occlusion is belief propagation

When an object becomes unobserved, its state should be predicted rather than deleted:

\[
p(z_{t+1}^{(k)}\mid h_t,a_t)
\]

until new evidence corrects the hypothesis. This is the direct bridge to an object-centric RSSM: recurrent/global memory carries context, per-object stochastic states preserve alternative futures, and observation-conditioned posteriors correct visible slots.

---

## 6. Supervision spectrum

| Approach | Source of object structure | Strength | Risk |
|---|---|---|---|
| Fully supervised | masks, boxes, identities, tracks | Task alignment and stable semantics | Annotation dependence; detector ceiling |
| Detection-seeded | external detector/tracker proposals | Practical integration and known interfaces | Inherits misses, false positives, and taxonomy |
| Weakly supervised | motion, temporal consistency, labels | Less annotation with some alignment | Biased proxy signals |
| Unsupervised slots | reconstruction and competition | Can discover reusable factors | Unstable decomposition; weak semantic alignment |
| Hybrid | detector anchors plus learned latent slots | Operational grounding with latent completion | Interface complexity and confirmation bias |

For this repository, a hybrid route is particularly credible: accept detections when available, but maintain latent hypotheses through misses and allow map-free operation. Object recognition remains an upstream stack; Payload Intelligence owns temporal evidence and observation decisions rather than redefining the detector.

---

## 7. Failure modes and diagnostics

| Failure | Symptom | Diagnostic or mitigation |
|---|---|---|
| Slot fragmentation | One object occupies several slots | Compare slots with instance masks/tracks and controlled occlusion |
| Slot merging | Multiple objects share one slot | Increase capacity; relational/decomposition losses; crowded-scene tests |
| Slot switching | Identity jumps between indices | Association metrics and long-horizon identity consistency |
| Background capture | Slots model texture rather than entities | Task-relevant prediction and motion/interaction supervision |
| Fixed-capacity overflow | Objects disappear when count exceeds (K) | Count-sweep evaluation; dynamic memory or overflow handling |
| Empty-slot instability | Unused slots hallucinate entities | Existence variables and calibrated birth/death processes |
| Ego-motion entanglement | Static objects appear dynamically complex | Condition on pose/gimbal actions; frame-factorization ablation |
| Appearance shortcut | Model predicts pixels but not object dynamics | Counterfactual motion and interaction tests |
| Semantic mismatch | Discovered components do not match mission objects | Detection anchors or downstream task supervision |
| Poor compositional transfer | New object counts/combinations fail | Systematic OOD composition benchmarks |

Object-centric models should not be judged only by reconstruction quality or segmentation ARI. The decisive test is whether the representation improves prediction, evidence acquisition, and decisions under controlled object-level variations.

---

## 8. Implications for Dream-to-Look

### 8.1 Proposed information state

An object-centric Dream-to-Look state can be written

\[
s_t^{\text{OC}}
=\left(
h_t^{\text{global}},
\{(h_t^{(k)},z_t^{(k)})\}_{k=1}^{K}
\right),
\]

where global memory encodes scene, sensor, and platform context while object states encode persistent hypotheses.

The state should support questions such as:

- Which objects are uncertain but mission-relevant?
- Which objects are predicted to leave the field of regard?
- Which pair of hypotheses can a new viewpoint distinguish?
- Which object has gone unseen longer than its predictive uncertainty allows?
- Will zooming on one object sacrifice a more valuable track?

### 8.2 Research hypotheses

1. Object-centric latent dynamics will generalize better than monolithic latents to changed object counts and arrangements.
2. Persistent slots will improve reacquisition after occlusion and field-of-view exit.
3. Per-object uncertainty and relevance will yield better sensing allocation than global uncertainty.
4. Explicit relational dynamics will improve observation decisions when objects interact or occlude one another.
5. Hybrid detector-seeded slots will outperform both raw detection memory and fully unsupervised slots in operationally realistic EO/IR scenes.

### 8.3 Baselines and metrics

Compare frame-wise detections, pooled scene vectors, deterministic recurrent object tracks, monolithic RSSM, and object-centric RSSM.

Measure identity consistency, existence calibration, per-object prediction error, occlusion recovery, data association, slot utilization, compositional OOD transfer, task return, time-to-evidence, and sensing allocation regret.

---

## 9. Durable takeaways

1. Object-centric representation is an inductive bias, not proof that learned slots are real objects.
2. Competitive binding separates observations into exchangeable latent components.
3. Persistent identity, birth/death, occlusion, and relational dynamics are essential for world modeling.
4. Ego-motion and sensor action must be modeled explicitly in airborne observation.
5. Unsupervised decomposition is not automatically aligned with mission semantics.
6. Hybrid detection-seeded latent objects are a practical bridge between existing perception stacks and learned world models.
7. The research value lies in better evidence-seeking decisions and compositional prediction, not prettier segmentation.

---

## 10. Primary references

- Greff, K. et al. (2019). [Multi-Object Representation Learning with Iterative Variational Inference (IODINE)](https://arxiv.org/abs/1903.00450).
- Burgess, C. P. et al. (2019). [MONet: Unsupervised Scene Decomposition and Representation](https://arxiv.org/abs/1901.11390).
- Locatello, F. et al. (2020). [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055).
- Kipf, T. et al. (2022). [Conditional Object-Centric Learning from Video (SAVi)](https://arxiv.org/abs/2111.12594).
- Lin, Z. et al. (2020). [Generative Modeling of Dynamic Visual Scenes (G-SWM)](https://arxiv.org/abs/2002.09405).
- Wu, Z. et al. (2023). [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861).
- Battaglia, P. W. et al. (2018). [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261).

