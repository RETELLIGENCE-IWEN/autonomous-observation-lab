# Foundation Note Writing Guide

## At a Glance

### Purpose

This guide defines how to write durable, human-readable notes in `docs/foundations/`. A foundation note should let an AI researcher return after months away and quickly recover:

- what a concept is;
- which problem it solves;
- how its central mechanism and mathematics work;
- what capability or value it creates;
- what it does **not** guarantee;
- why it matters to autonomous observation research.

Foundation notes are not exhaustive literature surveys, implementation manuals, or paper-by-paper histories. They are compact conceptual references built around enduring ideas.

### Intended reader

Assume a reader who is comfortable with machine learning and mathematical notation, but who may not remember the concept, its notation, or its relationship to this research program.

### What every note should contain

| Part | Question answered |
|---|---|
| Problem and intuition | Why does this concept exist? |
| Mathematical core | What is the smallest precise formulation? |
| Operational mechanism | What happens at inference and learning time? |
| Meaning and value | What new capability does it provide? |
| Limits and failures | What can go wrong or be misunderstood? |
| Research connection | How does it shape our hypotheses and experiments? |
| Primary references | Where should the reader go for authoritative detail? |

---

## 1. Writing Principles

### 1.1 Begin with the problem, not the machinery

Explain the information-processing problem before introducing an architecture, loss, or algorithm. A reader should know why the concept is necessary before seeing how it is implemented.

For example, an RSSM note should first establish partial observability, temporal memory, and predictive state—not begin with GRUs and latent variables.

### 1.2 Separate the enduring concept from one implementation

Architectures change faster than ideas. Identify which statements define the concept and which describe a particular realization. When useful, label variants explicitly rather than allowing one popular implementation to stand for the whole field.

### 1.3 Pair every important equation with meaning

An equation is incomplete until the note explains:

- what each variable represents;
- what dependence or assumption the equation expresses;
- what role it plays in learning or inference;
- what would change if the term were removed or altered.

The goal is not merely to reproduce mathematics from a paper, but to make the mathematics recoverable by thought.

### 1.4 Distinguish capability from guarantee

Use careful language. A learned latent state may *support* belief-like reasoning without being a calibrated Bayesian belief. An uncertainty signal may be useful for action selection without faithfully representing epistemic uncertainty.

Clearly separate:

- what the formulation intends;
- what training encourages;
- what the model can empirically do;
- what is theoretically guaranteed.

### 1.5 Treat limitations as part of the concept

Failure modes are not an appendix added for balance. They reveal the assumptions under which the concept is meaningful. Include representational failures, optimization failures, distribution-shift failures, and common interpretive errors when relevant.

### 1.6 Connect the concept to research decisions

A foundation note should end by changing how we formulate a research problem. The connection to Dream-to-Look or another research track must be concrete enough to suggest states, actions, objectives, baselines, ablations, or evaluation metrics.

---

## 2. Recommended Document Structure

Use this structure as a strong default, not an inflexible template. Merge or expand sections when the concept demands it.

### 2.1 At a Glance

Place this at the beginning. It should include:

- a one-sentence definition;
- why the concept matters;
- what the note covers and deliberately does not cover;
- a short reading map showing the document's major perspectives.

A reader should be able to decide in less than a minute whether the note answers their question.

### 2.2 The problem the concept addresses

Describe the setting, missing capability, and why simpler approaches are insufficient. State assumptions such as partial observability, stochasticity, non-stationarity, or limited sensing resources.

### 2.3 Core intuition

Explain the central idea in plain language before formalization. Use one compact example when it materially improves understanding.

### 2.4 Mathematical formulation

Introduce the minimum notation needed for precision. Present the generative model, state update, objective, decision rule, or other defining mathematics in conceptual order.

### 2.5 Operational process

Explain what happens step by step during inference, learning, or both. Distinguish observed quantities, latent quantities, predictions, actions, and supervision.

### 2.6 Learning and optimization

Explain how the desired behavior is induced. Interpret each important objective term and describe major optimization tensions or degeneracies.

### 2.7 Meaning and value

State what the learned representation or procedure makes possible. Prefer capability statements over architectural descriptions.

### 2.8 What it does not guarantee

Identify tempting but invalid conclusions. This section is mandatory when terminology such as *belief*, *uncertainty*, *causality*, *planning*, or *world model* can overstate what has actually been learned.

### 2.9 Failure modes and diagnostics

For each major failure, connect:

1. the underlying cause;
2. the observable symptom;
3. a diagnostic or experiment;
4. a possible mitigation, if known.

### 2.10 Alternatives and neighboring concepts

Compare only the alternatives needed to establish conceptual boundaries. Use a table when the distinctions involve repeated dimensions such as state representation, uncertainty, learning signal, or computational cost.

### 2.11 Implications for this research program

Translate the concept into the language of autonomous observation. Address the relevant subset of:

- representation: what the agent should remember or model;
- action: what constitutes a look, sensor choice, zoom, dwell, or request;
- objective: what makes one observation more valuable than another;
- uncertainty: what the agent knows and does not know;
- experiment: what hypothesis becomes testable;
- evaluation: which metric would demonstrate value.

### 2.12 Takeaways

End with a short list of durable claims—not a summary of every section.

### 2.13 Primary references

Prefer original papers, authoritative follow-up work, official implementations, and strong surveys used for orientation. References should be selective and annotated when their relevance is not obvious.

---

## 3. Mathematical and Notational Style

- Define notation locally before using it.
- Keep symbols consistent throughout the note.
- Prefer the smallest formulation that preserves the concept.
- Present equations in the order a reader needs to understand them, which may differ from the order in the source paper.
- Explain objectives term by term and identify competing pressures.
- State important modeling assumptions and approximations.
- Avoid long derivations unless the derivation itself reveals the core idea.
- When papers use conflicting notation, choose one system and provide a brief mapping only when needed.

The mathematical section should enable the reader to reconstruct the logic, not merely recognize familiar symbols.

---

## 4. Evidence and Source Policy

Use sources in roughly this order:

1. original or primary research papers;
2. official project pages and code repositories;
3. later primary papers that introduce important variants or corrections;
4. authoritative surveys or textbooks for synthesis and terminology.

Clearly distinguish among:

- claims directly supported by a source;
- widely accepted interpretation;
- our synthesis or inference;
- a research hypothesis proposed for this repository.

Do not inflate a note with loosely related citations. A smaller set of sources that supports the conceptual spine is preferable to a bibliography without narrative purpose.

---

## 5. Presentation Style

- Favor explanatory prose over disconnected bullets.
- Introduce the canonical English term even when adding a Korean explanation later.
- Use tables for exact, repeated comparisons.
- Use diagrams only when structure, information flow, or temporal dependence is materially clearer visually.
- Keep headings descriptive and navigable.
- Avoid unexplained jargon and promotional language.
- Write for rereading: definitions and key caveats should be easy to locate without reading linearly.

Length should follow conceptual need. Compactness means removing low-value material, not omitting the mathematics or criticism required for understanding.

---

## 6. Research-Connection Standard

The research section should go beyond saying that a concept is “useful for Dream-to-Look.” Where applicable, make the mapping explicit:

| Research element | Question |
|---|---|
| Observation | What information enters the agent? |
| Internal state | What must persist across time? |
| Action | What can the payload intelligence control or request? |
| Prediction | What future observation, state, or evidence is modeled? |
| Utility | Why is one look better than another? |
| Uncertainty | Which ignorance should influence action? |
| Baseline | What simpler policy tests whether the concept adds value? |
| Ablation | Which component carries the claimed capability? |
| Metric | What measurable result would validate the hypothesis? |

A good connection section should produce at least one falsifiable research question.

---

## 7. Quality Checklist

### Scope and orientation

- [ ] The opening defines the concept in one sentence.
- [ ] The reader can see what the note covers and excludes.
- [ ] The motivating problem appears before implementation detail.

### Explanation and mathematics

- [ ] The core intuition is understandable without the equations.
- [ ] Every central equation has a verbal interpretation.
- [ ] Assumptions, approximations, and symbols are explicit.
- [ ] The concept is separated from a particular architecture or paper.

### Critical analysis

- [ ] Capabilities are not presented as guarantees.
- [ ] Major failure modes and misleading interpretations are addressed.
- [ ] Neighboring concepts are distinguished where confusion is likely.

### Research relevance

- [ ] The note maps the concept to concrete research decisions.
- [ ] At least one testable hypothesis, baseline, ablation, or metric is suggested.
- [ ] The connection does not assume one fixed perception, map, or flight stack unless explicitly scoped.

### Sources and rereadability

- [ ] Primary and authoritative sources support the conceptual spine.
- [ ] Our inferences and proposals are identifiable as such.
- [ ] Definitions, caveats, and takeaways are easy to find later.

---

## 8. Suggested Writing Workflow

1. State the reader's recovery goal: what should they remember after reading?
2. Collect the small set of primary sources that define the concept.
3. Write the problem, one-sentence definition, and conceptual boundaries.
4. Establish one consistent notation system.
5. Write the mathematical and operational core together.
6. Add limitations and diagnostics before writing research implications.
7. Translate the concept into falsifiable autonomous-observation questions.
8. Perform a rereadability pass using the checklist above.

---

## 9. Recommended Sequence After RSSM

For the **Dream-to-Look with an Object-Centric RSSM** direction, the recommended sequence is:

1. **POMDPs and Belief States** — formalize hidden state, observation history, belief, action, and expected return.
2. **Active Sensing and Value of Information** — formalize why a particular look is worth taking.
3. **Object-Centric Representations and World Models** — decide what entities and relations the latent state should preserve.
4. **Uncertainty Estimation for Learned World Models** — distinguish actionable ignorance from ordinary predictive noise.
5. **Model-Based RL and Latent Imagination** — learn observation policies by evaluating possible looks in imagined futures.

This order moves from the formal problem, to the value of observation, to representation, trustworthy prediction, and finally policy learning. It does not prescribe a final architecture; it builds the conceptual tools needed to judge one.

