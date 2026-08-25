# Active Sensing and Value of Information

## At a Glance

> Active sensing is the deliberate selection of sensor actions to acquire evidence that improves a task-relevant belief or decision; Value of Information (VoI) measures whether that evidence is worth its acquisition cost.

### Why this concept matters

Passive perception asks what can be inferred from available measurements. Active sensing additionally asks what measurement should be acquired next. For Dream-to-Look, pan, tilt, zoom, modality, dwell, revisit, and occasional flight requests are not mere camera controls: they determine which uncertainty can be resolved, at what cost, and before which deadline.

### This note covers

- the active perception loop;
- expected information gain, Bayesian experimental design, and decision-theoretic VoI;
- why entropy reduction and mission value are different;
- myopic and non-myopic sensing;
- costs, stopping, observability, and failure modes;
- a concrete Dream-to-Look research formulation.

### Reading map

| Perspective | Main question |
|---|---|
| Active perception | How does action control data acquisition? |
| Information gain | Which action is expected to reduce uncertainty? |
| VoI | Will the new evidence improve a decision enough to justify its cost? |
| Planning | Should the observer optimize one look or a sequence? |
| Dream-to-Look | How should autonomous payload actions be valued? |

---

## 1. From passive perception to active sensing

Bajcsy's active perception view treats sensing as a controlled, goal-directed process. The agent repeatedly:

1. maintains a belief (b_t(x)=p(x\mid h_t)) about a hidden quantity (x);
2. chooses a sensing action (a_t);
3. receives (y_{t+1}\sim p(y\mid x,a_t));
4. updates (b_{t+1});
5. continues, stops, or takes a downstream task action.

The action-dependent likelihood

\[
p(y\mid x,a)
\]

is the mathematical core. A look changes field of view, resolution, occlusion geometry, noise, or modality, thereby changing which evidence is likely to arrive.

An **active sensor** in the hardware sense emits energy, such as radar or lidar. **Active sensing** here instead means actively controlling acquisition; a passive EO camera on a controllable gimbal qualifies.

---

## 2. Expected information gain

### 2.1 Entropy reduction

For belief (b(x)), entropy is

\[
H(X\mid b)=-\sum_x b(x)\log b(x).
\]

After action (a) and possible observation (y), the posterior is (b^{a,y}). Expected information gain is

\[
\operatorname{EIG}(a;b)
=H(X\mid b)
-\mathbb E_{y\sim p(\cdot\mid b,a)}
\left[H(X\mid b^{a,y})\right].
\]

Equivalently,

\[
\operatorname{EIG}(a;b)
=I(X;Y\mid a,b)
=\mathbb E_y
\left[D_{\mathrm{KL}}(b^{a,y}\Vert b)\right].
\]

These forms say the same thing: prefer measurements expected to move the posterior away from the prior and reduce uncertainty about (X).

### 2.2 The target variable determines the meaning

Information gain is only meaningful after choosing what (X) represents. It could be:

- object existence or identity;
- target position or velocity;
- association between detections and tracks;
- occlusion or visibility state;
- a mission hypothesis, such as “this object is the designated target.”

Reducing uncertainty about every pixel or nuisance variable can distract from mission evidence. Task-relevant latent variables should therefore be explicit.

### 2.3 Other uncertainty objectives

Common acquisition scores include posterior variance, expected KL divergence, mutual information, Fisher information, prediction entropy, and expected model disagreement. They are not interchangeable. Each assumes a particular belief representation and encodes a different notion of uncertainty.

---

## 3. Decision-theoretic Value of Information

### 3.1 Perfect information

Let (d\in\mathcal D) be a downstream decision and (U(d,x)) its utility. Without new evidence, the best expected utility is

\[
V(b)=\max_d\mathbb E_{x\sim b}[U(d,x)].
\]

The expected value of perfect information is

\[
\operatorname{EVPI}(b)
=\mathbb E_{x\sim b}\left[\max_d U(d,x)\right]
-\max_d\mathbb E_{x\sim b}[U(d,x)].
\]

EVPI is an upper bound: it asks how valuable it would be to know (x) exactly.

### 3.2 Sample information from a sensing action

A real look produces noisy evidence rather than perfect truth. Its expected value is

\[
\operatorname{VoI}(a;b)
=\mathbb E_{y\sim p(\cdot\mid b,a)}
\left[V(b^{a,y})\right]
-V(b)-C(a),
\]

where (C(a)) includes time, energy, motion, lost coverage, blur, risk, or coordination cost.

Acquire the observation when its expected improvement in downstream decisions exceeds its cost. Under consistent Bayesian decision theory and zero acquisition cost, expected sample information is nonnegative because the decision-maker can ignore unhelpful evidence. Approximate models and policies can violate this practical guarantee.

### 3.3 Why VoI differs from information gain

| Criterion | Values | Blind spot |
|---|---|---|
| Entropy reduction | Reduction in uncertainty about (X) | May learn irrelevant facts |
| Mutual information | Expected dependence between hidden state and observation | Does not encode decision consequences |
| Expected confidence | Sharper classifier/tracker output | Can reward miscalibration |
| Decision-theoretic VoI | Expected improvement in optimal task utility | Requires a utility model and downstream decision model |

A tiny belief change across a release/no-release or identify/ignore boundary can have high VoI. A large entropy reduction about an irrelevant object can have nearly zero VoI.

---

## 4. Sequential active sensing

### 4.1 Myopic selection

A one-step acquisition policy is

\[
a_t^*=\arg\max_a
\left[
\operatorname{EIG}(a;b_t)-\lambda C(a)
\right]
\]

or its decision-theoretic VoI counterpart. It is cheap and often useful, but can reject an individually weak look that enables a valuable later look.

### 4.2 Non-myopic selection

In a POMDP, sensing actions are valued through future beliefs:

\[
Q(b,a)=r(b,a)+\gamma
\mathbb E_y[V(\tau(b,a,y))].
\]

This supports sequences such as wide search → tentative detection → narrow zoom → track confirmation. The early action may be valuable mainly because it creates a better branch for later acquisition.

### 4.3 Receding-horizon sensing

Full belief-space planning is often infeasible. A practical compromise is:

1. imagine candidate sensing sequences for a short horizon;
2. predict evidence, belief change, utility, and cost;
3. execute only the first action;
4. replan after the real observation.

This is active-sensing model predictive control. Latent imagination can make it tractable, but model error becomes a central risk.

### 4.4 Stopping

An observation agent also needs a **stop/commit/return-to-search** decision. Continue sensing only while expected marginal VoI is positive:

\[
\max_a \operatorname{VoI}(a;b_t)>0.
\]

Without explicit stopping or resource cost, an uncertainty-seeking agent may inspect forever.

---

## 5. Observability, attention, and exploration

| Concept | Main object | Typical question |
|---|---|---|
| Active sensing | Physical or sensor acquisition action | Where/how should I measure? |
| Attention | Allocation of computational or representational capacity | Which available information should I process? |
| Exploration | Action for learning rewards, dynamics, or policies | What experience should I collect? |
| Active learning | Selection of examples to label | Which annotation is worth purchasing? |
| Bayesian experimental design | Selection of an experiment | Which experiment best informs parameters/hypotheses? |

These can overlap. Zoom is sensing; selecting one detected object for deeper processing is attention; visiting an unfamiliar region may be both active sensing and exploration.

Active sensing cannot solve structural unobservability. If every feasible action yields the same observation distribution under two critical hypotheses,

\[
p(y\mid x_1,a)=p(y\mid x_2,a)
\quad\forall a,
\]

then no sensing policy can distinguish them without a new modality, viewpoint authority, prior, or external information source.

---

## 6. Failure modes and diagnostics

| Failure | Symptom | Diagnostic or mitigation |
|---|---|---|
| Irrelevant curiosity | Agent inspects visually complex but unimportant regions | Measure downstream decision improvement, not global entropy alone |
| Confidence hacking | Zoom/dwell inflates confidence without accuracy | Calibration and counterfactual accuracy tests |
| Myopic trap | Greedy look misses useful multi-step acquisition sequence | Compare one-step and short-horizon planning |
| Cost omission | Excessive motion, zoom chatter, or flight requests | Explicit action, switching, dwell, and opportunity costs |
| Model exploitation | Planner selects looks where learned sensor model is falsely optimistic | Ensemble disagreement, conservative value, real-observation replanning |
| Belief collapse | Agent stops searching after premature hypothesis commitment | Maintain multimodality; test ambiguous histories |
| Redundant looks | Repeated observations provide little new evidence | Conditional rather than unconditional information gain |
| No stopping rule | Agent gathers evidence indefinitely | Stop/commit action and marginal-VoI criterion |
| Delayed-value credit failure | Useful early search actions appear unrewarded | Longer-horizon return, model-based backup, or outcome-conditioned supervision |

---

## 7. Implications for Dream-to-Look

### 7.1 Action space

The minimal payload action should express controllable acquisition variables rather than project-specific modes:

\[
a_t^P=(\text{view direction},\text{FOV/zoom},\text{modality},\text{dwell}).
\]

Optional flight cooperation is a priced request:

\[
a_t^F=(\text{desired viewpoint or visibility constraint},\text{priority},\text{validity horizon}).
\]

This keeps Payload Intelligence independent of the specific flight backend while allowing cooperation when the gimbal cannot create the necessary geometry.

### 7.2 A task-relevant acquisition objective

A useful conceptual objective is

\[
a_t^*=\arg\max_a
\mathbb E
\left[
\Delta U_{\text{evidence}}(b_t,a)
-\lambda_T C_{\text{time}}
-\lambda_M C_{\text{motion}}
-\lambda_R C_{\text{request}}
-\lambda_C C_{\text{lost coverage}}
\right].
\]

The central research problem is learning or approximating (Delta U_{\text{evidence}}): the expected improvement in mission-relevant decisions caused by a look.

### 7.3 Falsifiable hypotheses

1. Decision-aware VoI will outperform entropy-only acquisition when distractor uncertainty is high.
2. Short-horizon acquisition planning will outperform greedy next-best-view under staged evidence tasks.
3. Explicit sensing costs and stopping will reduce redundant looks without reducing decision accuracy.
4. A priced flight-request channel will be used sparsely and primarily when payload-only observability is insufficient.

### 7.4 Baselines and metrics

Baselines should include fixed scan, detection-centering, confidence-greedy zoom, entropy-greedy acquisition, one-step VoI, and short-horizon VoI.

Measure task decision accuracy, time-to-sufficient-evidence, cumulative sensing cost, redundant-look rate, calibration, coverage loss, request frequency, and regret relative to an oracle with privileged state.

---

## 8. Durable takeaways

1. Active sensing controls the evidence process, not merely the sensor pose.
2. Expected information gain measures belief reduction; decision-theoretic VoI measures expected improvement in action utility.
3. High information gain can have low mission value, and a small belief change can have high decision value.
4. Sequential sensing requires accounting for actions that enable later observations.
5. Costs and stopping are part of the problem, not implementation details.
6. Active sensing cannot recover information absent from every feasible observation channel.
7. Dream-to-Look should optimize evidence value under sensing and cooperation budgets, not camera centering alone.

---

## 9. Primary references

- Bajcsy, R. (1988). [Active Perception](https://doi.org/10.1109/5.5968). Foundational formulation of perception as controlled data acquisition.
- Aloimonos, J., Weiss, I., & Bandyopadhyay, A. (1988). [Active Vision](https://doi.org/10.1007/BF00133571). Early active-vision perspective connecting visual problems with controlled sensing.
- Lindley, D. V. (1956). [On a Measure of the Information Provided by an Experiment](https://doi.org/10.1214/aoms/1177728069). Foundational expected-information formulation for experimental design.
- Howard, R. A. (1966). [Information Value Theory](https://ieeexplore.ieee.org/document/4082121). Decision-theoretic treatment of the value of information.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). [Planning and Acting in Partially Observable Stochastic Domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf). Belief-space planning and implicit information value in POMDPs.
- Bajcsy, R., Aloimonos, Y., & Tsotsos, J. K. (2018). [Revisiting Active Perception](https://doi.org/10.1007/s11263-017-1071-7). Retrospective synthesis of the field and its enduring principles.

