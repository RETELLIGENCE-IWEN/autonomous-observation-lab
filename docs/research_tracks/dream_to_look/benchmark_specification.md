# Benchmark Specification: Staged Evidence Acquisition

## 1. Purpose

This benchmark isolates autonomous observation intelligence from image recognition and low-level flight control. It tests whether an agent can allocate gaze across multiple objects, accumulate identity evidence, maintain hypotheses through interrupted observation, reacquire the relevant object, and stop when sufficient evidence has been obtained.

It must be simple enough for causal analysis and rich enough that reactive centering, fixed scanning, and generic uncertainty reduction are insufficient.

---

## 2. Episode narrative

Each episode contains \(N\) moving objects, one target or optionally no target, a controllable sensor, a limited horizon, and a finite sensing budget.

### Stage 1 — Candidate discovery and triage

A wide view exposes noisy detections of several candidates. Coarse motion and appearance evidence are available, but target identity cannot normally be resolved immediately.

### Stage 2 — Disambiguation under competition

Narrow views or dwell provide stronger identity evidence for one object while reducing coverage of others. Distractors include visually uncertain but mission-irrelevant objects so entropy reduction and decision value lead to different choices.

### Stage 3 — Interruption and persistence

The likely target becomes occluded, leaves the FOV, or experiences detector dropout. Other objects compete for observation. The agent must propagate rather than discard the hypothesis.

### Stage 4 — Reacquisition and commitment

The object may reappear in one of several regions. The agent selects a reacquisition look, gathers final evidence, and commits to a target, declares absence, or abstains before the budget expires.

Stages are latent regimes, not labels exposed to the policy.

---

## 3. Hidden state

The simulator state is

\[
s_t=(X_t,G_t,M_t,B_t),
\]

where \(X_t\) is the object set, \(G_t\) sensor state, \(M_t\) environment and occlusion state, and \(B_t\) remaining time and sensing budget.

An object state contains

\[
x_t^{(i)}=
(\iota_i,c_i,q_i,p_t^{(i)},v_t^{(i)},\xi_t^{(i)},m_t^{(i)}),
\]

where \(\iota_i\) is privileged identity, \(c_i\) latent attributes, \(q_i\) the target predicate, \(p,v\) kinematics, \(\xi\) visibility, and \(m\) motion mode. Privileged identity and target label are never actor observations.

---

## 4. Mission predicate and evidence

Target identity requires multiple factors:

\[
q_i=\mathbf 1
\left[c_i^{\text{appearance}}=c^*
\land c_i^{\text{motion}}=m^*\right].
\]

Wide views reveal motion relatively well but appearance poorly. Zoom improves appearance evidence while reducing coverage. Dwell reduces noise but consumes time.

Scenario variants include appearance-only and motion-only distractors, high-uncertainty irrelevant objects, target-absent episodes, and several plausible candidates until late evidence arrives.

The generator must prevent bbox size, array order, handle, or spawn pattern from leaking the target label.

---

## 5. Observation space

The policy receives a variable-length detection set

\[
O_t=\{o_t^{(j)}\}_{j=1}^{D_t}
\]

and payload state \(g_t\).

\[
o_t^{(j)}=
[u,v,w,h,\rho,e_{1:d},\kappa,\ell,\delta].
\]

| Field | Meaning |
|---|---|
| \(u,v,w,h\) | normalized bbox center and size |
| \(\rho\) | detector confidence |
| \(e_{1:d}\) | noisy appearance-evidence vector |
| \(\kappa\) | visibility or measurement quality |
| \(\ell\) | optional noisy track-handle embedding |
| \(\delta\) | time since that handle was observed |

The handle is not ground-truth identity. It may reset, collide, or switch under configurable association noise.

Payload state contains view center, zoom/FOV, slew and dwell state, and remaining budget.

Separately configurable imperfections include missed and false detections, bbox noise, confidence miscalibration, missing appearance dimensions, association corruption, latency, stale observations, and later modality-specific evidence.

---

## 6. Action space

The first benchmark uses a discrete, object-aware but non-privileged interface.

Observation actions:

- WIDE_SCAN(sector)
- LOOK_AT(handle_or_slot, zoom)
- DWELL(handle_or_slot, duration)
- LOOK_AT_REGION(region, zoom)
- HOLD

Decision actions:

- COMMIT(handle_or_slot)
- DECLARE_ABSENT
- ABSTAIN

A vanished handle can be referenced only through agent memory and a predicted region. The environment must not resolve it using privileged identity for the actor.

After scientific separation is established, actions can become continuous pan, tilt, zoom, and dwell with explicit slew dynamics.

---

## 7. Dynamics and evidence

Object motion begins with simple but multimodal dynamics: constant velocity, controlled mode changes, crossing paths, and bounded maneuver alternatives. Occlusion can be scheduled or geometry-dependent.

Evidence quality is action-dependent:

\[
\sigma_{\text{appearance}}
=f(\text{zoom},\text{dwell},\text{visibility},\text{slew}).
\]

This causal dependency gives observation actions value. Repeated measurements should be conditionally correlated so dwell is not equivalent to unlimited independent samples.

---

## 8. Reward and utility

Use sparse decision utility plus explicit costs. Never reward hidden-target centering or privileged belief distance.

| Outcome | Initial utility |
|---|---:|
| Correct target commit | \(+1.00\) |
| Correct absence declaration | \(+1.00\) |
| Wrong commit or declaration | \(-1.00\) |
| Abstention | \(-0.15\) |
| Timeout without decision | \(-0.40\) |

\[
C(a_t)=
\lambda_\tau\Delta t+
\lambda_s C_{\text{slew}}+
\lambda_z C_{\text{zoom}}+
\lambda_d C_{\text{dwell}}+
\lambda_w C_{\text{lost coverage}}.
\]

Costs must produce a measurable accuracy–efficiency frontier. Privileged simulator targets may supervise diagnostic world-model heads, but must not enter actor observations or decision reward shaping.

---

## 9. Scenario generator

Training randomization:

- 3–6 objects;
- positions, velocities, and motion modes;
- target presence and identity combinations;
- distractor ambiguity;
- occlusion onset, duration, and reappearance region;
- detector and association noise;
- time/look budget;
- zoom-quality versus coverage trade-off.

Held-out axes:

- 7–10 objects;
- longer occlusions;
- unseen attribute combinations;
- shifted detector calibration;
- new motion modes;
- altered FOV/zoom;
- reduced budget;
- swapped normalized detector adapter.

Use non-overlapping deterministic seed blocks for training, validation/checkpoint selection, development evaluation, and a one-time untouched holdout. Store generator version and config hash with every artifact.

---

## 10. Oracle and sanity policies

Implement before learned models:

- random valid action;
- fixed cyclic scan;
- confidence-greedy look;
- entropy-greedy look using privileged generative probabilities;
- one-step decision-aware VoI oracle;
- short-horizon privileged planner;
- full-state target oracle upper bound.

The benchmark is not ready until a controlled case makes entropy-greedy inspect an irrelevant object while decision-aware VoI selects evidence capable of changing the mission decision.

---

## 11. Learned-model interface

World-model inputs are the detection set, payload state, previous action, and persistent object/global latent state.

Required prediction heads:

- object existence and association;
- position and motion distribution;
- visibility and detection probability;
- appearance-evidence distribution;
- target-decision evidence;
- reward, continuation, and budget consistency.

The actor receives only learned latent state and validity masks. It never receives ground-truth stage, identity, target label, or future occlusion schedule. The primary result should avoid privileged critic input; a privileged critic may appear only as a labeled optimization ablation.

---

## 12. Evaluation

Primary metrics:

- expected terminal utility;
- correct decision, wrong-commit, and abstention rates;
- sensing cost per correct decision;
- time or looks to sufficient evidence;
- target reacquisition rate and latency.

Diagnostic metrics:

- redundant-look and irrelevant-object observation rates;
- identity and existence calibration;
- belief accuracy during occlusion;
- visibility/evidence prediction NLL;
- open-loop error versus horizon;
- imagined versus realized action ranking;
- slot identity consistency;
- imagined-to-realized return gap.

Report confidence intervals, lower-tail performance, and paired-seed differences. Use identical unseen scenarios across policies and slice by target presence, ambiguity, occlusion duration, object count, detector quality, and budget.

---

## 13. Acceptance gates

### Gate 1 — Environment

- deterministic replay;
- no target leakage;
- intended oracle ordering;
- entropy and decision-aware VoI diverge in constructed cases.

### Gate 2 — World model

- filtering beats latest-observation prediction;
- the prior retains target hypotheses through occlusion;
- evidence quality is calibrated by action and horizon;
- slot identity remains useful through handle corruption.

### Gate 3 — Policy

- proposed policy beats fixed, reactive, entropy-greedy, and recurrent baselines;
- improvement remains at equal cost or on a superior Pareto frontier;
- multi-step advantage appears specifically in staged evidence;
- advantage survives untouched holdout seeds and at least two shifts.

---

## 14. Implementation order

1. deterministic simulator core and seeded generator;
2. observation/action adapters;
3. scripted policies and trajectory logger;
4. oracle belief and VoI evaluator;
5. benchmark validation suite;
6. replay dataset and set encoder;
7. deterministic recurrent baseline;
8. monolithic RSSM baseline;
9. object-centric RSSM;
10. imagination actor–critic;
11. ablations, shifts, and final holdout.

This order makes the scientific question testable before the most complex model is introduced.

