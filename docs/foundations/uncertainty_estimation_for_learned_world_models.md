# Uncertainty Estimation for Learned World Models

## At a Glance

> Uncertainty estimation asks not only what a learned world model predicts, but how strongly that prediction should be trusted, why it is uncertain, and whether the uncertainty is actionable through further observation.

### Why this concept matters

A Dream-to-Look agent should seek evidence because it recognizes consequential ignorance. A stochastic latent alone does not provide this capability. The system must distinguish ambiguous observations, unpredictable dynamics, insufficient data, distribution shift, and disagreement about long-horizon futures—and must connect those distinctions to safe observation actions.

### This note covers

- predictive, aleatoric, epistemic, state, and distributional uncertainty;
- probabilistic prediction and variance decomposition;
- ensembles, Bayesian approximations, latent distributions, and calibration;
- uncertainty propagation through imagined rollouts;
- OOD limits and failure modes;
- actionable uncertainty for Dream-to-Look.

### Reading map

| Perspective | Main question |
|---|---|
| Taxonomy | What kind of uncertainty is being represented? |
| Estimation | How can a neural world model express it? |
| Calibration | Do probabilities match empirical outcomes? |
| Rollout | How does uncertainty compound through imagination? |
| Actionability | Can a look reduce the uncertainty that matters? |

---

## 1. Uncertainty is not one scalar

### 1.1 Predictive uncertainty

Given dataset (D), input (x), and output (y), Bayesian predictive uncertainty is expressed by

\[
p(y\mid x,D)
=\int p(y\mid x,\theta)p(\theta\mid D)\,d\theta.
\]

The conditional model (p(y\mid x,\theta)) represents outcome variability under one model; the posterior (p(\theta\mid D)) represents uncertainty over plausible models.

### 1.2 Aleatoric uncertainty

Aleatoric uncertainty arises from irreducible or currently unresolved variability in the data-generating process: sensor noise, stochastic motion, ambiguous pixels, or genuinely multimodal futures.

It may be:

- **homoscedastic**, roughly constant across inputs;
- **heteroscedastic**, dependent on range, blur, weather, modality, or viewpoint.

A Gaussian predictor may output

\[
p_\theta(y\mid x)=
\mathcal N(\mu_\theta(x),\sigma_\theta^2(x)),
\]

and train by negative log likelihood:

\[
\mathcal L_{\mathrm{NLL}}
=\frac{(y-\mu_\theta(x))^2}{2\sigma_\theta^2(x)}
+\frac12\log\sigma_\theta^2(x)+C.
\]

The first term rewards fit relative to predicted noise; the log-variance term prevents unlimited variance inflation.

### 1.3 Epistemic uncertainty

Epistemic uncertainty reflects lack of knowledge about the model: insufficient coverage, ambiguous parameters, or extrapolation beyond training data. In principle, it can decrease with informative data.

This distinction is useful but not absolute. What looks aleatoric under one model class can become explainable after adding hidden variables, temporal context, or a better sensor. The taxonomy depends on what the model represents and conditions upon.

### 1.4 State uncertainty versus model uncertainty

In a POMDP, the agent may be uncertain about current hidden state even with known dynamics and sensor models:

\[
b_t(s)=p(s_t=s\mid h_t).
\]

This **state uncertainty** differs from uncertainty about (T), (Z), or learned parameters. Dream-to-Look may simultaneously face:

- uncertain object state because it is occluded;
- ambiguous future motion even under the correct model;
- epistemic disagreement because such scenes were rare in training;
- detector uncertainty because the observation stack is imperfect.

Combining them into one “confidence” can produce the wrong sensing action.

---

## 2. Predictive variance decomposition

For regression under a posterior over parameters, the law of total variance gives

\[
\operatorname{Var}(Y\mid x,D)
=\mathbb E_{\theta\mid D}
\left[\operatorname{Var}(Y\mid x,\theta)\right]
+\operatorname{Var}_{\theta\mid D}
\left(\mathbb E[Y\mid x,\theta]\right).
\]

The first term is commonly interpreted as aleatoric uncertainty; the second as epistemic uncertainty.

For an ensemble of (M) Gaussian predictors with means (mu_m) and variances (sigma_m^2),

\[
\bar\mu=\frac1M\sum_{m=1}^M\mu_m,
\]

\[
\widehat{\operatorname{Var}}(Y)
=\underbrace{\frac1M\sum_m\sigma_m^2}_{\text{within-model}}
+\underbrace{\frac1M\sum_m(\mu_m-\bar\mu)^2}_{\text{between-model}}.
\]

This decomposition is operationally useful, but an ensemble is not an exact posterior. Shared architecture, data, optimization bias, and correlated errors can make all members confidently wrong.

---

## 3. Estimation methods

### 3.1 Probabilistic output heads

The model predicts a distribution rather than a point. Options include Gaussian, categorical, mixture density, quantile, energy-based, or diffusion-style distributions. This primarily captures conditional outcome uncertainty, subject to the expressiveness and training objective of the chosen family.

### 3.2 Deep ensembles

Train independently initialized models, often with shuffled or bootstrapped data:

\[
\{p_{\theta_m}(y\mid x)\}_{m=1}^M.
\]

Ensembles are simple, strong, and parallelizable. They provide disagreement signals useful for model-based control, but multiply compute and memory and may underestimate uncertainty under shared blind spots.

### 3.3 Approximate Bayesian neural networks

Variational inference approximates (p(\theta\mid D)) with (q_\phi(\theta)), commonly optimizing

\[
\mathcal L
=-\mathbb E_{q_\phi(\theta)}[\log p(D\mid\theta)]
+D_{\mathrm{KL}}(q_\phi(\theta)\Vert p(\theta)).
\]

MC dropout interprets stochastic dropout passes as approximate Bayesian inference. These methods can be cheaper than full ensembles, but approximation quality depends strongly on assumptions and tuning.

### 3.4 Evidential methods

Evidential networks predict parameters of a higher-order distribution over output distributions. They offer single-pass uncertainty but can produce misleading evidence without careful regularization and OOD validation. “Evidence” is a parameterization, not proof of knowledge.

### 3.5 Latent stochastic world models

RSSMs and related models predict stochastic latent states:

\[
p_\theta(z_{t+1}\mid h_{t+1}),
\qquad
q_\theta(z_{t+1}\mid h_{t+1},o_{t+1}).
\]

Sampling these latents represents multiple possible trajectories within the learned model. It does not by itself separate aleatoric and epistemic uncertainty or detect model misspecification. Combining stochastic latents with model ensembles is one practical route to represent both trajectory variability and model disagreement.

### 3.6 Conformal prediction

Conformal methods can construct prediction sets with finite-sample marginal coverage under exchangeability assumptions. They are valuable for validating output coverage, but standard guarantees may fail under sequential distribution shift and do not directly produce causal explanations of uncertainty.

---

## 4. Calibration and sharpness

### 4.1 Calibration

A probabilistic classifier is calibrated when events assigned probability (p) occur approximately fraction (p) of the time. Calibration can be summarized by reliability diagrams, Expected Calibration Error (ECE), Brier score, NLL, and class-conditional variants.

For regression, evaluate prediction-interval coverage:

\[
P(Y\in I_{1-\alpha}(X))\approx 1-\alpha,
\]

alongside interval width.

### 4.2 Sharpness

A model can achieve coverage by predicting intervals so wide that they are useless. Good forecasts should be calibrated **and** sharp: as concentrated as possible while maintaining correct coverage.

### 4.3 Calibration is conditional on a distribution

A model calibrated on an average test set can be badly miscalibrated at long range, under blur, in IR, or after an occlusion. Report calibration across operational slices and rollout horizons, not only one global number.

Post-hoc temperature scaling can improve in-distribution classification calibration, but does not create epistemic awareness or guarantee OOD reliability.

---

## 5. Uncertainty in imagined rollouts

### 5.1 Compounding error

A learned transition

\[
\hat s_{t+1}=f_\theta(\hat s_t,a_t)
\]

is recursively fed its own predictions. Small one-step errors can move rollouts off the data manifold, where later predictions become less reliable. Long-horizon mean accuracy can hide multimodal divergence.

### 5.2 Propagation methods

| Method | Mechanism | Main trade-off |
|---|---|---|
| Deterministic mean rollout | Propagate one expected state | Cheap; collapses multimodality |
| Particle rollout | Sample multiple latent trajectories | Captures distributions; expensive |
| Ensemble trajectories | Roll out multiple models | Exposes disagreement; correlated blind spots |
| Moment propagation | Approximate mean/covariance analytically | Efficient only under restrictive approximations |
| Short receding horizon | Limit rollout and replan from reality | Reduces exploitation; may miss long-term value |

PETS combines probabilistic ensembles with trajectory sampling to capture both model uncertainty and stochastic dynamics during MPC.

### 5.3 Policy exploitation of uncertainty

A planner searches for high predicted return and can discover model errors more effectively than random validation data. This creates **model exploitation**: imagined trajectories look favorable precisely because they are unsupported.

Mitigations include uncertainty penalties,

\[
\tilde r(s,a)=\hat r(s,a)-\lambda u(s,a),
\]

conservative value estimates, ensemble disagreement constraints, short rollouts, support constraints, pessimistic planning, and frequent replanning from real observations.

Optimism can instead use uncertainty for exploration. Whether uncertainty should be a bonus or penalty depends on authorization, safety, data-collection phase, and the cost of failure.

---

## 6. OOD and unknown unknowns

Uncertainty estimation does not guarantee detection of every out-of-distribution input. Neural models can be confidently wrong far from training support; ensembles may agree because they share data and inductive biases.

Useful safeguards include:

- held-out shift suites over weather, sensor, detector, terrain, geometry, and object composition;
- feature- and density-based OOD signals;
- ensemble disagreement;
- prediction residual monitoring after real observations arrive;
- abstain, fallback, or request-more-evidence actions;
- explicit validity domains and operating envelopes.

The strongest operational question is not “does uncertainty rise on average?” but:

> Does the uncertainty signal reliably trigger a safer or more informative action before a consequential prediction failure?

---

## 7. Failure modes and diagnostics

| Failure | Symptom | Diagnostic or mitigation |
|---|---|---|
| Variance inflation | High likelihood with uselessly wide predictions | Sharpness and task utility alongside NLL |
| Variance collapse | Confident errors | Coverage, reliability, and adversarial/shift tests |
| Ensemble agreement bias | All members fail identically | Diversity interventions and structurally different models |
| Latent stochasticity misread as epistemic | Random samples interpreted as model ignorance | Controlled repeated-state data and ensemble decomposition |
| Horizon blindness | One-step calibration but poor rollout calibration | Error/coverage versus horizon curves |
| Aggregated calibration hides slices | Good global ECE, bad IR or long-range behavior | Conditional reliability by regime |
| Uncertainty reward hacking | Agent seeks inherently noisy scenes forever | Reducibility and task relevance tests |
| Uncertainty aversion | Agent never observes novel but important regions | Separate safe exploration policy and risk budget |
| Detector confidence leakage | World model treats upstream score as truth | Calibrate detector jointly and test detector swaps |

---

## 8. Implications for Dream-to-Look

### 8.1 Actionable uncertainty

Dream-to-Look should distinguish at least three questions:

1. **What is uncertain?** Object identity, existence, motion, visibility, association, or world-model prediction.
2. **Why is it uncertain?** Sensor noise, missing viewpoint, future stochasticity, insufficient training data, or OOD input.
3. **Can an allowed action reduce it?** Zoom, modality switch, revisit, dwell, or flight request.

Define reducible, task-relevant uncertainty conceptually as

\[
u_{\text{actionable}}(b,a)
=\mathbb E
\left[
L_{\text{decision}}(b)
-L_{\text{decision}}(b^{a,Y})
\right].
\]

This is closely related to VoI. It prevents the system from treating all predictive variance as a reason to look.

### 8.2 Proposed uncertainty channels

| Channel | Possible estimator | Possible response |
|---|---|---|
| Object-state ambiguity | posterior covariance/particles | revisit or change viewpoint |
| Observation noise | heteroscedastic likelihood | dwell, zoom, or switch modality |
| Dynamics ambiguity | multimodal latent rollout | monitor multiple hypotheses |
| Model epistemic uncertainty | ensemble disagreement | shorten horizon, gather data, fallback |
| OOD suspicion | density/feature/residual signal | abstain or request assistance |
| Decision sensitivity | counterfactual value spread | prioritize evidence near decision boundary |

### 8.3 Research hypotheses

1. Decomposed uncertainty will choose better sensing actions than a single confidence scalar.
2. Ensemble-plus-stochastic-latent models will better predict rollout failure than either alone.
3. Horizon-calibrated uncertainty will reduce model exploitation during latent look planning.
4. Object-level uncertainty will allocate sensing more efficiently than global scene uncertainty.
5. Decision-sensitive uncertainty will outperform entropy as a trigger for zoom, revisit, and flight requests.

### 8.4 Evaluation

Measure NLL, Brier score, calibration error, interval coverage and width, selective risk, OOD AUROC with caution, error–uncertainty rank correlation, rollout calibration by horizon, failure-prediction lead time, VoI regret, and downstream task return under sensing budgets.

The decisive ablation is behavioral: remove or corrupt uncertainty while holding mean predictions constant, then measure whether observation choices degrade.

---

## 9. Durable takeaways

1. Predictive uncertainty combines several causes that should not automatically share one response.
2. Aleatoric/epistemic is a useful decomposition, but depends on the model and represented variables.
3. Stochastic latent states are not automatically calibrated beliefs or epistemic uncertainty.
4. Ensembles are strong practical estimators, not exact Bayesian posteriors.
5. Calibration must be evaluated by regime and rollout horizon, together with sharpness.
6. Planners actively exploit world-model errors; uncertainty must influence imagination and fallback behavior.
7. Dream-to-Look needs task-relevant, reducible, actionable uncertainty—not uncertainty for its own sake.

---

## 10. Primary references

- Kendall, A., & Gal, Y. (2017). [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977).
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474).
- Gal, Y., & Ghahramani, Z. (2016). [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142).
- Guo, C. et al. (2017). [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
- Chua, K. et al. (2018). [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)](https://arxiv.org/abs/1805.12114).
- Ovadia, Y. et al. (2019). [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty under Dataset Shift](https://arxiv.org/abs/1906.02530).
- Angelopoulos, A. N., & Bates, S. (2022). [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).

