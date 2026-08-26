# Privileged Learning and Policy Distillation

## At a glance

Simulation exposes variables that a deployed robot cannot measure reliably: exact pose, target state, disturbance, contact, actuator parameters, delay, and complete scene geometry. **Privileged learning** uses this information to make training easier while enforcing a strict rule: the deployed policy may consume only deployment-available observations.

There are several distinct mechanisms:

- a privileged teacher demonstrates actions to an observable student;
- a privileged critic supplies better training signals to a deployable actor;
- privileged targets supervise a latent state or auxiliary prediction;
- an online adaptation module infers hidden environment parameters from recent history.

These mechanisms can be combined, but they solve different problems. Privileged information can accelerate learning and shape internal state; it cannot make fundamentally unobservable variables identifiable. The decisive engineering artifact is therefore an auditable training/deployment information contract.

## 1. The information asymmetry

Let

- \(x_k\) be full simulator state;
- \(z_k\) be a selected privileged variable such as line-of-sight rate, body disturbance, actuator time constant, or effective delay;
- \(o_k\) be the observation available at deployment;
- \(h_k=(o_{0:k},u_{0:k-1})\) denote observable history.

A privileged teacher may use

\[
u_k^T\sim\pi_T(u\mid x_k,z_k),
\]

while the student must use

\[
u_k^S\sim\pi_S(u\mid h_k).
\]

The purpose is not to reconstruct every privileged variable. It is to learn an action or internal representation that captures the parts of privileged state that are inferable from history and relevant to control.

This distinction is easy to violate accidentally. If normalized actor inputs are assembled from one simulator-state tensor, a supposedly deployable policy may receive true velocity, delay, object identity, or future-validity flags. A clean implementation builds the actor observation through the same interface used in deployment and separately constructs training-only inputs.

## 2. Policy distillation

Policy distillation trains a student to match a teacher. For a deterministic continuous-action teacher, a simple loss is

\[
\mathcal L_{\text{act}}
=
\mathbb E_{(h,x,z)\sim\mathcal D}
\left[
\left\|\mu_S(h)-\mu_T(x,z)\right\|_2^2
\right].
\]

For stochastic policies, one can minimize a KL divergence between teacher and student action distributions. Matching only the mean is often appropriate for deterministic deployment, but it discards information about teacher uncertainty or multimodality.

Distillation is useful when the teacher has access to a clean control state or can run an expensive optimizer. It converts privileged state estimation and online optimization into supervised learning. It is also limited by the data distribution: a student trained only on teacher trajectories may make small errors, enter unfamiliar states, make larger errors, and drift further away. This is **covariate shift**.

Dataset Aggregation (DAgger) addresses that problem by rolling out the current student, querying the teacher on the states the student visits, appending those labeled examples, and retraining. In simulation this is especially practical because teacher queries are cheap. Care is still needed when the teacher's corrective action assumes a state from which its own trajectory would never have arrived.

## 3. Asymmetric actor-critic learning

In asymmetric actor-critic learning, the actor remains deployment-compatible while the critic sees privileged state:

\[
u_k\sim\pi_\theta(u\mid h_k),
\qquad
Q_\phi=Q_\phi(x_k,z_k,u_k).
\]

The critic's task is to estimate training-time value, not to act at deployment. Full state can reduce ambiguity and variance in that estimate. Policy gradients then improve the observable actor through the privileged critic.

This asymmetry is weaker than giving privileged state to the actor because no privileged input is required for inference. It can nevertheless introduce problems:

- the critic may distinguish situations the actor cannot distinguish and produce incompatible gradients;
- actor improvement may overfit to simulator states or parameters;
- implementation leakage can place critic-only features in shared encoders;
- an excellent privileged critic does not make the actor's observation sufficient.

The architecture should keep actor and critic input paths explicit. Any shared representation must be checked for training-only inputs, including masks and normalization statistics.

## 4. Distilling a predictive state

Actions are not the only useful supervision. A recurrent student may learn an internal state \(r_k=g_\theta(h_k)\) and predict privileged quantities:

\[
\widehat z_k=p_\psi(r_k),
\qquad
\mathcal L_z=
\mathbb E\left[
\ell(\widehat z_k,z_k)
\right].
\]

Possible targets for visual control include:

- line-of-sight angular rate;
- carrier-induced image disturbance;
- achieved rather than requested gimbal rate;
- actuator lag or gain;
- measurement age or applied-command delay;
- future image error, target scale, or field-of-view margin.

Supervising the latent state can make memory purposeful and easier to diagnose. It asks the network to organize history around control-relevant predictions rather than only a delayed return. Auxiliary heads can be discarded at deployment if they are used only as training losses, or retained if their outputs support monitoring.

A combined training objective might be

\[
\mathcal L =
\lambda_{\text{act}}\mathcal L_{\text{act}}
+\lambda_z\mathcal L_z
+\lambda_{\text{pred}}\mathcal L_{\text{pred}}
+\lambda_{\text{RL}}\mathcal L_{\text{RL}}.
\]

The action term imitates a teacher, the latent term estimates selected hidden quantities, the predictive term forecasts future observations or risk, and the reinforcement-learning term optimizes long-horizon performance. Their weights define a curriculum and a scientific hypothesis. They should not be treated as free performance knobs without ablation.

## 5. Privileged teachers are not necessarily optimal

A teacher may be privileged yet weak. A PID controller with true image velocity has cleaner information than the student, but it can still be myopic. An MPC teacher may optimize an approximate model and cost. An unconstrained reinforcement-learning teacher may exploit simulation. Distillation transfers the teacher's biases along with its competence.

Useful teacher designs include:

- a tuned feedback controller augmented with true disturbance feedforward;
- an observer or MPC given true simulator state and parameter values;
- a high-capacity policy trained on full state;
- a mixture in which a safe conventional controller handles ordinary states and a privileged optimizer labels difficult ones.

Teacher quality should be reported on the same held-out scenarios as the student. If the student outperforms its teacher after reinforcement-learning fine-tuning, the experiment should distinguish improvement from altered observations, rewards, or action smoothing.

## 6. The observability boundary

Privileged training cannot violate information theory. If two hidden states \((x,z)\) generate identical observable histories \(h\) but require different optimal actions, no deterministic student \(\mu_S(h)\) can select both. It must choose a compromise, represent a distribution, gather more information, or rely on an additional sensor.

This boundary has practical consequences:

- a single bbox cannot reveal whether error comes from target motion or carrier motion;
- a short history may not identify a slow actuator time constant;
- constant unknown delay may be inferable from command-response history, while rapidly varying delay may not be;
- target motion behind a complete detection dropout is ambiguous without another cue.

Low prediction loss is not proof of correct identification. The training distribution may correlate hidden parameters with visible cues, letting the network shortcut. A controller trained with actuator type tied to visual background could appear to infer dynamics while actually recognizing the scene.

Tests should therefore break correlations, swap parameters independently, and compare against an oracle that receives true privileged state. The gap between observable student and oracle estimates the cost of partial observability under the tested distribution.

## 7. Online adaptation as privileged learning

Rapid Motor Adaptation separates a base policy conditioned on environment parameters from an adaptation module that estimates a useful latent from recent experience. In generic form,

\[
u_k=\pi(o_k,\widehat z_k),
\qquad
\widehat z_k=g(h_k),
\]

where the base policy is first trained with privileged \(z_k\), and the adaptation module is later trained to infer the corresponding latent from history.

For a gimbal, \(z\) could summarize actuator gain, time constant, backlash, delay, vibration spectrum, or payload configuration. This structure makes adaptation explicit and probeable. A monolithic recurrent actor may learn the same computation implicitly, with less architectural commitment but less interpretability.

The estimated latent need not equal a physical parameter. A task-relevant embedding can be sufficient, but then claims should say “adaptation latent” rather than “identified actuator parameters.” Physical identification requires validation against ground truth and identifiability analysis.

## 8. Capabilities and non-guarantees

Privileged learning can accelerate optimization, improve credit assignment, teach recurrent features, and reduce the burden on sparse reward. It also enables a clean experiment: compare the same deployed observation and architecture with and without training-only supervision.

It does not guarantee that the student can infer the privileged state, match the teacher under distribution shift, remain safe, or generalize to real hardware. A privileged critic supplies gradients, not deployment information. A distilled latent can be confidently wrong. Student performance remains bounded by observable evidence, policy capacity, data coverage, teacher behavior, and the match between simulated and real causal structure.

## 9. Failure modes and diagnostics

| Failure | Likely cause | Observable symptom | Diagnostic | Mitigation |
|---|---|---|---|---|
| Privileged leakage | actor path or normalization uses simulator-only fields | excellent simulation result, impossible runtime interface | export actor signature and replay with deployment schema only | separate data builders, automated feature audit, deployable-policy unit test |
| Student drifts off teacher data | behavior cloning covariate shift | compounding error after small disturbance | student rollouts labeled by teacher | DAgger or mixed student/teacher state collection |
| Teacher ceiling | teacher is myopic or model-biased | student reproduces known teacher failures | compare failure-conditioned trajectories | stronger/multiple teachers, RL fine-tuning, residual improvement |
| Unobservable target | history lacks information needed for label | latent prediction regresses to an average | conditional variance and oracle gap | longer history, extra telemetry, uncertainty output, reformulate target |
| Shortcut inference | nuisance cue correlates with privileged variable | prediction collapses when correlations are swapped | counterfactual parameter/context swaps | independent randomization and causal data design |
| Privileged critic destabilizes actor | critic separates states actor aliases | conflicting updates and high seed variance | group gradients by identical/near-identical actor observations | actor-compatible critic features, recurrent actor, regularization |
| Auxiliary loss dominates control | easy prediction displaces useful policy features | low auxiliary error but worse tracking | loss-gradient scales and weight ablation | normalize/adapt weights, choose control-relevant targets |
| Hidden deployment dependency | auxiliary head or reset logic requires ground truth | exported policy fails standalone | end-to-end cold-start and dropout test | explicit inference graph and deployment-contract test |

## 10. Neighboring concepts

| Concept | Relationship | Difference |
|---|---|---|
| Learning using privileged information | broad training-time use of unavailable features | may target classification or representation rather than sequential control |
| Behavioral cloning | learns actions from demonstrations | teacher need not have privileged state and data are usually fixed |
| DAgger | aggregates labels on learner-visited states | addresses covariate shift rather than hidden-state estimation itself |
| Asymmetric actor-critic | critic sees fuller state than actor | transfers value gradients, not necessarily teacher actions |
| Policy distillation | compresses or transfers a teacher policy | teacher may differ in size, task, or ensemble rather than observability |
| System identification | estimates physical parameters from input-output data | seeks interpretable plant quantities; privileged latent may be task-specific |
| Recurrent belief learning | summarizes observation history | can be learned without privileged labels |

## 11. Implications for predictive 1D gimbal servoing

The project should freeze an information table before training:

| Signal | Teacher | Training critic | Student actor | Real availability |
|---|---:|---:|---:|---|
| Timestamped bbox and validity | yes | yes | yes | yes |
| Previous requested/applied command | yes | yes | yes | yes, if instrumented |
| Payload IMU/body rate | yes | yes | yes only if carried in deployment | hardware-dependent |
| True line-of-sight rate | yes | yes | no | no |
| True target and vehicle pose | yes | yes | no | no |
| Actuator/delay parameters | yes | yes | no | normally no |
| Future target visibility | label only | label only | no | no |

The locked training sequence then has a defensible interpretation:

1. train or construct a strong privileged teacher;
2. collect teacher and student-visited trajectories across randomized conditions;
3. warm-start the recurrent actor by action and predictive-state distillation;
4. fine-tune the deployable actor with continuous-control reinforcement learning;
5. remove all privileged inputs and evaluate the exported actor alone;
6. compare against the teacher, an oracle actor, and a student trained without privilege.

The central research claim should be about **training-time supervision improving a deployment-identical recurrent servo**, not about the mere presence of simulator state. Required ablations include no privilege, action imitation only, auxiliary prediction only, privileged critic only, and the combined method. A bbox-only actor and a telemetry-enabled actor should be distinct experimental conditions rather than an ambiguous interface.

## Durable takeaways

1. Privileged information is permitted during training only when the deployed actor's observation contract remains unchanged and auditable.
2. Teacher imitation, asymmetric critics, predictive latent supervision, and online adaptation are related but distinct mechanisms.
3. Student-driven data collection is necessary when imitation errors change the visited state distribution.
4. Privileged targets cannot make unobservable state identifiable; the student can learn only what history contains or what the task can safely average over.
5. A privileged teacher transfers its biases, so teacher strength and failure modes must be measured.
6. The strongest evidence is an ablation showing better held-out control from the same deployable observations and architecture.

## Primary sources

- Andrei A. Rusu et al., [“Policy Distillation”](https://arxiv.org/abs/1511.06295), *ICLR*, 2016.
- Stéphane Ross, Geoffrey Gordon, and Drew Bagnell, [“A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning”](https://proceedings.mlr.press/v15/ross11a.html), *AISTATS*, 2011.
- Lerrel Pinto et al., [“Asymmetric Actor Critic for Image-Based Robot Learning”](https://doi.org/10.15607/RSS.2018.XIV.008), *Robotics: Science and Systems*, 2018.
- Dian Chen et al., [“Learning by Cheating”](https://proceedings.mlr.press/v100/chen20a.html), *CoRL 2019*, 2020 proceedings.
- Ashish Kumar et al., [“RMA: Rapid Motor Adaptation for Legged Robots”](https://arxiv.org/abs/2107.04034), *Robotics: Science and Systems*, 2021.
