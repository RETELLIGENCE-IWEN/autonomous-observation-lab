# Continuous-Control Actor-Critic Learning

## At a glance

Many physical controllers choose real-valued actions: angular rate, torque, thrust, or position setpoint. **Continuous-control actor-critic learning** represents that decision with an actor and learns its long-horizon value with a critic. It avoids manually discretizing a smooth action range and can learn anticipatory, state-dependent behavior when the useful control law is difficult to specify.

For a gimbal, the actor can map timestamped observation history to a bounded desired angular rate. A critic is useful during training but need not run on the vehicle. A compact deterministic actor can therefore be the deployed artifact even when training used stochastic exploration and two large critics.

Actor-critic methods are optimization machinery, not a control guarantee. They are sensitive to reward design, data distribution, value-estimation error, recurrent-state handling, and random seed. Their scientific value appears only against strong controllers and under held-out dynamics, delay, and maneuvers.

## 1. Why continuous actions need their own treatment

Suppose a policy selects a gimbal-rate command

\[
u_k \in [u_{\min},u_{\max}].
\]

Dividing this interval into a few discrete actions creates quantization and can produce chatter. Making the grid fine increases the number of action values without exploiting the fact that nearby commands usually have nearby physical effects. A continuous policy instead represents a distribution \(\pi_\theta(u\mid s)\) or a deterministic map \(\mu_\theta(s)\).

The standard discounted objective is

\[
J(\theta)=
\mathbb{E}_{\pi_\theta}
\left[
\sum_{k=0}^{\infty}\gamma^k r_k
\right],
\]

where \(r_k\) scores tracking, visibility, motion, and constraint use, and \(0\leq\gamma<1\) controls how strongly the future matters. The objective turns a control problem into an optimization problem: find policy parameters that produce high cumulative return over the training distribution.

For a partially observed system, the true Markov state is not the latest bounding box. The actor instead consumes an internal state

\[
h_k=f_\theta(h_{k-1},o_k,u_{k-1},\Delta t_k),
\]

which summarizes observation and action history. The action is then drawn from \(\pi_\theta(\cdot\mid h_k)\) or set to \(\mu_\theta(h_k)\). This recurrent state is an approximate belief or predictive control state, not automatically a physically identifiable state estimate.

## 2. Actor and critic

The action-value function of a policy is

\[
Q^\pi(s,u)=
\mathbb{E}\left[
r_k+\gamma r_{k+1}+\gamma^2r_{k+2}+\cdots
\mid s_k=s,u_k=u
\right].
\]

It obeys the Bellman relation

\[
Q^\pi(s,u)=
\mathbb{E}\left[
r_k+\gamma Q^\pi(s_{k+1},u_{k+1})
\right].
\]

The **critic** \(Q_\phi\) approximates this long-horizon value from experience. The **actor** changes its action toward values that the critic predicts will be better. This division enables learning from delayed consequences: a command that temporarily increases motion may still be valuable if it prevents target loss several frames later.

Critic error is consequential because the actor deliberately seeks actions with high predicted value. Function approximation can assign falsely high values to actions poorly represented in the data, and the actor can exploit those errors. Much of modern continuous-control algorithm design addresses this feedback loop.

## 3. Deterministic policy gradients, DDPG, and TD3

For a deterministic actor \(u=\mu_\theta(s)\), the deterministic policy-gradient result motivates

\[
\nabla_\theta J
\approx
\mathbb{E}_{s\sim\rho}
\left[
\nabla_\theta\mu_\theta(s)
\nabla_u Q_\phi(s,u)\big|_{u=\mu_\theta(s)}
\right].
\]

The actor is updated through the critic's gradient with respect to action. DDPG combines this idea with replay data, neural function approximation, slowly updated target networks, and exploratory noise. It is sample-efficient in principle because old transitions can be reused off-policy, but it can be brittle when the critic extrapolates badly.

Twin Delayed Deep Deterministic Policy Gradient (TD3) adds three stabilizing ideas:

1. learn two critics and use the smaller target value to reduce positive bias;
2. update the actor less often than the critics;
3. add clipped noise to the target action so the critic does not reward narrow action spikes.

A simplified target is

\[
y_k=r_k+\gamma(1-d_k)
\min_{i\in\{1,2\}}
Q_{\bar\phi_i}
\left(s_{k+1},
\mu_{\bar\theta}(s_{k+1})+\epsilon
\right),
\]

where \(d_k\) denotes a true terminal transition, bars denote target networks, and \(\epsilon\) is bounded smoothing noise. Time-limit truncation should not be mislabeled as physical termination; otherwise the critic learns an artificial value drop at the end of every training segment.

Deterministic methods give a natural deployment policy but require an explicit exploration process during training. Exploration noise that is harmless in normalized simulation can become unrealistic or unsafe on hardware, so hardware learning must remain inside a separate supervisory envelope.

## 4. Soft Actor-Critic

Soft Actor-Critic (SAC) learns a stochastic policy and augments return with entropy. One common actor objective is

\[
J_\pi(\theta)=
\mathbb{E}_{s\sim\mathcal D,\,u\sim\pi_\theta}
\left[
\alpha\log\pi_\theta(u\mid s)-Q_\phi(s,u)
\right].
\]

Minimizing this expression favors actions that have high value while retaining entropy. The temperature \(\alpha\) controls that tradeoff and can itself be tuned toward a target entropy. Twin critics are normally used to limit optimistic value estimates.

SAC is attractive for simulation training because it is off-policy, reuses experience, and maintains broad action exploration. A stochastic training policy does not require stochastic flight behavior. At deployment, a continuous controller can use the policy mean or another deterministic representative, followed by explicit rate and acceleration limits.

The train/deploy difference must be evaluated directly. The mean of a nonlinear or squashed action distribution is not always equivalent to its highest-probability action, and a policy trained to rely on sampling can perform differently when made deterministic.

## 5. Bounded actions and physical units

A common actor samples an unconstrained variable \(z\) and squashes it:

\[
u=u_{\max}\tanh z.
\]

This respects a symmetric rate bound, but it does not enforce acceleration, jerk, travel, current, or thermal limits. Those should be represented by a command filter, constrained action transformation, safety supervisor, or lower-level controller. The policy should also observe the **applied** action, not only the action it requested, when clipping or filtering is active.

Normalized actions aid optimization, but logs and interface contracts should retain physical units. An action of \(0.5\) is scientifically ambiguous unless it can be mapped to degrees per second, radians per second, or another actuator quantity.

Frequent saturation is a warning. It can mean the task exceeds actuator authority, the reward underprices saturation, the action range is incorrectly scaled, or the policy has learned bang-bang behavior. Clipping makes these cases look superficially safe while hiding a poor policy.

## 6. Designing a servo objective

A candidate dense reward for image tracking is

\[
r_k=
-q_e\,\rho(e_k)
-q_m\,\rho(\min(0,m_k))
-q_u u_k^2
-q_{\Delta u}(u_k-u_{k-1})^2
-q_s I_{\text{saturated}}
-q_l I_{\text{lost}},
\]

where \(e_k\) is image error, \(m_k\) is field-of-view margin, \(\rho\) is a chosen robust penalty, and the indicators mark saturation and target loss. The terms express tracking accuracy, boundary risk, effort, smoothness, authority use, and catastrophic loss.

Every term changes behavior:

- a large pointwise error penalty can make the controller aggressive and oscillatory;
- a large action penalty can reward inaction;
- an action-change penalty suppresses chatter but can slow response;
- a one-time loss penalty may be too sparse for learning;
- terminating immediately on loss can teach the critic little about recovery;
- bounding-box confidence can be gamed if controller motion changes detector behavior.

Reward is therefore an operational hypothesis about desired behavior, not a neutral metric. The benchmark should report physical metrics separately so a policy cannot be declared superior merely because it optimized its own scalar reward.

Imitation from PID or MPC is useful for warm-starting. It places the actor near a competent region before reinforcement learning explores improvements. The imitation weight should later be reduced or tested as an ablation; otherwise the learned policy may never exceed the teacher or may inherit its characteristic failure modes.

## 7. Recurrent off-policy learning

Recurrent policies are appropriate when line-of-sight rate, target intent, actuator state, or delay must be inferred from history. They also make replay more delicate.

A transition sampled in isolation does not contain the hidden state that existed when it was collected. Practical recurrent training therefore samples sequences and uses a **burn-in** prefix to reconstruct hidden state before calculating learning losses. The replay record should include observations, requested and applied actions, rewards, termination causes, validity masks, and timestamps.

Other recurrent hazards include:

- resetting hidden state at artificial chunk boundaries;
- mixing episodes or targets in one sequence;
- padding without masking losses;
- training with fixed \(\Delta t\) and deploying with jitter;
- allowing the critic privileged information while accidentally feeding it to the deployed actor;
- evaluating after a hidden-state reset that never occurs in normal operation.

Sequence length is an assumption about how much history matters. Too short removes relevant actuator or disturbance dynamics; arbitrarily long sequences increase compute and optimization difficulty without ensuring useful memory.

## 8. Capabilities and non-guarantees

Actor-critic learning can combine heterogeneous signals, infer hidden patterns from history, optimize delayed visibility outcomes, and amortize a complex control calculation into a small runtime network. Off-policy algorithms can also reuse simulation data efficiently.

It does not by itself provide closed-loop stability, constraint satisfaction, calibrated uncertainty, causal identification, or sim-to-real robustness. A high return can arise from exploiting simulator artifacts or reward omissions. Neural policies can change sharply outside their training support even when their in-distribution curve looks smooth.

Safety mechanisms must therefore be architectural: bounded commands, a verified inner loop, hard limits, stale-input handling, watchdogs, out-of-distribution monitoring where useful, and a tested fallback. These are not substitutes for a capable policy; they define the envelope in which one may operate.

## 9. Failure modes and diagnostics

| Failure | Likely cause | Observable symptom | Diagnostic | Mitigation |
|---|---|---|---|---|
| Critic divergence | large targets, poor scaling, recurrent replay error | exploding or highly drifting Q values | compare Q scale with realized returns | normalize inputs/rewards, check targets/masks, reduce update aggressiveness |
| Policy exploits critic | extrapolation error outside replay support | high predicted value, poor rollout return | evaluate critic on policy actions and held-out rollouts | twin critics, target smoothing, broader data, conservative updates |
| Chattering commands | reward underprices action change or sensor noise | high-frequency reversals near center | power spectrum and total variation | action-change cost, filtering, recurrent estimation, actuator-aware training |
| Passive policy | effort/smoothness penalty dominates | low motion and persistent error | term-by-term return decomposition | rescale objectives and use physical success constraints |
| Good simulation, poor transfer | policy uses simulator-specific cues or dynamics | sharp hardware degradation | parameter sweeps and hardware-in-the-loop replay | measured randomization, privileged-data audit, transfer ladder |
| Memory provides no benefit | sequences too short or task nearly Markov | recurrent and feedforward results match | recurrence/history ablation | simplify policy or improve history/timing inputs |
| Memory fails after dropout | hidden state drifts without valid measurements | wrong confident actions after reacquisition | controlled dropout tests and hidden-state probes | validity input, reset/recovery logic, dropout training |
| Large run-to-run variance | unstable optimization or insufficient evaluation | conclusions depend on seed | multi-seed confidence intervals and paired scenarios | tune robustly, increase data, report distribution not best run |

## 10. Neighboring approaches

| Approach | Strength | Limitation relative to actor-critic learning |
|---|---|---|
| Supervised controller imitation | stable and simple when a strong teacher exists | cannot naturally exceed teacher support and suffers covariate shift |
| System identification plus PID/LQR | interpretable and analyzable | depends on a useful compact model and estimator |
| Model predictive control | explicit constraints and look-ahead | online compute and model mismatch |
| Residual RL | preserves a nominal controller while learning corrections | residual can still destabilize unless bounded and analyzed |
| Model-based RL | can use learned prediction for planning or policy training | model error creates another failure path |
| Offline RL | avoids exploratory hardware interaction | sensitive to dataset coverage and out-of-distribution actions |

## 11. Implications for predictive 1D gimbal servoing

A credible first learned controller is a recurrent SAC-style outer loop trained in simulation, warm-started by a strong controller, and deployed deterministically. The recurrent actor receives timestamped visual features, previous applied commands, and permitted vehicle/gimbal telemetry. Training-only critics may receive fuller simulated state if that asymmetry is explicitly controlled and ablated.

The scientific comparison should include:

- tuned PID with filtering and anti-windup;
- estimator or disturbance-observer control;
- a predictive/MPC controller;
- feedforward actor-critic without memory;
- recurrent actor-critic without privileged training;
- the full recurrent, privileged, predictive-auxiliary method.

Report paired scenario results across multiple training seeds. Primary outcomes are loss-of-view probability, high-percentile absolute error, recovery time, command total variation, saturation duration, and compute latency. Mean return is a training diagnostic, not the headline result.

## Durable takeaways

1. Continuous actor-critic methods learn real-valued policies by using a critic to assign long-horizon value to actions.
2. DDPG and TD3 deploy deterministic actors; SAC learns a stochastic maximum-entropy policy that can be made deterministic for deployment.
3. Recurrent control requires sequence-aware replay, burn-in, timestamp handling, and careful hidden-state resets.
4. Bounded neural output is not equivalent to physical constraint enforcement.
5. Reward design determines behavior, so final claims must use independent physical metrics.
6. Actor-critic learning supplies optimization and adaptation, not stability or transfer guarantees.

## Primary sources

- David Silver et al., [“Deterministic Policy Gradient Algorithms”](https://proceedings.mlr.press/v32/silver14.html), *ICML*, 2014.
- Timothy P. Lillicrap et al., [“Continuous Control with Deep Reinforcement Learning”](https://arxiv.org/abs/1509.02971), *ICLR*, 2016.
- Scott Fujimoto, Herke van Hoof, and David Meger, [“Addressing Function Approximation Error in Actor-Critic Methods”](https://proceedings.mlr.press/v80/fujimoto18a.html), *ICML*, 2018.
- Tuomas Haarnoja et al., [“Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”](https://proceedings.mlr.press/v80/haarnoja18b.html), *ICML*, 2018.
- Nicolas Heess et al., [“Memory-Based Control with Recurrent Neural Networks”](https://arxiv.org/abs/1512.04455), 2015.
