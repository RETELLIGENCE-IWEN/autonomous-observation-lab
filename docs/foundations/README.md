# Foundation Notes

Foundation Notes are compact, human-readable references for concepts that repeatedly support the research.

They are not intended to exhaustively survey a field or replace the original papers. Each note focuses on the principles, mathematical formulation, meaning, value, limitations, and implications needed to recover the concept quickly after time away.

## Notes

- [Recurrent State-Space Model (RSSM)](rssm.md)
- [POMDPs and Belief States](pomdps_and_belief_states.md)
- [Active Sensing and Value of Information](active_sensing_and_value_of_information.md)
- [Object-Centric Representations and World Models](object_centric_representations_and_world_models.md)
- [Uncertainty Estimation for Learned World Models](uncertainty_estimation_for_learned_world_models.md)
- [Model-Based Reinforcement Learning and Latent Imagination](model_based_rl_and_latent_imagination.md)
- [Visual Servoing and Cascaded Gimbal Control](visual_servoing_and_cascaded_gimbal_control.md)
- [Continuous-Control Actor-Critic Learning](continuous_control_actor_critic_learning.md)
- [Privileged Learning and Policy Distillation](privileged_learning_and_policy_distillation.md)
- [Sim-to-Real for Learned Control](sim_to_real_for_learned_control.md)

## Writing new notes

- [Foundation Note Writing Guide](WRITING_GUIDE.md)

The guide defines the common structure, mathematical style, evidence policy, research-connection standard, and quality checklist for this folder.

## Recommended reading path for Dream-to-Look

1. [**POMDPs and belief states**](pomdps_and_belief_states.md) — the formal language for hidden state, accumulated evidence, uncertainty, and action under partial observability.
2. [**Active sensing and value of information**](active_sensing_and_value_of_information.md) — why one look can be more valuable than another.
3. [**Object-centric representations and world models**](object_centric_representations_and_world_models.md) — how persistent entities and relations can structure predictive state.
4. [**Uncertainty estimation for learned world models**](uncertainty_estimation_for_learned_world_models.md) — how to separate actionable ignorance from observation noise and model error.
5. [**Model-based reinforcement learning and latent imagination**](model_based_rl_and_latent_imagination.md) — how candidate observation actions can be evaluated through predicted futures.

Together, these notes form the initial conceptual stack for Dream-to-Look: formalize partial observability, value observations, structure the hidden world as persistent entities, estimate which predictions should be trusted, and evaluate candidate looks through imagined futures.

## Recommended reading path for predictive gimbal servoing

1. [**Visual servoing and cascaded gimbal control**](visual_servoing_and_cascaded_gimbal_control.md) — the geometry, feedback structure, delays, saturation, and classical reference architecture of the task.
2. [**POMDPs and belief states**](pomdps_and_belief_states.md) — why a single bounding box is not a complete control state and why observation history matters.
3. [**Continuous-control actor-critic learning**](continuous_control_actor_critic_learning.md) — how a policy can learn bounded, smooth, real-valued gimbal commands.
4. [**Privileged learning and policy distillation**](privileged_learning_and_policy_distillation.md) — how simulation-only platform state can improve training without leaking into deployment.
5. [**Sim-to-real for learned control**](sim_to_real_for_learned_control.md) — how actuator, timing, perception, and maneuver variation should shape training and validation.
6. [**Model-based reinforcement learning and latent imagination**](model_based_rl_and_latent_imagination.md) — the boundary between an actor with auxiliary predictions and a controller that actually uses a learned model to improve decisions.
7. [**Uncertainty estimation for learned world models**](uncertainty_estimation_for_learned_world_models.md) — how prediction confidence can support abstention, fallback, and out-of-distribution tests.

Together, these notes form the initial conceptual stack for predictive gimbal servoing: preserve the known control structure, represent hidden motion through history, learn the continuous outer-loop decision, exploit privileged state only during training, and test the resulting controller against the real distributions and failure modes it must survive.
