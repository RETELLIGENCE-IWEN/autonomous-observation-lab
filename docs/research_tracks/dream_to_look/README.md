# Dream-to-Look

## Core idea

An observation agent should imagine the future visibility and evidence produced by candidate gaze actions before moving the sensor.

Rather than reacting only to the current bounding box, an object-centric world model can compare futures such as continuing to track, zooming out early, looking toward a predicted reappearance region, switching modality, or waiting for a more informative view.

## Primary research question

> Can an object-centric RSSM predict how candidate observation actions will change mission-relevant evidence well enough for a latent-imagination policy to outperform policies that react only to current detections, confidence, or recurrent memory?

## Initial study

The first study uses object-feature observations and a staged scenario combining target identification, competition for sensing time, interrupted observation, and target reacquisition. It proceeds through an object-centric RSSM and latent-imagination policy while deliberately excluding pixel generation and low-level gimbal control.

- [Research Brief: Decision-Aware Dream-to-Look](research_brief.md)
- [Benchmark Specification: Staged Evidence Acquisition](benchmark_specification.md)

## Current foundations

- [Recurrent State-Space Model (RSSM)](../../foundations/rssm.md)
- [POMDPs and Belief States](../../foundations/pomdps_and_belief_states.md)
- [Active Sensing and Value of Information](../../foundations/active_sensing_and_value_of_information.md)
- [Object-Centric Representations and World Models](../../foundations/object_centric_representations_and_world_models.md)
- [Uncertainty Estimation for Learned World Models](../../foundations/uncertainty_estimation_for_learned_world_models.md)
- [Model-Based Reinforcement Learning and Latent Imagination](../../foundations/model_based_rl_and_latent_imagination.md)
- [Initial Research Candidates: Dream-to-Look](../initial_research_candidates.md#candidate-b-dream-to-look-with-an-object-centric-rssm)
- [Project Value and Roadmap](../../vision/research_roadmap.md)

## Status

Research framing and benchmark specification complete. Next milestone: implement and validate the seeded benchmark with scripted and oracle policies before training the learned world model.
