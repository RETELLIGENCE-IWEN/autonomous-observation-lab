# Dream-to-Look

## Position within Think–Dream–Look

Dream-to-Look (D2L) is a concrete research track inside the broader [Think–Dream–Look (TDL)](../../vision/think_dream_look.md) architecture.

TDL asks how an autonomous observation agent should allocate intelligence among three operations:

- **Think** — refine the current belief through internal computation;
- **Dream** — imagine future states and evidence consequences;
- **Look** — acquire new evidence from the physical world.

D2L focuses specifically on the **Dream → Look** link. Its purpose is to test whether a learned predictive world model can imagine the future evidence consequences of candidate observation actions accurately enough to improve real sensing decisions.

This intentionally keeps the current D2L benchmark narrower than the full TDL vision. D2L does not yet require adaptive recurrent reasoning depth, explicit Think-versus-Look compute allocation, or a unified TDL controller. Those are later research stages that can build on a validated D2L world model and benchmark.

## Core idea

An observation agent should imagine the future visibility and evidence produced by candidate gaze actions before moving the sensor.

Rather than reacting only to the current bounding box, an object-centric world model can compare futures such as continuing to track, zooming out early, looking toward a predicted reappearance region, switching modality, or waiting for a more informative view.

## Primary research question

> Can an object-centric RSSM predict how candidate observation actions will change mission-relevant evidence well enough for a latent-imagination policy to outperform policies that react only to current detections, confidence, or recurrent memory?

## Initial study

The first study uses object-feature observations and a staged scenario combining target identification, competition for sensing time, interrupted observation, and target reacquisition. It proceeds through an object-centric RSSM and latent-imagination policy while deliberately excluding pixel generation and low-level gimbal control.

- [TDL Concept: Think–Dream–Look](../../vision/think_dream_look.md)
- [Research Brief: Decision-Aware Dream-to-Look](research_brief.md)
- [Benchmark Specification: Staged Evidence Acquisition](benchmark_specification.md)
- [Technology Demo System Concept: Multi-UAV EO/IR Payload Embodiment](technology_demo_system_concept.md)
- [Gate 1 Quickstart](gate1_quickstart.md)
- [Gate 1 Validation Results](gate1_validation_results.md)
- [Gate 2 Protocol](gate2_protocol.md)
- [Gate 2 Quickstart](gate2_quickstart.md)
- [Gate 2 Reference Results](gate2_reference_results.md)

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

Gate 1 complete. Gate 2 evidence-belief milestone passes: object-centric RSSM shows a stable matched-capacity advantage in filtering, occlusion, open-loop identity prediction, and handle-corruption stress. Kinematic prediction remains near a trivial baseline and must be corrected before latent-imagination policy training.

Within the broader TDL progression, the current work remains at the **Dream-to-Look validation stage**. The next TDL-level extensions should only be introduced after the predictive model demonstrates that its imagined evidence rankings are causally useful for observation decisions.
