# Dream-to-Look

## Core idea

An observation agent should imagine the future visibility and evidence produced by candidate gaze actions before moving the sensor.

Rather than reacting only to the current bounding box, an object-centric world model can compare futures such as continuing to track, zooming out early, looking toward a predicted reappearance region, switching modality, or waiting for a more informative view.

## Primary research question

> Can an object-centric RSSM predict the epistemic consequences of candidate gaze actions well enough to produce useful anticipatory observation behavior under partial observability?

## Initial hypothesis

An RSSM-based policy that imagines future visibility, field-of-view margin, detection probability, and track-loss risk will outperform reactive recurrent policies under narrow FOV, occlusion, perception dropout, latency, and aggressive platform motion.

## Current foundations

- [Recurrent State-Space Model (RSSM)](../../foundations/rssm.md)
- [Initial Research Candidates: Dream-to-Look](../initial_research_candidates.md#candidate-b-dream-to-look-with-an-object-centric-rssm)
- [Project Value and Roadmap](../../vision/research_roadmap.md)

## Status

Concept formation and foundation study. No architecture has been selected yet.

