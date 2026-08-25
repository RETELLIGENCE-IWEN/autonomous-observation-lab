# Gate 1 Quickstart

## Goal

Gate 1 validates the benchmark before any learned world model is introduced. It checks deterministic replay, target-predicate integrity, randomized target handles, occlusion behavior, and the intended separation between generic entropy reduction and decision-aware Value of Information.

## Install

~~~bash
python -m pip install -e ".[dev]"
~~~

## Run tests

~~~bash
python -m pytest
~~~

## Run the reference evaluation

~~~bash
python -m autonomous_observation_lab.benchmark.evaluate \
  --episodes 500 --seed-start 10000 --json
~~~

The output contains the exact configuration, a constructed divergence case, and aggregate results for random, fixed-scan, entropy-greedy, and decision-aware VoI policies.

For the constructed case, the required result is:

~~~text
entropy_choice = 0
voi_choice = 1
~~~

Object 0 contains more reducible appearance uncertainty but is known not to satisfy the motion component of the target predicate. Object 1 contains less total uncertainty, but observing it can change the target decision.

## Interpretation boundary

The current factorized belief is a transparent oracle baseline, not the proposed learned model. Object handles are stable within an episode in Gate 1. Handle reset, collision, and identity switching are specified extensions required before the object-centric RSSM result.

See [Gate 1 Validation Results](gate1_validation_results.md) for the first reference run.
