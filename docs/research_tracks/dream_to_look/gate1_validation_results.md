# Gate 1 Validation Results

## At a Glance

The first benchmark implementation passes six deterministic and structural tests and produces the intended behavioral separation between entropy-seeking and decision-aware observation.

This is evidence that the benchmark expresses the research question. It is **not** yet evidence that an RSSM or learned policy solves it.

## Run definition

| Field | Value |
|---|---|
| Date | 2026-08-25 |
| Evaluation seeds | 10000–10499 |
| Episodes per policy | 500 |
| Objects | 5 |
| Signature bits | 3 |
| Horizon | 16 |
| Target-present probability | 0.85 |
| Focus evidence accuracy | 0.88 |
| Wide evidence accuracy | 0.62 |
| Detection miss probability | 0.05 |

## Structural tests

~~~text
6 passed
~~~

The tests cover deterministic observation replay, hidden target-predicate integrity, randomized target handles, occlusion/reappearance, constructed entropy/VoI divergence, and deterministic complete episodes.

## Constructed decision-relevance case

Object 0 has greater total reducible appearance uncertainty but is already known not to satisfy the motion component of the target predicate. Object 1 has less total uncertainty but is the only plausible target candidate.

| Selector | Choice | Score for object 0 | Score for object 1 |
|---|---:|---:|---:|
| Entropy reduction | 0 | 0.9959 | 0.4646 |
| Decision-aware VoI | 1 | -0.0200 | 0.3165 |

This satisfies the key Gate 1 requirement: generic information seeking and mission-decision value induce different observation actions.

## Reference policy evaluation

| Policy | Mean return | Correct | Wrong | Abstain | Mean steps |
|---|---:|---:|---:|---:|---:|
| Random | -0.0463 | 50.2% | 26.6% | 23.2% | 13.67 |
| Fixed scan | 0.4897 | 81.8% | 15.0% | 3.2% | 9.67 |
| Entropy greedy | 0.0148 | 51.2% | 23.0% | 25.8% | 12.71 |
| Decision-aware VoI | **0.6016** | **86.4%** | **12.8%** | **0.8%** | **8.46** |

Decision-aware VoI improves correct-decision rate by 4.6 percentage points over fixed scan while using 1.21 fewer steps on average. It also strongly separates from entropy-greedy, as intended.

## Interpretation boundary

The result supports three limited conclusions:

1. The environment is reproducible under fixed seeds.
2. The task distinguishes total uncertainty reduction from decision-relevant evidence acquisition.
3. A transparent one-step decision-aware oracle establishes a meaningful baseline above fixed scanning.

It does not yet support claims about learned beliefs, object-centric representation, RSSM prediction, multi-step latent imagination, association corruption, or transfer to real detector data.

## Gate status

| Requirement | Status | Evidence |
|---|---|---|
| Deterministic replay | Pass | repeated trace test |
| Unique target predicate | Pass | 200-seed structural test |
| Randomized target handle | Pass | 100-seed distribution test |
| Occlusion/reappearance | Pass | explicit interval test |
| Entropy versus VoI divergence | Pass | constructed case |
| Oracle above fixed scan | Pass on reference block | 0.6016 versus 0.4897 return |
| Statistical leakage audit | Pending | feature-label probe suite required |
| Handle corruption | Pending | next benchmark increment |
| Multi-step oracle separation | Pending | required before latent imagination |

Gate 1 is therefore **provisionally passed**, with three follow-up requirements before world-model training: automated leakage probes, association corruption, and a staged case where multi-step planning beats one-step VoI.

