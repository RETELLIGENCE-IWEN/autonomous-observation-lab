# Gate 1 Validation Results

## At a Glance

The completed Gate 1 benchmark passes nine deterministic, structural, leakage, association, and planning tests. It produces the intended behavioral separations between entropy-seeking and decision-aware observation, and between one-step and multi-step evidence valuation.

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
9 passed
~~~

The tests cover deterministic observation replay, hidden target-predicate integrity, randomized target handles, occlusion/reappearance, deterministic handle corruption and collision, constructed entropy/VoI divergence, nuisance-feature leakage, multi-step planning separation, and deterministic complete episodes.

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
| Fixed scan | 0.4282 | 78.8% | 17.4% | 3.8% | 10.00 |
| Entropy greedy | 0.0148 | 51.2% | 23.0% | 25.8% | 12.71 |
| Decision-aware VoI | **0.6016** | **86.4%** | **12.8%** | **0.8%** | **8.46** |

Decision-aware VoI improves correct-decision rate by 7.6 percentage points over fixed scan while using 1.54 fewer steps on average. It also strongly separates from entropy-greedy, as intended.

The fixed-scan result differs from the provisional run because the completed handle semantics no longer allow a policy to directly address a never-observed hidden object index. Revisit uses only a current or last-known observed association.

## Statistical leakage probe

The probe excludes legitimate appearance and motion evidence and attempts to predict the target label from handle, bbox geometry, confidence, quality, and remaining time.

| Metric | Result |
|---|---:|
| Samples | 9,524 |
| Positive rate | 17.13% |
| Nuisance-feature balanced accuracy | 0.5005 |
| Permutation-null P95 | 0.5217 |
| Decision | Pass |

The nuisance probe performs at chance and below the permutation-derived threshold. This does not prove that every possible learner is leakage-free, but it rejects the tested linear shortcut family and establishes an automated regression check.

## Multi-step evidence separation

The executable decision tree contains DIRECT, which improves confidence immediately but saturates, and SCOUT, which has no immediate information benefit but unlocks a decisive REVEAL observation.

| Horizon | DIRECT | SCOUT | Selected |
|---|---:|---:|---|
| One step | 0.130 | -0.020 | DIRECT |
| Two steps | 0.110 | 0.360 | SCOUT |

This establishes the minimum delayed-evidence structure required to test latent imagination. A greedy acquisition function cannot select the optimal first action.

## Association-corruption stress test

With independent observed-handle corruption probability 0.15 on seeds 11000–11499:

| Policy | Mean return | Correct | Wrong | Mean steps |
|---|---:|---:|---:|---:|
| Fixed scan | 0.1542 | 64.0% | 26.2% | 11.45 |
| Entropy greedy | -0.1865 | 42.6% | 34.4% | 12.99 |
| Decision-aware VoI | **0.4949** | **81.0%** | **16.0%** | **9.61** |

Independent reassignment produces handle resets, switches, and collisions. The degradation relative to stable handles is expected: the factorized oracle binds belief by observed handle and has no learned object association. This stress result creates a measurable target for the future object-centric model.

## Interpretation boundary

The result supports three limited conclusions:

1. The environment is reproducible under fixed seeds.
2. The task distinguishes total uncertainty reduction from decision-relevant evidence acquisition.
3. A transparent one-step decision-aware oracle establishes a meaningful baseline above fixed scanning.

It does not yet support claims about learned beliefs, object-centric representation, RSSM prediction, a learned multi-step policy, robust association, or transfer to real detector data.

## Gate status

| Requirement | Status | Evidence |
|---|---|---|
| Deterministic replay | Pass | repeated trace test |
| Unique target predicate | Pass | 200-seed structural test |
| Randomized target handle | Pass | 100-seed distribution test |
| Occlusion/reappearance | Pass | explicit interval test |
| Entropy versus VoI divergence | Pass | constructed case |
| Oracle above fixed scan | Pass on reference block | 0.6016 versus 0.4897 return |
| Statistical leakage audit | Pass | nuisance probe at 0.5005 balanced accuracy |
| Handle corruption | Pass as stress axis | deterministic reset/switch/collision and 500-seed run |
| Multi-step oracle separation | Pass | one-step DIRECT, two-step SCOUT |

Gate 1 is **complete**. The next gate is learned representation and world-model validity. The first learned milestone should still begin with the stable-handle distribution, then introduce association corruption as an explicit generalization and binding test.
