# Gate 2 Reference Results

## At a Glance

The first Gate 2 experiment provides a strong and initialization-stable positive result for object-centric target-belief learning and occlusion persistence. It does **not** yet demonstrate successful kinematic world modeling: position prediction remains near a trivial mean predictor.

Gate 2 is therefore passed for the learned evidence-belief milestone, with a required kinematic-model correction before observation-policy training.

## Run definition

| Field | Value |
|---|---|
| Training episodes | 800, seeds 40000–40799 |
| Validation episodes | 200, seeds 50000–50199 |
| Untouched test episodes | 300, seeds 60000–60299 |
| Corruption episodes | 300, seeds 70000–70299 |
| Training seeds | 7, 17, 29 |
| Epochs | 8 |
| Batch size | 32 |
| Learning rate | 0.0003 |
| Open-loop prefix | observations through step 4; prior from step 5 |
| Corruption probability | 0.15 |

All values below are mean ± sample standard deviation across the three training seeds.

## Parameter audit

| Model | Trainable parameters |
|---|---:|
| Deterministic recurrent | 90,093 |
| Monolithic RSSM | 69,837 |
| Object-centric RSSM | 70,825 |

The two RSSMs differ by only 1.4% in parameter count. The deterministic baseline is larger, so its weaker result cannot be attributed to a smaller capacity budget.

## Latest-frame baseline

| Metric | Result |
|---|---:|
| Overall target balanced accuracy | 0.629 |
| Overall target AUROC | 0.629 |
| Occlusion target balanced accuracy | 0.484 |

The latest-frame estimator loses the positive target during occlusion and establishes the minimum memory baseline.

## Filtering results

| Model | Balanced accuracy | AUROC | Brier ↓ | Occlusion balanced accuracy | Position RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Deterministic recurrent | 0.609 ± 0.011 | 0.624 ± 0.003 | 0.204 ± 0.003 | 0.559 ± 0.018 | 0.499 ± 0.002 |
| Monolithic RSSM | 0.608 ± 0.007 | 0.621 ± 0.008 | 0.208 ± 0.003 | 0.577 ± 0.012 | 0.498 ± 0.001 |
| Object-centric RSSM | **0.807 ± 0.006** | **0.903 ± 0.005** | **0.115 ± 0.002** | **0.830 ± 0.006** | **0.491 ± 0.005** |

The deterministic model exceeds the latest-frame baseline during occlusion, confirming that learned memory retains useful evidence. The object-centric RSSM produces a much larger and consistent improvement.

## Open-loop prior rollout from step 5

| Model | Balanced accuracy | AUROC | Brier ↓ | Occlusion balanced accuracy | Position RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Deterministic recurrent | 0.577 ± 0.015 | 0.594 ± 0.004 | 0.226 ± 0.003 | 0.572 ± 0.014 | 0.499 ± 0.001 |
| Monolithic RSSM | 0.549 ± 0.012 | 0.576 ± 0.017 | 0.210 ± 0.003 | 0.535 ± 0.015 | 0.497 ± 0.001 |
| Object-centric RSSM | **0.775 ± 0.003** | **0.866 ± 0.003** | **0.132 ± 0.004** | **0.796 ± 0.006** | **0.495 ± 0.003** |

The object-centric prior preserves most of its filtering-level identity information without future observations. The monolithic RSSM loses substantial decision information in open loop.

## Handle-corruption stress

| Model | Balanced accuracy | AUROC | Brier ↓ | Occlusion balanced accuracy | Position RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Deterministic recurrent | 0.579 ± 0.005 | 0.608 ± 0.001 | 0.208 ± 0.003 | 0.539 ± 0.010 | 0.503 ± 0.002 |
| Monolithic RSSM | 0.579 ± 0.006 | 0.599 ± 0.007 | 0.211 ± 0.002 | 0.548 ± 0.010 | 0.501 ± 0.000 |
| Object-centric RSSM | **0.741 ± 0.010** | **0.845 ± 0.007** | **0.136 ± 0.004** | **0.759 ± 0.018** | **0.496 ± 0.004** |

The object-centric model degrades under corrupted association, but retains a large advantage. This is robustness of shared object-wise processing, not proof that the current model solves data association: slots are still indexed by observed handles.

## What the result supports

1. Learned recurrent state contains more occlusion-time target information than the latest frame.
2. At matched parameter count, object-wise factorization learns target evidence much more effectively than a monolithic RSSM on this benchmark.
3. The object-centric prior retains decision-relevant identity information over the tested open-loop horizon.
4. The advantage is stable across three initialization/training seeds.
5. Shared object processing retains partial robustness under unseen handle corruption.

## What the result does not support

- successful object discovery or learned data association;
- variable-cardinality generalization beyond the fixed five-slot tensor;
- accurate kinematic imagination;
- calibrated epistemic uncertainty;
- a superior observation policy;
- transfer to image embeddings or a real detector.

## Kinematic failure

All three models produce position RMSE near 0.49–0.50, close to predicting the center of the normalized scene. The position head is therefore not yet a useful dynamics model.

Likely contributors include:

- position loss being underweighted relative to classification heads;
- observations providing bbox coordinates but no explicit ego/sensor geometry;
- the current recurrent objective favoring persistent identity over velocity inference;
- only eight epochs and no multi-step position-specific loss;
- action-dependent visibility being easier to learn than motion extrapolation.

The next increment should add position/velocity normalization, delta-state targets, horizon-weighted rollout loss, and a latest-velocity baseline before the latent-imagination policy is trained.

## Gate decision

| Gate 2 criterion | Decision |
|---|---|
| Deterministic data/tensor replay | Pass |
| Memory above latest-frame under occlusion | Pass |
| Useful RSSM prior rollout | Pass for object-centric target belief |
| Object-centric entity-level advantage | Pass for target evidence and corruption stress |
| Parameter-count and multi-seed audit | Pass |
| Kinematic prediction | Fail / corrective work required |

Gate 2 is **partially complete by capability**: evidence-belief learning passes; kinematic world modeling remains open. Gate 3 policy learning should not begin until the motion-model correction is validated, because a look policy that exploits inaccurate position rollouts could create a false positive result.

