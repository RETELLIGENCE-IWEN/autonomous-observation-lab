# Autonomous Observation Lab

Research on autonomous observation agents that recognize what they do not know and actively seek mission-relevant evidence.

Initial research directions include:

- **Hypothesis-Driven Active Observation** — selecting observations that best distinguish or falsify competing hypotheses.
- **Dream-to-Look** — imagining future visibility and evidence with an object-centric RSSM before acting.
- **Epistemic Distillation** — transferring belief, uncertainty, and observation strategies from privileged teachers to deployable partially observed agents.

This repository will host research notes, simulation environments, baselines, experiments, and demonstrations toward mission-aware and self-directed sensing.

See the [research documentation](docs/README.md) for the current vision, foundations, research tracks, ideas, and prior work.

> Early-stage research repository. Ideas and interfaces may evolve substantially.

## Gimbal causality demo

The predictive-gimbal track includes a deterministic comparison in which the
same image-plane bounding-box motion is produced by two different causes:

1. a stationary target and a moving position-controlled gimbal; and
2. a stationary body-forward gimbal and a moving target.

The Rerun view synchronizes a 3D world view, normalized 2D image frame, and
position/rate/bbox traces. Install and launch it with:

```bash
python3 -m pip install -e '.[visualization]'
aol-visualize-gimbal
```

For a portable recording instead of a live window:

```bash
aol-visualize-gimbal --output artifacts/gimbal_cause_demo.rrd
rerun artifacts/gimbal_cause_demo.rrd
```

The closed-loop benchmark compares proportional rate, proportional position,
and causal predictive-rate control while both the target and vehicle body move:

```bash
aol-visualize-gimbal --demo closed-loop
```

Save and share the synchronized benchmark with:

```bash
aol-visualize-gimbal --demo closed-loop --output artifacts/gimbal_closed_loop.rrd
```

On a Linux desktop, `scripts/open_gimbal_dashboard.sh closed-loop` launches the
viewer without a terminal. A desktop shortcut can point to this script for
mouse-only access.

The full configurable stress matrix is available with:

```bash
aol-visualize-gimbal --demo benchmark-suite
```

## Privileged gimbal dataset

Generate deterministic causal features and simulator-truth target-state labels
for the next learning stage with:

```bash
aol-generate-gimbal-dataset \
  --output artifacts/gimbal_target_state_train.npz \
  --split train \
  --seed-start 1000 \
  --episodes 8
```

Use `--domain-randomization` with disjoint seed ranges for learning experiments;
the seed then controls independently varied motion, camera, servo, timing, and
initial state, all recorded exactly in the manifest.

The paired JSON manifest records the schema, exact configurable scenario and
hardware values, split seeds, array shapes/dtypes, configuration hash, and optional
privileged rate/position ceiling results. See the
[dataset specification](docs/research_tracks/predictive_gimbal_servoing/privileged_dataset.md).

## Causal gimbal GRU

With the optional learning dependencies installed, train the first causal
bearing/rate predictor with:

```bash
aol-train-gimbal-gru \
  --train-data artifacts/gimbal_gru_train.npz \
  --validation-data artifacts/gimbal_gru_validation.npz \
  --test-data artifacts/gimbal_gru_test.npz \
  --profile o1_servo_aware \
  --checkpoint artifacts/gimbal_gru_o1.pt
```

The model predicts bearing, rate, and uncertainty at every configured horizon;
its streaming adapter plugs into either rate or position control. The initial
[GRU smoke experiment](docs/research_tracks/predictive_gimbal_servoing/gru_smoke_experiment.md)
shows the expected crossover: analytical geometry is better at the current
bearing, while the GRU becomes stronger at longer-horizon rate prediction.

Train a matched observation-profile ablation with:

```bash
aol-compare-gimbal-gru-profiles \
  --train-data artifacts/gimbal_randomized_train.npz \
  --validation-data artifacts/gimbal_randomized_validation.npz \
  --test-data artifacts/gimbal_randomized_test.npz \
  --checkpoint-directory artifacts/gimbal_profile_checkpoints \
  --output artifacts/gimbal_gru_profile_comparison.json
```

See the [randomized O0/O1/O2 experiment](docs/research_tracks/predictive_gimbal_servoing/observation_profile_experiment.md)
for the matched analytical comparison and current loss-of-view limitation.

Evaluate trained O1/O2 checkpoints as actual rate and position controllers with
validation-selected horizons and paired randomized test scenarios:

```bash
aol-evaluate-gimbal-gru-control \
  --train-data artifacts/gimbal_mixed_train.npz \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --o1-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o1.pt \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --output artifacts/gimbal_mixed_gru_closed_loop_comparison.json
```

The [closed-loop experiment](docs/research_tracks/predictive_gimbal_servoing/gru_closed_loop_experiment.md)
reports tracking error, loss of view, recovery, saturation, command effort, and
the negative result from a naive search fallback.
