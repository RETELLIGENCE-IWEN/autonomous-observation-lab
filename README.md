# Autonomous Observation Lab

Research on autonomous observation agents that recognize what they do not know and actively seek mission-relevant evidence.

Initial research directions include:

- **Hypothesis-Driven Active Observation** — selecting observations that best distinguish or falsify competing hypotheses.
- **Dream-to-Look** — imagining future visibility and evidence with an object-centric RSSM before acting.
- **Epistemic Distillation** — transferring belief, uncertainty, and observation strategies from privileged teachers to deployable partially observed agents.

This repository will host research notes, simulation environments, baselines, experiments, and demonstrations toward mission-aware and self-directed sensing.

See the [research documentation](docs/README.md) for the current vision, foundations, research tracks, ideas, and prior work.

The predictive-gimbal work to date is summarized as a paper-style retrospective
with exact-data figures in [Dream-to-Center: A Research Journey Toward Predictive One-Axis Gimbal Servoing](docs/research_tracks/predictive_gimbal_servoing/predictive_gimbal_servoing_journey_paper.md).

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

Replicate the O2 controller across independent model initializations while
holding datasets, optimizer, horizon-selection protocol, and paired test worlds
fixed:

```bash
aol-replicate-gimbal-gru-o2 \
  --train-data artifacts/gimbal_mixed_train.npz \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint-directory artifacts/gimbal_o2_replication_checkpoints \
  --output artifacts/gimbal_o2_replication.json \
  --training-seed 17 --training-seed 29 --training-seed 43 \
  --epochs 50 --batch-size 24
```

Inspect per-seed rate/position tracking and loss-of-view metrics with:

```bash
aol-visualize-gimbal --demo replication
```

The [multi-seed report](docs/research_tracks/predictive_gimbal_servoing/gru_multi_seed_replication.md)
shows that all three O2 initializations improve mean error, tail error,
loss-of-view time, and control cost over analytical control in both command
modes. Position horizon selection varies between 0.1 and 0.2 seconds, so that
specific horizon is not yet a stable finding. For mouse-only access, launch
`scripts/open_gimbal_replication_dashboard.sh`.

Open the consolidated baseline-versus-learned verification dashboard with:

```bash
scripts/open_gimbal_performance_dashboard.sh
```

It combines the frozen 24-variant paired comparison, per-scenario analytical
and O2 results, every paired episode delta, and the three-training-seed
replication. The accompanying [performance verification report](docs/research_tracks/predictive_gimbal_servoing/performance_verification.md)
separates the passed relative synthetic tracking gate from unresolved
smoothness, recovery, hardware, and absolute product requirements.

Evaluate the configurable `TRACK`/`COAST`/`SEARCH`/`REACQUIRE` manager on
scheduled detector outages, target re-entry, and a physically unreachable
ceiling with:

```bash
aol-evaluate-gimbal-recovery \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --output artifacts/gimbal_belief_recovery_comparison.json
```

The [belief-guided recovery experiment](docs/research_tracks/predictive_gimbal_servoing/belief_recovery_experiment.md)
compares estimator-native hold, blind travel sweep, and directed belief recovery
for analytical/O2 estimators and both hardware command modes.

Inspect an exact recorded recovery variant as synchronized hold, blind, and
belief-controller rows with:

```bash
aol-visualize-gimbal --demo recovery
```

The recovery dashboard includes the 3D geometry, normalized camera image,
target belief and uncertainty, detector/FOV visibility, actuator commands, and
the `TRACK`/`COAST`/`SEARCH`/`REACQUIRE` phase. Choose another recorded case or
adapter with `--recovery-scenario` and `--recovery-command-mode`. For mouse-only
launching, use `scripts/open_gimbal_recovery_dashboard.sh` directly or point a
desktop shortcut at it.

Calibrate the O2 GRU's predicted standard deviations on validation data, then
evaluate reliability once on the untouched test split:

```bash
aol-calibrate-gimbal-uncertainty \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --output artifacts/gimbal_o2_uncertainty_calibration.json \
  --batch-size 24
```

Open the reliability and per-horizon coverage dashboard with:

```bash
aol-visualize-gimbal --demo calibration
```

For mouse-only access, launch `scripts/open_gimbal_calibration_dashboard.sh`.
The [uncertainty calibration experiment](docs/research_tracks/predictive_gimbal_servoing/uncertainty_calibration_experiment.md)
reports the held-out improvement in Gaussian likelihood and 2σ coverage, the
full reliability-curve limitation, and the null effect on recovery behavior.

Evaluate the deployable context-aware calibration ablation with:

```bash
aol-calibrate-gimbal-contextual-uncertainty \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --output artifacts/gimbal_o2_contextual_uncertainty_calibration.json \
  --batch-size 24
```

Then run recovery threshold selection on development seeds and one frozen
evaluation on fresh test seeds:

```bash
aol-evaluate-gimbal-recovery-protocol \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --uncertainty-calibration artifacts/gimbal_o2_uncertainty_calibration.json \
  --output artifacts/gimbal_recovery_development_test_protocol.json \
  --test-output artifacts/gimbal_recovery_fresh_test.json
```

The [development/test report](docs/research_tracks/predictive_gimbal_servoing/contextual_calibration_and_recovery_protocol.md)
documents both negative gates: contextual scaling does not generalize, and the
development-selected recovery threshold improves averages but adds two
unrecovered fresh-test events. Mouse-only launchers are available at
`scripts/open_gimbal_contextual_calibration_dashboard.sh` and
`scripts/open_gimbal_fresh_recovery_dashboard.sh`.

Run the expanded seven-scenario recovery protocol with a per-scenario
recoverability gate against native hold:

```bash
aol-evaluate-gimbal-recovery-robustness \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --uncertainty-calibration artifacts/gimbal_o2_uncertainty_calibration.json \
  --output artifacts/gimbal_recovery_robustness_protocol.json \
  --test-output artifacts/gimbal_recovery_robustness_fresh_test.json
```

The [expanded recovery report](docs/research_tracks/predictive_gimbal_servoing/recovery_robustness_experiment.md)
shows that every current belief threshold fails development safety, chiefly
because stale constant-rate projection follows the wrong direction during a
target reversal. Native hold remains the default. Inspect the exact fresh-test
failure with `scripts/open_gimbal_robust_recovery_dashboard.sh`.

Evaluate the optional edge-conditioned recovery gate, which uses only the last
valid normalized bbox error and its outward image speed:

```bash
aol-evaluate-gimbal-edge-recovery \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --uncertainty-calibration artifacts/gimbal_o2_uncertainty_calibration.json \
  --output artifacts/gimbal_edge_recovery_protocol.json \
  --test-output artifacts/gimbal_edge_recovery_fresh_test.json
```

The [edge-conditioned report](docs/research_tracks/predictive_gimbal_servoing/edge_conditioned_recovery_experiment.md)
shows that the gate repairs all eight extra development failures from stale
belief search, but narrowly fails the new 46000-series acceptance gate. Native
hold remains selected. Inspect the exact detector-burst failure with
`scripts/open_gimbal_edge_recovery_dashboard.sh`.
