# Privileged Target-State Dataset

## Purpose

The first learned model will estimate body-relative target bearing and angular
rate from causal deployment observations. Simulator truth supervises that model,
but is never an actor input.

The implementation enforces three separate paths:

~~~text
GimbalObservation ---> O0/O1/O2 encoder ---> deployable features
        action   --------------------------> behavior-action record
simulator motion/diagnostics ---> oracle ---> labels and ceiling controller
~~~

`PrivilegedTargetStateOracle` accepts simulator motion definitions or
`GimbalDiagnostics`. Its controller intentionally does not implement the
deployable `act(GimbalObservation)` interface.

## Supervision

For each valid prediction horizon, the oracle records:

- wrapped body-relative target bearing in radians;
- body-relative target angular rate in radians per second.

It also records ideal normalized rate and position targets. These use the same
configurable servo limits and configurable oracle feedback/preview parameters as
the ceiling rollout. The oracle is an upper-bound diagnostic through the real
simulated actuator dynamics; it does not bypass command latency, bandwidth,
acceleration, rate, or travel limits.

## Deployable feature schema

Each observation profile uses the same fixed-width feature vector. Missing
capabilities are encoded as zero values with false validity masks, not as
apparently valid zeros.

Schema v2 contains:

- control interval and frame-update indicator;
- measurement age and validity;
- bbox center error in normalized and calibrated radians, width, height,
  confidence, and validity;
- normalized and calibrated-radian gimbal position/rate, with validity;
- normalized and calibrated-radian body rate, with validity;
- previous normalized action plus typed physical rate/position command channels;
- rate/position command-mode one-hot fields.

It contains no target bearing, target rate, body bearing, future motion, servo
queue state, or simulator episode identifier. Absolute simulator time is stored
as metadata for alignment, not as a model feature.

Keeping both representations is deliberate. Normalized values remain useful to
the controller, while physical values make labels in radians identifiable when
camera FOV, servo travel, and servo rate vary. The conversion uses the
deployment's configurable camera/servo calibration and does not require
privileged simulator state. Schema-v1 datasets remain loadable for historical
experiments.

## Array layout

| Array | Shape | Meaning |
|---|---:|---|
| `features` | `[episode, profile, time, feature]` | causal O0/O1/O2 model inputs |
| `sequence_mask` | `[episode, time]` | valid steps before padding |
| `actions` | `[episode, time, 4]` | rate/position channels plus mode masks |
| `targets` | `[episode, time, horizon, 2]` | privileged bearing/rate labels |
| `target_mask` | `[episode, time, horizon]` | horizons that remain inside the episode |
| `oracle_actions` | `[episode, time, 2]` | ideal rate and position targets |
| `time_s` | `[episode, time]` | alignment metadata, not a feature |
| episode indices | `[episode]` | seed, scenario, and behavior lookup indices |

All requested observation profiles are derived from the same rollout. Targets,
actions, and timestamps therefore have no profile dimension and cannot drift
between the O0/O1/O2 comparison conditions.

## Determinism and manifests

Generation takes the Cartesian product of requested seeds, scenarios, and
behavior policies. A JSON sidecar stores:

- schema version and exact array shapes/dtypes;
- split and deterministic seed block;
- scenarios, behavior policies, profiles, and prediction horizons;
- complete configurable camera, servo, timing, objective, and motion values;
- a SHA-256 configuration hash;
- privileged rate and position ceiling metrics, when enabled.

The loader disables NumPy pickle support and validates every loaded array
against the manifest. `validate_disjoint_seed_blocks` rejects reused simulator
seeds across manifests.

Without domain randomization, the six named scenarios remain fixed development
probes. With `--domain-randomization`, each split seed independently varies
target motion, body motion, maneuver pulses, camera, servo, timing, and initial
state around those authored families. The manifest stores every realized
variant, making the seed block a genuine motion/plant split rather than merely a
detector-noise split.

## Generate a dataset

After installing the package, run:

```bash
aol-generate-gimbal-dataset \
  --output artifacts/gimbal_target_state_train.npz \
  --split train \
  --seed-start 1000 \
  --episodes 8
```

Add `--domain-randomization` for learning splits. Use disjoint seed ranges for
train, validation, and test.

Repeat `--scenario`, `--behavior`, or `--profile` to select subsets. The
defaults include all six development scenarios, all three observation profiles,
and proportional-rate, predictive-rate, and privileged-oracle-rate collection
behaviors.
