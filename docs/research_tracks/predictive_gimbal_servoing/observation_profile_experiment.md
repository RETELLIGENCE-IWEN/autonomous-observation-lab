# Randomized Observation-Profile Experiment

## Question

How much do servo telemetry and body-rate telemetry contribute to a causal GRU
when motion, sensing, and actuation vary between episodes?

Three models use identical architecture, initialization seed, optimization,
trajectories, behavior actions, and privileged labels:

- O0: vision, timing, command mode, and previous action;
- O1: O0 plus measured gimbal position and rate;
- O2: O1 plus measured vehicle body rate.

The fixed-width encoder and explicit validity masks are unchanged between
profiles.

## Domain randomization

Every split seed now realizes a distinct variant of each development scenario.
Target and body random streams are independent. Configurable randomization
covers:

- sinusoid amplitude, frequency, phase, bias, and angular drift;
- independently sampled target and body rate pulses;
- gimbal travel, maximum rate/acceleration, time constant, command latency,
  deadband, quantization, and position-loop gain;
- camera FOV, frame rate, detector latency/jitter, center/size noise,
  confidence, and dropout;
- control rate, initial gimbal state, and target angular extent.

The manifest records both the distribution configuration and every realized
per-seed scenario. Analytical replay therefore uses the exact camera and servo
values for each episode.

The experiment contains 72 train, 24 validation, and 24 test world/plant
variants. Three behavior controllers expand those to 216/72/72 episodes. Split
seed blocks are disjoint.

## Calibration contract

Randomizing camera FOV and servo travel/rate exposed an identifiability issue in
the original feature schema: the same normalized image or gimbal value can mean
different physical angles on different hardware. Schema v2 therefore retains
both normalized values and calibrated physical values for image error, gimbal
angle/rate, body rate, and the previous rate/position command. The physical
values come from deployment camera and servo calibration, not simulator truth;
every associated hardware value remains configurable. Legacy schema-v1 datasets
remain loadable.

## Model and training

- one-layer unidirectional GRU;
- 64-dimensional embedding and recurrent state;
- 36,240 trainable parameters per profile;
- horizons: 0.0, 0.1, 0.2, and 0.3 seconds;
- 50 epochs with best-validation checkpoint restoration;
- identical optimizer and training seed for O0/O1/O2.

## Full test-set comparison

| Profile | Bearing RMSE | Rate RMSE | Bearing 2σ coverage | Rate 2σ coverage |
|---|---:|---:|---:|---:|
| O0 vision-only | 22.66° | 35.20°/s | 96.69% | 92.46% |
| O1 servo-aware | **21.47°** | 35.22°/s | 93.04% | 92.11% |
| O2 disturbance-aware | 22.01° | **31.21°/s** | 94.75% | 92.82% |

O1 reduces bearing RMSE by 5.2% relative to O0. O2 reduces rate RMSE by
11.3%. O2 does not dominate every metric: its full-set bearing result is
slightly worse than O1. The telemetry effects are state-specific rather than a
single monotonic ranking.

Rate uncertainty remains under-dispersed: two-standard-deviation coverage is
about 92% rather than the nominal Gaussian 95.45%. This should be calibrated on
validation data before uncertainty controls a safety decision.

## Matched analytical support

The constant-velocity estimator requires gimbal feedback and is valid for
76.0% of eligible labels. The following aggregate comparison uses exactly that
same support:

| Predictor | Bearing RMSE | Rate RMSE |
|---|---:|---:|
| Constant velocity | 11.12° | 47.11°/s |
| O1 GRU | 9.90° | 36.54°/s |
| O2 GRU | **8.20°** | **32.59°/s** |
| Privileged target state | 0.00° | 0.00°/s |

The horizon breakdown shows the causal crossover:

| Horizon | Analytical bearing | O2 bearing | Analytical rate | O2 rate |
|---:|---:|---:|---:|---:|
| 0.0 s | **4.39°** | 5.07° | 35.33°/s | **27.56°/s** |
| 0.1 s | 7.90° | **6.88°** | 44.54°/s | **30.01°/s** |
| 0.2 s | 12.05° | **8.82°** | 51.25°/s | **34.27°/s** |
| 0.3 s | 16.48° | **10.94°** | 55.23°/s | **37.76°/s** |

Explicit geometry remains strongest for immediate bearing. Body-rate-aware
recurrence becomes stronger once extrapolation is required.

## Failure boundary

On the remaining analytical-invalid portion, learned bearing error is 40.16°
for O1 and 42.52° for O2. These intervals include startup, prolonged
stale/dropout observations, and loss-of-view cases where current features do
not identify an off-screen target.
The GRU improves availability but cannot manufacture information. Closed-loop
loss-of-view recovery needs a declared search/fallback behavior rather than an
unqualified predictor-valid flag.

## Interpretation limits

This is stronger than the fixed-trajectory smoke test because test motion and
hardware realizations are untouched and seed-disjoint. It is still a synthetic,
single-training-seed experiment around six authored motion families. It does not
yet establish robustness to recorded flight dynamics, new motion families, or
real hardware.

## Reproduce

Generate the three non-overlapping randomized seed blocks:

```bash
aol-generate-gimbal-dataset \
  --output artifacts/gimbal_randomized_train.npz \
  --split train --seed-start 10000 --episodes 12 \
  --domain-randomization --no-oracle-ceilings

aol-generate-gimbal-dataset \
  --output artifacts/gimbal_randomized_validation.npz \
  --split validation --seed-start 20000 --episodes 4 \
  --domain-randomization --no-oracle-ceilings

aol-generate-gimbal-dataset \
  --output artifacts/gimbal_randomized_test.npz \
  --split test --seed-start 30000 --episodes 4 \
  --domain-randomization
```

Then train the matched models:

```bash
aol-compare-gimbal-gru-profiles \
  --train-data artifacts/gimbal_randomized_train.npz \
  --validation-data artifacts/gimbal_randomized_validation.npz \
  --test-data artifacts/gimbal_randomized_test.npz \
  --checkpoint-directory artifacts/gimbal_profile_checkpoints \
  --output artifacts/gimbal_gru_profile_comparison.json \
  --epochs 50 \
  --batch-size 24 \
  --hidden-dim 64 \
  --embedding-dim 64
```
