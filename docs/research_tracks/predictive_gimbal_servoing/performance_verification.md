# Baseline vs Learned Performance Verification

## What this verifies

The current primary research comparison is the disturbance-aware O2 GRU
against the analytical constant-velocity target-state controller. Both use the
same configurable rate or position adapter, simulated actuator, camera,
exogenous motion, detector randomness, and initial state. A proportional
controller remains visible as a simpler secondary baseline.

The frozen closed-loop artifact contains 24 paired test variants: four unseen
world/plant seeds across six scenario families. The replication artifact adds
three independently initialized O2 models without selecting the best seed.

## Relative synthetic tracking gate

The present research gate requires every O2 training seed to improve mean
error, episode P95 error, loss-of-view fraction, and declared control cost over
the analytical controller in both command modes. It passes:

| Mode / metric | Analytical | O2 mean ± seed SD | Delta |
|---|---:|---:|---:|
| Rate mean error | 22.84° | **18.57 ± 0.27°** | -4.27° |
| Rate P95 | 47.10° | **38.12 ± 0.58°** | -8.98° |
| Rate loss of view | 27.06% | **19.84 ± 0.43%** | -7.22 pp |
| Rate control cost | 1.742 | **1.287 ± 0.010** | -0.455 |
| Position mean error | 18.90° | **17.07 ± 0.12°** | -1.83° |
| Position P95 | 39.12° | **35.43 ± 0.68°** | -3.69° |
| Position loss of view | 21.28% | **17.51 ± 0.18%** | -3.77 pp |
| Position control cost | 1.275 | **1.124 ± 0.010** | -0.151 |

The gain is strongest under high latency and aggressive motion. On the frozen
seed-17 controller comparison, rate O2 reduces mean error by 8.55° in the
high-latency family and 5.95° under aggressive motion. Position O2 reduces it
by 2.50° and 3.48°, respectively.

## Weaknesses visible in the same evidence

- The physical travel-limit family is effectively tied. Neither estimator can
  recover a target outside the camera/mechanical reachable envelope.
- O2 command variation is higher than analytical control by 0.286/s for rate
  and 0.155/s for position in the frozen comparison. Smoothness is not a
  supported advantage.
- O2 prevents more losses but recovers more slowly after loss:
  event-weighted recovery is 1.07 to 1.58 s for analytical to O2 rate, and
  0.98 to 1.38 s for position.
- The rate adapter consistently selects the current-state GRU output. Its gain
  is evidence for improved recurrent bearing/rate state estimation and
  feed-forward, not a stable future-bearing lead.
- Position performance replicates, but the selected horizon varies between
  0.1 and 0.2 s across training seeds.
- Directed recovery policies fail their separate fresh safety gate, so native
  hold remains the accepted target-loss behavior.

## What is not yet verified

This result verifies a relative synthetic research target. It does not verify
an absolute deployment target because the project does not yet declare maximum
acceptable tracking error, P95 error, loss-of-view time, command activity, or
recovery time for the real mission.

Recorded quadcopter motion, identified camera/servo parameters, embedded
inference latency, real detector behavior, and hardware-in-the-loop safety also
remain untested. Those measurements and explicit mission thresholds are needed
before the words “meets requirements” are justified.

## Inspect

Open the consolidated dashboard with:

```bash
scripts/open_gimbal_performance_dashboard.sh
```

The view contains:

1. the aggregate analytical, proportional, and O2 metrics;
2. per-scenario analytical and O2 curves for both command modes;
3. all 24 paired O2-minus-analytical episode deltas, where values below zero
   favor O2; and
4. O2 mean-error stability across the three training initializations.

Generate a portable recording instead of opening the viewer with:

```bash
aol-visualize-gimbal --demo performance \
  --output artifacts/gimbal_performance_verification.rrd
```

The follow-on [adaptive position V2 experiment](adaptive_position_v2_experiment.md)
tests a multi-horizon, uncertainty-aware, jerk-limited position adapter against
this fixed-horizon learned controller. It reduces command variation by 13.6%
without aggregate tracking regression, but remains rejected after one
additional unrecovered aggressive-motion event.
