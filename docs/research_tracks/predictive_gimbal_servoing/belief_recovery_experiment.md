# Belief-Guided Loss-of-View Recovery

## Question

Can a deterministic recovery layer use the estimator's last target-state
belief to handle detector outages and target re-entry more safely than either
holding the native command or blindly sweeping the gimbal travel envelope?

This is a recovery-policy experiment, separate from the earlier question of
whether the O2 GRU prevents loss of view. The layer wraps either the analytical
constant-velocity estimator or the trained disturbance-aware O2 GRU and works
with both desired-rate and absolute-position adapters.

## Recovery contract

The manager has four explicit states:

1. `TRACK` uses the nominal target-state controller while detections arrive.
2. `COAST` tolerates a short measurement gap and projects the last fresh
   belief while uncertainty grows.
3. `SEARCH` moves toward the last belief's reachable boundary crossing, rather
   than scanning the full envelope. Search-time constant-rate propagation is
   bounded so an old bearing cannot wrap around and reverse direction.
4. `REACQUIRE` requires three fresh detector frames, then blends from the
   recovery command to nominal control over 0.25 s.

Detector capture time and detector-result arrival time are tracked separately.
This matters because a valid frame with 120 ms latency is fresh even though its
measurement timestamp is old. Only a genuinely fresh detection may re-anchor
the belief; recurrent outputs produced from missing inputs cannot silently
replace it.

All timing, uncertainty, search rate, boundary margin, confirmation, blending,
camera, and servo values are configurable. The logical mounting convention is
unchanged: zero gimbal angle is body forward.

## Constructed evaluation suite

The suite contains three authored eight-second events:

- `detector_burst_recovery`: the target remains physically reachable while
  detector outputs are suppressed during 2.0–3.4 s and 5.2–6.0 s;
- `travel_limit_reentry`: the target moves to 86°, beyond the configured
  travel/FOV envelope, then returns to body forward;
- `physically_unreachable`: the same departure occurs, but the target remains
  at 86° through episode end.

The exact event motion is fixed. Eight seeds randomize servo travel, rate,
acceleration, lag, latency, deadband, quantization, camera FOV/rate/noise,
control rate, and initial state. Controllers are paired on the same 24
world-and-hardware variants. O2 uses the forecast horizons already selected on
the earlier disjoint validation split: 0.0 s for rate and 0.2 s for position.

## Results

Each row averages all 24 variants, including the eight deliberately impossible
terminal cases.

| Estimator / adapter | Recovery | Mean error | Mean episode P95 | Loss of view | Cost | Unrecovered events |
|---|---|---:|---:|---:|---:|---:|
| Analytical rate | Hold | 26.36° | 50.86° | 43.63% | 2.427 | 8 |
| Analytical rate | Blind sweep | 31.04° | 69.69° | 42.59% | 3.246 | 8 |
| Analytical rate | **Belief** | **22.04°** | **43.50°** | 43.49% | **1.843** | 8 |
| O2 rate | Hold | 22.96° | 44.47° | **43.42%** | 2.021 | 8 |
| O2 rate | Blind sweep | 33.59° | 74.84° | 47.92% | 3.642 | 10 |
| O2 rate | **Belief** | **22.06°** | **43.03°** | 44.42% | **1.871** | 8 |
| Analytical position | **Hold** | **20.76°** | **42.81°** | **42.40%** | **1.746** | 8 |
| Analytical position | Blind sweep | 26.89° | 53.68° | 47.88% | 2.498 | 9 |
| Analytical position | Belief | 21.86° | 43.40° | 43.24% | 1.840 | 8 |
| O2 position | Hold | **20.42°** | **41.54°** | 41.97% | **1.728** | 9 |
| O2 position | Blind sweep | 33.55° | 67.77° | 57.50% | 3.426 | 16 |
| O2 position | Belief | 21.35° | 41.84° | **41.44%** | 1.789 | **8** |

For analytical rate control, belief recovery reduces mean error by 4.32° and
cost by 0.585 relative to hold, winning paired cost on 20/24 variants. For O2
rate control it reduces mean error by 0.90°, P95 by 1.44°, and cost by 0.150;
loss-of-view time rises by one percentage point. Both rate variants are far
safer than blind sweep.

Position control gives the important negative result. A native position
adapter already holds its last body-relative setpoint when its estimate
expires. Belief projection does not improve aggregate error or cost over that
strong default. O2 belief recovery does reduce loss time by 0.53 percentage
points and converts one terminally unrecovered event, but its cost is 0.062
higher. The belief layer should therefore remain optional by command mode, not
be enabled universally.

## Reachability interpretation

In `physically_unreachable`, every strategy has eight unrecovered events and
roughly 78.2% loss-of-view time. This is the intended physical ceiling: no
controller can point a finite-travel gimbal at the target. Belief-guided rate
control still avoids driving toward the wrong boundary, reducing O2 rate cost
from 3.84 to 3.34, whereas blind sweep raises it to 7.76.

In the two recoverable families, O2 belief position finishes every event; O2
position hold leaves one detector-burst variant unrecovered, and blind search
leaves eight recoverable variants unrecovered. The zero normalized command jump
at the instant of `REACQUIRE` confirms that the anchor-and-blend transition is
continuous. Search occurs while the target is actually visible on 5.5–6.5% of
visible O2 steps, exposing a useful next tuning target.

## Conclusion and next step

The state machine fixes the catastrophic blind-sweep behavior and provides a
clear, configurable recovery contract. It is beneficial for rate adapters, but
the position adapter's native hold remains the stronger aggregate default.

The follow-on [uncertainty calibration experiment](uncertainty_calibration_experiment.md)
fits per-horizon bearing/rate variance scales on validation data. It improves
held-out likelihood and 2σ coverage, but exposes conditional miscalibration and
does not change any recovery transition in this suite. Recovery parameters must
still be tuned on a development seed block before evaluation on a newly held-out
block; the present authored suite must not be reused as an untouched test claim.

The subsequent [development/test protocol](contextual_calibration_and_recovery_protocol.md)
uses disjoint 42000-series development and 43000-series test seeds. Its selected
O2 rate policy improves average tracking and cost on fresh test variants but
adds two unrecovered detector-outage events, so it fails the deployment safety
gate and does not replace native hold.

## Reproduce

```bash
aol-evaluate-gimbal-recovery \
  --o2-checkpoint artifacts/gimbal_mixed_checkpoints/gimbal_gru_o2.pt \
  --control-results artifacts/gimbal_mixed_gru_closed_loop_comparison.json \
  --output artifacts/gimbal_belief_recovery_comparison.json
```

The JSON records the complete recovery and domain-randomization configuration,
per-episode transitions and metrics, aggregate state occupancy, exact seed
block, selected O2 horizons, and paired hold/blind deltas.

## Visual inspection

Replay the seed-41000 detector-outage variant with O2 rate control:

```bash
aol-visualize-gimbal --demo recovery
```

The Rerun dashboard synchronizes hold, blind sweep, and belief recovery. Each
row contains the 3D target/gimbal geometry, 2D detector bbox, true and predicted
bearing with uncertainty, visibility signals, actuator response, and recovery
phase. The summary includes both selected-variant metrics and the full
24-variant aggregate.

Select other recorded cases and adapters without changing code:

```bash
aol-visualize-gimbal --demo recovery \
  --recovery-scenario travel_limit_reentry \
  --recovery-command-mode position \
  --seed 41003
```

For a portable recording, add
`--output artifacts/gimbal_recovery_dashboard.rrd`. The visualizer verifies the
checkpoint and source-result checksums before replaying the exact serialized
hardware variant.
