# Visibility-Risk Position V2.1 Experiment

## Motivation

Adaptive position V2 reduced command variation by 13.6% while slightly
improving aggregate tracking over fixed-horizon learned position control. It
nevertheless failed its safety gate because one aggressive-motion rollout
ended out of view.

Forensic replay localized that failure to the final sample at 7.992 s. After a
fast reversal, both controllers requested the same saturated target, but V2's
gimbal was 0.4° farther behind fixed horizon. V2 was outside the camera for
one frame and had no remaining episode time in which to recover. This is a
right-censored edge case, but the unrecovered-event gate was retained without
relaxation.

The resulting V2.1 question is:

> Can the controller temporarily spend some of V2's smoothness margin near the
> predicted image boundary, match fixed horizon's terminal-event count, and
> retain its tracking and command-activity advantages elsewhere?

## Controller

V2.1 retains the same multi-horizon O2 GRU, relative-uncertainty trust, native
hold behavior, and hardware-configured setpoint shaper. It adds a deployable
visibility-risk signal:

```text
predicted angular margin = |predicted bearing − gimbal angle|
                           + uncertainty sigma scale × bearing std

predicted FOV fraction = predicted angular margin / (camera FOV / 2)
```

Risk ramps linearly between configurable onset and full-risk FOV fractions.
The selected candidate begins at 0.55, reaches full risk at 0.85, and includes
one predicted bearing standard deviation. Risk can independently increase the
requested forecast horizon, setpoint rate limit, acceleration limit, and jerk
limit. All are neutral by default, and the camera FOV comes directly from the
scenario configuration.

The selected `preview_125` candidate changes only forecast timing: it adds up
to 125 ms of prediction near the image boundary. The forecast remains clipped
to trained GRU heads and uncertainty-weighted. Servo travel, physical rate,
acceleration, latency, deadband, quantization, and polarity remain enforced by
the independent hardware-configured plant.

```text
multi-horizon GRU ──► actuator-arrival forecast ──► uncertainty blend
                                                       │
camera FOV + predicted error/std ──► visibility risk ──┤
                                                       ▼
                                            risk-extended forecast
                                                       │
                                                       ▼
                                           shaped position command
```

## Development protocol

The previous 80000-series V2 fresh worlds were retained only as historical
evidence. V2.1 selection used eight new world seeds, 81000–81007, across all
six scenario families and all three independently trained GRU initializations:
48 variants and 144 rollouts per controller.

A candidate had to:

- remove at least one V2 unrecovered event;
- remain within +0.25° mean, +0.50° P95, and +0.5 percentage points loss of
  view versus V2;
- add no more than 10% command variation versus V2;
- remain at least 5% smoother than fixed horizon;
- have no more unrecovered events than fixed horizon; and
- introduce no scenario-level terminal-event regression.

The ablation sequence was informative:

- Additional shaping authority without preview improved aggregate tracking but
  removed no terminal event.
- Requiring the estimated image-error motion to point outward reduced guard
  activity but erased the safety benefit, suggesting that rate direction is
  too noisy around fast reversals for a hard gate.
- A 90 ms preview removed one of V2's two excess development events.
- Preview strengths of 125 and 175 ms removed both. The protocol selected
  125 ms because safety was tied and it had lower command variation.

On development, `preview_125` changes mean error by −0.030°, P95 by −0.100°,
loss of view by −0.107 percentage points, and control cost by −0.0036 versus
V2. It raises variation by 6.64% versus V2 but remains 6.38% below fixed
horizon. Unrecovered events change from 15 for V2 to 13, matching fixed.

## Untouched confirmation

After selection was frozen, the evaluator opened eight disjoint world seeds,
82000–82007. This again covers six scenario families and three GRU
initializations: 48 variants and 144 rollouts per controller. No candidate,
threshold, or controller parameter was changed after this block opened.

| Metric | Fixed horizon | Adaptive V2 | Risk V2.1 |
|---|---:|---:|---:|
| Mean absolute error | **13.292°** | 13.341° | 13.310° |
| P95 absolute error | 30.034° | 29.974° | **29.893°** |
| Loss of view | **10.759%** | 11.164% | 11.093% |
| Command variation/s | 1.106 | **0.979** | 1.039 |
| Command RMS | 0.567 | **0.562** | 0.567 |
| Actuator acceleration RMS | 0.566 | **0.546** | 0.552 |
| Mean control cost | 0.802 | 0.804 | **0.800** |
| Unrecovered events | 24 | 24 | 24 |

Relative to V2, V2.1 improves mean error by 0.031°, P95 by 0.081°, loss of
view by 0.071 percentage points, and control cost by 0.0038. It gives back
6.19% command variation.

Relative to fixed horizon, V2.1 is 6.03% smoother, improves P95 by 0.141° and
cost by 0.0017, but has 0.018° higher mean error and 0.334 percentage points
more loss of view. These small regressions remain within the declared
confirmation envelope. All seven aggregate checks, all three training-seed
checks, and all six scenario checks pass.

The aggressive-motion family shows the intended behavior most clearly. Versus
V2, V2.1 improves mean error by 0.221°, P95 by 0.488°, and loss of view by
0.487 percentage points while adding 0.169/s command variation. It adds much
less variation in the other scenario families.

The guard is active during 42.6% of valid confirmation steps. Although its
maximum preview is 125 ms, mean added preview is only 35 ms; the mean effective
horizon after uncertainty weighting is 152 ms.

## Verdict and limits

V2.1 passes its frozen synthetic confirmation gate and becomes the preferred
learned position-controller candidate for subsequent research. It repairs the
specific terminal-event deficit found in V2 while retaining a declared
smoothness advantage over fixed horizon.

This is not hardware deployment approval. All three controllers still record
24 unrecovered confirmation events, largely from aggressive motion and
travel-limit recovery. The absolute tracking, visibility, and recovery
requirements of the real mission remain undeclared, and the simulator has not
yet been fitted to a measured camera, servo, or quadcopter. Recorded motion,
servo identification, embedded inference timing, and hardware-in-the-loop
safety remain necessary.

## Reproduce and inspect

```bash
aol-evaluate-gimbal-visibility-risk-position \
  --validation-data artifacts/gimbal_mixed_validation.npz \
  --test-data artifacts/gimbal_mixed_test.npz \
  --checkpoint 17=artifacts/gimbal_o2_replication_checkpoints/gimbal_gru_o2_seed_17.pt \
  --checkpoint 29=artifacts/gimbal_o2_replication_checkpoints/gimbal_gru_o2_seed_29.pt \
  --checkpoint 43=artifacts/gimbal_o2_replication_checkpoints/gimbal_gru_o2_seed_43.pt \
  --output artifacts/gimbal_adaptive_position_v21.json
```

Open the interactive confirmation dashboard with:

```bash
scripts/open_gimbal_visibility_risk_dashboard.sh
```

It compares fixed horizon, V2, and V2.1 aggregates; shows all scenario deltas;
and replays aggressive motion with target/gimbal tracking, commands, visibility
risk, predicted FOV fraction, and adaptive horizon boost.

For direct physical inspection, open the synchronized controller arena:

```bash
scripts/open_gimbal_controller_arena.sh
```

Its three columns replay fixed horizon, adaptive V2, and risk V2.1 through the
same confirmation world. Each column contains a moving 3D body/gimbal/FOV view
and a normalized 2D camera view with true and delayed detector bounding boxes.
The common plots expose target/gimbal angles, absolute error, requested
position, target visibility, V2.1 risk, and forecast boost. Use
`--arena-scenario`, `--arena-world-seed`, and `--arena-training-seed` to inspect
any of the six scenarios, eight confirmation worlds, and three trained GRU
initializations.

Create a portable recording with:

```bash
aol-visualize-gimbal --demo visibility-risk \
  --output artifacts/gimbal_visibility_risk_v21.rrd
```

The equivalent portable arena recording is:

```bash
aol-visualize-gimbal --demo controller-arena \
  --output artifacts/gimbal_controller_arena.rrd
```
