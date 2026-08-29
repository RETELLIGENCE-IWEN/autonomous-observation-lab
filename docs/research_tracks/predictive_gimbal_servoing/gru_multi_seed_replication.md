# O2 GRU Multi-Seed Replication

## Question

Do the O2 GRU closed-loop gains survive independent model initialization, or
are they an accidental result of training seed 17?

## Protocol

Three 36,240-parameter O2 GRUs use training seeds 17, 29, and 43. Everything
else is frozen: model architecture, optimizer, 50-epoch budget, batch size,
train/validation/test datasets, controller gains, and scenario realization.
The datasets contain 432/144/144 episodes and have disjoint configuration
hashes:

- train: `a9d0a1da8564296ea85d387f53eec090f1dfdb99a7f756e05a9c219b1eba9423`;
- validation: `03368c8486c4bce0533515e97d619c25ca8f0c4a50f60dac0403a9294fa12927`;
- test: `4b014474527ae15952c734e3e855bc33cc9e590556ac73538b75a8e6bb0118`.

Each trained model independently selects its rate and position forecast
horizons on the same 24 validation world/plant variants. The selected model is
then evaluated once on the same 24 paired test variants. Analytical baselines
are replayed once and shared across replications. No best training seed is
selected; initialization seed is the replication unit, and dispersion below is
the sample standard deviation across the three seeds.

## Rate-control result

| Metric | Analytical | O2 mean ± seed SD | O2 range | Every seed improves? |
|---|---:|---:|---:|:---:|
| Mean error | 22.84° | **18.57 ± 0.27°** | 18.27–18.81° | yes |
| Mean episode P95 | 47.10° | **38.12 ± 0.58°** | 37.47–38.58° | yes |
| Loss of view | 27.06% | **19.84 ± 0.43%** | 19.39–20.25% | yes |
| Control cost | 1.742 | **1.287 ± 0.010** | 1.279–1.299 | yes |
| Rate saturation | 13.62% | 13.66 ± 2.16% | 11.76–16.01% | no |
| Command variation/s | 2.301 | 2.468 ± 0.109 | 2.374–2.587 | no |

The mean deltas from analytical control are -4.27° mean error, -8.98° P95,
-7.22 percentage points loss of view, and -0.455 control cost. All three
initializations select the current-state (`0.0 s`) output for rate control.
The replicated gain therefore comes from the learned bearing/rate state rather
than leading the rate loop with a future bearing target.

## Position-control result

| Metric | Analytical | O2 mean ± seed SD | O2 range | Every seed improves? |
|---|---:|---:|---:|:---:|
| Mean error | 18.90° | **17.07 ± 0.12°** | 16.95–17.20° | yes |
| Mean episode P95 | 39.12° | **35.43 ± 0.68°** | 35.03–36.22° | yes |
| Loss of view | 21.28% | **17.51 ± 0.18%** | 17.32–17.69% | yes |
| Control cost | 1.275 | **1.124 ± 0.010** | 1.117–1.136 | yes |
| Command variation/s | 1.032 | 1.012 ± 0.157 | 0.885–1.187 | no |

The mean deltas are -1.83° mean error, -3.69° P95, -3.77 percentage points
loss of view, and -0.151 control cost. Position horizon selection is not stable:
seed 17 selects `0.2 s`, while seeds 29 and 43 select `0.1 s`. The position-mode
performance gain replicates, but the earlier claim that a specific `0.2 s`
forecast is responsible does not.

## Decision

The useful closed-loop claim advances from a single initialization to a
three-initialization replication: O2 improves mean error, tail error,
loss-of-view time, and the declared control cost over analytical control in
both adapters for every training seed tested. The evidence does not support a
smoothness or saturation improvement, and it does not yet establish one stable
predictive position horizon.

This remains synthetic evidence on one frozen split and only three model
initializations. It reduces initialization risk; it does not test new motion
families, recorded flight disturbances, camera/servo system identification, or
hardware timing.

## Reproduce and inspect

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

Open the per-seed rate/position result dashboard with:

```bash
aol-visualize-gimbal --demo replication
```

For mouse-only access, launch `scripts/open_gimbal_replication_dashboard.sh`.
The portable recording is `artifacts/gimbal_o2_replication.rrd` when generated
with `--output`.
