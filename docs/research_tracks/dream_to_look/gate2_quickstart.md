# Gate 2 Quickstart

## Install learning dependencies

~~~bash
python -m pip install -e ".[all]"
~~~

## Run tests

~~~bash
python -m pytest
~~~

## Run the reference experiment

~~~bash
python -m autonomous_observation_lab.world_models.train \
  --train-episodes 800 \
  --validation-episodes 200 \
  --test-episodes 300 \
  --epochs 8 \
  --batch-size 32 \
  --learning-rate 0.0003 \
  --seed 7 \
  --device cpu \
  --output gate2_seed7.json
~~~

The command trains the deterministic recurrent model, monolithic RSSM, and object-centric RSSM on the same generated trajectories. It evaluates filtering, open-loop prior rollout from step 5, and held-out handle corruption.

For CPU runs, limiting BLAS/OpenMP threads may avoid oversubscription when launching more than one training process. Independent training seeds should normally be run sequentially unless CPU allocation is explicitly controlled.

## Reproduction seeds

The initial report uses model initialization/training seeds 7, 17, and 29. Dataset seed blocks are fixed by the training command:

- training: 40000–40799;
- validation: 50000–50199;
- test: 60000–60299;
- handle-corruption stress: 70000–70299.

Raw JSON should be retained with the exact code revision. The repository records aggregate reference results rather than committing model checkpoints.

