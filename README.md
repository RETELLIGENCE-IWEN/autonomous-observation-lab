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
