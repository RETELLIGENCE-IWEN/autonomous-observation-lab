# Sequence-Oracle Distillation V11.1/V11.2

## Question

V11 proves that a constrained privileged command sequence improves tracking,
visibility, smoothness, and saturation. V11.1 asks whether a causal recurrent
actor can imitate the episode-start oracle corrections from deployable O2
history and serialized hardware.

## Student

The student is a hardware-conditioned recurrent absolute-position actor. It
receives the 27 deployable O2 fields plus ten normalized configuration values:
FOV, asymmetric travel, rate and acceleration limits, position gain, command
latency, rate time constant, control period, and camera period. At every step,
logged previous-command fields are replaced with the student's own preceding
output. Privileged target state is used only to create teacher sequences.

Training emphasizes nonzero student/oracle disagreements and critical labels,
while retaining zero-correction cases at lower weight. Evaluation rolls the
student commands through the exact serialized plant from episode start.

## Results

The validation oracle remains useful on this 16-command slice: it improves
global tracking from 0.9232 to 0.9163 and critical tracking from 0.8537 to
0.8268. Neither student transfers that ceiling.

| Arm | Train cases | Global tracking | Critical tracking | Global visibility | Global smoothness | Global saturation |
|---|---:|---:|---:|---:|---:|---:|
| V11.1 student | 48 | +5.74% | +3.80% | +6.32% | **-36.26%** | **-21.05%** |
| V11.2 expanded student | 192 | +3.02% | +3.94% | +2.30% | **-9.77%** | **-11.12%** |

The expanded dataset materially reduces the global regression, so coverage is
part of the problem. Yet both actors reproduce the same failure: smoother,
lower-saturation commands leave the logged privileged trajectory and then
track worse. More teacher-forced epochs are not justified.

## Next gate

The next experiment must aggregate on-policy states. Roll the current student
through counterfactual image/servo observations, query the constrained oracle
around those student-induced trajectories, and add only disagreement and safe
retention examples. Validation must use the same counterfactual feedback loop;
open-loop logged-feature command imitation is no longer an acceptable selector.
The fresh test remains sealed and no V11.1/V11.2 student is promoted.

Reproduce with:

```bash
aol-distill-gimbal-sequence-oracle
```
