# Concept Lock: Predictive 1D Gimbal Servo

## Decision

**Locked on:** 2026-08-26  
**Working name:** Dream-to-Center  
**Project type:** deployable mini-project with a falsifiable research contribution

The mini-project will build a compact recurrent AI controller for the outer visual-servo loop of a one-axis camera gimbal. It will continuously command desired angular rate to keep a designated bbox centered under quadcopter oscillations and maneuvers.

The scientific proposition is that privileged predictive training can give a deployable controller a useful internal belief about image motion, platform disturbance, delay, and actuator response. The engineering proposition is that the resulting controller can run at camera rate on embedded hardware while an ordinary inner rate loop and independent safety layer retain physical control authority.

---

## Locked system boundary

~~~text
bbox/center-size + timestamps + deployment telemetry
                         |
                         v
        compact recurrent predictive policy
              20–60 Hz visual outer loop
                         |
             desired angular-rate command
                         |
                         v
          constraint projection and watchdog
                         |
                         v
       conventional embedded motor/rate servo
                 approximately 200–1000 Hz
                         |
                         v
                    1D gimbal
~~~

The learned policy owns the visual continuous-control decision. The conventional inner loop does not choose where to point; it makes the motor safely follow the requested rate.

---

## Locked decisions

| Topic | Decision |
|---|---|
| Controlled task | Keep one designated object's relevant bbox-center coordinate at image center |
| Degrees of freedom | One rotational gimbal axis per experiment |
| Perception boundary | Begin from bbox/center-size; object detection and identity selection are external |
| Deployment inputs | Use all telemetry genuinely available on the final system, including timestamps and gimbal feedback; use body-rate/attitude if the real interface provides it |
| Bbox-only condition | Required ablation, or primary condition only when imposed by the real interface |
| Learned state | Small recurrent state conditioned on current observation and previous action |
| Predictive supervision | Short-horizon future image error, scale, visibility/FOV-exit risk, and actuator response |
| Action | Continuous desired angular rate, normalized and bounded |
| Training stage 1 | Imitation warm-start from a competent PID/MPC teacher |
| Training stage 2 | Privileged distillation from true LOS, body motion, actuator state, and delay state |
| Training stage 3 | Continuous-control RL fine-tuning, initially SAC |
| Deployment policy | Deterministic recurrent actor using only the declared deployable observation profile |
| Low-level control | Existing motor/rate servo retained outside the learned policy |
| Safety | Independent angle/rate/acceleration constraints, observation watchdog, OOD handling, and fallback |
| Primary evaluation | Tail centering error, FOV loss, control effort, saturation, and held-out dynamics—not reward alone |
| Strong baselines | Tuned PID, estimator plus PID, disturbance-observer/predictive control, feed-forward RL, and recurrent model-free RL |
| Raw pixels | Excluded from the initial study |
| Online latent planning | Not the initial deployed controller; optional later diagnostic or ablation |

---

## Locked research identity

The project is not presented as:

- the first AI or RL gimbal controller;
- an end-to-end detector-to-motor system;
- a neural replacement for motor stabilization;
- a claim that recurrence alone is novel;
- a controller proven robust merely because training used randomization.

The candidate contribution is:

> A privileged-trained, deployable recurrent visual servo that internalizes short-horizon platform and actuator dynamics, directly controls a 1D gimbal outer loop, and is evaluated for causal predictive advantage and generalization across unseen maneuvers, delays, and payload parameters.

The novelty claim remains provisional until systematic literature and patent review and positive experimental gates.

---

## Deployment principles

1. Do not withhold telemetry that the real payload can reliably provide merely to make the learning problem harder.
2. Do not expose privileged simulator state to the deployed actor.
3. Do not ask the learned policy to reproduce motor commutation or electrical stabilization.
4. Keep inference comfortably inside the visual-loop period and measure tail latency.
5. Treat missing/stale detections, saturation, and timing jitter as normal operating conditions.
6. Validate first in deterministic simulation, then hardware-in-the-loop, an oscillating bench, restrained flight, and finally free flight.
7. Preserve a tested safe response when the learned policy is uncertain, late, invalid, or outside its operating envelope.

---

## Research mechanism that must be demonstrated

The predictive heads are not decorative auxiliary losses. The study must show that they shape a state used for control:

- action-conditioned recurrence must outperform action-agnostic recurrence where self-motion matters;
- prediction beyond the sensing/actuation delay must correlate with anticipatory commands;
- removing privileged predictive distillation must reduce the claimed generalization advantage;
- prediction improvement must translate into tail-error or FOV-retention improvement;
- unpredictable impulses must remain labeled as reactive recovery cases rather than anticipated successes.

---

## Decisions that remain open without changing the concept

- pan/yaw versus tilt/pitch;
- exact bbox coordinate convention and camera FOV;
- available encoder and vehicle telemetry;
- camera, detector, communication, and actuator rates;
- desired-rate command units and firmware interface;
- gimbal range, rate, acceleration, deadband, and latency;
- simulation backend and sources of recorded quadcopter motion;
- embedded compute target;
- hardware-test safety envelope.

These values instantiate the locked concept. They do not reopen its research identity.
