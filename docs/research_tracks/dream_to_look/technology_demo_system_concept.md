# Dream-to-Look Technology Demo System Concept

## 1. Purpose

This document records the current technology-demo concept for **Dream-to-Look** and connects it to the existing research direction in this repository.

The central idea is unchanged from the research track:

> Dream-to-Look is not primarily a gimbal tracker. It is an autonomous observation agent that maintains imperfect evidence about the world, identifies what remains unresolved, and actively selects the next observation expected to improve mission-relevant knowledge.

The technology demo makes that abstract research idea concrete for a UAV payload system with real perception interfaces, multi-UAV information sharing, and a controllable EO/IR gimbal.

---

## 2. Operational role

The payload is assumed to provide:

- 2-axis gimbal control;
- EO/IR modality switching;
- zoom in/out or equivalent FOV control.

Dream-to-Look receives payload authority only when there is no higher-priority explicit payload command.

Higher-priority functions may include:

- human-designated payload pointing;
- mission-defined area sweep;
- rule-based sensor operation;
- existing tracking logic;
- other explicit mission sensor tasks.

Therefore Dream-to-Look is an **autonomous observation layer for otherwise unallocated sensing time**, not a replacement for every payload mode.

A future Dream-to-Center or predictive servoing component may improve tracking execution, but the Dream-to-Look research question remains distinct: **what should be observed next, and why?**

---

## 3. System context and responsibility boundary

Three technical stacks are considered.

### 3.1 2D object detection stack

Provides image-plane detections such as:

- bounding box;
- class;
- confidence;
- objectness;
- coarse scene description.

Representative scene descriptions may include:

- forest;
- mountain;
- sea;
- desert;
- urban;
- other coarse environment categories.

The 2D stack does **not** provide a reliable global 3D object position.

### 3.2 3D object detection / fusion stack

Provides object-level 3D information such as:

- position: `x, y, z`;
- size: `l, w, h`;
- velocity: `vx, vy, vz`;
- orientation: `roll, pitch, yaw`;
- confidence;
- class.

For multi-UAV operation, the 3D stack is responsible for creating and maintaining the shared fused detection map.

Dream-to-Look treats that map as a **read-only perception product**. It does not own global 3D fusion or multi-sensor tracking.

### 3.3 Dream-to-Look stack

Dream-to-Look owns:

- projection of 2D detections into world-referenced viewing rays;
- temporal management of those rays;
- multi-UAV sharing of 2D ray evidence;
- maintenance of observation hypotheses and unresolved information;
- interpretation of relationships between 2D and 3D evidence;
- active selection of gimbal, zoom, and EO/IR actions.

This responsibility split preserves the broader repository principle that perception providers are replaceable interfaces and that the observation agent should not depend on one specific detector or fusion implementation.

---

## 4. Information available to Dream-to-Look

Dream-to-Look can observe four major information groups.

### 4.1 Platform state

- UAV position;
- UAV attitude;
- linear velocity;
- optionally angular rate and other motion state required for geometric projection.

### 4.2 Payload state

- gimbal azimuth/elevation;
- gimbal rates or limits when available;
- EO/IR mode;
- zoom or focal-length/FOV state;
- sensor latency or stabilization state when relevant.

### 4.3 2D observation information pool

Raw inputs originate as detections:

- bbox;
- class;
- confidence;
- objectness;
- timestamp/source UAV.

Dream-to-Look converts each useful detection into a world-referenced **viewing ray or bearing hypothesis** using:

- camera intrinsics;
- camera/gimbal geometry;
- current zoom/FOV;
- UAV position and attitude;
- payload orientation.

The resulting ray representation may carry:

- ray origin;
- direction;
- angular extent derived from the bbox;
- class belief;
- confidence/objectness;
- age;
- source UAV;
- association/cluster information;
- uncertainty and validity.

These rays are maintained over time and may be shared between UAVs.

### 4.4 3D fused object information pool

Dream-to-Look reads the shared 3D fused map maintained by the 3D stack.

It may use:

- object position and spatial extent;
- velocity and orientation;
- class/confidence;
- age/history/fusion metadata if exposed.

The key design point is that **2D rays and 3D fused objects remain different kinds of evidence**. A 2D ray is an uncertain spatial hypothesis with depth ambiguity; a fused 3D object is a stronger world-state estimate. Their mismatch is informative and should not be erased by prematurely forcing both into an identical representation.

---

## 5. Core observation-intelligence loop

The technology demo should implement the same perceptual-inquiry structure used by the broader Autonomous Observation Lab vision:

1. **Belief formation** — What is currently believed from 2D rays, 3D objects, scene context, and history?
2. **Gap/conflict detection** — Which evidence is missing, contradictory, stale, weak, or unexplained?
3. **Observation question** — What specifically should be checked next?
4. **Evidence requirement** — Which direction, zoom level, modality, or dwell condition would best answer that question?
5. **Payload action** — Move the gimbal, change zoom, switch EO/IR, or hold.
6. **Belief update** — Incorporate the resulting detection or non-detection.
7. **Sufficiency / attention transition** — Continue, revisit later, or move attention elsewhere.

The important output is therefore not merely `pan/tilt` but an observation decision that has an epistemic reason.

---

## 6. Candidate demo behaviors

The following are examples, not hard-coded rules that define Dream-to-Look.

### 6.1 2D evidence without matching 3D evidence

Example condition:

- one or more recent 2D detection rays indicate a spatial region;
- several rays may converge geometrically;
- no corresponding object exists in the 3D fused map.

Possible interpretation:

- unresolved object hypothesis;
- poor 3D observability;
- false 2D detection;
- stale/incomplete fusion;
- geometry that warrants another viewpoint.

Possible Dream-to-Look actions:

- slew toward the region;
- zoom in for higher-resolution evidence;
- zoom out for context;
- switch to IR if the context suggests complementary evidence;
- revisit from a different future viewing geometry.

The intelligent behavior is not "always zoom when rays intersect." The target behavior is to learn or estimate **whether another observation is expected to resolve a decision-relevant uncertainty**.

### 6.2 3D evidence without sufficient 2D confirmation

Example condition:

- the shared 3D map contains an object;
- current/recent 2D evidence is absent, weak, contradictory, or stale.

Possible causes include:

- object outside the current FOV;
- occlusion;
- insufficient image resolution;
- poor viewing aspect;
- illumination or modality mismatch;
- detector dropout;
- stale 3D track.

Possible actions:

- direct gaze toward the predicted object location;
- choose an appropriate zoom;
- switch EO/IR;
- reacquire from another angle when geometry changes.

### 6.3 Multi-UAV ray convergence

Example condition:

- UAV A, B, and C produce independent 2D rays;
- the rays geometrically support the same spatial region;
- the 3D map does not yet contain a confident object there.

This region may receive increased observation priority.

A Dream-to-Look instance could then choose to:

- obtain a high-resolution confirmation look;
- acquire a complementary aspect;
- change modality;
- preserve wide FOV instead if contextual evidence is more useful.

This is an important extension of the first-cycle research benchmark: multiple UAVs do not merely provide more detections; **their incomplete observations can change the expected value of another UAV's next observation**.

### 6.4 EO/IR modality selection

EO is expected to remain the primary detection modality, while IR provides complementary evidence.

Scene description can serve as context. For example, dense forest may increase the expected value of IR in some situations.

However:

> `forest -> IR` is a useful engineering rule, but it is not by itself the Dream-to-Look research contribution.

The desired research behavior is closer to:

> Given scene context, current EO evidence, observation history, 2D/3D disagreement, and predicted visibility, would switching to IR be expected to resolve an important uncertainty better than continuing with EO?

The same distinction applies to zoom selection and gaze direction.

---

## 7. Action space for the technology demo

The practical action interface should support at least:

### Gimbal

- azimuth command;
- elevation command.

This may be represented as desired angle, rate, or an abstract look target depending on the payload controller boundary.

### Zoom / FOV

- zoom in;
- zoom out;
- hold;
- optionally continuous zoom command.

### Sensor modality

- EO;
- IR.

A later interface may also expose dwell time, revisit timing, or structured observation purpose, but these are not required for the first integrated demo.

---

## 8. Relationship to the existing Dream-to-Look research track

The current technology-demo concept is not a new research direction. It is the **system embodiment** of the existing Dream-to-Look idea.

The existing first-cycle research intentionally simplified the problem to isolate the scientific question:

- object-feature inputs rather than full rendered perception;
- staged identification, observation allocation, occlusion, and reacquisition;
- object-centric RSSM;
- latent-imagination policy;
- no required 3D fusion map;
- no required flight backend;
- no continuous low-level gimbal control in the first benchmark.

The technology demo now introduces real system interfaces:

- 2D bbox detections;
- world-referenced 2D ray management;
- shared 3D fused detections;
- scene description;
- multi-UAV evidence sharing;
- actual 2-axis gimbal, zoom, and EO/IR actions.

The conceptual mapping is:

| Research concept | Technology-demo realization |
|---|---|
| object-feature observation | 2D detections + 3D fused objects + scene context |
| persistent object/evidence belief | temporal 2D ray hypotheses + 3D object evidence + learned latent state |
| evidence conflict / ignorance | 2D/3D mismatch, stale evidence, missing confirmation |
| gaze action | 2D gimbal pointing |
| observation-quality action | zoom/FOV selection |
| modality-dependent evidence | EO/IR switching |
| future visibility/evidence prediction | expected outcome of candidate looks |
| latent imagination | compare candidate observation futures before acting |

Therefore the original Dream-to-Look track should remain backend-independent, while this document defines one important **multi-UAV EO/IR payload embodiment** for demonstration and integration.

---

## 9. Internal-state design question

The next major architecture decision is how to convert the heterogeneous information pools into the learned information state used by Dream-to-Look.

A naive option would concatenate:

- recent 2D rays;
- 3D objects;
- scene context;
- platform/payload state.

A stronger option is to maintain explicit **object/hypothesis-centric latent state** in which the model reasons about:

- supporting 2D rays;
- matching or missing 3D evidence;
- source UAVs;
- class/identity belief;
- localization uncertainty;
- last-observed time;
- visibility/occlusion belief;
- contradiction state;
- expected evidence under candidate future observations.

This is where the existing object-centric RSSM and belief-state research becomes directly relevant.

The design should preserve `unknown`, `invalid`, `not observed`, and `contradictory` as distinct states rather than encoding missing information as ordinary zero-valued features.

---

## 10. Technology-demo success criterion

A weak demo would show:

> "The AI moved the gimbal and zoomed onto an object."

A stronger Dream-to-Look demo should make the causal reason for the action visible:

> "The agent noticed that the current evidence did not explain a spatial hypothesis, predicted that a particular look would be informative, changed its gaze/zoom/modality, and then updated or rejected that hypothesis from the resulting observation."

The desired audience reaction is therefore not simply:

> "It tracks automatically."

but:

> **"It recognized what still needed to be checked and chose how to check it."**

---

## 11. Current working definition

> **Dream-to-Look is an autonomous observation policy that operates when no higher-priority payload command is active. It uses shared 2D viewing-ray evidence, read-only 3D fused object information, scene context, and UAV/payload state to maintain an uncertain belief about the observable world, identify decision-relevant information gaps or conflicts, and actively select gimbal direction, zoom/FOV, and EO/IR modality to acquire the most useful next evidence.**

This definition should remain compatible with the broader repository goal: the observation intelligence should eventually generalize beyond this specific detector set, fusion implementation, UAV platform, and payload configuration.
