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

---

# 한국어 정리 — Dream-to-Look 기술 데모 시스템 개념

## 1. 개념과 목적

Dream-to-Look은 단순히 표적을 따라보는 자동 짐벌이나 tracking controller를 만드는 연구가 아니다.

핵심 목표는 다음과 같다.

> **현재 확보된 관측 정보로부터 무엇을 알고 있고 무엇이 아직 확인되지 않았는지를 판단하고, 그 모름을 해소하기 위해 다음에 어디를 어떤 방식으로 볼지를 스스로 결정하는 자율 관측 에이전트.**

이번 기술 데모는 기존 Dream-to-Look 연구에서 다루던 추상적인 active observation 개념을 실제 UAV 탑재장비와 perception stack에 연결하는 것을 목표로 한다.

탑재장비는 기본적으로 다음 기능을 제공한다고 가정한다.

- 2축 gimbal: azimuth / elevation;
- EO / IR 전환;
- zoom in / out 및 FOV 조절.

Dream-to-Look은 사람이 지정한 명시적인 탑재장비 운용 명령이나 기존 임무 로직이 존재하지 않는 시간에 탑재장비 운용 권한을 가진다.

즉 Dream-to-Look은 모든 기존 탑재장비 운용을 대체하는 것이 아니라, **명시적인 운용 명령이 없는 구간에서 센서를 지능적으로 활용하는 autonomous observation layer**로 본다.

기존 tracking 모델이나 향후 Dream-to-Center와 같은 predictive servoing은 표적을 잘 따라보는 실행 계층으로 활용할 수 있지만, Dream-to-Look의 핵심 질문은 별도이다.

> **지금 무엇을 더 봐야 하는가? 그리고 왜 그것을 봐야 하는가?**

---

## 2. 역할 분담

### 2.1 2D Object Detection 담당

2D OD stack은 영상에서 다음 정보를 제공한다.

- bounding box;
- class;
- confidence;
- objectness;
- scene description.

Scene description은 예를 들어 다음과 같은 coarse context를 제공할 수 있다.

- 산악;
- 산림;
- 해상;
- 사막;
- 도심 등.

2D detection만으로는 일반적으로 정확한 전역 3D 위치를 얻을 수 없다.

따라서 **2D bbox를 공간상의 ray로 변환하고, 그 ray를 시간적으로 관리하고 공유하는 것은 Dream-to-Look 측의 역할**로 둔다.

### 2.2 3D Object Detection / Fusion 담당

3D OD stack은 다음과 같은 객체 상태를 제공한다.

- `x, y, z` 위치;
- `l, w, h` 크기;
- `vx, vy, vz` 속도;
- `roll, pitch, yaw` 자세;
- confidence;
- class.

다수의 UAV가 존재하는 경우 각 UAV의 3D detection을 융합하여 **shared fused detection map**을 생성하고 관리하는 책임 역시 3D stack에 둔다.

Dream-to-Look은 이 fused map을 읽어 사용하지만, 직접 3D fusion이나 multi-sensor tracking을 수행하지 않는다.

### 2.3 Dream-to-Look 담당

Dream-to-Look의 주요 책임은 다음과 같다.

- 2D bbox를 world-referenced viewing ray로 변환;
- ray의 시간적 history, age, confidence, uncertainty 관리;
- multi-UAV 간 2D ray evidence 공유;
- 여러 ray 간 association / cluster / spatial hypothesis 관리;
- 2D evidence와 3D fused evidence 사이의 관계와 불일치 해석;
- 관측해야 할 대상 또는 공간의 우선순위 판단;
- gimbal, zoom, EO/IR action 선택.

이 역할 분리를 통해 특정 detector나 fusion algorithm에 Dream-to-Look이 종속되지 않도록 한다.

---

## 3. Dream-to-Look이 사용할 정보

Dream-to-Look의 입력은 크게 다음 네 그룹으로 볼 수 있다.

### 3.1 UAV state

- position;
- attitude;
- velocity;
- 필요 시 angular rate 등.

### 3.2 Payload state

- gimbal azimuth / elevation;
- gimbal rate 및 limit;
- EO / IR 상태;
- zoom / focal length / FOV;
- 필요 시 latency와 stabilization 상태.

### 3.3 2D Observation Pool

원본 detection 정보는 다음과 같다.

- bbox;
- class;
- confidence;
- objectness;
- timestamp;
- source UAV.

Dream-to-Look은 detection 당시의 camera intrinsic, zoom/FOV, UAV pose, gimbal orientation을 이용해 이를 공간상의 viewing ray 또는 bearing hypothesis로 변환한다.

Ray에는 예를 들어 다음 정보를 포함할 수 있다.

- origin;
- direction;
- bbox에서 계산한 angular extent;
- class belief;
- confidence / objectness;
- age;
- source UAV;
- association / cluster;
- uncertainty;
- validity.

이 ray는 단발성 정보가 아니라 시간적으로 관리되며, 다수 UAV 사이에서 공유될 수 있다.

### 3.4 3D Fused Object Pool

3D stack에서 관리하는 shared fused map을 read-only로 사용한다.

여기에는 다음과 같은 정보가 존재할 수 있다.

- position / extent;
- velocity / orientation;
- class / confidence;
- age / history / fusion metadata.

여기서 중요한 설계 원칙은 **2D ray와 3D fused object를 처음부터 같은 종류의 정보로 취급하지 않는 것**이다.

2D ray는 depth ambiguity가 존재하는 관측 evidence 또는 spatial hypothesis이고, 3D fused object는 보다 강한 world-state estimate다.

따라서 이 둘의 불일치 자체가 Dream-to-Look이 추가 관측을 수행해야 할 이유가 될 수 있다.

---

## 4. 핵심 동작 구조

Dream-to-Look의 내부 사고 흐름은 다음과 같이 정리할 수 있다.

1. **Belief Formation**  
   현재 2D ray, 3D object, scene context, history를 통해 무엇을 알고 있는가?

2. **Gap / Conflict Detection**  
   무엇이 빠져 있고, 모순되고, 오래되었으며, 확신이 부족한가?

3. **Observation Question**  
   지금 무엇을 확인해야 하는가?

4. **Evidence Requirement**  
   어느 방향, 어느 zoom, 어느 modality로 관측해야 그 질문에 답할 가능성이 높은가?

5. **Payload Action**  
   gimbal 이동, zoom 변경, EO/IR 전환 또는 hold.

6. **Belief Update**  
   새 detection뿐 아니라 예상했던 대상이 보이지 않은 경우까지 evidence로 반영.

7. **Sufficiency / Attention Transition**  
   추가 관측이 필요한지, 나중에 다시 볼지, 다른 대상으로 attention을 옮길지 판단.

즉 출력은 단순한 `pan/tilt` 값이 아니라, **왜 그 관측을 선택했는지가 존재하는 observation decision**이어야 한다.

---

## 5. 기술 데모에서 보여줄 수 있는 대표 상황

### 5.1 2D evidence는 존재하지만 3D object가 없는 경우

예를 들어 여러 최근 2D detection ray가 동일한 공간 영역을 지향하지만 3D fused map에는 대응 객체가 존재하지 않을 수 있다.

이 상태는 다음과 같이 해석될 수 있다.

- 아직 해결되지 않은 object hypothesis;
- 3D observability 부족;
- false 2D detection;
- stale/incomplete 3D fusion;
- 다른 viewpoint가 필요한 geometry.

Dream-to-Look은 이 상황에서 다음 행동 중 하나를 선택할 수 있다.

- 해당 영역으로 gimbal 이동;
- zoom-in하여 resolution 증가;
- 주변 context 확인을 위한 zoom-out;
- 보완적 증거를 얻기 위한 IR 전환;
- 현재가 아니라 미래의 더 좋은 viewing geometry에서 revisit.

중요한 것은 **ray가 겹치면 무조건 zoom한다**는 rule을 만드는 것이 아니다.

목표는 추가 관측이 현재의 decision-relevant uncertainty를 실제로 해소할 가능성이 있는지를 판단하는 것이다.

### 5.2 3D object는 존재하지만 2D confirmation이 부족한 경우

3D fused map에는 객체가 존재하지만 최근 2D observation에서는 충분히 확인되지 않을 수 있다.

원인은 예를 들어 다음과 같다.

- 현재 FOV 밖에 있음;
- occlusion;
- 낮은 image resolution;
- 불리한 viewing aspect;
- illumination 또는 modality mismatch;
- detector dropout;
- stale 3D track.

Dream-to-Look은 해당 위치를 다시 바라보고, 적절한 zoom이나 EO/IR modality를 선택하여 추가 evidence를 획득할 수 있다.

### 5.3 Multi-UAV Ray Convergence

UAV A, B, C에서 생성된 서로 독립적인 2D ray가 동일한 공간 영역을 지향하고 있지만 아직 3D fused object가 생성되지 않은 상황을 생각할 수 있다.

이 경우 해당 영역은 observation priority가 높아질 수 있다.

한 UAV가 다음과 같은 역할을 선택할 수 있다.

- 고배율 confirmation look;
- 다른 viewing aspect 확보;
- EO/IR cross-check;
- 주변 상황이 더 중요하면 wide FOV 유지.

이때 multi-UAV의 의미는 단순히 detector 개수가 늘어나는 데 있지 않다.

> **한 UAV의 불완전한 관측이 다른 UAV가 수행할 다음 관측의 가치를 바꿀 수 있다.**

이 점은 기존 단일-agent Dream-to-Look 연구에서 기술 데모로 확장될 때 중요한 요소다.

### 5.4 EO / IR 선택

기본적으로 EO를 main detection sensor로 사용하고 IR은 보완적 sensor로 보는 것이 현재 가정이다.

Scene description도 modality 선택의 context가 될 수 있다. 예를 들어 울창한 산림 환경에서는 특정 상황에서 IR 관측의 가치가 증가할 수 있다.

하지만 다음과 같은 rule 자체가 Dream-to-Look의 연구 결과는 아니다.

> `forest -> IR`

우리가 원하는 행동은 보다 다음에 가깝다.

> 현재 scene, EO evidence, observation history, 2D/3D disagreement, 예상 visibility를 고려할 때 EO를 유지하는 것보다 IR로 전환하는 것이 중요한 uncertainty를 더 잘 해소할 수 있는가?

Zoom과 gaze direction도 같은 원칙으로 본다.

---

## 6. Action Space

기술 데모에서 Dream-to-Look은 최소한 다음 action을 선택할 수 있어야 한다.

### Gimbal

- azimuth;
- elevation.

실제 interface boundary에 따라 angle command, rate command 또는 abstract look-point 형태로 구현할 수 있다.

### Zoom / FOV

- zoom in;
- zoom out;
- hold;
- 필요 시 continuous zoom.

### Sensor Modality

- EO;
- IR.

향후 dwell time, revisit timing, observation purpose 같은 action을 추가할 수 있지만 first integrated demo의 필수조건으로 두지는 않는다.

---

## 7. 기존 Dream-to-Look 연구와의 관계

이번 기술 데모는 기존 Dream-to-Look과 별개의 새로운 연구가 아니다.

기존 연구에서는 scientific question을 깨끗하게 분리하기 위해 의도적으로 다음과 같이 단순화했다.

- pixel 대신 object-feature 입력;
- identification / attention allocation / occlusion / reacquisition staged scenario;
- object-centric RSSM;
- latent-imagination policy;
- 3D fusion map에 대한 dependency 없음;
- 특정 flight backend dependency 없음;
- 초기 benchmark에서는 continuous low-level gimbal control 제외.

이번 기술 데모에서는 여기에 실제 시스템 interface가 붙는다.

| 기존 연구 개념 | 기술 데모에서의 구체화 |
|---|---|
| object-feature observation | 2D detection + 3D fused object + scene context |
| persistent evidence belief | temporal ray hypothesis + 3D object evidence + latent state |
| uncertainty / ignorance / contradiction | 2D/3D mismatch, stale evidence, missing confirmation |
| gaze action | 2축 gimbal pointing |
| observation-quality action | zoom / FOV |
| modality-dependent evidence | EO / IR switching |
| future visibility/evidence prediction | candidate look의 예상 관측 결과 |
| latent imagination | 행동하기 전에 여러 observation future 비교 |

따라서 기존 Dream-to-Look research core는 계속 detector, fusion stack, flight backend와 독립적으로 유지하는 것이 좋다.

이번 문서는 그 core를 실제 **multi-UAV EO/IR payload system에 적용하는 technology-demo embodiment**를 정의한다.

---

## 8. 다음 핵심 아키텍처 문제

다음으로 결정해야 할 가장 중요한 문제는 서로 다른 정보 pool을 Dream-to-Look 내부의 learned information state로 어떻게 변환할 것인가이다.

가장 단순한 방식은 다음을 그대로 concatenate하는 것이다.

- recent 2D rays;
- 3D objects;
- scene context;
- UAV / payload state.

하지만 보다 강한 방향은 **object/hypothesis-centric latent state**를 명시적으로 유지하는 것이다.

각 hypothesis가 다음과 같은 정보를 가질 수 있다.

- supporting 2D rays;
- matching 또는 missing 3D evidence;
- source UAV;
- class / identity belief;
- localization uncertainty;
- last-observed time;
- visibility / occlusion belief;
- contradiction state;
- candidate future observation에서 기대되는 evidence.

이 지점에서 기존에 연구하고 있는 **Object-Centric RSSM, belief state, latent imagination**이 실제 기술 데모 구조와 직접 연결된다.

또한 `unknown`, `invalid`, `not observed`, `contradictory`는 서로 다른 의미이므로 missing value를 단순히 `0`으로 넣어 정상적인 관측값처럼 표현하지 않는 것이 중요하다.

---

## 9. 기술 데모의 성공 기준

약한 데모는 다음과 같다.

> "AI가 객체를 발견하고 짐벌을 돌린 뒤 zoom-in 했다."

Dream-to-Look의 연구 가치를 보여주는 데모는 다음에 가까워야 한다.

> **"현재 정보로 설명되지 않는 관측 가설을 발견했고, 어떤 추가 관측이 그 불확실성을 해소할 수 있을지 판단한 뒤 gaze / zoom / modality를 선택했으며, 새롭게 얻은 증거를 통해 그 가설을 강화하거나 기각했다."**

즉 보여주고 싶은 것은 단순한 자동 tracking이 아니다.

> **"얘가 지금 무엇을 더 확인해야 하는지를 스스로 판단하고, 어떻게 확인할지도 선택한다."**

이 반응이 나오는 것이 기술 데모의 핵심 목표다.

---

## 10. 현재 Working Definition

> **Dream-to-Look은 상위 우선순위의 명시적인 탑재장비 운용 명령이 없는 동안 동작하는 자율 관측 정책이다. 공유된 2D viewing-ray evidence, read-only 3D fused object 정보, scene context, UAV 및 payload state를 이용해 현재 관측 가능한 세계에 대한 불확실한 belief를 유지하고, 임무 판단에 중요한 information gap이나 conflict를 찾아낸 뒤, 가장 가치 있는 다음 evidence를 획득하기 위해 gimbal direction, zoom/FOV 및 EO/IR modality를 능동적으로 선택한다.**

장기적으로 이 observation intelligence는 현재의 특정 2D detector, 3D fusion implementation, UAV platform 또는 payload hardware에 종속되지 않는 방향으로 일반화되어야 한다.
