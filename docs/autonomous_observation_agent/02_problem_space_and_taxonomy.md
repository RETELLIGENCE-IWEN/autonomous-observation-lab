# 2. Problem Space and Taxonomy

## 2.1 특정 setup이 아닌 capability로 정의하기

연구대상은 다음과 같은 특정 연결구성이 아니다.

- `3D Fusion Map → Payload Intelligence → Flight Intelligence`
- `edgeTAM bbox → Payload Intelligence → GP-G-`

이를 다음의 추상적 provider/contract로 표현한다.

| Contract | 핵심 질문 |
|---|---|
| Perception Provider | 앞단은 무엇을 얼마나 정확하게 알려주는가? |
| Mission Provider | 무엇을 관측하거나 판단해야 하는가? |
| Platform State | 기체가 현재 어떻게 움직이고 있는가? |
| Payload Capability | 어떤 센서 행동이 가능한가? |
| Motion Provider | 관측조건 요청을 받을 수 있는가? |

자율 관측 에이전트는 제공자 이름이나 내부 알고리즘을 몰라야 한다. edgeTAM, 3D fusion stack, GP-G-, Flight Intelligence는 각각 교체 가능한 구현체다.

## 2.2 Perception capability levels

3D fusion map은 필수 입력이 아니라 가장 풍부한 정보 형태다.

| Level | 정보 | 대표 예시 |
|---|---|---|
| P0: Image/Feature Only | 영상 또는 feature map | raw EO/IR, vision embedding |
| P1: 2D Detection | bbox, class, confidence | edgeTAM detection |
| P2: 2D Tracklet | track ID, bbox history, image velocity | detector + tracker |
| P3: Bearing-Aware Track | LOS, angular rate, range estimate/uncertainty | camera calibration + ownship state |
| P4: Local 3D Track | 상대 3D position/velocity/covariance | monocular estimator, local fusion |
| P5: Fused World Model | 전역 object map, history, multi-sensor belief | 3D fusion map |

공통 policy core는 각 입력을 object-centric belief로 변환하는 adapter 뒤에 위치할 수 있다. 공통 표현의 후보는 다음과 같다.

- direction or image position
- apparent size and scale trend
- relative motion
- class/identity belief
- localization uncertainty
- last-observed time and age
- visibility/occlusion belief
- supporting and contradicting evidence
- validity mask and information-source identity

값이 없는 필드를 0으로 채워 정상값처럼 보이게 하면 안 된다. `validity mask`, `uncertainty`, `source token`으로 모름을 명시해야 한다.

## 2.3 bbox-only 조건은 POMDP다

bbox만으로는 다음 원인을 구분하기 어렵다.

- 표적이 멀어진 것과 zoom-out의 차이
- 표적의 실제 이동과 ownship/gimbal motion의 차이
- 실제 차폐와 detector dropout의 차이
- 객체가 멈춘 것과 동일한 영상위치를 유지하는 상대운동의 차이

따라서 내부 belief가 필요하다.

\[
b_t=f(b_{t-1},z_t,a^{sensor}_{t-1},x^{platform}_t,x^{payload}_t)
\]

- \(z_t\): 현재 perception observation
- \(a^{sensor}_{t-1}\): 이전 gaze/sensor action
- \(x^{platform}_t\): 플랫폼 motion state
- \(x^{payload}_t\): 짐벌·센서 state

외부 3D fusion map이 있으면 풍부한 belief가 제공되는 것이고, bbox-only에서는 제한적인 belief를 에이전트가 자체적으로 형성한다.

## 2.4 Payload capability profiles

장비구성은 task taxonomy가 아니라 action/observation capability를 정의한다.

\[
c_{payload}=\{N_{axis},\theta_{range},\dot\theta_{max},EO,IR,zoom,FOV,latency\}
\]

대표 구성:

- Fixed EO + fixed FOV
- Fixed EO/IR + selectable modality
- 1-axis EO + fixed FOV
- 1-axis EO/IR + zoom
- 2-axis EO + zoom
- 2-axis EO/IR + continuous zoom

장비별 의미:

| 구성 | 가능한 자율성 |
|---|---|
| 2축 + EO/IR + zoom | gaze, modality, resolution, dwell의 완전한 능동 운용 |
| 2축 + fixed FOV | gaze와 modality 중심 |
| 1축 + zoom | 한 축의 부족을 zoom 또는 motion request로 보완 |
| 1축 + fixed FOV | 제한된 gaze와 예측적 request가 중요 |
| fixed + zoom | spatial gaze 대신 FOV, ROI, compute attention 관리 |
| completely fixed | 관측정책의 직접 action이 적어지므로 motion request 또는 adaptive processing이 필요 |

지원하지 않는 action을 dummy output으로 학습시키기보다 capability mask로 제거하는 것이 적절하다.

## 2.5 Motion capability levels

| Level | 관계 |
|---|---|
| M0: No Influence | 비행은 외생적이며 요청 불가능 |
| M1: Request-Aware | 관측조건을 요청할 수 있으나 수용은 외부 결정 |
| M2: Coordinated | 비행정책이 관측 요청을 명시적으로 고려 |
| M3: Jointly Optimized | 비행·관측을 공동 목적함수로 학습 |

기본 연구범위는 M0이며 M1을 선택적 확장으로 둔다. M2/M3는 cooperation의 이점이 독립성 상실 비용보다 크다는 증거가 있을 때만 다룬다.

## 2.6 상위 UAV mission taxonomy

관측 에이전트는 다양한 항공 임무에서 사용될 수 있다. 상위 임무는 다음과 같이 포괄적으로 정리할 수 있다.

| Family | 대표 임무 |
|---|---|
| Move | Transit, Route Following, NOE, Loiter, Patrol, RTB, Recovery |
| Observe | Search, Reconnaissance, Surveillance, Acquisition, Tracking, BDA |
| Coordinate | Formation, Flocking, Escort, Cooperative Search/Tracking |
| Engage | A2A Interception, Precision Strike, Strafe, OWA, Designation |
| Survive | Terrain/Threat Avoidance, Evasion, Deception, Lost-Link 대응 |
| Support | Cueing, Relay, Overwatch, Localization, Handover |

다만 `Precision Strike`, `BDA`, `Search and Rescue` 같은 임무명을 관측 policy의 고정 mode로 직접 학습하면 재사용성이 떨어진다. 상위 임무는 observation objective primitive로 컴파일하는 편이 좋다.

## 2.7 Observation objective primitives

| Primitive | 목적 |
|---|---|
| Find | 관심조건에 맞는 객체 발견 |
| Acquire | 후보를 안정적 관심대상으로 획득 |
| Keep-in-View | 대상을 FOV 안에 유지 |
| Center/Compose | 지정 영상위치 또는 구도로 정렬 |
| Track | 상태와 identity를 연속 유지 |
| Identify | 분류·식별에 필요한 관측품질 확보 |
| Inspect | 특정 부분 또는 주변 맥락을 자세히 관찰 |
| Monitor | 일정 주기로 상태 갱신 |
| Reacquire | 유실 대상을 다시 발견 |
| Cover | 지정 영역의 관측 누락 최소화 |
| Confirm Event | 특정 사건의 발생 여부 확인 |
| Compare/Assess | 이전 상태와 현재 상태 비교 |
| Handover | 다른 센서·플랫폼에 전달 가능한 상태 확보 |

상위 임무의 조합 예:

| Mission | Primitive sequence |
|---|---|
| Area Search | Cover → Find → Acquire → Identify |
| Moving Target Tracking | Acquire → Track → Keep-in-View → Reacquire |
| Precision Strike Support | Acquire → Identify → Track → Confirm Event |
| BDA | Reacquire → Inspect → Compare/Assess |
| Search and Rescue | Cover → Find → Identify → Handover |
| Nightfall Gate Observation | Acquire → Keep-in-View → Predict → Reacquire |

## 2.8 Payload-native behavioral taxonomy

관측 에이전트가 실제로 수행하는 행동은 다음과 같이 분류할 수 있다.

### Orient

- Cue-to-Look
- stabilized viewing
- predictive pointing
- non-central target composition

### Search

- area scanning
- local hypothesis search
- reacquisition search
- anomaly-triggered search

### Observe

- target inspection
- context observation
- change inspection
- evidence-specific observation

### Track

- single-target tracking
- multi-object attention
- predictive tracking
- identity-preserving tracking

### Identify and Assess

- recognition-view acquisition
- confidence refinement
- multimodal cross-check
- event confirmation
- BDA observation

### Transition

- target switching
- stop/sufficiency decision
- prospective revisit
- handover support

## 2.9 입력과 출력의 기본 contract

### 입력 후보

- mission intent, assigned ROI/target set, priority
- perception observations and source identity
- object history, confidence, uncertainty
- ownship position, attitude, velocity, angular rate
- gimbal angles, rates, limits and latency
- EO/IR/zoom/FOV state
- occlusion/visibility estimate
- unresolved questions and prospective memory

### 직접 출력

- attention subject or look-point
- gimbal target/rate
- EO/IR selection
- zoom/FOV/ROI
- search or reacquisition pattern
- dwell and revisit timing
- observation purpose/query
- sufficiency/termination decision

### 선택적 출력

- observation feasibility
- desired viewing envelope
- deadline/urgency
- expected mission-value gain
- reason code and unresolved knowledge gap

비행명령 자체는 출력하지 않는다.

## 2.10 성공 metric의 계층

단순 짐벌 metric만으로는 연구가치를 증명할 수 없다.

### Control metrics

- LOS error, image-centering error
- gimbal saturation and rate violation
- motion blur, image motion

### Observation metrics

- time-to-acquire/reacquire
- dwell completion
- FOV retention
- identification-quality exposure

### Epistemic metrics

- belief calibration
- hypothesis entropy/posterior separation
- false certainty rate
- contradiction resolution rate
- question resolution efficiency
- stopping quality

### Mission metrics

- correct target/decision rate
- false engagement or missed-event rate
- time-to-decision
- mission success under observation budget
- operator intervention reduction

연구의 최종 비교는 control metric이 아니라 epistemic/mission metric에서 이루어져야 한다.

