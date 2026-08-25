# 5. Project Value and Research Roadmap

## 5.1 프로젝트별 연구환경의 성격

### Flight Intelligence 관련 프로젝트

강점:

- Precision Strike, Tracking, BDA, Search, Escort 등 다양한 mission context
- 복수 표적과 identity ambiguity
- EO/IR와 관측각·거리의 의미
- 기체의 큰 자세·각속도 변화
- deception, decoy, occlusion으로 확장 가능
- 관측결과가 실제 mission decision에 미치는 영향 평가 가능

따라서 고차원 가설, mission-value, semantic inquiry를 검증하는 본무대로 적합하다.

### Project Nightfall

강점:

- monocular bbox 중심의 명확한 partial observation
- narrow FOV와 빠른 terminal image motion
- perception latency/dropout을 통제 가능
- simulator ground truth가 풍부함
- gate pose, crossing point, clearance, future visibility를 정확히 평가 가능
- GP-G- 등 독립 비행정책과의 interface 검증 가능

따라서 prediction, memory, RSSM, privileged-to-partial transfer를 정밀하게 검증하는 controlled testbed로 적합하다.

## 5.2 세 연구의 적합도

| Research | Flight Intelligence | Nightfall | 주요 이유 |
|---|---:|---:|---|
| Hypothesis-Driven Active Observation | 매우 높음 | 낮음~중간 | Flight 임무에는 복수 상황가설이 자연스럽지만 현재 gate는 단순함 |
| RSSM Dream-to-Look | 높음 | 매우 높음 | ownship motion과 미래 visibility 예측이 두 프로젝트 모두 중요 |
| Epistemic Distillation | 매우 높음 | 매우 높음 | sim privileged state와 deployable limited perception 사이 전이 |

## 5.3 Hypothesis-Driven Observation의 프로젝트 가치

### Flight Intelligence 관련

#### Precision Strike

- real target vs decoy
- designated identity 유지
- target swap 탐지
- attack 전 evidence sufficiency
- release/abort/wait 판단의 신뢰성 향상

#### Tracking and Surveillance

- 구조물 뒤 이동가설 유지
- 가장 가능성 높은 위치가 아닌 가장 구분력 높은 위치 관측
- 장기 identity continuity
- 재획득 시간과 잘못된 association 감소

#### BDA

- destroyed vs obscured vs still-functional vs wrong-target
- EO/IR 교차검증
- 재공격 필요성 판단 지원

#### Adversarial scenarios

- decoy와 attention diversion
- confirmation bias 억제
- 반증 증거 탐색

Flight Intelligence는 observation agent로부터 point estimate만 받는 대신 hypothesis distribution, confidence, unresolved ambiguity, sufficiency를 받을 수 있다.

### Nightfall

현재 단일 명확한 gate에서는 경쟁 가설이 적어 연구가 억지스러울 수 있다. 다음 확장에서는 의미가 생긴다.

- multiple/decoy gate candidates
- partial gate occlusion
- gate frame vs aperture confusion
- detector ID switch/false positive
- severe dropout and ambiguous geometry

Nightfall을 이 연구만을 위해 과도하게 복잡하게 만드는 것은 추천하지 않는다.

## 5.4 RSSM Dream-to-Look의 프로젝트 가치

### Flight Intelligence 관련

- 선회·급강하·회피 전에 future LOS와 FOV exit 예측
- preemptive zoom-out
- gimbal-limit 도달 전 target composition 변경
- occlusion 반대편으로 선행 지향
- 공격기동 중 track continuity
- 미래 observation failure를 근거로 platform request 생성

### Nightfall

- 현재 bbox와 GP-G- motion으로 future gate bbox/size 예측
- gate corner/FOV margin과 dropout probability 예측
- terminal phase 이전 zoom/FOV/ROI 전환
- edgeTAM bbox 누락 구간에서 belief 유지
- image motion, late alignment, frame-strike 위험의 사전 탐지
- observation-risk request 전달

단, completely fixed camera이고 zoom/ROI/request action도 없다면 RSSM은 visual predictor로는 가치가 있지만 gaze research로서의 직접 action이 부족하다.

## 5.5 Epistemic Distillation의 프로젝트 가치

### Flight Intelligence 관련

- 3D fusion map 유무에 종속되지 않는 deployable policy
- simulator truth를 학습 teacher로 활용
- 서로 다른 vision stack과 sensor modality로 전략 전이
- compact edge policy
- fusion-rich teacher가 배운 관측전략을 bbox/bearing-only student로 이전
- uncertainty와 `unknown`을 포함한 안전한 knowledge transfer

### Nightfall

Teacher:

- true gate pose
- exact vehicle-gate relative state
- future crossing point
- clearance and frame-strike margin
- actual FOV geometry
- simulator dynamics

Student:

- edgeTAM bbox/confidence/history
- ownship/gimbal telemetry
- limited features

전달 가능한 target:

- latent gate pose belief
- future bbox/FOV margin
- time-to-FOV-exit
- crossing feasibility
- uncertainty
- gaze/zoom action and reason

Nightfall의 asymmetric information, DAgger 경험, sim-to-real 구조와 매우 잘 맞는다.

## 5.6 권장 연구 진행 순서

### Phase 0. Common research substrate

목적은 범용 제품 완성이 아니라 세 가설의 공정한 비교 기반 마련이다.

- configurable perception levels P1–P5
- fixed/1-axis/2-axis payload profiles
- swappable motion provider
- object/evidence logging
- deterministic evaluation seeds
- epistemic and mission metrics
- baseline rule/tracker/controller

### Phase 1. Nightfall-based temporal belief benchmark

연구질문:

> bbox-only 조건에서 미래 visibility를 예측하는 world model이 reactive recurrent policy보다 유의미한가?

비교:

- geometric predictor
- GRU policy
- recurrent SAC/PPO
- object-centric RSSM
- Dreamer/planning on RSSM

핵심 scenario:

- narrow FOV
- variable latency/dropout
- motion blur
- fast closing rate
- 1-axis/2-axis/zoom profiles

### Phase 2. Nightfall epistemic distillation

연구질문:

> privileged gate truth의 belief와 uncertainty를 증류하면 action-only DAgger보다 제한 관측 student가 좋아지는가?

비교:

- student from scratch
- action BC
- action DAgger
- belief distillation
- full epistemic distillation

### Phase 3. Flight mission hypothesis benchmark

초기에는 지나치게 큰 전술임무보다 가설구분이 명확한 소규모 시나리오를 사용한다.

- target disappears behind structure
- identity crossing/target swap
- EO/IR contradictory evidence
- pre/post strike state assessment

비교:

- center tracker
- confidence policy
- entropy/information-gain policy
- mission-weighted hypothesis policy
- oracle planner

### Phase 4. Cross-project generalization

- perception adapter 교체
- payload profile 교체
- motion provider 교체
- observation objective 교체
- zero-shot/few-shot/generalization evaluation

범용성은 기능목록이 아니라 연구원리가 다른 조건에서도 성립한다는 증거로 사용한다.

### Phase 5. Integrated autonomous inquiry

- hypothesis/evidence graph
- RSSM imagination
- query generation
- epistemic distillation
- prospective memory
- value-bearing platform request

## 5.7 단계별 시연 메시지

### Demo 1: Predictive Eye

> “현재 보이는 곳을 따라가는 것이 아니라, 기체운동과 차폐를 고려해 다음에 보일 곳을 미리 봅니다.”

### Demo 2: Privileged Knowledge Without Privileged Input

> “배치 모델은 bbox만 보지만, 3D 세계를 본 teacher에게 배운 belief와 uncertainty를 내부적으로 형성합니다.”

### Demo 3: Hypothesis-Testing Eye

> “가장 그럴듯한 표적을 따라가는 대신, 자신의 판단이 틀렸음을 드러낼 수 있는 증거를 먼저 확인합니다.”

### Demo 4: Autonomous Inquiry

> “사람은 판단해야 할 임무만 지시하고, 센서는 필요한 질문과 관측절차를 스스로 구성합니다.”

## 5.8 연구 gate와 중단 기준

새로운 모델이 복잡하다는 이유만으로 연구가치를 인정하지 않는다.

### RSSM 연구 gate

- multi-step prediction이 geometric/GRU baseline보다 우수
- prediction 향상이 실제 preemptive behavior로 연결
- model error exploitation이 통제됨
- tail FOV loss/reacquisition 개선

### Distillation gate

- action-only BC/DAgger 대비 mission improvement
- uncertainty calibration 또는 OOD robustness 개선
- privileged target이 false certainty를 증가시키지 않음

### Hypothesis 연구 gate

- generic information gain보다 mission decision 개선
- 더 적은 관측으로 가설 resolution
- counterevidence behavior가 실제로 확인됨
- open-world/unknown hypothesis에서 과신하지 않음

## 5.9 장기 통합 비전

최종 시스템은 다음 역할을 가진다.

1. 제한된 perception으로 object-centric belief를 유지한다.
2. 복수의 상황가설과 모순된 증거를 표현한다.
3. RSSM으로 후보 관측의 미래 결과를 imagination한다.
4. mission-weighted hypothesis discrimination으로 query/action을 선택한다.
5. prospective memory로 미래 재관측을 예약한다.
6. 충분한 증거가 모이면 inquiry를 종료한다.
7. 자신의 payload로 불가능하면 value-bearing request를 보낸다.
8. 풍부한 teacher에서 배운 이 능력을 제한된 student가 실시간 수행한다.

## 5.10 핵심 메시지

> Nightfall에서는 **보이지 않는 미래를 예측하는 눈**을 증명한다.  
> Flight Intelligence 계열에서는 **무엇을 확인해야 하는지 스스로 판단하는 눈**을 증명한다.  
> Epistemic Distillation은 이 지능을 실제 제한 센서로 옮기는 공통 전이 기술이다.

