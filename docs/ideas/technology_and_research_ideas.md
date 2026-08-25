# 3. Background Technology and Research Idea Map

## 3.1 기술을 기능에 배치하기

특정 AI 키워드 하나를 중심으로 연구를 정의하지 않는다. 에이전트에 필요한 인지기능을 먼저 두고 기술을 배치한다.

| Cognitive function | 유력 기술 |
|---|---|
| Mission intent 이해 | sLLM, VLM, ontology, neuro-symbolic reasoning |
| Object/scene representation | Set Transformer, Graph Transformer, GNN, scene graph |
| Temporal belief | Bayesian filter, RNN, Transformer memory, RSSM |
| Future imagination | RSSM/world model, model-based RL, diffusion/flow model |
| Uncertainty and ignorance | ensemble, evidential learning, Bayesian methods, conformal prediction |
| Knowledge-gap discovery | anomaly/OOD, contradiction graph, epistemic uncertainty |
| Observation query generation | sLLM/VLM, hierarchical policy, program synthesis |
| Gaze/sensor decision | RL, POMDP planning, active inference, MPC |
| Fast actuation | classical control, feed-forward, MPC, constraint guard |
| Sim-to-real/limited sensing | privileged learning, distillation, DAgger, domain randomization |

## 3.2 RL and POMDP

관측행동은 미래 관측분포를 바꾸며 효과가 지연된다. zoom-in은 현재 식별을 돕지만 주변 coverage와 재획득 가능성을 낮출 수 있다. 한 대상을 오래 보는 행동은 해당 target belief를 개선하지만 다른 대상의 정보 age를 증가시킨다. 따라서 순차적 의사결정과 장기 opportunity cost가 핵심이다.

유력 계열:

- recurrent model-free RL
- hierarchical RL and options
- distributional RL
- constrained/safe RL
- offline RL from operator or heuristic trajectories
- model-based RL
- meta-RL for new payloads
- POMDP and belief-space planning

RL의 action은 반드시 motor command일 필요가 없다. `inspect target A`, `wide search`, `verify with IR`, `wait for reappearance` 같은 observation option을 선택하고, 하위 controller가 이를 실행할 수 있다.

## 3.3 RSSM and world models

RSSM은 deterministic history와 stochastic latent state를 결합한다.

\[
h_t=f(h_{t-1},z_{t-1},a_{t-1}), \quad z_t\sim q(z_t|h_t,e_t)
\]

관측이 없을 때 prior로 상태를 유지하고, 관측이 들어오면 posterior로 수정할 수 있어 차폐·dropout·부분관측과 잘 맞는다.

권장 방향은 pixel-perfect video generation보다 **decision-oriented object-centric world model**이다. 예측대상은 다음과 같다.

- future bbox/LOS and apparent size
- visibility/occlusion probability
- detection and identity confidence
- gimbal-limit/FOV margin
- track-loss probability
- event occurrence
- mission value and continuation

`Dream before you look`이라는 형태로 후보 gaze sequence를 latent space에서 rollout하고 예상 정보가치·mission value·risk를 비교할 수 있다.

## 3.4 Transformer

Transformer는 연구질문 그 자체가 아니라 강력한 표현·융합 도구다.

가능한 token:

- mission token
- payload capability token
- ownship/gimbal token
- object tokens
- evidence-source token
- outstanding-question token
- historical event token

장점:

- variable number of objects
- multimodal cross-attention
- long history and sparse events
- target/mission-conditioned attention
- capability-conditioned universal policy

단, attention weight는 calibrated uncertainty가 아니며 Transformer만으로 앎과 모름을 해결했다고 간주하면 안 된다.

## 3.5 GNN and scene graphs

관측질의는 객체의 독립 속성보다 관계에 의해 생기는 경우가 많다.

### 노드 예

- aircraft, payload, target, other object
- building, road, occlusion region
- evidence item, hypothesis, observation query

### edge 예

- occluded-by
- expected-to-emerge-at
- possibly-same-as
- supports/contradicts
- moving-toward
- observed-by

Temporal GNN, Graph Transformer, probabilistic node state를 사용하면 `target_3가 building_2 뒤로 사라졌고 road_7에서 재출현할 수 있다` 같은 구조를 직접 표현할 수 있다.

## 3.6 sLLM, VLM, VLA

### sLLM/VLM의 적합한 역할

- mission intent를 observation objective로 변환
- structured observation query 생성
- 새로운 객체 관계와 semantic context 해석
- 질문·행동의 이유 요약
- 미리 정의되지 않은 임무에 대한 high-level generalization

### 부적합하거나 위험한 역할

- 고주파 continuous gimbal control
- calibration이 필요한 확률 belief
- 제한을 엄격히 만족해야 하는 actuation
- 설명과 실제 policy causal reason의 동일성 보장

VLA는 영상·언어·행동을 통합한다는 점에서 장기 후보지만, 데이터 규모와 continuous precision, confidence calibration이 문제다. 초기에는 `VLM/sLLM inquiry executive + learned belief/world model + RL gaze policy + classical control`의 계층구조가 더 타당하다.

구조화 query 예:

```yaml
subject: target_03
property: identity
required_evidence:
  modality: EO
  minimum_pixel_size: 80
  dwell_time: 1.5
stop_condition:
  posterior_probability: 0.90
```

## 3.7 Uncertainty quantification

구분해야 할 uncertainty:

- aleatoric: 영상 노이즈, 날씨, 저해상도
- epistemic: 미경험 상황, 데이터 부족
- state: position, velocity, identity
- model: future prediction의 신뢰도
- task: 어떤 증거가 필요한지의 모호성

후보 기술:

- deep ensemble
- Bayesian neural network
- MC dropout
- evidential learning
- calibrated classifier
- conformal prediction
- energy/OOD score
- ensemble world models

detector confidence를 belief confidence로 그대로 사용하면 안 된다. 관측 에이전트의 핵심 안전문제는 **틀린 확신(false certainty)**이다.

## 3.8 Active sensing, value of information, active inference

Active sensing은 다음 sensor state를 선택해 정보량 또는 task utility를 최적화한다.

- entropy/mutual information
- Fisher information
- expected KL divergence
- probability of detection
- covariance reduction
- mission time and cost

본 연구에서는 일반 information gain보다 **mission- or decision-relevant value of observation**가 중요하다.

\[
VoO(a_t)=\mathbb{E}[V_{mission}(b_{t+1})-V_{mission}(b_t)|b_t,a_t]-C(a_t)
\]

Active inference는 pragmatic value와 epistemic value를 하나의 expected free energy 관점으로 통합할 수 있어 철학적으로 잘 맞는다. 다만 구현과 비교가 어려우므로 RL/VoI baseline과 병행하는 탐색 연구가 적절하다.

## 3.9 Causal inference and experimental design

관측은 경쟁 가설을 구분하기 위한 실험으로 볼 수 있다.

- 어떤 증거가 현재 belief를 반증할 수 있는가?
- 어떤 관측이 두 가설에서 서로 다른 결과를 만들어내는가?
- 이 관측이 실제 mission decision을 바꿀 수 있는가?

후보 기술:

- Bayesian experimental design
- hypothesis testing
- causal graph
- counterfactual prediction
- expected posterior separation
- falsification-oriented reward

## 3.10 Generative models

### Diffusion / flow matching

가능한 역할:

- multi-modal future trajectory/hypothesis generation
- multiple gaze-plan generation
- occlusion 이후 재출현 분포 생성
- operator behavior imitation
- generative world model

짐벌 action이 낮은 차원이고 단일모드라면 과도할 수 있다. 미래가 여러 모드로 갈라지거나 장기 gaze trajectory를 생성할 때 의미가 커진다. flow matching은 적은 sampling step이 필요한 실시간 후보로 볼 수 있다.

### GAN

중심 policy보다는 보조 도구다.

- EO↔IR domain translation
- rare/deceptive scenario generation
- weather/noise/domain randomization
- sim-to-real adaptation

## 3.11 Memory architectures

필요한 memory는 단순 hidden state보다 다양하다.

- working memory: 현재 belief와 active query
- episodic memory: 이전 사건과 관측
- semantic memory: 객체와 상황의 일반 지식
- prospective memory: 미래에 다시 확인할 약속
- question memory: 아직 해결되지 않은 질의

Prospective memory는 특히 자율 관측에 중요하다. `지금은 A를 보지만 5초 후 B의 예상 출구를 확인한다`는 행동은 과거기억이 아니라 미래 task scheduling이다.

## 3.12 Wild and wide research ideas

### Self-Questioning Sensor

모름을 구조화된 observation query로 변환한다.

### Counterfactual Gaze

`계속 본다면`, `다른 곳을 본다면`, `IR로 바꾼다면` belief와 mission decision이 어떻게 달라질지 비교한다.

### Contradiction-Seeking Gaze

현재 믿음을 확증하는 증거가 아니라 틀렸음을 드러낼 수 있는 관측을 선택한다.

### Mission-Grounded Curiosity

\[
r_{curiosity}=novelty\times mission\ relevance\times resolvability
\]

임무와 무관한 시각적 노이즈에 집착하지 않고 decision-relevant unknown을 탐색한다.

### Learned Prospective Memory

언제·어디를·무슨 센서로·왜 다시 볼지 기록하고 실행한다.

### Deception-Aware Gaze

decoy, camouflage, signature manipulation, attention diversion을 전제로 반증과 교차검증 행동을 학습한다.

### Adaptive Perception Budgeting

짐벌뿐 아니라 detector resolution, ROI, inference frequency, VLM 호출 여부 등 연산 자원도 action으로 관리한다.

### Sensor Self-Discovery

새 payload가 연결되면 움직여보고 FOV, latency, limit, zoom response, blind zone을 스스로 식별한다.

### Epistemic Tool Use

고비용 VLM, 3D fusion request, external sensor cue를 언제 호출할지 epistemic necessity에 따라 결정한다.

### Semantic Observation Request

플랫폼 측에 heading command가 아닌 필요한 evidence, observation envelope, deadline, expected value를 전달한다.

## 3.13 유력 계층형 architecture

### Slow inquiry loop: 약 0.2–2 Hz

- mission interpretation
- knowledge-gap detection
- query generation
- sufficiency and explanation
- sLLM/VLM or neuro-symbolic reasoner

### Mid-level policy loop: 약 5–25 Hz

- belief update
- attention target
- gaze strategy, zoom, EO/IR
- future imagination
- object-centric RSSM/GNN/Transformer + RL/planner

### Fast control loop: 약 50–200 Hz

- LOS stabilization
- pan/tilt rate control
- limit and rate protection
- classical controller/MPC

## 3.14 기술 우선순위

| 기술 | 우선도 | 이유 |
|---|---:|---|
| POMDP/RL | 매우 높음 | 순차적 관측과 opportunity cost |
| probabilistic belief/uncertainty | 매우 높음 | 앎과 모름의 핵심 |
| RSSM/world model | 매우 높음 | 차폐, 예측, imagination |
| causal/hypothesis reasoning | 매우 높음 | 필요한 증거 선택 |
| GNN/Graph Transformer | 높음 | object/scene/evidence relation |
| Transformer | 높음 | variable object, temporal/multimodal fusion |
| sLLM/VLM | 높음 | intent와 query generation |
| VLA | 중간~높음 | 장기 generalist 방향 |
| active inference | 중간~높음 | epistemic/pragmatic value 통합 |
| diffusion/flow | 중간 | multi-modal future/plan |
| GAN | 낮음 | 데이터·domain 보조 |

