# 4. Three Research Candidates: Deep Dive

이 문서는 첫 연구 포트폴리오로 선정한 세 후보를 독립적인 연구과제로 정의한다. 세 연구는 서로 중복되는 기술을 사용할 수 있으며, 장기적으로 하나의 자율 관측 에이전트로 통합될 수 있다.

---

# Candidate A. Hypothesis-Driven Active Observation

## A.1 핵심 명제

> 자율 관측 에이전트는 가장 가능성 높은 상태를 따라보는 대신, 경쟁하는 상황 가설을 가장 효과적으로 구분하거나 반증할 수 있는 관측을 선택해야 한다.

예를 들어 표적이 건물 뒤로 사라진 경우:

- \(H_1\): 좌측 출구로 이동
- \(H_2\): 우측 출구로 이동
- \(H_3\): 건물 뒤 정지
- \(H_4\): target swap 또는 false detection

가장 확률이 높은 \(H_1\) 위치를 바로 보는 것은 exploitation이다. 그러나 두 출구와 정지 가설을 가장 효율적으로 구분할 관측은 다른 위치·FOV·시점일 수 있다.

## A.2 연구 질문

1. 현재 belief에서 경쟁 가설을 자동으로 구성할 수 있는가?
2. 특정 sensor action이 각 가설 아래 어떤 observation을 만들지 예측할 수 있는가?
3. 단순 entropy 감소가 아니라 mission decision을 바꾸는 증거를 선택할 수 있는가?
4. 자신의 belief를 확증하는 대신 반증 가능한 관측을 선택할 수 있는가?
5. 추가 관측이 무가치해지는 순간을 판단할 수 있는가?

## A.3 검증 가설

> Mission-weighted hypothesis discrimination을 최대화하는 관측정책은 center-tracking, confidence-driven observation, generic information-gain policy보다 제한된 관측시간과 센서 자유도 아래에서 더 높은 올바른 판단률과 낮은 false-certainty를 달성한다.

## A.4 형식적 정의

belief는 가설과 확률의 집합이다.

\[
B_t=\{(H_k,p_k)\}_{k=1}^{K}, \quad \sum_k p_k=1
\]

sensor action \(a_t\)에서 예상되는 관측은:

\[
p(o_{t+1}|H_k,a_t,c_{payload})
\]

일반적인 information objective는:

\[
IG(a_t)=H(B_t)-\mathbb{E}_{o_{t+1}}[H(B_{t+1})]
\]

그러나 모든 가설 구분이 임무에 동일하게 중요하지 않다. mission decision \(d\)와 loss \(L(d,H)\)를 두면 observation value를 다음처럼 정의할 수 있다.

\[
VoO(a_t)=R(B_t)-\mathbb{E}_{o_{t+1}}[R(B_{t+1})]-C(a_t)
\]

여기서 Bayes risk는:

\[
R(B)=\min_d\sum_k p(H_k|B)L(d,H_k)
\]

이 정의는 단순히 uncertainty가 큰 것을 보는 대신 실제 decision loss를 줄일 관측을 선택한다.

## A.5 핵심 기술 스택

### Belief and tracking

- Bayesian multi-hypothesis tracking
- particle filter / Gaussian mixture / IMM
- learned probabilistic belief network
- identity and target-existence belief

### Scene representation

- object-centric tokens
- graph/scene graph
- Set Transformer or Graph Transformer
- evidence and hypothesis nodes

### Observation model

- analytical camera/FOV model
- detection and identification probability model
- learned modality/zoom-dependent likelihood
- occlusion and visibility model

### Decision

- POMDP/belief-space planning
- MCTS or limited tree search
- recurrent or belief-conditioned RL
- hierarchical RL with observation options

## A.6 상태, 행동, 출력

### State/belief

- object state distributions
- identity/existence hypotheses
- visibility and occlusion
- evidence support/contradiction
- unresolved question
- observation history and cost

### Actions

- select subject/ROI
- select query type
- pan/tilt/look-point
- EO/IR/zoom/FOV
- dwell/wait/revisit
- terminate inquiry

### Explainable outputs

- active hypothesis set
- selected discriminating question
- expected posterior change
- selected action reason
- stopping reason

## A.7 대표 시나리오

### Occlusion exit

이동 표적이 구조물 뒤로 사라진다. 출구 여러 개와 정지 가능성이 존재한다. 정책은 가장 가능성 높은 출구만 고집하지 않고 적은 시선 전환으로 가설들을 구분한다.

### EO/IR contradiction

EO는 차량으로 분류하지만 IR thermal signature가 맞지 않는다. 정책은 `real vehicle`, `cold abandoned vehicle`, `decoy` 가설을 유지하고 구분 가능한 거리·각도·sensor mode를 선택한다.

### BDA

타격 후 `destroyed`, `temporarily occluded`, `still functional`, `wrong target` 가설을 구분하기 위해 attack-before/after evidence와 주변 activity를 관측한다.

### Target identity preservation

유사 객체가 교차한 뒤 target swap 가능성이 생긴다. 정책은 단순 nearest-box가 아니라 identity를 구분하는 특징이 보이는 관측을 선택한다.

## A.8 Baselines

1. fixed scan/finite-state machine
2. center-tracking
3. maximum detector confidence
4. highest state uncertainty first
5. generic entropy/information-gain policy
6. mission-weighted VoO policy
7. oracle belief-space planner

## A.9 Metrics

### Epistemic

- correct-hypothesis posterior
- Brier score, NLL, ECE
- false-certainty rate
- posterior separation
- observations-to-resolution
- contradiction resolution rate

### Operational

- time-to-correct-decision
- target swap and false identification
- reacquisition rate
- sensor-motion/dwell cost
- missed-event and false-engagement rate

## A.10 Ablations

- mission-weighting 제거
- explicit hypothesis nodes 제거
- counterevidence reward 제거
- observation model oracle vs learned
- graph relation 제거
- modality/zoom action 제거
- stopping action 제거

## A.11 예상 실패모드

- hypothesis set에 진실이 포함되지 않는 open-world failure
- observation model이 틀린데 과도하게 확신
- entropy만 줄이는 값싼 관측에 수렴
- 질문을 너무 자주 바꾸는 thrashing
- sensor cost가 약해 과도한 움직임
- target identity와 detection confidence 혼동

## A.12 주요 연구기여 가능성

- mission-weighted falsification objective
- object/evidence/hypothesis graph belief
- active stopping/sufficiency decision
- adversarial or deception-aware hypothesis testing

---

# Candidate B. Dream-to-Look with an Object-Centric RSSM

## B.1 핵심 명제

> 관측 에이전트는 실제로 시선을 움직이기 전에 후보 gaze action이 만들어낼 미래 visibility와 evidence를 latent world model에서 상상하고 선택할 수 있어야 한다.

반응형 tracker는 현재 bbox를 따라간다. Dream-to-Look는 `계속 추적`, `미리 zoom-out`, `예상 출현점 관측`, `IR 전환`, `잠시 기다림`을 RSSM 안에서 rollout하여 미래 관측가치와 실패위험을 비교한다.

## B.2 연구 질문

1. 부분관측·차폐·dropout에서 object belief를 유지할 수 있는가?
2. ownship motion과 sensor action이 미래 영상상태에 주는 영향을 분리해 학습할 수 있는가?
3. pixel reconstruction 없이 decision-relevant future만 예측해도 좋은 gaze planning이 가능한가?
4. model uncertainty를 고려해 자신 있게 틀리는 imagination을 억제할 수 있는가?
5. model-free recurrent RL보다 예측적 선행 행동이 실제로 나타나는가?

## B.3 검증 가설

> Object-centric RSSM에서 미래 visibility, detection, FOV margin, identity confidence를 imagination하는 정책은 recurrent model-free RL보다 좁은 FOV, 빠른 ownship maneuver, perception latency, 장기 occlusion 조건에서 낮은 track loss와 빠른 재획득, 높은 mission success를 보인다.

## B.4 RSSM 구성

관측 encoder:

\[
e_t=Enc(o_t, x^{platform}_t,x^{payload}_t,c_{payload})
\]

deterministic dynamics:

\[
h_t=f_{GRU}(h_{t-1},z_{t-1},a_{t-1})
\]

prior and posterior:

\[
p(z_t|h_t), \qquad q(z_t|h_t,e_t)
\]

관측이 없으면 prior로 belief를 전개하고, 새 evidence가 들어오면 posterior로 수정한다.

## B.5 Object-centric design

단일 global latent는 여러 객체의 identity와 개별 uncertainty를 섞을 수 있다. 권장 형태는:

- per-object stochastic latent
- global scene/context latent
- graph interaction or cross-attention
- explicit visibility/existence variable
- target-slot association mechanism

각 object latent의 예:

\[
z^i_t=\{position,image\ geometry,motion,identity,visibility,uncertainty\}
\]

## B.6 Task-oriented prediction heads

필수에 가까운 head:

- future bbox center/size distribution
- LOS and LOS-rate
- visibility/occlusion probability
- detection probability
- FOV and gimbal-limit margin
- class/identity posterior
- track-loss probability
- reward/mission-value prediction
- continuation/event probability

선택적 head:

- low-resolution semantic frame
- optical flow/image motion
- future relation graph
- observation sufficiency

## B.7 Policy learning

### Dreamer-style latent actor–critic

posterior state에서 시작해 RSSM prior로 imagined trajectory를 생성한다.

\[
\hat z_{t+1:t+H}\sim p_{RSSM}(\cdot|z_t,a_{t:t+H-1})
\]

actor와 critic은 imagined λ-return으로 학습한다.

### Online planning alternative

- CEM/MPC over RSSM
- MCTS over observation options
- hybrid: actor proposal + short-horizon CEM refinement

정책과 planner를 비교하는 것 자체가 연구질문이 될 수 있다.

## B.8 Uncertainty-aware imagination

단일 world model은 OOD에서 자신 있게 틀릴 수 있다.

후보:

- RSSM ensemble disagreement
- epistemic/aleatoric separated heads
- posterior predictive variance
- OOD detector
- pessimistic value or uncertainty penalty
- risk-sensitive distributional critic

정책 목적의 예:

\[
J=\mathbb{E}[R_{mission}+\beta R_{epistemic}-\lambda C_{sensor}-\eta U_{model}]
\]

## B.9 대표 시나리오

### Predictive zoom-out

표적이 아직 중앙에 잘 보이지만 ownship 선회와 closing rate 때문에 곧 FOV를 채울 것을 예측해 미리 zoom-out한다.

### Occlusion pre-look

표적을 끝까지 따라가지 않고 구조물 반대편 예상 출현점으로 선행 지향한다.

### Gimbal-limit anticipation

기체 선회로 짐벌이 limit에 도달할 것을 예측하고 화면 내 표적 위치를 편향하거나 되감기/reacquisition 전략을 시작한다.

### Nightfall gate

현재 bbox와 GP-G-의 motion으로 미래 gate bbox, corner/FOV margin, dropout 가능성을 예측한다. terminal phase 전에 zoom/FOV/ROI를 조정하거나 observation-risk request를 보낸다.

## B.10 Baselines

1. geometric constant-velocity predictor
2. frame stack MLP
3. GRU/LSTM recurrent policy
4. recurrent SAC/PPO without world-model loss
5. RSSM + model-free policy head
6. Dreamer-style imagination policy
7. oracle-dynamics planner

## B.11 Metrics

### World-model quality

- multi-step bbox/LOS prediction
- visibility Brier/NLL
- calibration under dropout/occlusion
- identity preservation
- model disagreement vs actual error

### Behavioral quality

- preemptive action lead time
- FOV exit and track-loss rate
- time-to-reacquire
- gimbal saturation
- zoom switching cost
- decision/mission success

중요한 것은 prediction loss가 낮은 모델이 반드시 좋은 policy를 만드는 것은 아니라는 점이다. 최종 판정은 behavior와 mission metric으로 해야 한다.

## B.12 Ablations

- stochastic latent 제거
- object-centric slot 제거
- task-oriented head별 제거
- image reconstruction 유무
- ensemble uncertainty 제거
- imagination horizon 변화
- platform state/action 제거
- privileged training target 제거

## B.13 예상 실패모드

- latent rollout error accumulation
- posterior collapse
- high-dimensional multi-object RSSM instability
- policy exploitation of world-model error
- zoom change와 range change의 confounding
- rare tail event 예측 부족
- 좋은 reconstruction, 나쁜 decision representation

## B.14 주요 연구기여 가능성

- airborne payload를 위한 object-centric sensing RSSM
- future visibility를 직접 최적화하는 Dream-to-Look policy
- model uncertainty-aware gaze imagination
- fixed/1-axis/2-axis capability-conditioned world model

---

# Candidate C. Epistemic Distillation

## C.1 핵심 명제

> 시뮬레이션에서 완전한 세계를 본 teacher가 배운 ‘보는 법’을, 실제 배치에서 제한된 bbox·영상 정보만 보는 student가 belief와 uncertainty까지 포함해 내재화할 수 있는가?

일반 action distillation은 teacher의 행동만 복제한다. 그러나 같은 행동도 서로 다른 이유로 선택될 수 있고, student가 teacher가 방문하지 않은 상태에 들어가면 행동의 의미를 잃는다. Epistemic Distillation은 action뿐 아니라 teacher의 내부 지식상태를 전달한다.

## C.2 연구 질문

1. privileged 3D state로 학습한 observation strategy를 bbox-only student가 재현할 수 있는가?
2. action 외에 belief·uncertainty·query·future visibility를 증류하면 어떤 이점이 있는가?
3. teacher가 아는 것 중 student 관측으로 원리적으로 알 수 없는 정보를 어떻게 처리할 것인가?
4. epistemic target이 covariate shift와 OOD calibration을 개선하는가?
5. detector/fusion stack이 바뀌어도 distilled strategy가 유지되는가?

## C.3 검증 가설

> Action-only BC보다 action, belief, uncertainty, observation query, future visibility를 공동 증류한 recurrent student가 bbox noise, dropout, occlusion, unseen motion에서 높은 관측성능과 낮은 false certainty를 보인다.

## C.4 Teacher information

시뮬레이션 teacher가 사용할 수 있는 privileged state:

- true 3D position/velocity/identity
- full occlusion and visibility
- geometry and terrain
- true detector error and target association
- future trajectory or simulator dynamics
- exact FOV/gimbal feasibility
- target hit/BDA ground truth
- Nightfall gate pose, crossing point, clearance

Teacher 후보:

- oracle/POMDP planner
- graph Transformer policy
- privileged actor–critic
- RSSM/Dreamer policy
- heuristic expert with access to ground truth

Teacher가 반드시 하나일 필요는 없다. 여러 expert의 mixture 또는 oracle label generator도 가능하다.

## C.5 Student information

- bbox/class/confidence
- 2D track history
- ownship and gimbal telemetry
- sensor mode/zoom
- limited image/feature embedding
- capability mask

memory 후보:

- GRU/LSTM
- temporal Transformer
- recurrent GNN
- compact RSSM/state-space model

## C.6 Distillation targets

### Action distillation

\[
L_{action}=\|a^S_t-a^T_t\|^2 \quad \text{or}\quad D_{KL}(\pi_T\|\pi_S)
\]

### Belief distillation

\[
L_{belief}=D(b^T_t,\hat b^S_t)
\]

대상:

- target state distribution
- identity/existence probability
- visibility and occlusion
- hypothesis distribution
- outstanding knowledge gap

### Uncertainty distillation

- posterior variance/distribution matching
- evidential parameter matching
- calibration loss
- teacher ensemble disagreement target

### Future/predictive distillation

- future bbox/LOS
- time-to-FOV-exit
- track-loss probability
- future observation quality

### Query/rationale distillation

- current observation question
- required evidence type
- expected value
- stop condition
- reason code

## C.7 전체 loss

\[
L=\lambda_aL_{action}+\lambda_bL_{belief}+\lambda_uL_{uncertainty}+\lambda_fL_{future}+\lambda_qL_{query}+\lambda_rL_{RL}
\]

모든 loss를 처음부터 동일하게 넣기보다 staged curriculum과 ablation이 필요하다.

1. belief/future representation pretraining
2. action BC
3. DAgger or mixed rollout
4. limited RL fine-tuning with BC anchor
5. calibration and OOD evaluation

## C.8 원리적으로 알 수 없는 정보 처리

teacher가 완전한 state를 알고 student가 bbox만 본다면 일부 정보는 관측적으로 식별 불가능하다. student에게 teacher의 point estimate를 강제하면 false certainty가 생긴다.

대안:

- distributional target, not point target
- information mask and observability label
- teacher posterior marginalized to student information
- multi-hypothesis student output
- confidence ceiling under ambiguity
- predict `unknown/unresolvable` state

이 부분은 Epistemic Distillation의 핵심 연구기여가 될 수 있다.

## C.9 Dataset aggregation

offline BC만 사용하면 student가 작은 오류로 teacher distribution 밖에 나간다.

후보:

- DAgger
- scheduled teacher intervention
- teacher/student mixed rollout
- uncertainty-triggered teacher query
- replay prioritization for epistemic failure
- offline pretraining + anchored RL fine-tuning

Nightfall의 기존 DAgger/anchored fine-tuning 경험은 이 연구의 강점이 될 수 있다. 단, action anchor가 mission improvement를 막지 않도록 belief/query target과 분리해서 설계해야 한다.

## C.10 대표 시나리오

### Fusion-rich teacher to bbox student

teacher는 3D fusion map과 occlusion geometry로 최적 관측을 선택하고, student는 bbox와 ownship history만으로 이를 근사한다.

### Nightfall

teacher는 gate pose, clearance, future crossing point를 보고 future visibility와 gaze/zoom을 선택한다. student는 edgeTAM bbox만으로 teacher의 latent gate belief와 time-to-FOV-exit를 복원한다.

### Cross-perception-stack transfer

서로 다른 detector noise/latency/confidence calibration에서 teacher의 epistemic representation을 통해 strategy가 유지되는지 평가한다.

## C.11 Baselines

1. student RL from scratch
2. action-only BC
3. action BC + DAgger
4. feature distillation
5. action + belief distillation
6. full epistemic distillation
7. privileged asymmetric critic without distillation
8. teacher upper bound

## C.12 Metrics

- teacher action agreement
- mission performance gap to teacher
- belief NLL/Brier/ECE
- uncertainty-error correlation
- false-certainty rate
- dropout/occlusion robustness
- recovery after covariate shift
- unseen detector/payload generalization
- edge inference cost

## C.13 Ablations

- belief target 제거
- uncertainty target 제거
- query target 제거
- future target 제거
- DAgger 제거
- privileged teacher feature 수준 변화
- teacher/student capacity 변화
- point target vs distribution target

## C.14 예상 실패모드

- impossible knowledge를 강제해 hallucinated certainty 발생
- teacher의 shortcut과 bias 증류
- belief loss가 action-relevant feature를 압도
- student가 auxiliary target은 잘 맞추지만 행동 개선 없음
- DAgger query cost 증가
- sim detector model과 real detector mismatch

## C.15 주요 연구기여 가능성

- action이 아닌 epistemic state를 증류하는 방법론
- observability-aware uncertainty distillation
- privileged fusion teacher에서 minimal perception student로의 전이
- cross-sensor/cross-detector generalization benchmark

---

# 4.4 세 연구의 관계와 장기 통합

세 연구가 답하는 질문은 다르다.

| 연구 | 핵심 질문 |
|---|---|
| Hypothesis-Driven Observation | 무엇을 보면 가설을 가장 잘 구분할 수 있는가? |
| RSSM Dream-to-Look | 그곳을 보면 앞으로 무엇이 일어날 것인가? |
| Epistemic Distillation | 완전한 세계를 본 teacher의 보는 법을 제한된 센서로 재현할 수 있는가? |

장기 통합 형태:

1. privileged teacher가 scene graph와 복수 가설을 유지한다.
2. object-centric RSSM이 각 후보 gaze의 미래 observation을 imagination한다.
3. mission-weighted hypothesis discrimination으로 action을 선택한다.
4. teacher의 belief, uncertainty, query, imagined value, action을 bbox-only student에 증류한다.
5. student는 실제 배치에서 독립적으로 행동하고 필요시 semantic observation request를 보낸다.

통합 연구문장의 예:

> A privileged hypothesis-aware teacher imagines the future epistemic consequences of candidate gaze actions through an object-centric RSSM, and distills this active observation strategy into a partially observed deployable student.

처음부터 통합하지 않는다. 각 가설을 독립적으로 검증한 뒤, 실패원인과 실제 추가가치가 확인된 구성만 결합한다.

