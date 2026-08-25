# 1. Research Vision: From Smart Gimbal to Epistemic Autonomy

## 1.1 출발점

연구의 최초 출발점은 단순하고 직관적이었다.

> 1축/2축 짐벌, EO/IR, 줌 유무가 다른 항공 탑재 관측장비를 지능적·자율적으로 운용할 수 있는가?

Flight Intelligence가 항공기의 상태와 임무를 해석하여 비행명령을 생성한다면, 이 모델은 센서와 짐벌을 운용한다. 직관적으로는 **똑똑하게 눈동자를 굴리는 모델**이다.

그러나 이 문제를 `bbox → pan/tilt` 또는 자동추적기로 정의하면 연구의 가치가 빠르게 소진된다. 자동 안정화, auto-tracking, target centering은 이미 널리 연구되고 제품화되어 있다. 잘 구현된 범용 센서 프레임워크 역시 중요한 엔지니어링 성과이지만, 그 자체만으로는 선행·선도 연구가 요구하는 새로운 가치에 충분하지 않다.

따라서 연구의 중심은 다음과 같이 이동한다.

> 센서를 자동으로 움직이는 방법이 아니라, **임무를 이해하기 위해 무엇을 관측해야 하는지 스스로 판단하는 방법**을 연구한다.

## 1.2 궁극적 연구 정체성

이 연구가 지향하는 시스템은 다음과 같다.

> **자신이 무엇을 알고 무엇을 모르는지 판단하고, 임무에 필요한 증거를 스스로 찾아보는 자율 관측 에이전트**

영문 개념은 다음 표현들이 가깝다.

- Autonomous Observation Agent
- Autonomous Perceptual Inquiry
- Epistemic Sensing Intelligence
- Mission-Aware Active Observation
- Hypothesis-Driven Active Perception

프로젝트명 또는 모델명은 추가 선행명칭·상표 조사가 필요하다. 초기 논의에서 사용한 `Gaze Intelligence`는 인간 eye-tracking 분야의 기존 회사명과 충돌 가능성이 있고, `SCOPE` 역시 흔한 약어이므로 현재는 작업명으로만 사용한다.

## 1.3 자동화와 연구의 차이

### 자동화 중심 접근

- 지정된 표적을 영상 중앙에 유지한다.
- 표적이 멀면 확대하고 가까우면 축소한다.
- detection confidence가 낮으면 IR로 전환한다.
- 표적을 놓치면 미리 정의된 sweep pattern을 수행한다.

이러한 기능은 유용하지만 대부분 잘 설계된 rule, controller, tracker로 구현할 수 있다.

### 연구 중심 접근

- 현재 증거로 어떤 가설들이 가능한지 유지한다.
- 어떤 모름이 임무결정을 바꿀 수 있는지 판단한다.
- 현재 믿음을 확증하는 장면뿐 아니라 반증할 장면을 선택한다.
- 예상한 사건이 일어나지 않았음을 중요한 증거로 해석한다.
- 어떤 센서·시점·줌이 경쟁 가설을 가장 잘 구분할지 판단한다.
- 추가 관측의 가치가 낮아지는 순간을 인식하고 관측을 종료한다.
- 현재는 보지 않더라도 미래에 다시 확인해야 할 대상을 기억한다.
- 자체 행동만으로 관측이 불가능할 때 platform-motion provider에 관측조건을 요청한다.

이 차이는 `sensor automation`과 `epistemic autonomy`의 차이다.

## 1.4 책임 경계

연구의 독립성을 유지하기 위해 다음 기능은 원칙적으로 외부에 둔다.

| 기능 | 기본 담당 |
|---|---|
| 무인기별 임무·역할 할당 | 상위 Mission Manager |
| 무인기별 표적·탐색구역 할당 | 상위 Mission Manager |
| 항공기 기동·비행명령 생성 | Flight Intelligence, GP-G-, autopilot 등 |
| 객체 탐지·분류 | Vision Stack, edgeTAM 등 |
| 전역 3D fusion 및 multi-sensor tracking | Fusion/Tracking Stack, 존재할 경우 |
| 관심대상·관측목적의 실시간 선택 | Autonomous Observation Agent |
| 짐벌, EO/IR, zoom, FOV 운용 | Autonomous Observation Agent |
| 관측 실패 예측, 재획득, 충분성 판단 | Autonomous Observation Agent |

관측 에이전트는 특정 비행모델을 전제로 하지 않는다. 비행은 기본적으로 외생적 조건이다. 다만 자신의 payload 행동만으로 관측목적 달성이 불가능하거나 급격히 비효율적일 경우, 다음과 같은 요청을 전달할 수 있다.

- desired LOS region
- preferred observation aspect/distance
- maximum LOS or image rate
- required dwell time
- gimbal-limit margin
- observation deadline and urgency
- expected mission-value gain

요청을 수용할지는 Flight Intelligence, GP-G-, conventional autopilot 또는 상위 mission manager가 결정한다. 이를 통해 정책을 결합하지 않고도 필요한 경우 느슨한 협력이 가능하다.

## 1.5 핵심 폐루프: Perceptual Inquiry

자율 관측은 다음의 폐루프로 정의한다.

1. **Mission Intent**: 무엇을 판단하거나 유지해야 하는가?
2. **Belief Formation**: 현재 세계에 대해 무엇을 알고 있는가?
3. **Gap/Conflict Detection**: 무엇이 부족하고, 모순되며, 예상과 다른가?
4. **Observation Question**: 무엇을 확인해야 하는가?
5. **Evidence Requirement**: 어떤 센서와 관측조건이 답을 줄 수 있는가?
6. **Gaze/Sensor Action**: 어디를 어떻게 얼마나 오래 볼 것인가?
7. **Belief Update**: 새 증거가 가설과 uncertainty를 어떻게 바꾸는가?
8. **Sufficiency Decision**: 이제 임무판단에 충분한가?

이를 식으로 표현하면 단순한 policy는 다음과 같다.

\[
\pi_{payload}: (b_t, g_t, c_{payload}, x_{platform,t}) \rightarrow a^{sensor}_t
\]

- \(b_t\): 현재 belief
- \(g_t\): mission/observation goal
- \(c_{payload}\): 센서와 짐벌 capability
- \(x_{platform,t}\): 플랫폼 상태
- \(a^{sensor}_t\): gaze, sensor mode, zoom, dwell, attention action

상위 inquiry policy는 관측목적 자체를 생성한다.

\[
q_t = \pi_{inquiry}(b_t, g_t, m_t)
\]

- \(q_t\): 해결해야 할 구조화된 관측질의
- \(m_t\): unresolved question 및 prospective memory

## 1.6 uncertainty를 넘어 ignorance로

에이전트는 최소한 다음 상태를 구분해야 한다.

| 상태 | 의미 | 예시 행동 |
|---|---|---|
| Uncertainty | 후보들 사이의 확신이 낮음 | 구분력이 높은 관측 선택 |
| Ignorance | 관련 정보가 아예 관측되지 않음 | 탐색 또는 새로운 query 생성 |
| Contradiction | 센서·시점·모델의 증거가 충돌 | 교차검증 또는 반증 관측 |
| Prediction Error | 보여야 할 사건이 나타나지 않음 | 가설 수정 및 대안 경로 관측 |
| Insufficiency | 임무결정을 내리기에 증거 부족 | 추가 dwell, zoom, modality 변경 |
| Sufficiency | 추가 관측이 결정을 바꿀 가능성 낮음 | 관측 종료 및 attention 전환 |

모든 uncertainty를 줄이는 것은 목표가 아니다. 감소시켜야 하는 것은 **decision-relevant uncertainty**다. 차량의 색상을 모르는 것은 중요하지 않을 수 있지만, 실제 차량인지 decoy인지, 엔진이 가동 중인지, 지정 표적과 동일한 객체인지, 공격 후 기능이 상실되었는지는 임무결정을 바꿀 수 있다.

## 1.7 고객에게 보여줄 새로운 가치

연구 성공의 증거는 짐벌이 부드럽게 움직였다는 것이 아니다. 다음과 같은 행동이 명시적인 지시 없이 나타나야 한다.

- 표적이 차폐되기 전에 미래 출현 위치를 미리 본다.
- 현재 표적을 잠시 놓고 더 중요한 불확실성을 해소한다.
- EO와 IR의 결론이 충돌하자 다른 관측조건으로 재검증한다.
- 추적 중인 대상이 아니라, 나타나야 하지만 나타나지 않은 위치를 본다.
- 추가 관측이 무가치하다고 판단하여 스스로 종료한다.
- 지금은 보지 않지만 5초 후 다시 확인할 대상을 기억한다.
- 자신의 행동만으로 관측이 불가능해질 것을 예측하고 기체 측에 요청한다.

고객에게 기대하는 반응은 다음과 같다.

> “짐벌이 자동으로 따라가네요”가 아니라,  
> **“얘가 지금 무엇을 확인해야 하는지를 스스로 판단하네요?”**

## 1.8 연구의 중심 질문

가장 넓은 질문은 다음과 같다.

> 부분관측 환경에서 자율시스템이 임무 수행에 필요한 지식의 결손을 스스로 발견하고, 이를 해소하기 위한 관측질의를 생성·수행·종료할 수 있는가?

이를 뒷받침하는 하위 질문은 다음과 같다.

1. 복수의 상황 가설 중 무엇을 구분해야 하는지 판단할 수 있는가?
2. 관측행동의 미래 결과를 상상하여 선제적으로 시선을 운용할 수 있는가?
3. 풍부한 시뮬레이션 지식을 제한된 실제 센서 정책에 전달할 수 있는가?
4. 센서 embodiment가 바뀌어도 관측전략이 일반화되는가?
5. 독립적인 비행정책과 최소한의 semantic request만으로 협력할 수 있는가?
6. 의도적 기만과 attention hijacking에도 자신의 belief를 검증할 수 있는가?

