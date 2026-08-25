# 6. Prior Work, Industry Signals, and References

## 6.1 조사 결론

관련 분야는 존재하며, 산업계도 자율 센서 운용 방향으로 움직이고 있다. 완전히 비어 있는 문제는 아니다. 그러나 다음 요소를 하나의 공개 범용 연구체계로 통합한 사례는 뚜렷하지 않다.

- heterogeneous perception level: bbox to fused 3D belief
- fixed/1-axis/2-axis and EO/IR/zoom capability conditioning
- mission-conditioned gaze and sensor policy
- uncertainty, contradiction, prospective memory and sufficiency
- future observation imagination
- optional semantic request to independent motion provider
- privileged-to-partial epistemic transfer

따라서 정직하고 강한 연구 메시지는 `아무도 시도하지 않은 완전 신기술`이 아니라 다음과 같다.

> Active sensing, autonomous sensor management, learned gaze control에 흩어진 이론과 기능을 바탕으로, 임무에 필요한 지식결손을 스스로 발견하고 증거를 찾아보는 deployable autonomous observation agent를 연구한다.

## 6.2 Active sensing and active perception

Active sensing은 센서가 수동적으로 데이터를 받는 대신 다음 sensor state를 선택하여 information 또는 task utility를 최적화한다. PTZ configuration, platform displacement, detection probability, covariance, Fisher information, entropy, mutual information 등이 주요 대상이다.

- L. Varotto et al., **Active Sensing for Search and Tracking: A Review**  
  <https://arxiv.org/pdf/2112.02381>

본 연구의 차별화 포인트는 generic information gain을 mission-relevant hypothesis resolution과 sufficiency로 확장하는 것이다.

## 6.3 Learned PTZ and active camera control

Eagle은 image-to-PTZ action을 end-to-end RL로 학습하고 embedded deployment와 sim-to-real을 보였다. 표적 중심 유지와 고해상도 tracking이 중심이므로, self-generated inquiry보다 좁은 문제다.

- S. S. Sandha et al., **Eagle: End-to-end Deep Reinforcement Learning based Autonomous Control of PTZ Cameras**  
  <https://arxiv.org/abs/2304.04356>

UAV gimbal active tracking 연구는 기체운동과 영상정보를 결합해 stabilization, detection, tracking을 개선한다.

- J. G. Hansen et al., **Active Object Detection and Tracking Using Gimbal Mechanisms for UAVs**  
  <https://www.mdpi.com/2504-446X/8/2/55>

## 6.4 Learning when and where to zoom

항공영상에서 RL로 처리할 영역과 확대시점을 선택하여 탐지성능과 계산비용을 절충한 연구가 있다. 실제 optical zoom과 동일하지는 않지만, wide search와 narrow inspection의 trade-off를 학습한다는 점에서 직접 관련된다.

- B. Uzkent and S. Ermon, **Learning When and Where to Zoom With Deep Reinforcement Learning**, CVPR 2020  
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Uzkent_Learning_When_and_Where_to_Zoom_With_Deep_Reinforcement_Learning_CVPR_2020_paper.html>

## 6.5 Visual active tracking

D-VAT 계열은 단안 영상에서 vehicle thrust/angular velocity를 직접 생성해 target visibility를 유지한다. 이는 플랫폼을 직접 움직이는 active perception으로, payload를 독립시키고 필요시 request만 전달하려는 본 연구와 인접하지만 책임범위는 다르다.

- **End-to-End Visual Active Tracking for Micro Aerial Vehicles**  
  <https://arxiv.org/html/2308.16874v2>

## 6.6 Autonomous sensor management in defence

영국 Dstl/DASA는 autonomous sensor management를 search, detection, identification, recognition, tracking, situational awareness를 위해 pointing direction, FOV, sensitivity, power, sensor location 등을 선택하는 문제로 정의했다. 정보이론, 게임이론, RL, 모듈형 센서모델을 관심영역으로 제시했다.

특히 학계에는 연구가 많지만 실제 시나리오 적용은 큰 action space와 실용문제 때문에 heuristic과 human context에 의존한다고 진단한다.

- Dstl/DASA, **Autonomous Sensor Management and Sensor Counter Deception: Competition Document**, 2023  
  <https://www.gov.uk/government/publications/autonomous-sensor-management-and-sensor-counter-deception/autonomous-sensor-management-and-sensor-counter-deception-competition-document>

- Dstl/DASA, **Frequently Asked Questions**, 2023  
  <https://www.gov.uk/government/publications/autonomous-sensor-management-and-sensor-counter-deception/autonomous-sensor-management-and-sensor-counter-deception-frequently-asked-questions>

FAQ는 특정 센서가 아닌 abstract sensor, 다양한 reward policy의 교체, multi-platform generality, information-theoretic generation-after-next 연구를 명시한다. 이는 본 문서의 capability-contract와 generality 방향을 강하게 뒷받침한다.

2025–2026 Phase 2는 representative sensing network에서 TRL 6 시연을 요구한다.

- UKDI/Dstl, **Autonomous Sensor Management and Sensor Counter Deception Phase 2**  
  <https://www.gov.uk/government/publications/autonomous-sensor-management-and-sensor-counter-deception-phase-2/autonomous-sensor-management-and-sensor-counter-deception-phase-2-competition-document>

## 6.7 Industry direction

전통적 EO/IR turret은 stabilization, optical zoom, automatic video tracker, multi-target tracker, geolocation, image blending을 제공한다.

- L3Harris, **WESCAM MX-15 Air Surveillance and Reconnaissance**  
  <https://www.l3harris.com/all-capabilities/wescam-mx-15-air-surveillance-and-reconnaissance>

최근 L3Harris WESCAM과 Overwatch Imaging의 Automated Sensor Operator는 wide-area scan, detect, classify, automatic track initiation, simultaneous search and multi-target tracking, edge autonomy를 공개적으로 설명한다.

- L3Harris, **From Payload to Platform: Autonomous ISR Where It Actually Matters**, 2026  
  <https://www.l3harris.com/newsroom/editorial/2026/05/payload-platform-autonomous-isr-where-it-actually-matters>

따라서 `산업계는 단순 auto-tracking뿐`이라는 주장은 성립하지 않는다. 공개자료만으로는 다음은 불분명하다.

- mission-conditioned zoom/EO/IR policy
- uncertainty and contradiction handling
- predictive occlusion/reacquisition
- cross-embodiment policy
- bbox-to-fused-map adaptability
- observation sufficiency
- semantic request to platform control
- learning/planning method

## 6.8 Emerging active VLA direction

최근 manipulation과 robotic camera 분야에서는 VLA에 active view/resolution selection을 결합하는 연구가 나타나고 있다.

- **ActiveVLA: Injecting Active Perception into Vision-Language-Action Models**  
  <https://arxiv.org/html/2601.08325v1>

- **The Robotic Eyeball for Embodied Perception / EyeVLA**  
  <https://arxiv.org/html/2511.15279>

이는 언어목적과 카메라 행동을 연결하는 추세를 보여주지만, airborne EO/IR의 uncertainty-aware belief, fast dynamics, sensor constraints, independent motion interface를 그대로 해결하지는 않는다.

## 6.9 연구 공백을 표현하는 안전한 방식

피해야 할 주장:

- 최초의 autonomous sensor management
- 최초의 RL-based PTZ control
- 시장에 autonomous EO/IR operation이 전혀 없음
- 하나의 통합 프레임워크 자체가 주요 과학적 novelty

검증 가능한 주장 후보:

- mission-weighted hypothesis falsification을 위한 active gaze
- decision-oriented object-centric RSSM for future visibility
- observability-aware epistemic distillation from privileged world state
- cross-perception/cross-payload generalization
- independent-but-requesting payload/motion architecture

최종 novelty claim은 체계적인 논문·특허·제품 선행조사를 거쳐 축소 또는 수정해야 한다.

## 6.10 검색 키워드

- active sensing / active perception
- autonomous sensor management
- sensor scheduling / sensor tasking
- belief-space planning
- value of information / Bayesian experimental design
- active camera / PTZ control
- UAV gimbal active tracking
- learned zoom control
- epistemic planning / hypothesis-driven perception
- predictive gaze / anticipatory gaze
- object-centric world model / RSSM
- privileged learning / asymmetric actor critic
- belief distillation / uncertainty distillation
- active VLA / embodied camera control
- sensor counter-deception

