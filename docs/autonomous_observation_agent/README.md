# Autonomous Observation Agent Research Notes

이 문서 세트는 항공 탑재 EO/IR 관측장비의 자율 운용을 출발점으로, **자신이 무엇을 알고 무엇을 모르는지 판단하고 임무에 필요한 증거를 스스로 찾아보는 자율 관측 에이전트**에 관한 연구 방향을 정리한다.

이 자료의 목적은 특정 짐벌·센서 조합을 구현하기 위한 상세 설계서를 만드는 것이 아니다. 연구의 목적과 가치, 문제정의, 선행 분야, 가능한 기술적 접근, 검증 가능한 연구가설, 시연 시나리오를 보존하고 이후 연구기획·제안서·논문·실험설계의 기준점으로 사용하는 것이다.

## 문서 구성

1. [01_research_vision.md](01_research_vision.md)  
   출발점, 책임 경계, 연구적 전환, 궁극적 비전과 핵심 용어를 정리한다.

2. [02_problem_space_and_taxonomy.md](02_problem_space_and_taxonomy.md)  
   다양한 perception 수준, payload 구성, platform-motion provider, 임무와 관측 primitive를 하나의 문제공간으로 정리한다.

3. [03_background_technology_and_ideas.md](03_background_technology_and_ideas.md)  
   RL, POMDP, RSSM, Transformer, GNN, sLLM/VLM/VLA, uncertainty, active inference, causal reasoning, diffusion 등 가능한 배경기술과 wild research ideas를 폭넓게 정리한다.

4. [04_three_research_candidates_deep_dive.md](04_three_research_candidates_deep_dive.md)  
   다음 세 연구를 가설, 수학적 정의, 모델 구조, 학습목표, baseline, 실험, metric, 위험요인까지 심층적으로 다룬다.
   - Hypothesis-Driven Active Observation
   - Dream-to-Look with an Object-Centric RSSM
   - Epistemic Distillation

5. [05_project_value_and_roadmap.md](05_project_value_and_roadmap.md)  
   세 연구가 Flight Intelligence 관련 프로젝트와 Project Nightfall에 주는 가치, 적합도, 단계적 연구 로드맵과 장기 통합 방향을 정리한다.

6. [06_prior_work_and_references.md](06_prior_work_and_references.md)  
   관련 학술 분야와 산업 동향, 공개 선행사례, 연구 공백 및 참고 링크를 정리한다.

## 한 문장 비전

> 임무 수행에 필요한 지식과 현재 관측 사이의 결손을 스스로 인식하고, 이를 해소하기 위한 관측질의를 생성·수행·종료하는 자율 관측 에이전트를 연구한다.

## 연구의 중심적 전환

| 단계 | 중심 질문 |
|---|---|
| 짐벌 제어 | 지정된 방향을 정확히 볼 수 있는가? |
| 자동 추적 | 지정된 표적을 놓치지 않을 수 있는가? |
| 센서 관리 | 탐색·추적·줌·EO/IR를 효율적으로 선택할 수 있는가? |
| 임무가치 기반 관측 | 어떤 관측이 미래 임무판단에 가장 유용한가? |
| 인식적 자율성 | 무엇을 모르는지 발견하고, 필요한 증거를 스스로 찾아볼 수 있는가? |

범용 프레임워크, 입력 adapter, 장비 capability profile은 중요한 기반이지만 그 자체를 최종 연구기여로 간주하지 않는다. 이들은 연구가 특정 시나리오의 구현 요령이나 reward hacking에 불과하지 않음을 검증하기 위한 실험 기반이다.

