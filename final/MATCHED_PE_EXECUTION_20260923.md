# 동일 조건 재학습 실행 계약 — 2026-09-23

사용자 요청 “아니 실험 1과 ㅅ실험2 제대로 돌려보라고”로 실험1·2 실행을 요청받았다.
비교 복구 계약의 H256/L6 및 full-data 학습 조건을 따른다. 실험3은 승인 범위 밖이다.
기존 전체 variant 확장과 PE 우선 비교 중 어느 쪽이든 필요한 공통 선행 작업으로
실험1 기존 02의 matched revision부터 시작한다. 과거 02 및 완료된 SignNet 결과는
새 결과로 재명명하거나 이어 학습하지 않는다.

## 이번 실행 가능한 비교군

- 실험1 공개 02: 기존 no-cluster backbone, H256/L6, PE 없음. 별도 no-PE 번호 없음.
- 실험1 공개 06: 같은 backbone + SignNet H128/L8, k32.
- 실험1 공개 07: 같은 backbone + R-PEARL H128/L8, K12/M120.
- 이번 seed는 42. 동일 비교군에서 모든 설정과 source SHA를 고정한다.
- source model/operator, target/loss/score는 기존 승인 설계와 동일하다.
- 기존 actual-view .25 cache와 precomputed graph를 checksum 검증 후 공통 사용한다.
- 각 120 epochs, train 6,471/val 548 전체, batch32/microbatch1, float32.
- AdamW lr=.001, weight_decay=.0001, betas=(.9,.999), eps=1e-8.
- Seed/graph 순서/backbone 초기화/loss 및 최종 NMS 동일. Scheduler/early stopping 없음.
- Val-loss 최저 3개만 COCO 평가한 뒤 highest AP 선택. 매 epoch portable checkpoint 보존.
- Raw feature37/edge24, candidate/edge/target/weight 및 score fusion은 생산 설계 §3을 따른다.
- PE parameter 증가는 명시한다. 구조 정보만의 pure-PE 효과라고 주장하지 않는다.
- 각 run은 `experiment_1/runs/matched_pe_v2/{공개번호_이름}/seed_42/`에 새로 저장한다.
- CLI/환경변수 가드, CPU 및 actual-graph CUDA conformance, config comparison을 통과해야 한다.
- foreground 프로세스만 사용한다. 실패 시 다음 run을 시작하지 않고 자동 재시작하지 않는다.

실험2는 실행 요청을 받았지만 새 production 구현 및 actual-view .05 cache가 준비되지 않았다.
작동하지 않는 구형 96×3 runner를 승인 해제하거나 crop 정보를 합성하지 않는다.
별도 detector cache 재생성 승인은 아직 확인되지 않았다. 실험2를 실행 중이라고 쓰지 않는다.

원본 `graph_pe_production_v1` 실행 승인은 보류 상태로 유지한다. 이번 승인은 새
`matched_pe_v2` study, 지정 mode·seed·alignment에만 적용한다. 모델 용량/입력 정책을
바꾸지 않으며 architecture를 새로 창작하거나 축소하지 않는다.
