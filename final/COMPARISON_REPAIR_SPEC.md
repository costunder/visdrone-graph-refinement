# 비교 조건 및 번호 정정 — 2026-09-23

상태: **2026-09-23 실험1·2 실행 요청 수신**. 실험1 matched seed42 비교군의 정확한
실행 승인은 `MATCHED_PE_EXECUTION_20260923.md` 및 해당 manifest를 따른다.
실험2 input/구현 및 실험3 단계 경계 보류는 유지한다. 이 문서는 재학습 완료 보고가 아니다.
공개 번호의 기준은 `manifests/current_experiments.json`이다. 기존 manifest의
local/internal ID 및 run directory 이름은 과거 실행을 추적하는 artifact ID로 보존한다.
기존 checkpoint, run_config, metric, prediction, cache를 이동하거나 덮어쓰지 않는다.

## 전체 공개 번호

| 실험 1 | 항목 | 상태 |
|---|---|---|
| 00 | Full-image 기준선 | 기존 결과 |
| 01 | 로컬 GOIS 기준선 | 기존 결과 |
| 02 | No-cluster GNN | 과거 결과 검증 미완료; matched revision 미실행 |
| 03 | DBSCAN GNN | 과거 결과 검증 미완료 |
| 04 | Size refinement GNN | 과거 결과 검증 미완료 |
| 05 | Same-pool pruning | fusion까지 변경된 비교; cutoff-only 주장 금지 |
| 06 | SignNet PE | seed42 학습·평가 완료; matched baseline 없음 |
| 07 | R-PEARL PE | 구현됨; 학습·평가 미실행 |

| 실험 2 | 항목 | 상태 |
|---|---|---|
| 00 | GNN confidence rescue | 기존 결과 |
| 01 | Size refinement rescue | 기존 결과 |
| 02 | Cluster-token HGNN | 기존 결과 |
| 03 | Sparse EdgeSAGE | 기존 결과 |
| 04 | PPR-GATv2-SAGE | 기존 결과 |
| 05 | Full-image PPR 비교 | 후보 생성 조건을 바꾸는 비교 |
| 06 | Class-relation PPR-GATv2-SAGE | 기존 결과; matched PE revision 미실행 |
| 07 | Heterogeneous GNN | target/loss/score도 변경된 비통제 비교 |
| 08 | Controlled heterogeneous GNN | 06 protocol 유지; capacity는 근사 대응 |
| 09 | SignNet PE | 새 production 구현·실행 없음 |
| 10 | R-PEARL PE | 새 production 구현·실행 없음 |

별도의 no-PE 번호는 없다. baseline은 실험 1의 02, 실험 2의 06이다.
실험 1 공개 06의 실제 artifact는 `experiment_1/runs/graph_pe_production_v1/10/signnet/aligned/seed_42`다.
실험 2 공개 09는 SignNet이며, 과거 zero-PE 09와 동일 항목이 아니다.
구형 PE runner는 차단 상태를 유지한다. 공개 번호 변환은 `scripts/map_local_variants.py`가
담당하지만 변환 성공은 실행 승인이나 새 production 구현 완료를 뜻하지 않는다.

## 현재 비교 불일치

실험 1의 기존 02는 H96/L3, epoch당 1,024 graph, val-loss 최대 512 graph다.
완료된 SignNet은 H256/L6, epoch당 전체 6,471 graph, val-loss 전체 548 graph다.
둘 다 120 epochs/seed42여도 PE만의 통제 비교가 아니다. 기존 runner source SHA
검증도 미완료이므로 과거 결과를 matched baseline으로 승격하지 않는다.

## PE 비교에 고정할 공통 계약 (재학습 전 승인 대상)

baseline 번호를 추가하지 않고 기존 02/06의 **새 revision**과 PE를 비교한다.
기존 H96/L3 모델을 PE 모델로 오인하게 이름만 바꾸거나, PE를 축소하지 않는다.

- Backbone H256/L6, PE H128/L8: 기존 승인 architecture를 유지한다.
- Detector weight, dataset/class mapping, train/val image 목록과 GT checksum 고정.
- 각 비교군 안에서 raw candidate cache checksum, 수·순서·view geometry 고정.
- Graph feature, edge/relation, target, sample weight, loss 및 score/NMS 고정.
- 실험 1은 .25 후보, no-cluster node37/edge24와 기존 score fusion을 유지.
- 실험 2는 .05 후보, 06의 node40/edge27 및 PPR/class relation, score와 cutoff 유지.
- 각 run 120 epochs, 매 epoch 전체 train 6,471장, full-val 548장. empty graph는 기록.
- AdamW lr .001, weight decay .0001, betas (.9,.999), eps 1e-8, effective batch32,
  float32, scheduler/early stopping/gradient clipping 없음. OOM 시 architecture 축소 금지.
- Seed별 동일 backbone 초기화·graph 순서. planned seeds 42/43/44; 실행 예약 아님.
- Val-loss 최저 3개만 COCO 평가, 그중 최고 AP 선택, 동률 이른 epoch.
- PE encoder 추가에 따른 parameter 차이는 명시한다. 구조 정보만의 순수 효과나
  동일 capacity라고 주장하지 않는다. alignment control은 별도 설계 승인 없이 실행하지 않는다.

전체 실험을 모두 하나의 동일 조건 ablation으로 묶지 않는다. slicing 제거(실험2/05),
size stage 추가, heterogeneous backbone 등은 각 연구 질문에 맞는 변경 요인을 명시해야 한다.
실험 1의 03/04/05 및 실험 2의 다른 과거 run들이 위 계약으로 재학습되었다고 쓰지 않는다.
전체 variant를 이 protocol로 확장하는 구현은 본 문서로 자동 승인되지 않는다.

완료된 SignNet의 재사용 여부도 자동 확정하지 않는다. 새 baseline과 input identity,
원본 code/config, 초기화, loss/selection, 완료·provenance를 대조한 뒤 결정한다.
`scripts/check_pe_comparison.py`는 저장 config의 보수적 비교 게이트이며,
통과해도 실제 tensor/gradient/완료/provenance 검증이나 실행 승인을 대신하지 않는다.

## 실험 3 — 연결 의미 미확정, 기존 실행 차단

기존 코드는 frozen 실험1 → 5 prior → 실험2다. 사용자가 이 연결을 거부했으므로
기존 00/01은 승인된 active 실험으로 취급하지 않는다. source와 synthetic test는 보존한다.
대체 구조를 번호만 뒤집어서 구현하지 않는다. 다음 경계는 사용자 확인이 필요하다.

1. 실험2의 출력이 후보/점수/embedding 중 무엇인가?
2. GOIS 단계와 실험1 처리의 정확한 위치는 어디인가?
3. 전달되는 후보의 confidence 분포와 실험1 재학습 입력을 어떻게 일치시키는가?
4. 중간 cutoff/NMS 유무, 최종 scoring, freeze/trainable 범위는 무엇인가?

이 항목이 미확정이므로 실험3은 production 승인 대상이 아니며 GPU 실행 금지다.
현재 runner는 main/run_variant/train_epoch에서 CLI/환경변수와 무관하게 실패한다.

## 완료 범위

이번 정정은 번호·상태·실행 차단·config 비교 검사다. 새 model/loss/candidate policy는
구현하지 않았고 학습·detector inference·COCO 평가는 실행하지 않았다.
비교 가능한 새 학습 결과가 생성되었다거나 전체 연구가 복구 완료되었다고 주장하지 않는다.
