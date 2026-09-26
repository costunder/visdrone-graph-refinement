# Relation-aware Proposal Refinement for Tiny Object Detection

`final/`은 실험 1, 실험 2, 두 분기를 결합하는 실험 3을 물리적으로 분리하며,
**각 실험은 독립적으로 `00`부터 번호를 시작한다**. 이전 통합 runner의 번호는
shared runner의 내부 호환 정보로만 보존한다.

**2026-09-23 정정:** 공개 번호는 [현재 목록](manifests/current_experiments.json),
비교 조건과 실행 보류는 [비교 복구 계약](COMPARISON_REPAIR_SPEC.md)이 기준이다.
별도 no-PE 번호는 제외했다. 기존 폴더 번호는 provenance를 위해 보존하며 공개 번호와
명시적으로 매핑한다. SignNet 한 run의 완료는 matched PE 비교 완료를 뜻하지 않는다.
2026-09-23 후속 실행 요청으로 실험1 matched seed42 비교군은
[별도 실행 계약](MATCHED_PE_EXECUTION_20260923.md)에 따라 재승인되었다.
기존 study의 재실행 및 실험3 차단은 유지하며 실험2는 입력/구현 준비 전 시작하지 않는다.

실행 관찰: 실험1 공개 02의 새 `matched_pe_v2` seed42 run이 2026-09-23 19:30 KST
GPU 학습을 시작했다. 다음 순서는 공개 06/07이며 실패 시 중단한다. 최신 상태는
`experiment_1/runs/matched_pe_v2/02_gnn_no_cluster/seed_42/progress.json`이다.
기존 02의 역사 검증 미완료 상태와 새 run의 학습 상태를 혼동하지 않는다.

2026-09-15 검증 상태 정정: 아래 실험 1의 02-04 수치는 원본 역사 artifact에 기록된
값이지만 실행 당시 runner SHA 확인이 끝나지 않은 `unverified_historical`이다.
05는 fusion 조건까지 달라 `invalid_control_comparison`이다. 새 PE 비교의 검증된
baseline으로 사용하지 않는다. 원본 수치를 삭제하거나 새 값으로 덮어쓰지 않았다.

승인된 새 study는 `graph_pe_production_v1`이다. 256×6 backbone + 128×8 PE의
구현과 검증은 [설계](GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md),
[승인·상태 manifest](manifests/graph_pe_production_proposal.json)를 따른다.
구형 96×3 PE 실행기는 계속 차단되며 새 결과와 혼합하지 않는다.

## Local Variant Order

### Experiment 1: Fixed Candidate Pool Ablation

| Local ID | Variant | AP |
|---|---|---:|
| 00 | `00_full_inference` | 0.104598 |
| 01 | `01_gois_reimplementation` | 0.117279 |
| 02 | `02_gnn_no_cluster` | 0.135093 |
| 03 | `03_gnn_dbscan_cluster` | 0.134531 |
| 04 | `04_size_refinement_gnn` | 0.135060 |
| 05 | `05_gnn_prune_same_pool` | 0.147710 |
| 06 | SignNet PE (artifact `10/signnet`) | seed42 completed; matched comparison incomplete |
| 07 | R-PEARL PE (legacy artifact ID `11`) | implemented, not trained |

### Experiment 2: Low-confidence Candidate Rescue

| Local ID | Variant | AP / Status |
|---|---|---|
| 00 | `00_gnn_conf_rescue` | 0.154986 |
| 01 | `01_size_refinement_conf_rescue` | 0.155612 |
| 02 | `02_low_conf_cluster_token_hgnn_refinement` | 0.152964 |
| 03 | `03_sparse_edge_sage` | 0.175821 |
| 04 | `04_ppr_gatv2_sage` | 0.177066 |
| 05 | `05_ppr_gatv2_sage_no_gois` | 0.116618 |
| 06 | `06_class_relation_ppr_gatv2_sage` | 0.177150 (relational homogeneous) |
| 07 | `07_hetero_detection_class_view_gatv2` | 0.152326 (non-controlled heterogeneous) |
| 08 | `08_controlled_hetero_ppr_gatv2_sage` | 0.175064 (controlled heterogeneous) |
| 09 | SignNet PE (legacy artifact ID `10`) | model CPU verified; input/runner pending; not trained |
| 10 | R-PEARL PE (legacy artifact ID `11`) | model CPU verified; input/runner pending; not trained |

### Experiment 3: Stage Design Unconfirmed — Execution Blocked

| Local ID | Variant | Status / Role |
|---|---|---|
| legacy 00 | `00_zero_prior_control` | blocked; old implementation only |
| legacy 01 | `01_exp1_graph_prior_exp2_ppr` | blocked; user-rejected stage direction |

아래는 승인된 새 설계가 아니라 보존된 기존 구현의 설명이다. 실험 3의 단계 경계는
재확인이 필요하며 번호 순서만 반대로 바꿔 구현하지 않는다. 기존 코드는 하나의 raw `0.05`
coarse+fine pool에서 `>=0.25` core에 실험 1 graph scorer를 먼저 적용하고, 그
NMS 이전 score를 전체 pool의 node feature로 전달한 뒤 실험 2 sparse PPR graph가
후처리한다. 후보 제거와 class-wise NMS는 모든 graph 처리가 끝난 뒤 한 번만 한다.

## Layout

```text
final/
├── experiment_1/   # historical 00-05 with audited status; separate production PE study
├── experiment_2/   # completed 00-08; configured PE 09-11
├── experiment_3/   # local IDs 00-01; sequential pre-NMS graph pipeline
├── runs/           # shared detector weight and raw detector cache only
├── manifests/
└── scripts/
```

현재 공개 번호와 legacy variant 대응은 `manifests/current_experiments.json`에 기록한다.
`manifests/experiment_variant_ids.json`은 기존 실행기/artifact 호환 기록이다.

## Execution

아래 명령은 역사 진입점 기록이다. 현재 실행 승인이 아니며, 실험3은 환경변수와
무관하게 차단된다. 재승인된 실험1은 별도 `--study matched_pe_v2` 범위만 허용한다.
구형 PE의 숫자 09/10/11은 현재 공개 번호가 아니므로 그대로 실행 요청에 쓰지 않는다.

**구형 PE 실행은 계속 차단 상태다.** 아래 PE 명령은 기존 진입점 기록이며 승인
플래그가 있어도 96×3 모델을 실행하지 않는다. 새 architecture는
[`GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`](GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md)의
승인된 설계다. 별도 새 실행기는 `run_graph_pe_production.py`이며 구현·실행 상태는
manifest와 `experiment_1/runs/graph_pe_production_v1/`의 실제 audit/progress를 따른다.
실험 2의 low-confidence actual-view cache 재생성은 별도 승인이 필요하다.

`RELGRAPH_VARIANTS`도 각 실험의 local ID를 받는다.

```bash
RELGRAPH_EXPERIMENT=1 RELGRAPH_VARIANTS=00,01 \
  final/scripts/run_table6_yolo11_10class_extra_ablation.sh

RELGRAPH_EXPERIMENT=2 RELGRAPH_VARIANTS=00,01,02 \
  final/scripts/run_table6_yolo11_10class_extra_ablation.sh

RELGRAPH_EXPERIMENT=2-followup RELGRAPH_VARIANTS=08 \
  final/scripts/run_table6_yolo11_10class_extra_ablation.sh

# 아래 명령은 승인 없이는 즉시 종료한다.
ALLOW_MODEL_TRAINING=1 RELGRAPH_VARIANTS=00,01 \
  final/experiment_3/run_experiment_3.sh

# PE runner는 학습과 COCO 평가를 각각 승인해야 한다.
ALLOW_MODEL_TRAINING=1 ALLOW_COCO_EVALUATION=1 GRAPH_PE_VARIANTS=09,10,11 \
  final/experiment_1/run_graph_pe_ablation.sh

ALLOW_MODEL_TRAINING=1 ALLOW_COCO_EVALUATION=1 GRAPH_PE_VARIANTS=09,10,11 \
  final/experiment_2/run_graph_pe_ablation.sh
```

학습 variant는 `ALLOW_MODEL_TRAINING=1` 없이는 실행되지 않는다. 실험 2의 `06`,
`07`, `08`은 기존 low-confidence detector cache를 재사용해 각각 40 epochs 학습 및
COCO 평가를 완료했다. candidate/target/loss/scoring을 `06`과 동일하게 고정한
통제형 heterogeneous `08`은 AP 0.175064로 `06`보다 0.002086 낮고, 비통제형
`07`보다 0.022738 높다. `07`의 같은 checkpoint에서 수행한 inference-policy
ablation 최고 AP 0.155987은 evaluation split에서 선택된 post-hoc 결과로
분리한다.

Graph PE `09-11`은 masked zero control, SignNet adaptation, R-PEARL adaptation 순서다. 등록 parameter 수와 실행 operator 수는 같지만 활성 학습 용량은 같지 않다. 2026-09-15에 pooling 순서, top-3-only 평가, 원본 precision, 직접 호출 가드 및 cache-only 경로를 정정하고 CPU 계약 검사를 통과했다. 학습·detector inference·COCO 평가는 실행하지 않았다. 상세 설계와 한계는 `GRAPH_POSITIONAL_ENCODING_SPEC.md`에 있다.

```bash
CUDA_VISIBLE_DEVICES='' /home/kim-sanghwa/miniconda3/envs/GNN/bin/python -B final/test_graph_pe_contract.py
CUDA_VISIBLE_DEVICES='' /home/kim-sanghwa/miniconda3/envs/GNN/bin/python -B final/test_graph_pe_runner_contract.py
```

실험 3은 현재 코드와 통제 조건만 구성했으며 학습·평가를 실행하지 않았다. 기존
detector cache가 없으면 inference로 대체하지 않고 실패하며, 두 variant는 다섯
Experiment-1 prior channel의 값만 달라진다. 과거 post-NMS 후기 융합 구현과 수치는
의도한 실험이 아니므로 `experiment_3/archive/`에 무효 보관했다.

## Claim Boundary

`experiment_1/01`은 공식 GOIS 실행 결과가 아니라 local matched-condition
reimplementation이다. 실험 3은 실험 1의 frozen pre-NMS graph score를 실험 2의
full low-confidence graph에 전달하는 순차 결합이며, 아직 성능 수치는 없다. PE `09-11` 역시 실제 run 전에는 성능 주장을 하지 않으며 zero control을 거치지 않은 기존 `02`/`06`과의 직접 차이를 PE 효과로 해석하지 않는다. 자세한
provenance와 라이선스 범위는 `THIRD_PARTY_NOTICES.md`를 따른다.
