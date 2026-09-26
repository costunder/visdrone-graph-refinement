# Experiment 1: Fixed Candidate-pool Ablation

공개 ID는 다른 실험과 독립적으로 `00`부터 시작한다. 기준은
[`current_experiments.json`](../manifests/current_experiments.json)이다.
기존 run 폴더 번호는 보존한다. 사용자 후속 실행 요청에 따라
[matched 실행 계약](../MATCHED_PE_EXECUTION_20260923.md)의 seed42 비교군을 재승인했다.
완료된 과거 SignNet과 기존 02는 여전히 matched comparison이 아니다.

| Local ID | Variant | Internal legacy name | Role |
|---|---|---|---|
| 00 | `00_full_inference` | same | full-image control |
| 01 | `01_gois_reimplementation` | same | local GOIS-Det reimplementation |
| 02 | `02_gnn_no_cluster` | same | typed relation GNN, no cluster relation |
| 03 | `03_gnn_dbscan_cluster` | same | 02 + normalized DBSCAN relation |
| 04 | `04_size_refinement_gnn` | `05_size_refinement_gnn` | 02 + size-only stage 2 |
| 05 | `05_gnn_prune_same_pool` | `08_gnn_prune_same_pool` | 02 + score pruning |
| 06 | `gnn_no_cluster_signnet_pe` | artifact ID `10` | H256/L6 + SignNet H128/L8; seed42 120 epochs and 3 COCO checkpoints completed; matched baseline absent |
| 07 | `gnn_no_cluster_rpearl_pe` | artifact ID `11` | production implementation exists; training/evaluation not run |

## Corrected pipeline (graph schema v2)

```text
coarse tiles ─┐
              ├─ raw candidate pool (no tile/cross-scale NMS)
 fine tiles ──┘
                       │
                       ├─ 01: detector score ───────────────┐
                       └─ 02-05: typed relation GNN/rerank ─┤
                                                           ▼
                                              final class-wise NMS
```

- 겹치는 후보도 GNN 입력 전에는 지우지 않는다. 서로 다른 crop의 박스는 위치와 크기가 비슷해도 별도 노드다.
- 각 노드는 detector predicted class와 view type(`full/coarse/fine/ROI`), 원본 이미지에서의 view bbox 및 view 내부 상대 좌표를 가진다. GT class를 node type으로 쓰지 않는다.
- edge type은 `self`, `spatial`, `overlap`, `containment`, `same_cluster`, `same_view`, `cross_view`, `cross_class_context`다.
- 선택된 non-self edge는 항상 양방향으로 만든다. 거리는 이미지 크기로 정규화하고, sparse spatial grid를 사용해 `N x N` 관계 행렬을 만들지 않는다.
- 01과 02-05는 동일한 raw coarse+fine detector pool에서 출발한다. 차이는 01이 바로 NMS하고 02-05가 graph rerank 뒤 NMS한다는 점이다.

그래프 스키마의 node dim은 `27 + num_classes`, edge dim은 `24`다. cache schema는 `gois_graph_schema_v4`, pipeline code version은 `48`이다.
2026-09-15 감사 정정: 02의 config에 code48과 120 epochs가 있지만 실행 당시 runner SHA를
복원하지 못했으므로 02-04는 `unverified_historical`이며 완료/검증 주장을 하지 않는다.
05는 fusion까지 변경되어 `invalid_control_comparison`이다. 기존 산출물은 보존한다.

검증:

```bash
/home/kim-sanghwa/miniconda3/envs/GNN/bin/python final/experiment_1/test_relation_graph_schema.py
/home/kim-sanghwa/miniconda3/envs/GNN/bin/python final/test_graph_pe_contract.py
/home/kim-sanghwa/miniconda3/envs/GNN/bin/python final/test_graph_pe_runner_contract.py
```

- Results: `runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/`
- Publication summary: `reports/publication_results_best_ap.csv`
- Existing runner: `run_experiment_1.sh` (local ID `00-05`)
- Legacy PE runner: `run_graph_pe_ablation.sh` (blocked; public PE IDs `06,07` map to legacy names)
- PE specification: `../GRAPH_POSITIONAL_ENCODING_SPEC.md`

## Historical masked-PE configuration (blocked, not current public enumeration)

아래 09/10/11은 보존된 구형 실행기 ID이며 공개 실험 번호가 아니다. 별도 no-PE
항목은 공개 목록에서 제외했다. 현재 06 SignNet 완료 run은
`runs/graph_pe_production_v1/10/signnet/aligned/seed_42/`다.

현재 아래 구형 96×3 구성은 CPU 회귀 검사용이며 production 실행은 차단했다.
학습·평가 플래그로도 우회할 수 없다. 256×6 backbone과 128×8 PE의 승인된 설계는
[`GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`](../GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md)에
있으며, 새 구현은 `../graph_pe_production.py`, CPU 계약 검사는
`../test_graph_pe_production_contract.py`에 분리한다. 새 실행 및 conformance 상태는
`../manifests/graph_pe_production_proposal.json`과 해당 run artifact를 따른다.
아래 96×3 설명은 legacy CPU 회귀 경로만 기술하며 새 production 모델 설명이 아니다.

`09/10/11`은 모두 SignNet과 R-PEARL module을 instantiate하므로 등록된 trainable parameter 수는 `628,810`으로 같다. activation mask는 각각 `[0,0]`, `[1,0]`, `[0,1]`이다. raw candidate pool, node/edge, 96-wide 3-layer base GNN, target, loss, seed, score fusion과 final NMS는 동일하다. ReZero 초기 출력은 같은 seed의 새 02 backbone과 동일하며 과거 학습 checkpoint를 로드한다는 뜻은 아니다.

다만 gate가 열린 뒤 loss에 연결 가능한 PE parameter는 각각 `0 / 84,676 / 94,948`이므로 active-capacity-matched 비교가 아니다. PE 활성화와 추가 학습 용량의 효과를 분리했다고 주장하지 않는다. 2026-09-15에 모델 계약 및 실제 runner import/tensor/NMS 경로 CPU 검사를 통과했다. 정식 checkpoint, AP/AR, score plot은 생성하지 않았다.

`RELGRAPH_VARIANTS`는 기존 runner의 active local ID `00-05`에만 사용한다. PE runner는 별도의 `GRAPH_PE_VARIANTS=09,10,11`을 사용한다.
