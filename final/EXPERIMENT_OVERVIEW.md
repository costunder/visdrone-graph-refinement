# 전체 실험 설계·결과·검증 현황

- 기존 결과 기준일: 2026-09-11
- PE 구현·검증 및 주장 경계 정정일: 2026-09-15 (기존 실험 수치 재평가 없음)
- 대상: `final/experiment_1`, `final/experiment_2`, `final/experiment_3`
- 데이터셋: VisDrone Detection train/validation, 10-class mapping
- 평가: pycocotools COCO AP/AR
- 결과 원본: [`reports/experiment_results.csv`](reports/experiment_results.csv)
- 번호 기준: [`manifests/variant_order.json`](manifests/variant_order.json)

이 문서는 현재 저장소의 specification, manifest, run config, checkpoint·metric
산출물 및 구현 점검 내용을 한곳에 정리한 연구 현황 문서다. 문서 작성 과정에서는
학습, detector inference, COCO evaluation 또는 GPU 작업을 실행하지 않았다.

2026-09-15 추가 상태: 기존 96×3 masked-branch PE runner는 production 실행을
차단하고 CPU 회귀 검사용으로만 보존했다. 256×6 backbone/128×8 PE와 full-data,
활성 alignment control, 폭·깊이 비교 제안은
[`GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`](GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md)에
분리했으며 사용자 설계 승인 전이다. 새 architecture나 실험 결과가 추가된 것은 아니다.

> **해석 주의:** 아래의 `기록 완료`는 checkpoint 또는 평가 수치가 저장되어 있다는
> 뜻이지, 곧바로 논문에 사용할 수 있다는 뜻이 아니다. 문서·manifest·코드·산출물이
> 충돌하거나 통제 변인이 오염된 경우에는 저장소 규칙에 따라 `unverified`,
> `historical_nonconformant`, `invalid_control`로 별도 표시한다.

## 1. 연구 전체 구조

이 연구의 공통 목표는 다중 해상도 detector가 만든 후보를 최종 NMS 전에 graph로
구조화하여, 작은 객체 후보의 점수를 더 잘 보정하고 낮은 confidence 후보를 구조적
문맥으로 복구할 수 있는지 확인하는 것이다.

| 구분 | 핵심 연구 질문 | 후보 pool | graph 역할 | 현재 상태 |
|---|---|---|---|---|
| 실험 1 | 고정된 후보 pool에서 typed relation GNN이 proposal scoring을 개선하는가? | confidence 0.25 coarse + fine | 후보 재점수화, cluster·size·cutoff ablation | `00–05` 수치 존재, 일부 unverified/invalid |
| 실험 2 | 낮은 confidence 후보를 sparse relation/PPR graph로 복구할 수 있는가? | confidence 0.05 coarse + fine | sparse rescue, PPR, class relation, heterogeneous graph | `00–08` 수치 존재, 현재 계약별 유효성 상이 |
| 실험 3 | 실험 1의 pre-NMS graph score를 prior로 주면 실험 2 rescue가 개선되는가? | 하나의 confidence 0.05 coarse + fine pool | frozen 실험 1 prior → 실험 2 sparse PPR | configured, not run |
| Graph PE | candidate·edge·backbone을 고정했을 때 PE 분기 활성화가 어떤 영향을 주는가? | 실험 1/2의 각 기준 pool | zero control, SignNet, R-PEARL; active capacity는 다름 | configured, not run |

세 실험의 관계는 다음과 같다.

```text
Experiment 1
raw conf=0.25 coarse+fine candidates
        -> typed sparse graph reranking
        -> one final class-wise NMS

Experiment 2
raw conf=0.05 coarse+fine candidates
        -> sparse local/PPR graph rescue
        -> score cutoff 0.25
        -> one final class-wise NMS

Experiment 3
one raw conf=0.05 coarse+fine pool
        -> conf>=0.25 core만 frozen Experiment-1 scorer에 입력
        -> [core, object, small, large, ROI] prior를 full pool에 부착
        -> Experiment-2/06 형식의 sparse PPR stage
        -> score cutoff 0.25
        -> one final class-wise NMS
```

실험 3은 실험 1과 실험 2의 최종 prediction을 합치는 late fusion이 아니다. 두 graph
stage 사이에 후보 제거, 중간 NMS 또는 COCO prediction 생성이 없어야 한다.

## 2. 공통 데이터·평가 조건

| 항목 | 현재 계약 |
|---|---|
| Detector | `final/runs/table6_yolo11_10class_extra_ablation/detector/weights/best.pt` |
| 학습 split | `Full/data/visdrone_det_yolo_10class/images/train` |
| 평가 split | `Full/data/visdrone_det_yolo_10class/images/val` |
| 평가 GT | `Full/data/visdrone_det_yolo_10class/annotations/val_coco_gt.json` |
| Class mapping | YOLO class `0..9` → COCO category id `1..10` |
| Coarse slicing | size 640, overlap 0.20 |
| Fine slicing | size 256, overlap 0.20 |
| Detector IoU | 0.70 |
| Final class-wise NMS IoU | 0.40 |
| `max_det` | 300 |
| 기본 seed | 42 |
| 평가 지표 | AP, AP50, AP75, AP-small/medium/large, AR |

학습형 GNN의 공통 기록값은 batch size 32, epoch당 최대 1,024 graph update,
validation-loss subset 최대 512 graph, AdamW `lr=1e-3`, `weight_decay=1e-4`다.
주요 target 설정은 positive IoU 0.50, ROI-support IoU 0.10,
ROI center margin 0.75다. 단, 실험 2의 초기 variant와 비통제형 `07`은 현재
단일-factor 계약과 다른 target·loss·policy를 사용하므로 같은 통제군으로 묶으면 안 된다.

Detector cache에는 full class probability가 아니라 hard `category_id`와 score만 있다.
따라서 현재 class-relation 실험은 detector의 전체 class uncertainty를 입력으로 쓰지
못한다.

## 3. 실험 1 — Fixed Candidate-pool Ablation

### 3.1 연구 질문과 승인된 파이프라인

실험 1의 핵심 질문은 동일한 raw coarse + global-fine 후보를 그대로 유지한 상태에서,
최종 NMS 전에 typed relation GNN을 넣는 것이 proposal score를 개선하는지다.

- `01`과 `02–05`는 동일한 raw pre-cross-view-NMS 후보 pool에서 시작해야 한다.
- graph 전에 겹치는 crop 후보를 병합하거나 제거하지 않는다.
- 최종 class-wise NMS는 모든 graph 처리 뒤 한 번만 수행한다.
- GT class는 node type 또는 inference feature로 사용하지 않는다.
- node feature dim은 `37 = 27 + 10`, edge feature dim은 24다.
- cache schema는 `gois_graph_schema_v4`, pipeline code version은 48이다.
- relation은 `self`, `spatial`, `overlap`, `containment`, `same_cluster`,
  `same_view`, `cross_view`, `cross_class_context`를 구분한다.
- 선택된 non-self edge는 양방향이며 거리값은 image size로 정규화한다.
- spatial relation은 sparse grid로 만들고 dense `N x N` relation matrix를 만들지 않는다.

기본 graph scorer는 hidden width 96, 3개의 `EdgeGatedLayer`, residual/skip path를
사용하며 현재 계산된 trainable parameter 수는 449,187이다. `04`의 size-only
stage 2는 2개 layer 설정을 사용한다. 이 숫자는 모델이 detector 전체가 아니라
proposal postprocessor라는 점을 고려해야 하지만, 현재 scale/saturation ablation이
없으므로 충분한 capacity라고 입증된 상태도 아니다.

### 3.2 Variant 정의

| ID | Variant | 의도된 단일 변인 |
|---:|---|---|
| 00 | `00_full_inference` | full-image detector control |
| 01 | `01_gois_reimplementation` | coarse 640 + fine 256 local GOIS-style baseline |
| 02 | `02_gnn_no_cluster` | fixed pool에 cluster 없는 3-head typed GNN 추가 |
| 03 | `03_gnn_dbscan_cluster` | `02` 대비 normalized DBSCAN context만 추가 |
| 04 | `04_size_refinement_gnn` | `02` posterior를 입력으로 쓰는 size-only stage 2 추가 |
| 05 | `05_gnn_prune_same_pool` | 원래 의도는 `02` score 뒤 cutoff 0.25만 추가 |
| 09 | `09_gnn_no_cluster_zero_pe` | masked zero-PE control; 등록 수만 동일 |
| 10 | `10_gnn_no_cluster_signnet_pe` | SignNet project adaptation 활성화 |
| 11 | `11_gnn_no_cluster_rpearl_pe` | R-PEARL project adaptation 활성화 |

### 3.3 저장된 평가 수치

아래 수치는 실제 결과표에 보존된 값이다. `검증 상태`는 수치 존재 여부와 논문용
유효성을 구분한다.

| ID | Epoch | Predictions | AP | AP50 | AP75 | AP-small | AP-medium | AP-large | 검증 상태 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 00 | 0 | 22,369 | 0.104598 | 0.168043 | 0.109274 | 0.036672 | 0.170474 | 0.333570 | recorded baseline |
| 01 | 0 | 64,333 | 0.117279 | 0.212564 | 0.112201 | 0.086742 | 0.163154 | 0.177452 | recorded; 문서 간 metric 충돌 |
| 02 | 116 | 64,270 | 0.135093 | 0.242665 | 0.130074 | 0.104773 | 0.184753 | 0.189976 | `unverified` |
| 03 | 109 | 64,270 | 0.134531 | 0.242602 | 0.129257 | 0.103990 | 0.183989 | 0.188046 | `unverified` |
| 04 | 120 | 64,264 | 0.135060 | 0.243248 | 0.129798 | 0.104138 | 0.184419 | 0.189825 | `unverified` |
| 05 | 116 | 58,383 | 0.147710 | 0.261229 | 0.144015 | 0.104332 | 0.203742 | 0.211145 | `invalid_control` + `unverified` |
| 09 | — | — | — | — | — | — | — | — | `configured_not_run` |
| 10 | — | — | — | — | — | — | — | — | `configured_not_run` |
| 11 | — | — | — | — | — | — | — | — | `configured_not_run` |

### 3.4 현재 수치에서 읽을 수 있는 것

- `02 - 01 = +0.017814 AP`가 기록되어 있으나, 현재 문서·manifest 충돌이 해결되기
  전에는 schema-v4 GNN의 확정 개선치로 승격할 수 없다.
- `03 - 02 = -0.000562 AP`다. 현재 단일-seed 기록은 DBSCAN cluster context의
  이점을 보이지 않는다.
- `04 - 02 = -0.000033 AP`로 사실상 동일하다. size-only stage 2가 전체 AP를
  개선했다고 말할 근거가 없다.
- `05 - 02 = +0.012617 AP`지만 cutoff 효과로 해석할 수 없다. `02` 평가는
  `max(detector, 0.5*detector + 0.5*graph)`를 쓰고, `05` 평가는
  `0.75*detector + 0.25*graph` 뒤 cutoff를 사용한다. 즉 cutoff뿐 아니라 score
  fusion까지 바뀌어 `05`는 현재 통제 실험으로 무효다.

### 3.5 실험 1 검증 문제

1. [`GOIS_FINAL_REPRO_SPEC.md`](GOIS_FINAL_REPRO_SPEC.md)와
   [`manifests/variant_order.json`](manifests/variant_order.json)은 `02–05`를
   graph schema v2 재학습 필요 상태로 남겨 두었지만,
   [`experiment_1/README.md`](experiment_1/README.md)와 publication metadata는
   schema v4/code 48 재학습 완료 및 publication-ready라고 기록한다.
2. 실제 run config와 checkpoint는 code 48, 120-epoch 계열 산출물임을 보여 주지만,
   저장된 runner SHA에 대응하는 source snapshot이 보존되어 있지 않아 현재 코드와
   exact implementation identity를 복원할 수 없다. 규칙상 이 충돌은 `unverified`다.
3. 공통 specification의 historical `01` AP는 0.117158이고 현재 결과표는 0.117279다.
   어느 prediction artifact를 canonical로 삼는지 명시적으로 정리해야 한다.
4. `05`는 위에서 설명한 score-fusion 오염 때문에 AP 0.147710을 pruning 단독
   효과로 인용하면 안 된다.
5. base capacity에 대한 module별 parameter inventory, operator call count,
   message-path/gradient audit 및 width/depth scale ablation이 없다.

따라서 현재 실험 1에서 가장 높은 숫자는 `05`지만, 유효한 최고 성능이나 논문
결론으로 사용할 수 없다.

## 4. 실험 2 — Low-confidence Candidate Rescue

### 4.1 연구 질문과 현재 sparse 계약

실험 2는 confidence 0.05의 coarse + fine 후보를 먼저 넓게 확보하고, sparse graph
안에서 문맥적으로 의미 있는 후보를 재점수화한 뒤 cutoff 0.25와 final NMS를 적용한다.

- VisDrone 10-class 기준 node dim은 40, edge dim은 27이다.
- local edge는 self-loop와 최대 same-class 12, cross-class 4 neighbor를 유지한다.
- PPR은 alpha 0.15, 8 propagation steps, frontier 64, node당 top-8 neighbor를 쓴다.
- 기본 scorer는 hidden width 96, depth 3, 4 attention heads, dropout 0.10과
  residual/skip update를 사용한다.
- `06`은 detection node 하나의 tensor만 쓰는 class-relation-aware homogeneous
  graph다. formal heterogeneous graph가 아니다.
- `07`과 `08`만 `detection`, `class`, `view` node store를 갖는 heterogeneous
  graph다.
- `08`은 observed class만 materialize하고 detection당 하나의 class membership
  edge를 쓴다. `07`의 all-class support와 class-correction head는 사용하지 않는다.

### 4.2 Variant 정의

| ID | Variant | 핵심 구성 |
|---:|---|---|
| 00 | `00_gnn_conf_rescue` | 초기 pairwise low-confidence GNN rescue |
| 01 | `01_size_refinement_conf_rescue` | `00` 뒤 size-only refinement |
| 02 | `02_low_conf_cluster_token_hgnn_refinement` | 더 낮은 후보 pool과 cluster-token HGNN |
| 03 | `03_sparse_edge_sage` | sparse local EdgeSAGE, PPR 없음 |
| 04 | `04_ppr_gatv2_sage` | sparse PPR-GATv2-SAGE |
| 05 | `05_ppr_gatv2_sage_no_gois` | full-image-only no-GOIS dependency 진단 |
| 06 | `06_class_relation_ppr_gatv2_sage` | ordered class-pair relation을 더한 homogeneous GNN |
| 07 | `07_hetero_detection_class_view_gatv2` | non-controlled true heterogeneous graph + class correction |
| 08 | `08_controlled_hetero_ppr_gatv2_sage` | `06` protocol을 고정한 sparse heterogeneous backbone |
| 09 | `09_class_relation_ppr_zero_pe` | masked zero-PE control; 등록 수만 동일 |
| 10 | `10_class_relation_ppr_signnet_pe` | SignNet project adaptation 활성화 |
| 11 | `11_class_relation_ppr_rpearl_pe` | R-PEARL project adaptation 활성화 |

### 4.3 저장된 40-epoch 계열 평가 수치

| ID | Epoch | Predictions | AP | AP50 | AP75 | AP-small | AP-medium | AP-large | 검증 상태 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 00 | 40 | 106,179 | 0.154986 | 0.292166 | 0.143345 | 0.118195 | 0.211473 | 0.216065 | `historical_nonconformant` |
| 01 | 12 | 104,972 | 0.155612 | 0.293011 | 0.144791 | 0.119332 | 0.213236 | 0.198478 | `historical_nonconformant` |
| 02 | 38 | 95,670 | 0.152964 | 0.287680 | 0.142720 | 0.116245 | 0.212173 | 0.212586 | `historical_nonconformant` |
| 03 | 35 | 91,540 | 0.175821 | 0.324647 | 0.167234 | 0.126308 | 0.235104 | 0.245616 | recorded; provenance incomplete |
| 04 | 39 | 92,229 | 0.177066 | 0.326150 | 0.169085 | 0.128963 | 0.236865 | 0.246917 | recorded; provenance incomplete |
| 05 | 34 | 47,293 | 0.116618 | 0.205475 | 0.114024 | 0.047286 | 0.190595 | 0.366672 | 별도 no-GOIS 진단 |
| 06 | 40 | 88,260 | 0.177150 | 0.327180 | 0.168583 | 0.130209 | 0.236234 | 0.240657 | recorded; provenance incomplete |
| 07 | 40 | 56,403 | 0.152326 | 0.276638 | 0.146536 | 0.102563 | 0.210700 | 0.230325 | exploratory, non-controlled |
| 08 | 34 | 93,045 | 0.175064 | 0.324153 | 0.166801 | 0.127815 | 0.233630 | 0.233929 | controlled design; audit incomplete |
| 09 | — | — | — | — | — | — | — | — | `configured_not_run` |
| 10 | — | — | — | — | — | — | — | — | `configured_not_run` |
| 11 | — | — | — | — | — | — | — | — | `configured_not_run` |

### 4.4 결과 해석

- `00–02`는 결과가 존재하지만 현재 sparse pre-NMS 계약의 직접 선행 결과로 사용할
  수 없다. `00` run config에는 `final_nms_pre_gnn_rescue`가 명시되어 있고, shared
  legacy graph builder는 DBSCAN distance와 IoU/containment/distance relation을
  dense `N x N` matrix로 계산한다.
- `03 → 04`의 AP 증가는 `+0.001245`다. 다만 EdgeSAGE에서 PPR-GATv2-SAGE로
  backbone과 PPR을 함께 바꾸므로 PPR 단독 효과로 해석할 수 없는 composite 비교다.
- `04 → 06`은 `+0.000084 AP`다. 이는 practical parity이며 의미 있는 전체 성능
  향상으로 표현하면 안 된다. AP50과 AP-small은 약간 올랐지만 AP75와 AP-large는
  내려갔다.
- `06 → 08`은 `-0.002086 AP`다. 현재 controlled heterogeneous backbone은
  homogeneous `06`보다 낮으며 heterogeneous 구조의 우월성을 뒷받침하지 않는다.
- `07`은 backbone, target, auxiliary class loss, class correction과 inference gate를
  동시에 바꾼 exploratory variant다. `06`과의 단일-factor 비교가 아니다.
- `07` epoch 40의 같은 checkpoint에서 고른 post-hoc 최고 policy는 AP 0.155987이지만,
  evaluation split에서 선택된 결과이므로 primary AP 0.152326을 대체할 수 없다.
- `05`는 full-image-only 후보를 사용하므로 coarse+fine low-confidence pool의
  `04/06`과 동일 조건 AP 비교로 사용하면 안 된다.

### 4.5 `06`과 `08`의 architecture/capacity

| 항목 | `06` homogeneous | `08` controlled heterogeneous |
|---|---:|---:|
| Hidden width | 96 | 56 |
| Graph layers | 3 | 3 |
| Attention heads | 4 | 4 |
| Trainable parameters | 237,087 | 224,021 |
| Parameter ratio (`08/06`) | — | 0.9449 |
| Detection features/edges | 기준 | `06`와 동일하도록 설계 |
| Target/weight/loss/scoring/NMS | 기준 | `06`와 동일하도록 설계 |

폭 56은 메모리 부족을 피하려고 임의 축소한 값이 아니라, 이 비교에서 parameter
capacity를 근접시키기 위해 specification에 명시된 값이다. 다만 저장된
`model_capacity.json`에는 총 parameter, width, layer, head만 있고 module별 inventory,
active operator call, message-path 및 gradient audit가 없다. README의 real-image
bitwise audit 설명도 현재 JSON에서 실제 tensor equality를 재계산한 증거가 아니라
문자열 주장만 남아 있어 canonical audit로는 부족하다.

### 4.6 120-epoch extension

| Variant | Epoch | AP | AP50 | AP75 | AP-small | 상태 |
|---|---:|---:|---:|---:|---:|---|
| `06_class_relation_ppr_gatv2_sage` | 120 | 0.181971 | 0.336158 | 0.173229 | 0.132985 | historical only |
| `08_controlled_hetero_ppr_gatv2_sage` | 120 | 0.181004 | 0.334153 | 0.171657 | 0.131658 | historical only |

숫자상 `08 - 06 = -0.000967 AP`지만 유효한 controlled extension으로 사용할 수 없다.
runner가 seed를 variant loop 전에 한 번만 설정하고 `06`, `08`을 순서대로 실행하여
두 모델 초기화·shuffle의 실효 RNG 상태가 같지 않다. 두 run config에 seed 42가
기록됐다는 사실만으로 같은-seed 비교가 성립하지 않는다. 또한 전용 shell script가
`ALLOW_MODEL_TRAINING=1`을 내부에서 직접 설정하므로 현재 승인 규칙에도 맞지 않는다.

### 4.7 실험 2 검증 문제

1. `00–02`는 현재 sparse pre-NMS 실험 계열이 아니라 역사적 결과로 격리해야 한다.
2. `03–06` run config는 code version 4/5/5/6이고 현재 runner는 version 7이다.
   source hash, cache checksum 및 exact command가 없어 현재 코드와 동일하다고 증명할
   수 없다.
3. `07`은 비통제 탐색이고 `08`만 intended controlled comparison이지만, real-image
   tensor identity와 full capacity audit가 canonical artifact로 충분히 남지 않았다.
4. publication exporter는 실험 2에 대해 검증 항목과 무관하게
   `publication_ready=true`를 기록하므로 이 flag를 논문 사용 가능 판정으로 믿으면
   안 된다.
5. 120-epoch `06/08`은 same-seed control이 아니므로 main 결과에 합치면 안 된다.

## 5. 실험 3 — Sequential Experiment-1 Prior to Experiment-2 Rescue

### 5.1 연구 질문

실험 3은 GOIS 전에 의미 있는 후보 graph prior를 만들고, 그 정보를 낮은 confidence
후보를 복구하는 다음 graph stage에 전달하는 순차 결합이다. 두 실험의 최종 박스를
합치는 실험이 아니다.

### 5.2 단계별 데이터 흐름

1. confidence 0.05의 coarse + fine detector cache를 하나의 raw candidate pool로
   합치고 안정적인 node order를 부여한다.
2. detector score가 0.25 이상인 core subset만 frozen 실험 1
   `02_gnn_no_cluster/epoch_116.pt`에 입력한다.
3. 실험 1 단계에서는 후보를 제거하거나 NMS·COCO prediction을 만들지 않는다.
4. core 출력으로 `[core mask, object, small, large, ROI]` 다섯 prior channel을 만든다.
5. 같은 node order의 full low-confidence pool에 이 prior를 붙인다. node dim은
   `40 + 5 = 45`, edge dim은 27이다.
6. trainable stage 2는 실험 2 `06`의 class-relation PPR-GATv2-SAGE 계약을 사용한다.
7. 모든 graph 처리가 끝난 뒤 score cutoff 0.25와 class-wise NMS 0.40을 한 번만
   수행한다.

현재 default stage 2는 width 96, depth 3, attention heads 4이며 실험 2의 residual
구조를 상속한다. 그러나 실제 run의 `model_capacity.json`이 아직 없고 expected
parameter total과 module별 inventory가 specification에 고정되어 있지 않아 production
capacity 검증은 미완료다.

### 5.3 Controlled variants

| ID | Variant | 다섯 prior channel | 결과 |
|---:|---|---|---|
| 00 | `00_zero_prior_control` | 전부 0 | not run |
| 01 | `01_exp1_graph_prior_exp2_ppr` | frozen 실험 1 확률 | not run |

두 variant는 raw candidates, node order, sparse/PPR edges, target, sample weight, loss,
architecture, parameter count, seed, checkpoint selection, scoring 및 final NMS가 같아야
한다. 오직 다섯 prior 값만 다르다. 따라서 실행 후 해석 대상은 `AP(01)-AP(00)`이다.
기존 `06`과 `01`을 직접 비교해 prior 효과를 주장해서는 안 된다.

### 5.4 현재 상태와 선행 blocker

- active `experiment_3/runs/`가 없고 `00/01` 모두 `configured_not_run`이다.
- detector cache가 없으면 inference로 대체하지 않고 실패하도록 구현되어 있다.
- 과거 post-NMS late-fusion 구현과 수치는
  `experiment_3/archive/invalid_post_nms_late_fusion_20260813/`에 무효 보관되어 있으며
  active 결과가 아니다.
- frozen source인 실험 1 `02/epoch_116.pt`가 현재 `unverified` 상태이므로, 실험 1의
  schema/provenance 충돌을 먼저 해소하지 않으면 실험 3도 publication-valid 결과를
  만들 수 없다.
- checkpoint에는 model state만 저장되고 optimizer/scheduler/RNG state가 저장되지
  않아 현재 resume는 동일 trajectory를 재현하지 못한다.

실험 3에는 아직 AP/AR가 없으며 0, 추정치 또는 placeholder score를 채우면 안 된다.

## 6. Graph Positional Encoding 확장 — 실험 1/2의 `09–11`

PE 실험의 질문은 candidate pool, edge, base backbone, target, loss, seed, scoring 및
NMS를 고정했을 때 PE 분기 활성화가 미치는 영향이다. 추가 활성 학습 용량도 개입에
포함되므로 topology 정보만의 순수 효과를 분리하는 실험은 아니다.

| ID | 역할 | 활성 branch | 왜 필요한가 |
|---:|---|---|---|
| 09 | zero PE control | 없음 | 등록 parameter·실행 연산 수를 유지한 PE 분기 OFF 기준 |
| 10 | SignNet adaptation | SignNet | Laplacian eigenvector의 sign-invariant 구조 정보 검증 |
| 11 | R-PEARL adaptation | R-PEARL | deterministic random probe와 graph filtering 정보 검증 |

`09`는 성능 기법이 아니라 통제군이다. `10/11`과 동일한 두 encoder를 instantiate하고
동일한 operator를 실행하지만 activation mask를 `[0,0]`으로 두어 PE 기여와 gradient를
막는다. 따라서 총 등록 수와 forward 호출 수는 같지만, 활성 학습 용량은 통제되지
않는다. 기존 문서의 ‘추가 parameter 효과까지 분리한다’는 설명은 철회한다.

### 6.1 PE 입력과 모델

- raw PE는 8개 normalized-Laplacian eigenvector, 8개 validity mask, 16개
  deterministic random probe의 총 32 channel이다.
- SignNet branch는 `phi(u) + phi(-u)`로 eigenvector sign invariance를 만든다.
- R-PEARL branch는 `[W, SW, ..., S^8W]` polynomial filter bank 뒤 MLP와 GIN을
  적용하고, **각 sample에 rho를 적용한 다음 합산**한다. 2026-09-15에 코드의 반대
  순서를 기존 설계에 맞게 정정했다.
- 두 branch는 각각 width 96, depth 3이며 bias-free fusion과 zero-initialized ReZero
  gate를 사용한다.
- base width/depth/head와 기존 residual/skip connection은 줄이지 않는다.
- zero initialization으로 세 variant 모두 해당 base model과 같은 함수에서 시작한다.

### 6.2 Capacity 통제

| 실험 | Base parameters | PE parameters | `09/10/11` total | PE sparse operator calls |
|---|---:|---:|---:|---:|
| 실험 1 | 449,187 | 179,623 | 628,810 | 17 |
| 실험 2 | 237,087 | 179,623 | 416,710 | 17 |

위 수치는 등록 parameter와 실행 operator 수다. gate가 열린 뒤 loss에 연결 가능한
PE parameter는 `09=0`, `10=84,676`, `11=94,948`이며 실제 batch의 nonzero gradient
수와는 별도 개념이다. 활성 encoder는 각각 0 / 75,459 / 85,731개 parameter를 가진다.
즉 active-capacity-matched 비교가 아니며 96×3 backbone의 capacity 충분성도 미검증이다.
모두 `configured_not_run`이며 checkpoint, AP/AR 또는 score plot이 없다. 기존 `02/06`과
PE treatment를 바로 비교하지 말고 반드시 `10-09`, `11-09`로 해석해야 한다.

SignNet과 R-PEARL 구현은 공식 repository exact run이 아니라 논문의 핵심 원리를
재구현한 project-specific adaptation이다. repeated-eigenvalue eigenspace에 대한
basis invariance는 현재 SignNet branch의 범위가 아니다. R-PEARL도 고정 16개 probe로
재생성한 결과의 정확한 topology-only 순열 등변성을 보장하지 않는다. PE를 노드와
함께 옮겼을 때의 조건부 일관성과 PE 재생성 시의 한계를 별도로 검사한다.

### 6.3 2026-09-15 수정 및 CPU 검증

- R-PEARL rho-before-pool 순서와 실제 sparse operator 17회를 hook으로 검증했다.
- raw PE 재생성, independent eigenvector sign, empty/isolated graph, ReZero 초기
  gate gradient, inactive fusion gradient 및 PE model+optimizer resume를 검사했다.
- 두 runner를 별도 CPU process에서 실제 import했다. 실험 2의 필수
  `graph_radius_ratio` parser 누락을 수정하고 모든 PE mode의 base tensor 보존과
  graph 뒤 최종 NMS 1회를 확인했다.
- 평가 계획은 반올림하지 않은 validation loss 상위 3개만 checksum과 함께 고정한다.
  마지막 epoch 자동 추가를 제거했으며 AP 동률은 이른 epoch를 선택한다.
- Python 직접 호출에도 학습·평가의 환경변수와 flag를 각각 요구한다. PE의 cache
  경로는 detector를 생성하지 않고 checksum이 확인된 파일만 읽는다.
- 기존 실험 코드·결과 및 detector cache는 수정하지 않았다. 성능 결과는 없으며
  실제 데이터 전체 검증, 자원 검증과 실행 승인은 별도로 필요하다.

## 7. 모델 규모와 skip connection 현황

| 모델 | Width × depth | Heads | Trainable parameters | Skip/residual | 판정 |
|---|---|---:|---:|---|---|
| 실험 1 `02` base | 96 × 3 | — | 449,187 | 있음 | total count 확인, 세부 capacity audit 미완료 |
| 실험 2 `06` | 96 × 3 | 4 | 237,087 | 있음 | total count 확인, 세부 audit 미완료 |
| 실험 2 `08` | 56 × 3 | 4 | 224,021 | 있음 | capacity-matched 설계, 세부 audit 미완료 |
| 실험 3 stage 2 | 96 × 3 | 4 | run artifact 없음 | 있음 | configured, capacity 미검증 |
| 실험 1 PE | base + PE 96 × 3 | — | 628,810 | base + PE 모두 있음 | configured, not run |
| 실험 2 PE | base + PE 96 × 3 | 4 | 416,710 | base + PE 모두 있음 | configured, not run |

현재 코드에는 실험 1 `EdgeGatedLayer`, 실험 2의 EdgeSAGE/PPR-GATv2-SAGE update,
controlled hetero update 및 PE sparse GIN에 residual/skip connection이 있다.

다만 `96 × 3`이라는 숫자만으로 toy model이라고 단정할 수도, 충분한 production
capacity라고 단정할 수도 없다. 이 모델은 detector backbone이 아니라 candidate graph
postprocessor지만, width/depth/head scale ablation과 saturation curve가 없기 때문에
현재 공통 판정은 `capacity_unverified`다.

## 8. 검증 및 재현성 현황

### 8.1 확인된 구현 특성

- 실험 1의 현재 graph builder는 37-node/24-edge schema, typed relation, 양방향
  non-self edge와 sparse spatial construction을 구현한다.
- 실험 1 `02/03` run config의 의도된 차이는 cluster mode `none/dbscan`이다.
- 실험 2 `03–06`은 sparse local/PPR edge를 사용하고 `06`은 homogeneous graph다.
- 실험 2 `08`은 실제 `HeteroData` 형태이며 observed active class와 detection당 하나의
  membership edge를 사용한다.
- 실험 3 active 코드는 shared pool, five-prior transfer, single final NMS 및 cache
  fail-closed 흐름을 구현한다.
- PE runner는 training과 COCO evaluation에 이중 guard를 둔다.

다음은 CPU/synthetic 검증 현황이다. PE 검사는 2026-09-15에 확장해 재실행했고,
나머지는 2026-09-11 문서의 기존 검증 기록을 유지한다.

- 실험 1 relation graph schema test
- 실험 2 sparse forward/backward and controlled-tensor smoke test
- 실험 3 synthetic sequential contract test 5개
- Graph PE 확장 model contract 및 별도 runner contract 11개 검사 (CPU, 2026-09-15)
- 실험 1 `02/epoch_116`, 실험 2 `06/epoch_40`, `08/epoch_34` strict CPU checkpoint load

CPU/synthetic test 통과는 실제 학습 성능, 통계적 유의성 또는 publication validity를
뜻하지 않는다.

### 8.2 공통 provenance·실행 문제

현재 기존 runner에는 다음 문제가 남아 있다.

1. 실험 1/2의 일부 cache loader는 cache miss 시 detector inference를 자동 실행할 수
   있어 현재 cache-only/fail-closed 규칙과 충돌한다.
2. 기존 checkpoint는 `model.state_dict()`만 저장한다. optimizer, scheduler, CPU/CUDA
   RNG 및 dataloader state가 없어 중단 후 resume trajectory가 동일하지 않다.
3. checkpoint save가 atomic하지 않은 경로가 있으며, E2의 일부 config write도 atomic
   provenance 계약을 만족하지 않는다.
4. command, source/cache/checkpoint checksum, 시작·종료 시각, PID/file lock,
   atomic `progress.json`, GPU/RSS/swap/resource log 및 non-finite 상태 기록이 완전하지
   않다.
5. 실험 1 runner는 config 변경 시 canonical metric/prediction/checkpoint를 지울 수 있는
   경로가 있어 새 조건은 새 run directory에 저장한다는 규칙과 충돌한다.
6. publication exporter는 E1에서는 code version만으로 current 여부를 판정하고,
   E2에서는 검증 없이 publication-ready를 true로 둔다.
7. GraphSAGE, GATv2, PPR/HeteroConv의 연구 provenance와 인용은 현재
   [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에 충분히 정리되어 있지 않다.

## 9. 현재 주장 가능한 범위

### 말할 수 있는 것

- 이 저장소는 GOIS 논문의 slicing 아이디어를 비교 기준으로 삼은 local
  matched-condition reimplementation과 relation-aware graph postprocessing 실험을
  구성했다.
- 실험 1/2에서 graph 기반 candidate reranking과 low-confidence rescue에 대한 여러
  수치가 기록되어 있다.
- 실험 2 `04`와 `06`의 single-seed AP는 practical parity다.
- 실험 2 controlled `08`은 현재 `06`보다 낮아 heterogeneous backbone 우월성을
  보여 주지 않는다.
- 실험 3과 PE 실험은 설계와 코드만 구성되었고 성능 결과가 없다.

### 말하면 안 되는 것

- `01_gois_reimplementation`을 공식 GOIS 코드 실행, 공식 결과 또는 논문 Table 6의
  exact reproduction이라고 표현하는 것
- 실험 1 `05`의 AP 상승을 post-rerank cutoff 단독 효과라고 표현하는 것
- 실험 2 `00–02`를 현재 sparse pre-NMS 계약의 유효 결과로 합치는 것
- `03 → 04` 차이를 PPR 단독 효과라고 표현하는 것
- `04 → 06`의 `+0.000084 AP`를 의미 있는 전체 성능 향상이라고 과장하는 것
- `07`의 post-hoc policy를 primary result로 교체하는 것
- `08` 결과를 heterogeneous graph 우월성 또는 causal importance의 근거로 쓰는 것
- 120-epoch `06/08`을 동일-seed controlled comparison으로 쓰는 것
- 아직 실행하지 않은 실험 3 또는 PE의 성능을 예측값·0·placeholder로 채우는 것
- 단일 seed와 단일 split 결과로 statistical significance, generalization, robustness,
  novelty, SOTA를 주장하는 것

## 10. 논문용 결과로 승격하기 전 우선순위

1. 실험 1 `02–05`의 schema 상태를 specification, manifest, source snapshot,
   checkpoint, config, metric과 하나로 정합화한다.
2. 실험 1 `05`를 `02`와 완전히 같은 score fusion으로 고치고 cutoff만 바꾸는 진짜
   single-factor control로 다시 정의한다.
3. 실험 2 `00–02`와 120-epoch invalid-seed 결과를 active publication table에서
   격리한다.
4. 실험 2 `06/08`의 real candidate identity, tensor equality, module별 capacity,
   operator call 및 gradient path를 계산된 artifact로 남긴다.
5. 기존 runner를 cache fail-closed, atomic full-state checkpoint, deterministic resume,
   source/cache checksum과 resource logging 계약에 맞춘다.
6. 유효한 핵심 비교를 여러 seed로 반복하고 mean, standard deviation 및 사전 고정된
   통계 검정을 보고한다.
7. 위 선행 조건이 해결된 뒤에만 사용자 승인하에 실험 3과 PE `09–11`을 실행한다.

## 11. 기준 문서와 산출물 위치

- 공통 설계·GOIS 경계: [`GOIS_FINAL_REPRO_SPEC.md`](GOIS_FINAL_REPRO_SPEC.md)
- 전체 번호 안내: [`README.md`](README.md)
- 실험 1 설명: [`experiment_1/README.md`](experiment_1/README.md)
- 실험 2 설명: [`experiment_2/README.md`](experiment_2/README.md)
- 실험 2 sparse spec: [`experiment_2/SPARSE_FOLLOWUP_SPEC.md`](experiment_2/SPARSE_FOLLOWUP_SPEC.md)
- 실험 3 설명: [`experiment_3/README.md`](experiment_3/README.md)
- PE 설계: [`GRAPH_POSITIONAL_ENCODING_SPEC.md`](GRAPH_POSITIONAL_ENCODING_SPEC.md)
- canonical ID manifest: [`manifests/experiment_variant_ids.json`](manifests/experiment_variant_ids.json)
- 상태·경로 manifest: [`manifests/variant_order.json`](manifests/variant_order.json)
- 제3자 provenance·license: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- 통합 원본 수치: [`reports/experiment_results.csv`](reports/experiment_results.csv)
- 사용자 확인용 sheet: [`reports/current_experiment_scores.xlsx`](reports/current_experiment_scores.xlsx)

## 12. 상태 표기 정의

| 상태 | 의미 |
|---|---|
| `recorded baseline` | 실행·평가 산출물은 있으나 별도 provenance 한계를 함께 확인해야 함 |
| `recorded; provenance incomplete` | 수치와 checkpoint가 있으나 현재 source·cache·command identity가 완전하지 않음 |
| `unverified` | 문서, manifest, 코드 또는 산출물이 충돌하여 fail-closed 상태 |
| `historical_nonconformant` | 과거에는 실행됐지만 현재 승인된 graph/candidate 계약과 다름 |
| `invalid_control` | 비교에서 의도한 단일 변인 외의 조건도 함께 바뀜 |
| `exploratory, non-controlled` | 진단 가치는 있으나 인과적 단일-factor 비교가 아님 |
| `configured_not_run` | 설계·코드만 있고 실제 training/evaluation 수치는 없음 |
