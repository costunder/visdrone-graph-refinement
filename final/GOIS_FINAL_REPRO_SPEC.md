# Multi-scale Relational GNN Experiment Spec

2026-09-23 현재 상태: 공개 번호는 `manifests/current_experiments.json`, 비교 복구와
실험3 실행 보류는 `COMPARISON_REPAIR_SPEC.md`가 우선한다. 기존 번호는 역사 ID다.
실험1 SignNet seed42는 완료했지만 matched baseline 비교는 미완료다. 새 실행은 보류한다.

작성일: 2026-05-17

정정일: 2026-09-05
> Numbering note: 이 문서 본문의 `00-08`은 과거 통합 실행기의 internal ID다. 현재 사용자-facing artifact와 실행 번호는 각 실험에서 `00`부터 다시 시작하며, canonical order는 `manifests/variant_order.json`을 따른다.

## 2026-09-15 상태 감사 및 승인된 production PE revision

실험 1 local 02-04의 과거 결과 상태는 `unverified_historical`이다. 02의 실제
config에는 schema v4/code48이 있으므로 모든 산출물을 schema v1이라고 단정한
아래 역사 설명은 현재 판정이 아니다. 반대로 실행 당시 runner SHA를 복원하지 못했으므로
README의 완료/검증 주장도 철회한다. 05는 score fusion까지 달라진
`invalid_control_comparison`이며 cutoff-only 비교로 인용하지 않는다.
원본 checkpoint/config/metric은 보존하고 새 production 비교에 재사용하지 않는다.
근거는 `GRAPH_PE_INPUT_READINESS_AUDIT_20260915.md` 및 기존 전체 감사 문서다.

새 study `graph_pe_production_v1`은 `GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`를 따른다.
256×6 backbone·128×8 PE 구현 및 실험 1 GPU 검증/학습/평가가 명시적으로 승인되었다.
실제 conformance 통과 전 학습 금지, 구형 96×3 masked-branch runner 차단은 유지한다.
실험 1의 actual-view .25 cache만 사용하며 실험 2 .05 cache 재생성은 별도 미승인이다.


## Graph PE extension (2026-09-05)

Experiment 1 local `09-11`과 Experiment 2 local `09-11`은 이 문서의
기존 variant를 덮어쓰지 않는 별도 positional-encoding ablation이다. 정확한
candidate identity, PE tensor, SignNet/R-PEARL adaptation, capacity, operator,
cache-only runner 및 claim boundary는 `GRAPH_POSITIONAL_ENCODING_SPEC.md`가
우선한다. 여섯 variant는 `configured_not_run`이며 학습·detector inference·COCO
평가 결과가 없다. local ID와 상태의 canonical source는
`manifests/variant_order.json`이다.

## Graph Schema v2 Correction (2026-07-23)

02/03 및 이를 재사용하는 Experiment 1 local 04/05는 이제 cross-view NMS 이전의 동일한 raw `coarse + global fine` detector pool로 graph를 만든다. 각 detection은 predicted class type과 `full/coarse/fine/ROI` view type, crop frame geometry를 가지며, typed relation edge는 양방향이다. 최종 class-wise NMS는 graph rerank/pruning 뒤에 수행한다. DBSCAN 및 graph radius는 이미지 크기로 정규화한다.

이 변경은 node/edge tensor 의미와 차원을 바꿨다. 과거 02-05 metric의 실제 검증 상태는
위 2026-09-15 감사 정정을 따른다. 00/01 결과 파일과 과거 graph artifact는 삭제하거나 덮어쓰지 않았다.


## Canonical Goal

이 문서는 GOIS 논문을 비교 기준으로 인용하되, 공식 GOIS 코드의 원본 실행이나 논문 Table 6의 exact reproduction을 주장하지 않는다. 01은 이 저장소 안에서 구현한 coarse/fine slicing + class-wise NMS 기준선이다. 공식 코드 라이선스와 공개 범위는 `THIRD_PARTY_NOTICES.md`를 따른다.

`/mnt/AI/팀플/final`은 GOIS 논문의 YOLO11 실험을 참고해 공통 detector, dataset, class-space, evaluation protocol로 비교하는 로컬 실험 폴더다.

00은 로컬 FI-Det control이고, 01은 로컬 GOIS-Det 재구현 baseline이다. 02/03은 같은 detector, dataset, class-space, inference/eval 조건에서 붙이는 추가 ablation이다. 02/03은 논문 원본 Table 6 행이라고 쓰지 않는다.

02는 01과 같은 raw `coarse + global fine` detector candidate pool을 쓰되 cross-view NMS 전에 graph를 만들고, cluster relation을 쓰지 않으며, supervised ROI head 없이 `obj/small/large` 3-head만 직접 학습한다. `roi_like=max(small, 0.7*obj+0.3*small)`, `stage_score=max(obj, small, large)`를 적용해 score fusion한 뒤 final NMS를 수행한다.

03은 02에 DBSCAN cluster context만 붙인 버전이다. candidate pool, 3-head 출력, ROI-like 유도식, score fusion은 02와 같고, 차이는 graph에 cluster context가 들어가는지뿐이다.

04는 detector 후보를 low-confidence까지 넓힌 뒤 GNN 점수로 재정렬하고, 재점수화된 confidence 기준 `0.25` 컷을 적용한 다음 GOIS final NMS에 넣는 별도 추가 실험이다. 04는 논문 Table 6 재현 행이라고 쓰지 않는다.

05는 02의 개선 ablation이다. 01/02와 같은 후보 pool을 유지하고, 02의 GNN posterior를 다시 feature로 넣은 learned stage2 size-only refinement만 본다. object/keep score는 02 stage1 값을 보존하고, stage2는 small/large size posterior만 재정렬한다.

06은 04의 개선 ablation이다. 04와 같은 low-confidence 후보 pool을 유지하고, 04 GNN의 object/keep score를 보존한 상태에서 learned stage2가 small/large size posterior만 재정렬한다. 이후 보존된 04 score와 보정된 size posterior로 `0.25` 컷과 GOIS final NMS를 적용한다.

07은 04의 HGNN 확장이다. 04보다 더 낮은 confidence의 더 많은 후보 pool을 만들고, DBSCAN cluster를 prediction 생성 규칙으로 쓰지 않고 cluster token/hyperedge context로만 넣어 HGNN이 빠르게 재랭킹한다. 이후 `0.25` 컷과 GOIS final NMS를 적용한다.

08(internal; Experiment 1 local 05)은 02의 같은 raw pre-NMS 후보 pool과 같은 GNN checkpoint를 그대로 쓰고, score fusion 이후 낮게 재랭킹된 후보를 `0.25` 기준으로 pruned한 뒤 final NMS를 수행한다. 후보 생성, detector, GNN 학습은 바꾸지 않고 post-rerank pruning만 본다.

02/03만 cluster 유무 비교로 유지하고, 같은 의미의 duplicate variant는 두지 않는다.

중요: 01_gois_reimplementation가 large-preserve 없이 `coarse + fine + NMS`인 GOIS-Det을 참고한 로컬 조건이므로, 02/03도 large-preserve를 끈 상태에서만 유효한 ablation으로 취급한다.

추가 중요: 01/02/03은 같은 raw `coarse + global fine` detector pool에서 출발해야 한다. 01은 이 pool에 바로 final NMS를 적용하고, 02/03은 NMS 전에 graph를 구성한 뒤 rerank 결과에 final NMS를 적용한다. GNN variant가 coarse-only 후보나 GNN-selected ROI fine만 쓰면 small-object 후보를 덜 받는 불공정 비교라서 유효한 ablation으로 쓰지 않는다.

## Publication Claim Boundary

```text
GOIS paper citation: required
GOIS official code execution claim: no
GOIS Table 6 exact reproduction claim: no
01 implementation label: local reimplementation
legacy artifact alias: 01_gois_original_repo
public redistribution of upstream GOIS code: prohibited unless written permission is obtained
```

## Common Conditions For 00-03

아래 조건은 00/01/02/03 모두에 동일하게 적용한다.

```text
ablation detector:
  final/runs/table6_yolo11_10class_extra_ablation/detector/weights/best.pt

dataset:
  /mnt/AI/팀플/Full/data/visdrone_det_yolo_10class/dataset.yaml

train images:
  /mnt/AI/팀플/Full/data/visdrone_det_yolo_10class/images/train

train labels:
  /mnt/AI/팀플/Full/data/visdrone_det_yolo_10class/labels/train

eval images:
  /mnt/AI/팀플/Full/data/visdrone_det_yolo_10class/images/val

eval labels:
  /mnt/AI/팀플/Full/data/visdrone_det_yolo_10class/labels/val

eval ground truth:
  /mnt/AI/팀플/Full/data/visdrone_det_yolo_10class/annotations/val_coco_gt.json

class space:
  VisDrone 10-class
  YOLO class 0..9 -> COCO category id 1..10

evaluation:
  pycocotools COCO AP/AR
```

고정 inference 설정:

```text
full_conf: 0.25
coarse_conf: 0.25
fine_conf: 0.25
model_iou: 0.7
final_nms_iou: 0.4
coarse_slice_size: 640
coarse_overlap: 0.2
fine_slice_size: 256
fine_overlap: 0.2
max_det: 300
large_preserve: disabled for 00-03 same-condition comparison
raw_candidate_pool_for_01_03: identical coarse + global fine detections before cross-view NMS
candidate_pipeline_for_02: typed graph before NMS, no cluster relation, legacy 3-head fusion, then final NMS
candidate_pipeline_for_03: same as 02, with normalized DBSCAN relation only
candidate_pipeline_for_experiment_1_local_05: same raw pool and 02 checkpoint, post-rerank pruning, then final NMS
```

04_gnn_conf_rescue만 다른 점:

```text
role: additional low-conf GNN rerank-before-GOIS experiment
base structure: 03-style size-aware GNN structure
candidate pool before GNN: low-conf coarse + low-conf global fine
rescue_coarse_conf: 0.05
rescue_fine_conf: 0.05
GNN score alpha: 1.0, so GNN score is the confidence used for the next cutoff
post-GNN cutoff: stage1_keep_conf 0.25
GOIS postprocess after cutoff: final classwise NMS 0.4, max_det 300
large_preserve: disabled
interpretation: low-conf rerank -> 0.25 cut -> GOIS experiment, not same-condition Table 6 ablation
```

05/06/07 개선 요소:

```text
learned stage2 size-only refinement for 05/06:
  Stage1 GNN sees detector confidence, bbox geometry, class, IoU/containment/distance, and source relation.
  Stage1 object/small/large/ROI posterior is frozen and appended back to each node feature.
  Stage2 GNN learns only small/large size posterior from the enriched graph.
  Stage2 does not learn keep/background/drop and does not replace the object/keep score.

cluster-token HGNN for 07:
  A lower-confidence candidate pool than 04 is grouped into same-class DBSCAN clusters.
  Each cluster becomes a hypergraph context token connected to its member detection nodes.
  Cluster tokens summarize member confidence, geometry, stage1 posterior, source composition, and local density.
  Cluster tokens do not directly create predictions; final keep/drop remains node-level HGNN reranking.

same-pool rerank pruning for 08 (Experiment 1 local 05):
  It reuses the shared raw pre-NMS detector pool and the 02 no-cluster GNN checkpoint.
  It changes only the final ranking score rule: fused_score = (1 - alpha) * detector_score + alpha * gnn_score.
  Candidates with fused_score < 0.25 are removed.
  No new candidate generation, low-conf rescue, cluster token, or extra GNN training is added.

no-ROI-head GNN for 02:
  02 uses the same raw detector candidates as 01 and a no-cluster typed graph before final NMS.
  The GNN output heads are obj/small/large only.
  ROI is not directly supervised as an independent head.
  roi_like_score = max(small, 0.7 * obj + 0.3 * small).
  stage_score = max(obj, small, large).
  Final score fusion follows the legacy 3-head fixed-pool rule, without cluster or extra candidate injection.
  This isolates whether the noisy ROI head is useful under the same 02 candidate pool.

cluster-context GNN for 03:
  03 is the same as 02 except size_graph_cluster_mode=dbscan.
  It uses the same raw pre-NMS candidates, obj/small/large heads, ROI-like formula, score fusion, and final NMS.
  The ablation isolates DBSCAN cluster context.

```

## Variant Order

```text
00_full_inference:
  role: local FI-Det control under the matched protocol
  path: final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/00_full_inference
  status: completed

01_gois_reimplementation:
  role: local GOIS-Det reimplementation under the matched protocol
  condition: coarse 640 overlap 0.2 + fine 256 overlap 0.2 + final classwise NMS 0.4
  canonical output path: final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/01_gois_reimplementation
  completed legacy artifact path: final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/01_gois_original_repo
  status: completed

02_gnn_no_cluster:
  role: additional same-condition ablation
  condition: shared raw coarse+fine pool, typed GNN before NMS, no cluster relation, obj/small/large heads only, ROI-like derived from obj/small, score fusion, final NMS
  path: final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/02_gnn_no_cluster
  status: graph schema v1 artifact exists; schema v2 retraining required

03_gnn_dbscan_cluster:
  role: additional same-condition ablation
  condition: same as 02 plus DBSCAN cluster context only
  path: final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/03_gnn_dbscan_cluster
  status: graph schema v1 artifact exists; schema v2 retraining required

04_gnn_conf_rescue:
  role: additional low-conf GNN rerank-before-GOIS experiment
  condition: low-conf coarse+fine candidates, GNN score rerank, 0.25 cutoff, then GOIS final NMS
  path: final/experiment_2/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/00_gnn_conf_rescue
  status: completed

05_size_refinement_gnn:
  role: 02 improvement, learned stage2 size-only refinement
  condition: same candidate pool as 02, 02 posterior reinserted as stage2 features, object/keep score preserved
  path: final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/04_size_refinement_gnn
  status: graph schema v1 artifact exists; schema v2 retraining required

06_size_refinement_conf_rescue:
  role: 04 improvement, learned stage2 size-only refinement on low-conf pool
  condition: same low-conf candidate pool as 04, 04 object/keep score preserved, only small/large posterior refined, 0.25 cutoff, final GOIS NMS
  path: final/experiment_2/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/01_size_refinement_conf_rescue
  status: completed

07_low_conf_cluster_token_hgnn_refinement:
  role: 04 HGNN ablation, cluster-token hypergraph rerank on same low-conf pool
  condition: same low-conf candidate pool as 04, DBSCAN cluster tokens as context, HGNN rerank, 0.25 cutoff, final GOIS NMS
  path: final/experiment_2/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/02_low_conf_cluster_token_hgnn_refinement
  status: completed

08_gnn_prune_same_pool:
  role: 02 same-pool post-rerank pruning only
  condition: same raw pre-NMS pool as 02, reuse 02 GNN checkpoint, fused-score 0.25 cutoff, final NMS
  path: final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/05_gnn_prune_same_pool
  status: graph schema v1 artifact exists; schema v2 retraining required

```

## Historical Completed 00/01 Metrics

```text
00_full_inference:
  AP: 0.104598
  AP50: 0.168043
  AP75: 0.109274
  AP_small: 0.036672
  AP_medium: 0.170474
  AP_large: 0.33357
  AR_1: 0.048467
  AR_10: 0.106497
  AR_100: 0.132806
  AR_small: 0.051549
  AR_medium: 0.222747
  AR_large: 0.416511

01_gois_reimplementation:
  AP: 0.117158
  AP50: 0.212481
  AP75: 0.11211
  AP_small: 0.086793
  AP_medium: 0.162835
  AP_large: 0.177456
  AR_1: 0.058534
  AR_10: 0.183906
  AR_100: 0.244775
  AR_small: 0.179996
  AR_medium: 0.3181
  AR_large: 0.287256
```

## Valid Files

현재 유효한 연구용 항목이다. `GOIS_00_03_colab_ablation*.ipynb`는 프로젝트 제출 스냅샷이며 현재 연구 실행기 및 provenance 판단 대상에서 제외한다:

```text
/mnt/AI/팀플/final/GOIS_FINAL_REPRO_SPEC.md
/mnt/AI/팀플/final/GRAPH_POSITIONAL_ENCODING_SPEC.md
/mnt/AI/팀플/final/manifests/variant_order.json
final/runs/table6_yolo11_10class_extra_ablation/detector/weights/best.pt
final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/00_full_inference
final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/01_gois_original_repo  # legacy completed artifact, normalized on read
final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/02_gnn_no_cluster
final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/03_gnn_dbscan_cluster
final/experiment_2/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/00_gnn_conf_rescue
final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/04_size_refinement_gnn
final/experiment_2/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/01_size_refinement_conf_rescue
final/experiment_2/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/02_low_conf_cluster_token_hgnn_refinement
final/experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/05_gnn_prune_same_pool
/mnt/AI/팀플/final/experiment_1/scripts/run_gois_two_stage_gnn_ablation.py
/mnt/AI/팀플/final/experiment_1/scripts/run_gois_paper_ablation_curves.py
/mnt/AI/팀플/final/experiment_1/run_graph_pe_ablation.py
/mnt/AI/팀플/final/experiment_2/run_graph_pe_ablation.py
/mnt/AI/팀플/final/graph_positional_encoding.py
/mnt/AI/팀플/final/scripts/run_table6_yolo11_10class_extra_ablation.sh
/mnt/AI/팀플/final/scripts/my_package
```
