# AGENTS.md

이 저장소의 작업은 **DESIGN-FIRST / IMPLEMENTATION-CONFORMANCE / CAPACITY-AND-CLAIM-GOVERNANCE** 원칙을 따른다. 이 문서는 루트 전체에 적용된다.

## 1. 작업 범위와 기준 문서

- 2026-09-23 정정: 공개 번호는 `final/manifests/current_experiments.json`을 따른다.
  기존 ID manifest는 artifact/internal 경로 호환 기록이며 공개 번호로 출력하지 않는다.
  `final/COMPARISON_REPAIR_SPEC.md`가 비교 복구 및 실험3 보류 상태의 최신 기준이다.
  아래 실험3의 exp1-prior→exp2 계약은 사용자가 연결 방향을 거부한 기존 구현 설명이며
  새 production 승인으로 취급하지 않는다. 단계 경계 재확인 전 실행하지 않는다.

- 2026-09-23 후속 실행 요청: 실험1 matched seed42 비교군은
  `final/MATCHED_PE_EXECUTION_20260923.md` 및 해당 실행 manifest의 범위로 재승인되었다.
  구형 study 차단을 해제하지 않으며 새 `matched_pe_v2`에만 적용한다.
  실험2 입력/구현 준비와 detector cache 재생성 승인 확인은 별도로 필요하다.

- 현재 연구 대상은 `final/experiment_1`, `final/experiment_2`, `final/experiment_3`이다.
- `CP/`, `Full/`, `GOIS/`는 데이터·기존 구현·비교 기준을 제공하는 참조 영역이다. 사용자가 명시하지 않는 한 수정, 이동, 삭제하지 않는다.
- 작업 전 최소한 다음 문서 중 해당 범위를 읽는다.
  - 공통 조건과 주장 경계: `final/GOIS_FINAL_REPRO_SPEC.md`, `final/README.md`
  - 실험 1: `final/experiment_1/README.md`
  - 실험 2: `final/experiment_2/README.md`, `final/experiment_2/SPARSE_FOLLOWUP_SPEC.md`
  - 실험 3: `final/experiment_3/README.md`
  - 번호와 경로: `final/manifests/experiment_variant_ids.json`, `final/manifests/variant_order.json`
  - 외부 코드와 명칭: `final/THIRD_PARTY_NOTICES.md`
- 설계 의미는 실험별 최신 specification, ID와 경로는 manifest, 완료 여부와 수치는 실제 checkpoint·run config·provenance·평가 파일의 일치로 판단한다.
- 문서, manifest, 코드, 산출물이 충돌하면 임의로 하나를 선택하지 않는다. 해당 상태를 `unverified`로 취급하고 GPU 작업과 성능 주장을 중단한 뒤 충돌을 보고한다.

## 2. 절대 규칙

- 문서·코드 수정 요청은 학습, detector inference, COCO evaluation 또는 GPU 실행 권한을 뜻하지 않는다.
- 새 실험 실행은 사용자의 명시적 승인과 기존 실행 가드를 모두 만족할 때만 허용한다. 특히 학습은 `ALLOW_MODEL_TRAINING=1` 또는 해당 runner의 `--allow_training` 없이 시작하지 않는다.
- 설계 승인을 받기 전에는 새 architecture, loss, target, candidate policy, score rule 또는 evaluation policy를 구현하거나 실행하지 않는다.
- 출처 없는 architecture/operator를 창작하지 않는다. 각 구성 요소는 다음 중 하나로 provenance를 가져야 한다.
  1. 공식 baseline 또는 논문에 충실한 구현
  2. 저장소의 승인된 기존 구현
  3. 사용자가 승인한 project-specific adaptation
- 메모리·시간 문제를 hidden width, layer 수, attention head, edge type, candidate pool 또는 graph stage 축소로 해결하지 않는다.
- 승인된 설계와 실제 constructed model·tensor pipeline이 다르면 fail-closed 한다.
- screen, synthetic test, 단일 checkpoint, post-hoc evaluation 결과를 architecture failure/success 또는 연구 결론으로 확대하지 않는다.
- 설계 의미가 바뀌면 관련 specification과 manifest를 먼저 갱신하고 재승인을 받는다.

## 3. Design-first 계약

새 variant 또는 기존 variant의 의미 변경 전 다음을 문서화한다.

- 연구 질문과 비교 대상
- baseline provenance와 변경되는 단 하나의 주요 변인
- detector weight, dataset split, VisDrone 10-class mapping, seed
- raw candidate 생성부터 최종 prediction까지의 전체 순서
- confidence threshold, slice size/overlap, IoU, `max_det`, NMS 위치와 횟수
- node·edge·token·prior의 정의, shape, dtype, mask, node order
- spatial/overlap/containment/view/class/cluster/PPR relation의 방향과 생성 규칙
- model width/depth/head/FFN, module별 parameter 수와 operator call count
- frozen/trainable scope, initialization, loss, optimizer, scheduler, checkpoint selection
- capacity 대응, ablation, 진단, 예상 실패 모드와 미해결 위험
- 결과 경로, resume 규칙, provenance 및 claim boundary

`TBD`, `later`, `optional` 또는 암묵적 fallback이 남아 있으면 production experiment로 승인하지 않는다.

## 4. 실험별 불변 계약

### Experiment 1 — fixed candidate-pool ablation

- `01`과 `02-05`는 동일한 raw coarse + global fine detector pool에서 시작한다.
- cross-view NMS 전에 graph를 구성하며 최종 class-wise NMS는 graph 처리 뒤 한 번만 수행한다.
- graph 입력 전에 겹치는 crop의 후보를 합치거나 제거하지 않는다.
- node type은 detector predicted class와 `full/coarse/fine/ROI` view를 사용한다. GT class를 node type이나 입력 feature로 누설하지 않는다.
- 현재 계약은 node dim `27 + num_classes`, edge dim `24`, cache schema `gois_graph_schema_v4`, pipeline code version `48`이다.
- relation은 `self`, `spatial`, `overlap`, `containment`, `same_cluster`, `same_view`, `cross_view`, `cross_class_context`를 구분한다. 선택된 non-self edge는 양방향이며 거리는 image size로 정규화한다.
- sparse spatial construction을 유지하고 `N x N` dense relation matrix로 바꾸지 않는다.
- `02`와 `03`의 유효한 차이는 DBSCAN cluster context 유무다. 후보 pool, heads, fusion, final NMS를 함께 바꾸지 않는다.
- `05_gnn_prune_same_pool`은 `02`의 scorer와 동일 pool을 사용하며 post-rerank cutoff 효과만 다룬다.

### Experiment 2 — low-confidence candidate rescue

- low-confidence coarse + fine 후보를 sparse relation/PPR graph에서 재점수화한 뒤 cutoff와 final class-wise NMS를 적용한다.
- `06_class_relation_ppr_gatv2_sage`는 detection node 하나의 tensor를 쓰는 **class-relation-aware homogeneous graph**다. formal heterogeneous graph로 부르지 않는다.
- `07`과 `08`만 detection/class/view node store를 갖는 heterogeneous graph다.
- `08`은 `06`의 detection features, detection-to-detection edges, targets, weights, loss, scoring, cutoff, NMS, seed와 selection policy를 고정한 controlled comparison이다.
- `08`의 class support는 detection당 한 membership edge이며 observed class만 materialize한다. `07`의 all-class support를 몰래 재도입하지 않는다.
- capacity 비교는 실제 trainable parameter inventory로 검증한다. 현재 기준은 `06 = 237,087`, `08 = 224,021`이며 canonical audit은 해당 run의 `model_capacity.json`이다.
- detector cache에 full class probability가 없으면 있다고 가정하거나 합성하지 않는다. hard `category_id`와 score라는 입력 한계를 명시한다.

### Experiment 3 — sequential Experiment-1 prior to Experiment-2 rescue

- 실험 1·2의 최종 prediction을 합치는 late fusion이 아니다.
- 하나의 raw confidence `0.05` coarse + fine pool을 유지한다.
- detector score `>= 0.25` core에 frozen Experiment-1 `02_gnn_no_cluster/epoch_116.pt` scorer를 적용하되, 이 단계에서는 후보 제거·NMS·COCO prediction 생성을 하지 않는다.
- `[core mask, object, small, large, ROI]` 다섯 prior channel을 동일 node order의 full low-confidence pool에 붙여 Experiment-2 `06` 계약의 sparse PPR stage로 전달한다.
- `00_zero_prior_control`과 `01_exp1_graph_prior_exp2_ppr`은 다섯 prior channel의 값만 달라야 한다. raw candidates, edges, targets, weights, loss, architecture, parameter count, seed, scoring, selection, final NMS는 같아야 한다.
- graph 처리가 모두 끝난 뒤 threshold와 final class-wise NMS를 한 번만 수행한다.
- detector cache가 없으면 inference로 대체하지 말고 실패한다.
- 현재 실험 3은 `configured_not_run`이다. 수치 칸은 비워 두며 0, 추정치 또는 placeholder 그래프를 만들지 않는다.
- `archive/invalid_post_nms_late_fusion_20260813`은 무효 역사 자료이며 active 구현·결과·주장에 사용하지 않는다.

## 5. Implementation conformance

GPU 실행 전 코드와 승인 계약을 다음 항목으로 대조한다.

- experiment-local ID, internal legacy ID, output path
- detector weight와 dataset/class mapping
- candidate pool identity, candidate count/order, crop/view geometry
- pre-NMS/post-NMS 단계와 final NMS 횟수
- node/edge/prior shape와 feature 의미
- relation type, direction, PPR/local-neighbor construction
- hidden/depth/head/FFN과 module별 parameter count
- trainable/frozen inventory와 strict checkpoint reload
- graph operator call count와 실제 사용 여부
- target, sample weight, loss, score fusion, cutoff, selection policy
- missing/empty graph, padding/mask, non-finite 값 처리
- control variant 사이의 비의도적 차이

다음 shortcut은 금지한다.

- fixed-pool 비교에서 variant별 candidate pool 변경
- graph 전 cross-view NMS 또는 실험 3의 중간 NMS
- GT class/box를 inference feature로 사용
- sparse graph를 dense all-pairs graph로 대체
- cache miss 시 무단 detector inference
- checkpoint schema mismatch의 non-strict load
- 실험 3을 post-NMS prediction merge로 구현
- control과 treatment의 architecture·capacity·seed·evaluation 동시 변경

기존 CPU/synthetic contract test는 실행할 수 있다. 변경된 경로에는 같은 수준의 deterministic test와 shape/gradient/parameter audit를 추가한다. 테스트가 GPU 학습·detector inference·전체 평가를 호출하면 별도 실행 승인이 필요하다.

## 6. Capacity와 resource governance

- screen에서도 production architecture를 유지한다. 줄일 수 있는 것은 sample 수, update 수, evaluation subset뿐이며 screen 결과는 정식 결과와 분리한다.
- 허용되는 resource 대응은 batch size 조정, gradient accumulation, chunking, sparse construction, 기존 cache 재사용, checkpoint resume, 검증된 mixed precision이다.
- OOM을 이유로 width/depth/head/relation/prior channel을 줄이려면 별도 architecture ablation으로 사전 승인받는다.
- capacity가 비교의 confound가 될 수 있으면 parameter 수뿐 아니라 active operator, message path, call count와 gradient flow를 함께 기록한다.
- 추가 block은 기존 함수를 보존하는 identity 또는 zero-output initialization을 우선한다.

## 7. 로컬 실행 규칙

- systemd service/unit, cgroup controller, watcher/heartbeat daemon, automatic restart, remote scheduler 전제를 추가하지 않는다.
- foreground process, PID + file lock, atomic `progress.json`, portable checkpoint, 명시적 manual resume를 사용한다.
- 실행 시 command/config, seed, code/schema version, 입력 cache/checkpoint checksum, 시작·종료 시각을 기록한다.
- resource log에는 `torch.cuda.max_memory_allocated/reserved`, process RSS/swap, `nvidia-smi` snapshot, OOM/CUDA/non-finite 상태를 남긴다. Xid는 관찰 가능할 때만 informational로 기록한다.
- canonical 결과를 덮어쓰지 않는다. 새 조건은 새 run directory에 저장하고 완료 후에만 publication summary로 승격한다.

## 8. 평가와 결과 선택

- 평가는 VisDrone 10-class의 pycocotools COCO AP/AR 계약을 따른다.
- primary metric, 평가 checkpoint 수, validation-loss 기반 후보 선택과 tie-break를 실행 전에 고정한다.
- evaluation split에서 고른 inference policy는 post-hoc ablation으로 표시하며 primary 결과를 대체하지 않는다.
- AP, AP50, AP75, AP-small/medium/large와 AR을 원본 precision으로 보존한다.
- not-run, failed, partial, screen, completed를 명시적으로 구분한다. 빈 결과를 0으로 채우지 않는다.
- `reports/publication_results_best_ap.csv`는 canonical run artifact와 provenance에서 생성하며 수치를 손으로 꾸며 넣지 않는다.

## 9. Claim governance

- `01_gois_reimplementation`은 local matched-condition reimplementation이다. 공식 GOIS 코드 실행, 공식 결과, 논문 Table 6 exact reproduction으로 표현하지 않는다.
- 서로 다른 candidate pool이나 inference policy의 AP를 동일 조건 ablation처럼 직접 비교하지 않는다.
- Experiment 2 `06`의 `04` 대비 변화는 practical parity 범위이며 의미 있는 전체 성능 향상으로 과장하지 않는다.
- `07`의 post-hoc best policy는 primary result가 아니다.
- controlled `08`은 `06`보다 낮은 AP이므로 heterogeneous structure 우월성의 증거가 아니다. learned relation mixture는 사용 여부의 진단이지 causal importance가 아니다.
- Experiment 3은 실행·평가 전이므로 성능 결론이 없다.
- 단일 seed 또는 단일 split 결과로 통계적 유의성, 일반화, SOTA를 주장하지 않는다.
- `novel`, `first`, `significant`, `robust`, `state-of-the-art`는 직접 뒷받침하는 비교·반복·통계가 있을 때만 사용한다.

## 10. 산출물과 정리

- checkpoint, detector/prediction cache, run config, provenance, 원본 metric과 archive를 `dummy`로 간주하지 않는다.
- cache나 archive 삭제 전 코드·provenance의 참조 여부와 재생성 비용을 확인하고 사용자의 삭제 범위를 확정한다.
- `__pycache__`, `.ruff_cache`, 빈 temp directory와 실제 placeholder만 정리 대상으로 본다. 가능하면 휴지통으로 이동한다.
- 실험 전 placeholder metric, 가짜 CSV row, 가짜 score plot을 만들지 않는다.
- 과거 무효 결과는 active report에서 제외하되, 사용자가 명시하지 않으면 역사 provenance 자체를 삭제하지 않는다.

## 11. 완료 조건

- 변경 파일, 승인 설계와의 일치 여부, 수행한 CPU/synthetic test, 실행하지 않은 GPU 작업, 남은 위험을 명확히 보고한다.
- 결과를 추가했다면 run config·checkpoint·metric·provenance가 서로 일치하는지 확인한다.
- 설계 불일치, cache/checkpoint 부재, 비교 조건 오염 또는 claim 근거 부족이 하나라도 있으면 결과를 승격하지 말고 fail-closed 한다.
