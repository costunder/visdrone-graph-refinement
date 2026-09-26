# Graph Positional-Encoding Ablation Specification

이 문서는 Experiment 1의 fixed-pool `02_gnn_no_cluster`와 Experiment 2의
low-confidence PPR `06_class_relation_ppr_gatv2_sage`에 graph positional
encoding(PE)을 추가하는 사전 고정 계약이다. 아래 여섯 variant는 모두
`configured_not_run`이며, 학습·detector inference·COCO 평가는 수행되지 않았다.

## 현재 실행 상태 — 2026-09-15 production 재설계 대기

**아래 96×3 masked-branch 구성은 CPU 회귀 검사용으로 보존하며 production 실행은
차단했다.** 두 runner의 `main`, `run_variant`, `train_epoch`는 학습·평가 승인
플래그가 모두 있어도 새 설계 승인·구현 전에는 실행할 수 없다. runner version은
3이며 encoder code version 2와 기존 checkpoint schema는 바꾸지 않았다.

새 256×6 backbone/128×8 PE 안은 `GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`에
`design_pending_approval`로 분리했다. 그 문서는 아직 구현된 architecture 또는
GPU 실행 계약이 아니다. 아래 ID·shape·parameter 수는 구형 구성의 기록이며,
미구현 production 모델의 검증 수치로 사용하지 않는다.

## 1. 연구 질문과 local ID

연구 질문은 **후보·edge·backbone·loss·score rule을 고정했을 때 PE 분기 활성화가
검출 후보 재점수화에 미치는 영향**이다. 활성 분기의 추가 학습 용량도 이 개입에
포함된다. 구조 정보만의 효과를 추가 학습 용량과 분리하는 실험은 아니다.

### 2026-09-15 implementation-conformance 정정

- 기존에 명시한 `rho → sample sum`과 `validation-loss top 3 only` 계약으로 코드를
  복원한다. 후보·edge·loss·score·seed·width/depth/head 및 local ID는 바꾸지 않는다.
- PE raw schema는 1을 유지하고 PE code/runner version은 2로 올린다. checkpoint는
  `graph_pe_training_v2`로 구분하며 이전 코드 또는 다른 PE mode를 재개하지 않는다.
- 등록 parameter 수와 활성 loss path를 별도 기록한다. 총 parameter 수 일치가
  활성 학습 용량 일치 또는 충분한 backbone capacity의 증거라는 설명은 철회한다.
- 순열 검증은 아래의 조건부 보장과 한계를 각각 검사한다. 이를 통과하기 위해
  random probe 생성 규칙이나 node order를 변경하지 않는다.
- 이 정정은 코드·CPU 검증 범위다. 학습·detector inference·COCO 실행 승인은 아니다.

| Experiment | Local ID | Variant | 활성 PE | 기준 variant |
|---|---:|---|---|---|
| 1 | 09 | `09_gnn_no_cluster_zero_pe` | 없음 | `02_gnn_no_cluster` |
| 1 | 10 | `10_gnn_no_cluster_signnet_pe` | SignNet adaptation | `02_gnn_no_cluster` |
| 1 | 11 | `11_gnn_no_cluster_rpearl_pe` | R-PEARL adaptation | `02_gnn_no_cluster` |
| 2 | 09 | `09_class_relation_ppr_zero_pe` | 없음 | `06_class_relation_ppr_gatv2_sage` |
| 2 | 10 | `10_class_relation_ppr_signnet_pe` | SignNet adaptation | `06_class_relation_ppr_gatv2_sage` |
| 2 | 11 | `11_class_relation_ppr_rpearl_pe` | R-PEARL adaptation | `06_class_relation_ppr_gatv2_sage` |

Experiment 1의 `06`은 과거 폐기된 adaptive-rescan archive와 연결되어 있으므로
재사용하지 않고 active PE ID를 `09-11`로 배정한다. 두 실험에서 같은 local ID는
같은 PE intervention을 뜻하지만 결과와 checkpoint 경로는 실험별로 완전히 분리한다.

## 2. 공통 데이터와 평가 계약

- detector: `final/runs/table6_yolo11_10class_extra_ablation/detector/weights/best.pt`
- dataset: VisDrone train/validation, 10-class mapping
- seed: `42`; 각 variant의 model 초기화와 graph sampling 직전에 다시 설정한다.
- detector는 학습하지 않는다. coarse/fine cache가 정확한 signature로 이미 존재해야
  하며 cache miss는 detector inference로 대체하지 않고 실패한다.
- dtype: node/edge/target/weight/PE는 `float32`, `edge_index`는 `int64`다.
- node order는 기존 tensor builder가 정한 detection candidate order를 그대로 쓴다.
  PE 생성이나 batching 과정에서 node를 정렬·병합·삭제하지 않는다.
- 학습 target(`label_iou=0.50`, `roi_label_iou=0.10`, `roi_center_margin=0.75`), sample weight, loss, optimizer, scheduler 부재, checkpoint selection,
  confidence rule 및 NMS는 각각 기준 `02`/`06`과 같다.
- primary checkpoint selection은 validation loss가 가장 낮은 세 checkpoint를 미리
  선택한 뒤 그 세 개만 COCO 평가하고, 그중 AP가 가장 높은 checkpoint를 보고한다.
  동률은 이른 epoch를 선택한다.
- loss와 metric을 반올림하지 않고 보존한다. 평가 전에 checkpoint epoch/checksum과
  loss를 `checkpoint_selection.json`에 고정한다. 마지막 epoch 자동 추가는 금지한다.
  완료 시 세 평가만을 대상으로 `primary_result.json`을 만들며 AP 동률도 이른 epoch다.
- metrics는 VisDrone 10-class pycocotools AP, AP50, AP75,
  AP-small/medium/large 및 AR을 원본 precision으로 보존한다.
- 단일 seed 결과는 유의성·일반화·SOTA 근거로 사용하지 않는다.

### Experiment 1 고정 조건

- coarse/fine confidence `0.25`, coarse tile `640`/overlap `0.20`, fine tile
  `256`/overlap `0.20`, detector IoU `0.7`, `max_det=300`이다.
- `01-03`과 동일한 raw coarse + global-fine pre-cross-view-NMS pool을 쓴다.
- graph는 NMS 전에 만들며 graph 입력 전에 겹친 crop 후보를 병합하거나 제거하지 않는다.
- node dim은 `37 = 27 + 10`, edge dim은 `24`, graph schema는
  `gois_graph_schema_v4`, base pipeline code version은 `48`이다.
- cluster mode는 `none`, pixel radius는 `256`, normalized radius는 `0.12`, cross-class normalized radius는 `0.08`, same-class kNN은 6, cross-class kNN은 2로 고정하고 relation은 `self`, `spatial`, `overlap`,
  `containment`, `same_cluster`, `same_view`, `cross_view`,
  `cross_class_context`의 기존 typed sparse edge를 그대로 쓴다.
- 3-head object/small/large weighted BCE와 `02`의 sample weight를 그대로 쓴다.
- 각 후보의 최종 pre-NMS score는 `02`와 같이 detector score와 3-head graph
  score의 평균 중 detector score보다 큰 값을 사용한다. 중간 cutoff는 없고 모든
  graph 처리가 끝난 뒤 class-wise NMS를 IoU `0.4`로 한 번만 수행한다.
- training budget은 120 epochs, epoch당 최대 1,024 non-empty graphs,
  validation-loss subset 최대 512 graphs, batch size 32, AdamW
  (`lr=1e-3`, `weight_decay=1e-4`)이다.

### Experiment 2 고정 조건

- coarse/fine confidence `0.05`, coarse tile `640`/overlap `0.20`, fine tile
  `256`/overlap `0.20`, detector IoU `0.7`, `max_det=300`이다.
- `06`과 동일한 raw low-confidence coarse + fine pool, 40-node-feature /
  27-edge-feature tensor, sparse local relation 및 truncated PPR edge를 쓴다.
- local spatial radius는 `256` pixels, normalized edge-feature radius는 `0.12`, spatial cell size는 radius와 같은 자동값(`0.0` 설정)이다. local neighbor는 same-class 12, cross-class 4, PPR top-k 8,
  alpha `0.15`, propagation 8 steps, frontier 64로 고정한다.
- ordered class-pair key/value/bias, neighbor-class context, 4 attention heads,
  dropout `0.10`, 3-head target/weight/BCE를 포함해 `06`을 그대로 보존한다.
- graph score만 사용하도록 fusion alpha는 `1.0`이고 cutoff `0.25` 뒤
  class-wise NMS를 IoU `0.4`로 한 번만 수행한다.
- training budget은 40 epochs이고 나머지 batch·optimizer·selection 조건은
  Experiment 1과 같다.

## 3. PE 입력 정의

PE는 target이나 GT를 사용하지 않고 detection graph topology와 고정 seed의 random
probe에서 만든다.
기존 directed edge에서 self-loop를 제거하고 역방향을 추가한 뒤 중복을 합친 binary
undirected support를 공통 PE support로 쓴다. 기존 message edge와 edge attribute는
수정하지 않는다.

각 node에는 다음 32개 raw PE channel을 같은 순서로 붙인다.

\[
X_{PE}=[U_{1:8}\;\Vert\;M_{1:8}\;\Vert\;W_{1:16}],
\qquad X_{PE}\in\mathbb R^{N\times32}.
\]

- `U`: symmetric normalized Laplacian
  \(L=I-D^{-1/2}AD^{-1/2}\)의 가장 작은 non-zero eigenvalue에 대응하는
  eigenvector 최대 8개다.
- `M`: 유효 eigenvector channel은 1, padding은 0인 mask다. 각 column의 mask는
  모든 node에 동일하다.
- `W`: image ID와 seed에서 결정적으로 생성한 16개의 i.i.d. standard-normal
  random probe다. 동일 raw graph는 세 variant에서 bitwise-identical한 `W`를 쓴다.
- 이 결정성은 고정 candidate order에 대한 것이다. 이미 생성한 `W`를 노드와 함께
  순열하면 encoder는 순열에 일관되지만, 순서를 바꾼 graph에 같은 seed로 `W`를
  새로 생성하면 동일 candidate에 같은 probe가 붙는다는 보장은 없다. 고정 16-sample
  결과를 정확한 topology-only 순열 등변성으로 주장하지 않는다.
- isolated node만 있는 graph는 `U=M=0`으로 명시한다. random probe는 유지한다.
- connected component별 non-trivial eigenpair 후보를 구한 뒤 eigenvalue,
  component의 최소 원래 node index, component 내부 eigen-index 순서로 정렬한다.
- component node 수가 64 이하이면 그 component의 작은 bounded Laplacian에 exact
  symmetric EVD를 사용한다. 65 이상이면 SciPy sparse `eigsh`만 사용한다.
  sparse solver 실패나 non-finite 값은 fallback 없이 fail-closed한다. 이 bounded
  EVD는 PE 계산용이며 graph relation/message passing을 dense all-pairs로 바꾸지 않는다.

Experiment 1의 model 입력 tensor는 `N x 69`, Experiment 2는 `N x 72`지만,
base backbone에는 각각 앞의 37/40 channel만 전달한다. 따라서 base input layer와
head skip concatenation의 shape 및 의미는 기존 checkpoint architecture와 같다.

## 4. PE encoder와 삽입 위치

세 variant는 모두 아래 두 encoder 전체를 instantiate하고 optimizer에 등록한다.
오직 고정 activation mask만 다르다.

| Variant kind | `[SignNet, R-PEARL]` mask |
|---|---|
| zero control | `[0, 0]` |
| SignNet | `[1, 0]` |
| R-PEARL | `[0, 1]` |

### SignNet project-specific adaptation

- 8개 Laplacian eigenvector를 각각 scalar node signal로 본다.
- shared 96-wide, 3-layer sparse GIN \(\phi\)를 `u`와 `-u`에 각각 적용하고
  \(\phi(u)+\phi(-u)\)로 exact sign invariance를 만든다.
- mask 뒤 eigenvector axis를 sum-pool하고 2-layer 96-wide DeepSets rho MLP를
  적용한다.
- SignNet 논문의 sign-invariant 원리를 따르는 local adaptation이며 공식 repository의
  SetTransformer rho를 그대로 vendoring한 구현은 아니다. repeated-eigenvalue
  eigenspace에 대한 basis invariance는 제공하지 않으므로 이 한계를 결과에 명시한다.

### R-PEARL project-specific adaptation

- normalized adjacency \(S=D^{-1/2}AD^{-1/2}\)와 probe `W`로
  `[W, SW, ..., S^8W]`의 9-step polynomial filter bank를 만든다.
- filter axis의 2-layer 96-wide MLP, 3-layer shared sparse GIN sample encoder,
  2-layer 96-wide rho MLP를 순서대로 적용하고 16개 random sample을 sum-pool한다.
  즉 `H: N×16×96 → rho(H): N×16×96 → sum_samples: N×96`이다.
- PEARL의 random-feature graph filtering과 sample aggregation을 따르는 local
  R-PEARL adaptation이다. random probes는 재현성을 위해 run 내에서 고정한다.

각 sparse GIN layer는 self term, non-self sparse neighbor sum, 2-layer MLP,
residual connection 및 LayerNorm을 사용한다. 두 96-d output을 이어 붙인 후 bias 없는
`Linear(192,96)`으로 base hidden width에 투영한다. 투영값은 affine parameter가 없는
LayerNorm을 거쳐 scalar ReZero gate `tanh(g)`와 곱해진다. `g=0`으로 초기화하므로
모든 variant는 해당 base model과 정확히 같은 함수에서 시작한다.

- Experiment 1: node-type projection 뒤, 첫 `EdgeGatedLayer` 전에 PE residual을 더한다.
- Experiment 2: class-context residual 뒤, 첫 PPR-GATv2-SAGE layer 전에 더한다.
- base residual/skip connection, width 96, depth 3, Experiment 2 attention head 4는
  줄이거나 교체하지 않는다.

## 5. Capacity와 operator inventory

모든 PE variant의 encoder parameter inventory는 동일하다.

| Module | Registered parameters (`requires_grad=True`) |
|---|---:|
| SignNet adaptation | 75,459 |
| R-PEARL adaptation | 85,731 |
| bias-free fusion + ReZero gate | 18,433 |
| PE total | 179,623 |

Expected full trainable count는 Experiment 1 `628,810` (`449,187 + 179,623`),
Experiment 2 `416,710` (`237,087 + 179,623`)이다. runner는 model을 만든 직후
module별 실제 count를 재계산하며 하나라도 다르면 학습 전에 실패한다.

각 forward에서 mask와 무관하게 두 encoder를 모두 호출한다. PE sparse operator
inventory는 SignNet GIN 6회(positive/negative 각각 3회), R-PEARL normalized-adjacency
filter 8회와 GIN 3회로 총 17회다. base graph layer는 별도로 각 3회다. zero control의
두 encoder도 계산되지만 mask 뒤 기여와 gradient는 0이다. treatment의 inactive
branch도 동일하다. 즉 총 parameter 수와 operator call count는 같고, 의도된 차이는
활성 PE signal과 그 gradient path뿐이다.

등록 수와 별도로 gate가 열린 뒤 loss에 연결 가능한 PE parameter element 수는
`09=0`, `10=84,676` (`75,459 + 9,216 + 1`), `11=94,948`
(`85,731 + 9,216 + 1`)이다. fusion의 비활성 절반도 loss gradient가 0이다.
이는 구조적으로 가능한 경로의 inventory이며, 특정 batch에서 측정한 nonzero
gradient 수와 같다는 뜻은 아니다. gate가 0인 초기 step에는 활성 encoder도 loss
gradient가 0이고 treatment의 gate가 먼저 학습되어야 한다.

따라서 이 비교는 **등록 parameter·실행 operator 수를 맞춘 masked-branch ablation**이지
active-capacity-matched ablation이 아니다. `09` 대비 개선을 구조 정보만의 순수 효과로
주장하지 않는다. 새로운 active control이나 backbone scale 실험은 별도 설계·승인
대상이며 이번 정정에서 추가하지 않는다.

OOM이나 시간 문제는 width/depth/head/filter order/sample 수/후보/edge를 줄여 해결하지
않는다. batch size, gradient accumulation, graph precompute/cache, sparse construction,
checkpoint resume만 조정할 수 있다.

## 6. 초기화, checkpoint, provenance

- base와 두 encoder는 같은 seed에서 PyTorch 기본 Linear/LayerNorm 초기화를 쓴다.
- ReZero scalar만 정확히 0으로 초기화한다.
- detector/cache는 frozen input이고 base GNN과 두 PE encoder는 optimizer에 등록한다.
  mask가 0인 branch는 계산되지만 loss gradient가 차단된다.
- 각 training epoch 시작 시 `seed + epoch * 1,000,003`으로 Torch CPU/CUDA RNG를 다시 설정해 resume 전후의 shuffle/dropout을 고정한다.
- checkpoint schema `graph_pe_training_v2`는 epoch, strict model state, optimizer state,
  PE code/config/mode 계약을 하나의 atomic file로 저장한다. 이전 PE 코드나 다른 PE
  mode, 기존 `02`/`06` checkpoint는 거부하며 non-strict load를 사용하지 않는다.
- run directory에는 command/config, seed, source cache 및 detector checksum,
  PE schema/code checksum, model capacity, 시작·종료 시각과 resource log를 남긴다.
- canonical 결과를 덮어쓰지 않고 local ID별 새 directory에 쓴다.

External design provenance:

- SignNet/BasisNet: Lim et al., *Sign and Basis Invariant Networks for Spectral
  Graph Representation Learning*, ICLR 2023,
  <https://openreview.net/forum?id=Q-UHqMorzil>, MIT reference code
  <https://github.com/cptq/SignNet-BasisNet>.
- PEARL: Kanatsoulis et al., *Learning Efficient Positional Encodings with
  Graph Neural Networks*, ICLR 2025,
  <https://cs.stanford.edu/people/jure/pubs/efficient-positional-encodings-iclr25.pdf>,
  MIT reference code <https://github.com/ehejin/Pearl-PE>.

저장소 구현은 위 원리를 명시적으로 재구현한 project-specific adaptation이며 공식
checkpoint나 source file을 복사하지 않는다. `THIRD_PARTY_NOTICES.md`에 이 경계를
기록한다.

## 7. 실행·검증·주장 경계

- 전용 runner만 이 variant를 허용하며 `ALLOW_MODEL_TRAINING=1`과 Python의
  `--allow_training`이 모두 있어야 한다. COCO 평가도 `ALLOW_COCO_EVALUATION=1`과
  `--allow_evaluation`이 모두 있어야 한다. Python 직접 호출에도 네 guard를 적용한다.
- runner는 정확한 detector cache가 없으면 즉시 실패한다. detector inference를
  자동 수행하지 않는다. PE 경로는 detector를 생성하거나 legacy generate-or-load
  함수를 호출하지 않고, preflight에서 검증한 cache를 checksum 재확인 후 직접 읽는다.
- CPU synthetic contract test는 shape, finite output, deterministic input,
  SignNet sign invariance, PE를 함께 옮기는 조건부 permutation consistency,
  simple-spectrum graph의 SignNet PE 재생성, R-PEARL 재생성의 순서 의존 한계,
  disjoint two-graph batching,
  모든 mode의 exact ReZero base identity, parameter equality, active/inactive
  gradient path, 실제 operator hook count, rho/sample-pool 순서 및 PE model+optimizer
  checkpoint round-trip을 검사한다. runner test는 각 실험의 실제 import 경로에서
  tensor builder, 실행 guard, cache miss/변조, top-3-only selection, 동률 처리 및
  원본 precision을 검사하며 detector/학습/평가는 호출하지 않는다.
- CPU test 통과는 학습 또는 성능 검증이 아니다. 여섯 variant의 AP/AR 칸은 실제
  canonical run 전까지 비워 두며 0이나 placeholder를 쓰지 않는다.
- primary comparison은 Experiment 1에서 `09/10/11`, Experiment 2에서 `09/10/11`
  내부 비교다. 기존 `02`/`06`과는 PE wrapper의 추가 inactive capacity와 새 random
  initialization protocol이 있으므로 zero control을 거치지 않은 직접 성능 차이를
  PE 효과로 주장하지 않는다.
