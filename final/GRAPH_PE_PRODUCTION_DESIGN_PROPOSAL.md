# 실험 1·2 Graph PE — production 재설계 승인안

작성일: 2026-09-15 · 설계 ID: `pe_production_v1`

**상태: `approved_implementation_in_progress`. 2026-09-15 명시적 승인을 받았다.**
사용자 답변 “승인한다고 그리고 제대로 구현한거맞지?”는 직전 요청한
256×6 backbone·128×8 PE 구현 및 실험 1 GPU 검증·학습·COCO 평가에 대한 승인이다.
승인은 구현 완료나 conformance 통과를 뜻하지 않는다. 아래 검증을 통과하기 전
학습하지 않는다. 실험 2 detector cache 재생성은 별도 미승인 상태다.
이 문서는 새 `graph_pe_production_v1` study의 architecture 계약이며 구형 실행기는 해제하지 않는다.
사용자의 “toy 모델로 만들지 말라”는 요구에 따라 기존 96×3 masked-branch PE
runner는 실행 차단했다. 학습·평가 플래그가 있어도 차단하며 CLI 우회 옵션은 없다.
승인 후에도 구형 runner를 해제하지 않고, 이 설계에 맞는 별도 구현과 검증이 필요하다.
설계/구현 승인과 GPU·학습·detector inference·COCO 실행 승인은 별개다.

기계 판독용 승인안: `manifests/graph_pe_production_proposal.json`.
현재 모델의 역사 계약: `GRAPH_POSITIONAL_ENCODING_SPEC.md`.

## 1. 해결할 문제와 변경 범위

기존 PE 구성은 backbone 96채널·3층, PE 96채널·3층이었다. 두 encoder를 모두
등록하고 사용하지 않는 분기를 mask로 끄므로 등록 parameter 수가 같아도 실제
학습 용량은 같지 않았다. 이 수치로 충분한 용량 또는 공정한 active-capacity
통제를 주장하지 않는다. 반대로 채널 수만으로 기존 연구 전체를 무효로 단정하지도 않는다.

이번 제안은 다음 세 질문을 분리한다.

1. 같은 후보와 후처리에서 backbone의 폭·깊이가 성능을 제한하는가?
2. 고정된 더 큰 backbone에 SignNet 또는 R-PEARL을 추가하면 도움이 되는가?
3. 같은 PE encoder가 만든 정보를 **올바른 detection node에 연결하는 것**이 중요한가?

주 실험은 256×6 backbone으로 고정한다. 96×3은 이름이 명시된 capacity 진단에만
남기고 production 기본값·OOM fallback·screen 대체 architecture로 사용하지 않는다.
256×6 자체가 충분하다는 보장은 없으며 실제 학습 곡선과 capacity 비교가 필요하다.
실험 3, 기존 00-08 결과, CP/Full/GOIS 참조 코드는 변경 범위 밖이다.

## 2. 출처와 project-specific adaptation의 경계

공식 저장소를 2026-09-15에 다음 commit으로 확인했다.

- SignNet: `07f31187823ff8d42ed2f61eabe54344aea7cf24`.
  [공식 SignNet 구현](https://github.com/cptq/SignNet-BasisNet/blob/07f31187823ff8d42ed2f61eabe54344aea7cf24/GINESignNetPyG/core/sign_net.py)은
  부호 대칭 GIN과 SetTransformer를 사용한다.
  [기본 설정](https://github.com/cptq/SignNet-BasisNet/blob/07f31187823ff8d42ed2f61eabe54344aea7cf24/GINESignNetPyG/core/config.py)의
  hidden 128/SignNet 4층 및
  [ZINC 설정](https://github.com/cptq/SignNet-BasisNet/blob/07f31187823ff8d42ed2f61eabe54344aea7cf24/GINESignNetPyG/train/config/zinc.yaml)의
  downstream 6층을 확인했다. 8층 PE는 아래 PEARL 논문의 RelBench SignNet 설정에도 있다.
- PEARL: `cc2826b65f59d11b38c013736377713435c799ab`.
  [논문 Appendix I](https://cs.stanford.edu/people/jure/pubs/efficient-positional-encodings-iclr25.pdf)는
  ZINC R-PEARL에 8층 GIN, K=12, 50-120 samples를 기술한다.
  [공개 Peptides 설정](https://github.com/ehejin/Pearl-PE/blob/cc2826b65f59d11b38c013736377713435c799ab/PEARL/configs/peptides/RPEARL-peptides.yaml)은
  9층 PE/6층 backbone/200 samples이고,
  [공개 ZINC 500k 설정](https://github.com/ehejin/Pearl-PE/blob/cc2826b65f59d11b38c013736377713435c799ab/PEARL/configs/zinc/RPEARL-500k.yaml)은
  8층 PE/120 samples/K=1이다. 논문의 K=12와 해당 YAML을 동일 설정이라고 쓰지 않는다.

따라서 여기의 256×6 backbone, 128×8 PE, K=12/M=120 조합은 **원 논문 전체 실험의
exact reproduction이 아니라 출처를 명시한 VisDrone용 adaptation 제안**이다.
256은 이 프로젝트에서 제안하는 폭 확대값이지 논문이 VisDrone에 보증한 최소 폭이 아니다.

유지하는 연산은 기존 `EdgeGatedLayer`, class-relation `PPRGATv2SAGELayer`,
residual `SparseGINLayer`다. SignNet rho는
[공식 attention/FFN 구성](https://github.com/cptq/SignNet-BasisNet/blob/07f31187823ff8d42ed2f61eabe54344aea7cf24/GINESignNetPyG/core/model_utils/transformer_module.py)을
참고한다. PE 내부 BatchNorm 대신 기존 residual LayerNorm을 유지하는 것은
graph microbatch와 sample chunk 간 통계 결합을 없애기 위한 명시적 adaptation이다.
미사용 eigenvalue encoder나 항상 꺼진 다른 PE encoder는 등록하지 않는다.

R-PEARL sample mean은 논문의 통계적 정의를 따른다. 공개
[PEARL 코드](https://github.com/ehejin/Pearl-PE/blob/cc2826b65f59d11b38c013736377713435c799ab/PEARL/src/pe.py)의
기본 sum 경로와 같다고 주장하지 않는다. 사용하지 않는 upstream module이나
dense Laplacian 경로를 복사하지 않는다. 실제 코드 재사용 시 MIT notice를 보존한다.
SignNet을 BasisNet으로 부르지 않으며 BasisNet은 이번 승인 범위에 새로 추가하지 않는다.

## 3. 데이터·후보·평가 불변 조건

- detector: `final/runs/table6_yolo11_10class_extra_ablation/detector/weights/best.pt`, frozen.
- dataset: `Full/data/visdrone_det_yolo_10class`, train 및 val, YOLO 0..9 → COCO 1..10.
- ground truth: 같은 dataset의 `annotations/val_coco_gt.json`; feature에는 사용하지 않는다.
- 기존 coarse/fine detector cache만 읽는다. signature, SHA256, image 목록, candidate
  수·순서·view geometry를 검증하고 하나라도 불일치하면 실패한다. cache miss 시 inference 금지.
- raw 후보 순서는 coarse 다음 fine이며 각 cache 내부 순서를 유지한다. 중복 crop
  후보도 node로 남긴다. 이미지당 후보 상한이나 edge 축소를 새로 넣지 않는다.
- coarse 640/overlap 0.20, fine 256/overlap 0.20, detector IoU 0.7,
  detector `max_det=300`, large-preserve 비활성이다.
- 실험 1: raw confidence 0.25, node 37/edge 24, cache schema v4/code 48,
  no-cluster 02 tensor builder. spatial radius 256px/0.12, cross-class radius 0.08,
  same/cross-class kNN 6/2, 기존 8 relation 의미와 non-self 양방향 edge를 보존한다.
  `[unknown, full, coarse, fine, ROI]`의 5-view channel과 predicted class 10개를 유지한다.
- 실험 2: raw confidence 0.05, node 40/edge 27, homogeneous 06 tensor builder.
  spatial radius 256px, normalized feature radius 0.12, cell-size 설정 0.0,
  local same/cross 12/4, PPR top-k 8/alpha 0.15/8 steps/frontier 64를 유지한다.
  class-pair key/value/bias와 neighbor-class context를 보존한다.
- 실험 1 score는 `max(detector, 0.5*detector + 0.5*max(obj,small,large))`.
  중간 cutoff 없이 graph 이후 class-wise NMS IoU 0.4, max_det 300을 한 번 적용한다.
- 실험 2 score는 기존 06처럼 `max(obj, size_score)`다. size_score는 candidate
  union area가 96² 이상이면 large, 미만이면 small이다. fusion alpha 1.0,
  cutoff 0.25 이후 같은 final NMS를 한 번 적용한다. class relabeling은 없다.
- target IoU 0.50, ROI IoU 0.10/center margin 0.75와 기존 target/weight 생성 규칙을 유지한다.

2026-09-15 감사 후 공통 spec/manifest/README의 상태를 정정했다. 실험 1 02-04는
실행 당시 runner SHA 확인 미완료로 `unverified_historical`, 05는 fusion 조건 오염으로
`invalid_control_comparison`이다. 어느 쪽도 이번 study의 검증된 baseline으로 채택하지
않는다. 새 구현은 승인된 현재 tensor builder와 actual-view cache에서 처음부터 학습한다.

## 4. 정확한 architecture 제안

| 항목 | 실험 1 | 실험 2 |
|---|---|---|
| backbone | 기존 typed EdgeGated, H=256, L=6 | 기존 class-relation PPR-GATv2-SAGE, H=256, L=6 |
| attention | 기존처럼 없음 | 4 heads, head dimension 64 |
| residual/skip | 매 층 residual/LN + head의 원본 feature concat 유지 | class-context residual + 매 층 residual/LN + head 원본 feature concat 유지 |
| backbone 내부 MLP | message/gate 각각 `(2H+24)→H→H`; 별도 Transformer FFN 없음 | update `2H→H`; 별도 Transformer FFN 없음 |
| dropout | 기존처럼 0 | 기존처럼 0.10 |

PE는 base feature와 별도 tensor로 전달한다. `base_x[N,D]`와 `edge_attr[E,F]`는
float32, `edge_index[2,E]`/node identity는 int64다. GT는 target/weight에만 존재한다.
base graph의 방향·edge attribute·PPR score는 수정하지 않는다.
PE support만 self 제거/역방향 추가/중복 제거한 binary undirected sparse graph다.
raw PE schema는 `graph_pe_production_raw_v1`이며 구형 8-vector/16-probe cache를
읽지 않는다. no-PE mode에는 PE tensor나 encoder 계산을 요구하지 않는다.

### SignNet-k32

- raw: `U[N,32]` float32와 `mask[N,32]` bool. normalized Laplacian의 가장 작은
  nonzero eigenpair를 기존 component 방식으로 선택하고 부족하면 padding한다.
  component ≤64는 bounded float64 EVD, 그 외 sparse eigsh(tol=1e-6)이며 실패 시 중단한다.
  nonzero 판정은 eigenvalue>1e-6, eigsh iteration limit은 기존처럼
  `max(1000, 20*component_nodes)`다. 기존 deterministic solver 초기 벡터와
  eigenvalue/component 최소 원래 index/eigen-index tie-break를 유지한다.
  eigenvalue는 감사에 저장하되 model 입력으로 추가하지 않는다.
- scalar input projection `1→128` + ReLU 뒤 shared residual GIN 8층.
  각 층은 `MLP(128→128→128, ReLU)`와 trainable eps(초기 0), residual + LN이다.
  같은 phi를 `u/-u`에 적용해 합한다. padding은 매 층 끝에서 0으로 만든다.
- rho: **SetTransformer 1 block**, 4 heads, Q/K/V/out bias-free 128→128,
  FFN 128→128→128(ReLU, bias=True), attention/FFN residual + affine LN,
  dropout 0. eigenvector axis attention 후 masked sum, bias-free 128→128 + LN.
  attention은 node별 32×32이지 detection node 사이의 N×N attention이 아니다.
  all-padding node의 rho output은 명시적으로 0이다.
- encoder LN eps=1e-5, attention/FFN LN eps=1e-6, rho output LN eps=1e-5.
- sign invariance는 검사한다. 반복 eigenvalue의 basis rotation이나 k32 경계의
  eigenspace 분할에 대한 invariance는 제공하지 않는다. full-spectrum SignNet으로 쓰지 않는다.

### R-PEARL-M120

- raw: `W[N,120]`, i.i.d. standard normal, float32.
  training은 seed/experiment/split/image_id/epoch를 담은 SHA256 기반 별도 CPU RNG
  substream으로 매 epoch 새로 만들고, evaluation은 epoch 대신 고정 `eval_probe_v1`을 쓴다.
  candidate identity/order와 함께 저장해 chunking/resume 시 같은 샘플을 재사용한다.
  모든 별도 RNG는 정렬된 key의 compact JSON(UTF-8, allow_nan=False)을 SHA256하고
  digest 첫 8 bytes를 little-endian 정수로 바꾼 뒤 `2^63-1`과 bitwise AND해
  `torch.Generator(device='cpu')` seed로 쓴다. namespace는 `backbone`,
  `signnet_init`, `rpearl_init`, `probe`, `alignment`로 분리한다.
- normalized adjacency S의 `[W,SW,...,S^11 W]`, 즉 **K=12/filter order=11**.
  power matrix S^k를 만들지 않고 sparse apply를 반복한다.
- filter MLP 12→128→128(ReLU) + LN, residual GIN 128채널·8층,
  sample별 rho 128→128→128(ReLU) + LN 후 **120 samples mean**.
  rho는 mean 전에 적용하며 registered-only 모듈은 없다. LN eps=1e-5, dropout=0.
- node와 W를 함께 순열한 조건부 등변성과 독립 probe를 재추출한 Monte Carlo
  변동을 분리 검사한다. 유한 M120에 exact topology-only invariance를 주장하지 않는다.

### 삽입·초기화

각 treatment에는 선택한 encoder **하나만** 있다. PE 128→base 256 bias-free
projection + affine-free LN(eps=1e-5) 후 `tanh(g)`를 곱해 base hidden에 더한다.
g는 scalar 0으로 초기화한다. 실험 1 node-type projection 뒤, 실험 2 class-context
뒤이며 둘 다 첫 graph layer 앞 한 번이다. no-PE baseline에는 PE 모듈/gate가 없다.
backbone은 공통 seed, encoder는 별도 고정 RNG substream으로 초기화해 backbone
초기 tensor가 treatment/control에서 같게 한다. pretrained GNN checkpoint는 로드하지 않는다.
Linear/LN은 PyTorch 기본 초기화를 사용하고 base class-pair/attention 초기화는
기존 구현대로 보존한다. gate와 GIN eps만 명시한 0 초기화다.
ReZero 첫 step의 encoder gradient=0은 정상이나, gate가 열린 뒤에도 계속 0이면 실패다.

## 5. 용량: 설계 계산과 실제 검증을 구분

아래 표는 설계에서 계산한 **등록 parameter 수 계약**이다. 2026-09-15 실험 1의
no-PE/SignNet/R-PEARL constructed model이 각각 6,073,347 / 6,488,332 / 6,424,076으로
정확히 일치함을 확인했다. 실험 2 표는 아직 계산값이며 constructed-model 검증값이 아니다.
맞추려고 dummy tensor를 추가하거나 module을 줄일 수 없다.

| Backbone module | 실험 1 H256/L6 | 실험 2 H256/L6 |
|---|---:|---:|
| input projection + LN | 10,240 | 11,008 |
| node-type 또는 class-context/gate/norm | 4,352 | 77,056 |
| message block 1개 | 997,120 | 387,220 |
| message block 6개 | 5,982,720 | 2,323,320 |
| 3-output head | 76,035 | 76,803 |
| backbone 합계 | **6,073,347** | **2,488,187** |

| PE module | SignNet-k32 | R-PEARL-M120 |
|---|---:|---:|
| input/filter projection | 256 | 18,432 |
| residual GIN 8개 | 266,248 | 266,248 |
| SetTransformer block | 99,072 | 없음 |
| output/rho MLP 및 norm | 16,640 | 33,280 |
| 128→256 projection + gate | 32,769 | 32,769 |
| 추가 PE 합계 | **414,985** | **350,729** |
| 실험 1 전체 예상 등록 수 | **6,488,332** | **6,424,076** |
| 실험 2 전체 예상 등록 수 | **2,903,172** | **2,838,916** |

각 backbone의 H/L 용량식은 다음과 같다(head=4, classes=10).

- 실험 1: `H² + 98H + 3 + L*(15H² + 55H)`.
- 실험 2: `2H² + 132H + 3 + L*(5H² + 231H + 404)`.

기존 backbone의 조건부 relation/class parameter까지 모두 매 batch 활성이라고
간주하지 않는다. 특히 실험 1 no-cluster의 same_cluster relation은 항상 0일 수 있다.
이를 제거해 graph 계약을 바꾸지 않고 구조적 비활성/관측되지 않은 relation/실제
nonzero gradient를 별도 기록한다. detector의 frozen parameter를 GNN 용량에 합산하지 않는다.

논리적 sparse aggregation은 backbone 6회, SignNet phi 16회 또는 R-PEARL
filter 11회+GIN 8회다. E1의 8 relation projection과 E2의 각 class-pair 경로도
별도 hook으로 센다. chunk/recomputation에 따른 실제 호출 횟수는 이 논리 횟수와
분리 기록하며, 총 호출 횟수가 같다는 이유로 active capacity가 같다고 쓰지 않는다.

## 6. 비교 설계와 번호

기존 local ID를 몰래 재해석하지 않는다. 현재 09/10/11 manifest는 보존하며,
이 승인을 받으면 새 study directory와 revision을 manifest에 함께 등록한다.
새 숫자 ID는 만들지 않는 제안이다.

| 기존 ID에 연결할 study row | mode/profile | 의미 |
|---|---|---|
| 09 | `no_pe/main` | H256/L6 backbone만; 꺼진 PE module 없음 |
| 10 | `signnet/aligned` | H256/L6 + SignNet-k32 |
| 11 | `rpearl/aligned` | H256/L6 + R-PEARL-M120 |
| 10 | `signnet/node_permuted_control` | 같은 encoder·parameter·operator로 PE-node 연결만 변경 |
| 11 | `rpearl/node_permuted_control` | 위와 동일한 통제 |
| 09 | `capacity_h96_l3`, `capacity_h96_l6`, `capacity_h256_l3` | 주 H256/L6과 함께 폭/깊이 2×2 비교 |

node-permuted control은 **학습부터 별도 run**이다. 각 이미지에서 encoder output을
만든 뒤 projection 전에 고정 permutation으로 node 연결만 바꾼다. 후보/edge/label/
weight는 순열하지 않고 loss gradient도 detach하지 않는다. permutation은 node identity와
별도 seed로 재현하며 N>1이면 fixed point 없는 permutation을 사용한다. N=1은 identity,
N=0은 empty이고 해당 graph 수를 보고한다. method의 graph-global 정보까지 제거하는
통제는 아니므로 “모든 구조 정보 vs 용량”이 아니라 **node alignment 효과**를 검증한다.
permutation은 고정 `alignment` substream의 randperm을 fixed point가 없을 때까지
재추출한다. model seed/experiment/split/image_id/node identity를 key에 포함하되
epoch는 포함하지 않는다. 같은 method의 aligned/control 초기 weight는 같게 한다.

no-PE 대비 차이는 encoder 추가를 포함한 실용적 성능 차이다. 두 PE 사이의 parameter 수를
억지로 맞추지 않으며 pure-PE 비교 또는 동일 capacity라고 주장하지 않는다.

각 row는 seeds **42/43/44**를 사용한다. 실험당 8 rows×3 seeds=24 runs,
실험 1·2 합계 **48 runs**의 완전 비교 제안이다(주 비교 18, 나머지 진단 30).
이는 실행 예약/승인이 아니다. 작은 96×3 row는 명시된 capacity 진단만을 위한 것이며
큰 모델을 대신하는 결과로 승격하지 않는다. 일부만 완료되면 나머지는 미실행으로 표시한다.

## 7. 학습·자원·평가 계약 제안

- 두 실험 모두 120 epochs. 매 epoch train 전체의 모든 non-empty graph를 한 번씩
  방문한다. 현재 cache 이름에 나타나는 train 6,471/val 548개는 실행 전 dataset/cache
  목록으로 검증하며 empty graph도 누락 목록과 함께 기록한다. 1,024 graphs screen을
  production이라고 부르지 않는다. validation loss도 전체 val에서 계산한다.
- full-data budget과 실험 2의 epoch 수는 기존 실험과 달라진다. capacity_h96_l3도
  새 budget으로 처음부터 학습하므로 과거 checkpoint와 혼합한 PE ablation은 아니다.
- optimizer AdamW lr=1e-3, weight_decay=1e-4, betas=(0.9,0.999), eps=1e-8;
  scheduler 없음, early stopping 없음, gradient clipping 없음, float32.
- effective batch는 고정 32 graphs, 기본 microbatch=1. 마지막 batch는 남은 graph
  전체를 사용한다. accumulation은 단순히 micro-loss를 32로 나누는 방식이 아니다.
  virtual batch의 weight 합으로 각 microbatch BCE numerator를 정규화한다.
  head weight는 기존 [1,2,1.5]; 실험 1 pos_weight는 train 전체에서 고정하고,
  실험 2 pos_weight는 virtual batch 전체 target의 positive/negative 수에서 계산해
  모든 microbatch에 동일하게 적용한다. 마지막 불완전 batch도 같은 정의를 쓴다.
- val의 virtual batch는 image ID 순으로 32개씩 고정한다. 같은 BCE numerator와
  denominator를 누적해 full-val loss를 구한다. batch/accumulation 변경 때문에
  class balance나 graph weighting이 달라지지 않는지를 CPU 비교한다.
- PE signal chunk=8, message edge chunk=4,096, SignNet attention node chunk=256.
  모든 eigenvector/sample/edge를 처리하고 attention은 32 eigenvectors 전체를 본다.
  activation checkpointing으로 backward에 필요한 중간값을 재계산한다.
  chunked softmax는 destination별 전체 edge의 max/sum을 사용해야 한다.
  micro/chunk 조정은 동일 함수의 수치·gradient 동등성 검사 후에만 허용한다.
- OOM 발생 시 더 작은 micro/chunk, activation recomputation, 기존 cache 사용만
  허용한다. width/depth/heads/K/M/eigenvector 수/후보/edge/graph stage 축소는 금지한다.
  허용된 방법으로도 안 되면 resource-blocked로 중단하고 승인 없이 다른 모델을 실행하지 않는다.
- 첫 실행 전 실제 candidate N/E의 median/p95/max, encoder variance, estimated work와
  GPU max allocated/reserved, RSS/swap, nvidia-smi snapshot을 측정해야 한다.
  부분 실데이터 CUDA 진단은 `experiment_1/runs/graph_pe_production_v1/audits/`에 기록했다.
  정식 학습은 전체 입력 분포의 quantile/max graph 검증 후 시작한다. 일부 graph의 실행
  가능성만으로 전체 epoch wall time이나 모든 크기의 resource feasibility를 확정하지 않는다.
- val-loss 최저 3개 checkpoint만 SHA/epoch/loss를 고정한 뒤 COCO 평가한다.
  primary는 그 3개의 최고 AP이며 loss/AP 동률은 이른 epoch. last 자동 추가 금지.
  AP/AR 원본 precision 보존; 3-seed 평균/표준편차와 각 seed를 모두 보고한다.
  같은 val을 selection에 쓰므로 independent test나 SOTA 증거로 표현하지 않는다.
- 최적화 미수렴, 큰 일반화 gap, deep-layer representation collapse가 있으면
  capacity의 충분성 또는 PE 우열을 결론내리지 않는다. budget 변경은 재설계 대상으로 남긴다.

## 8. 구현 승인 후 반드시 통과할 검사

1. 실제 constructed model의 H/L/heads/FFN/module count를 승인안과 대조. 96×3을
   main에 넣거나 선언값과 다른 model을 반환하면 실패. disabled encoder/dummy parameter 금지.
2. base feature/edge/target/weight/candidate identity가 main 세 모델 및 alignment
   control 사이 bitwise 동일. final NMS 한 번, 중간 cutoff/NMS 없음(실험 2 최종 cutoff 제외).
3. residual/head skip 실제 호출, active encoder와 각 layer의 finite gradient,
   초기 gate=0 identity 및 gate가 열린 뒤 gradient를 검사. 동일 모듈을 반복 호출해
   독립적인 8층이라고 속이지 않도록 서로 다른 parameter storage도 검사.
4. SignNet의 독립 column sign flip, all-padding, simple-spectrum 재생성;
   repeated-eigenvalue rotation은 보장 범위 밖인 진단으로 표시.
5. R-PEARL rho-before-mean 및 probe 전달 순열 검사. 16개 고정 synthetic graph에서
   8개 독립 probe bank로 M120/240/480 출력 변동을 기록하되, primary M120을
   평가 결과에 따라 교체하지 않는다. 큰 변동이면 안정성 주장 보류.
6. CPU float64 작은 fixture에서 non-chunked/chunked forward 및 gradient 비교
   (atol=1e-9, rtol=1e-7); float32는 atol=1e-5/rtol=1e-4.
   loss 축적은 dropout을 끈 동일 함수로 비교하고 dropout/RNG resume도 따로 검사.
7. strict model/optimizer/RNG/epoch resume, 동일 다음 update, wrong architecture/
   mode/seed/cache/design SHA 거부. 원본 graph/cache 없으면 fail-closed.
8. 학습·detector·평가 실행 가드, sparse-only 제약, atomic output/lock,
   no-overwrite, top-3-only, metric precision 및 실패 상태 기록.

실험 1 새 모델과 실행기 CPU 검사 14개를 통과했다. 원본 연산 대비 chunked
forward/gradient, 각 PE 층 gradient, sign/conditional permutation, 원본 target/feature/NMS,
model/optimizer/RNG의 동일 다음 update, 승인 가드를 검증한다. 구형 실행 차단 회귀도 유지한다.
실험 1의 부분 실데이터 GPU forward/backward도 확인했다. 전체 입력 GPU 감사 및 정식
학습/평가 완료 여부는 해당 run의 audit/progress/checkpoint/metric으로 판단하며,
실험 2나 미구현 capacity 진단까지 통과했다고 확대하지 않는다.

## 9. 경로·승격·승인

승인 후 결과는 `final/experiment_{1,2}/runs/graph_pe_production_v1/`
아래 `{local_id}/{profile}/seed_{42,43,44}`에 분리한다. 그 전에는 해당 run directory,
checkpoint, metric 또는 placeholder CSV/plot을 생성하지 않는다.
새 checkpoint schema는 `graph_pe_production_training_v1`; 구형 v2 checkpoint는 거부한다.
model/optimizer/RNG, config/design/code/cache/detector SHA, seed, candidate order,
capacity/gradient/operator/resource audit, 시작·종료/오류, selection과 원본 metric을 보존한다.
foreground PID+file lock와 atomic progress, manual resume만 사용한다.
resume은 완료된 epoch checkpoint에서만 허용한다. 중간 epoch에서 중단되면 마지막
완료 epoch로 복귀하고 그 epoch 이후의 부분 산출물은 완료 결과로 취급하지 않는다.

위 256×6 backbone + 128×8 PE의 구현과 실험 1 GPU 검증·학습·평가가 명시적으로
승인되었다. GPU 실행은 실제 conformance와 입력/상태 충돌 해소 후 실행 가드까지
통과해야 한다. 48개 run 전체 자동 예약은 하지 않는다. 실험 1 SignNet/aligned/seed42를
첫 실행 대상으로 하고, 나머지는 실제 실행 전까지 not-run이다.
실험 1은 `GRAPH_PE_INPUT_READINESS_AUDIT_20260915.md`의 experiment-local .25 cache만
사용한다. 과거 checkpoint/metric은 새 study에 재사용하지 않으며 역사 실행의 runner SHA
미확인 문제는 `unverified_historical`로 격리한다. 실험 2 .05 cache의 실제 view geometry
부재를 합성 metadata로 메우지 않는다.
