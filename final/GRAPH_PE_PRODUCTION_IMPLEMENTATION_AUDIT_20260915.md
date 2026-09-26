# Production PE 구현 감사 — 2026-09-15

## 범위와 결론

사용자의 명시적 승인에 따라 실험 1의 새 H256/L6 backbone + H128/L8 PE를
구현했다. 아래 검증은 **실험 1의 구현 conformance**에 대한 근거이며, 성능 우월성,
충분한 모델 용량, 논문급 결과 또는 실험 2 완료를 뜻하지 않는다.

- 승인 설계: `GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`
- 모델: `graph_pe_production.py`
- cache-only 데이터 준비: `graph_pe_production_data.py`
- foreground GPU 실행기: `run_graph_pe_production.py`
- 테스트: `test_graph_pe_production_contract.py`, `test_graph_pe_production_runner.py`
- 활성 상태: `manifests/graph_pe_production_proposal.json` 및 실제 run의 `progress.json`

## 용량과 실제 연산

| 실험 1 모델 | 실제 등록 trainable parameters | Backbone | PE |
|---|---:|---|---|
| no-PE | 6,073,347 | 256채널, 6층 | encoder/gate 없음 |
| SignNet-k32 | 6,488,332 | 동일 | 128채널, 독립 GIN 8층, 32 eigenvectors, 4-head signal attention |
| R-PEARL-M120 | 6,424,076 | 동일 | 128채널, 독립 GIN 8층, K12/M120, rho 이후 mean |

한 encoder만 instantiate한다. PE layer 8개가 서로 다른 parameter storage를
갖는지 검사했다. 초기 ReZero gate=0에서는 같은 seed의 backbone과 출력이 같고,
gate를 열면 각 PE layer와 projection까지 finite/nonzero gradient를 확인했다.
node-type residual, backbone 매 층 residual, 원본 feature head concat을 보존했다.

기존 no-cluster backbone의 `same_cluster` relation은 실제 graph에서 비활성이다.
부분 실데이터 GPU 감사에서 각 backbone layer의 relation 4 weight만 zero gradient였다.
이 여섯 tensor를 활성 용량이라고 주장하지 않으며, 제거해 원래 graph 계약을 바꾸지도 않는다.
R-PEARL과 SignNet을 서로 동일 parameter 수 또는 pure-PE capacity 통제로 표현하지 않는다.

## 통과한 검사

새 모델/실행기 CPU unittest 14개:

- 구성된 실제 파라미터 수, H/L, 고유 layer storage, 공통 backbone 초기 weight.
- gate=0 출력 보존, gate를 연 뒤 모든 PE parameter gradient.
- chunked / unchunked forward와 gradient: float64 atol 1e-9/rtol 1e-7,
  float32 atol 1e-5/rtol 1e-4.
- 원본 EdgeGated backbone과 출력 동등성, relation/skip 실제 hook 호출.
- SignNet column별 부호 뒤집기, padding/empty/isolated 처리, 단순 고유값 graph의
  topology 순열 후 eigensolver 재실행, 70-node sparse eigsh 경로.
- R-PEARL rho-before-mean, 120 probes 유지, node와 probe 동시 순열.
- probe/초기화 RNG 격리 및 alignment control의 동일 초기 weight·gradient.
- 원본 weighted BCE와 virtual-batch accumulation의 loss/gradient 동등성.
- GT 변경 시 input/edge/PE는 동일하고 target만 변경됨, 후보 보존, 최종 NMS 1회.
- 학습/평가 CLI+환경변수 guard, cache metadata 누락 시 실패.
- full model/AdamW/RNG/epoch strict resume 후 다음 update의 state_dict 및 optimizer 일치.
- mode/seed/code/cache/design/epoch/구형 schema 불일치와 checkpoint 덮어쓰기 거부.

기존 PE runner 회귀 13개도 통과했다. 구형 96×3 runner는 별도 production 승인을
받은 뒤에도 차단되며, CP/Full/GOIS와 기존 backbone·checkpoint·metric은 수정하지 않았다.

## 실제 GPU 진단

RTX 3090에서 실제 train image 71의 후보 **1,166개 / edge 10,900개**를
그대로 사용해 두 PE 모델의 forward/backward를 확인했다. activation recomputation,
signal chunk 8, edge chunk 4,096으로 모든 신호·edge를 처리했다.

- SignNet: peak allocated 약 480.62 MiB, 진단 forward/backward 약 0.612초.
- R-PEARL: peak allocated 약 272.59 MiB, 약 0.316초.

이는 warm-up을 포함할 수 있는 한 graph의 진단값이며 throughput benchmark,
전체 epoch 시간 또는 전체 데이터의 최대 메모리 보증이 아니다.
진단 가중치는 학습에 재사용하지 않는다. 위 값의 원본은
`experiment_1/runs/graph_pe_production_v1/audits/*_partial_real_gpu_conformance.json`이다.
초기 메모리 측정 호출의 CUDA 초기화 순서 오류를 발견해 수정했고, 이후 검사가 통과했다.

정식 실행기는 전체 입력 graph의 N/E median·p95·max 사례에서 다시 GPU 검사를 수행해
해당 run의 `gpu_conformance.json`에 기록한 다음 새 weight로 학습한다.

## 입력·학습·평가 계약

실험 1의 actual-view .25 detector cache 4개 SHA와 이미지 목록, class mapping,
crop/view geometry, 후보 수·순서를 검증한다. shared legacy cache의 가짜 view fallback을
사용하지 않는다. graph cache와 원본 detector cache를 구분하며 detector inference는 없다.

정식 설정은 train 전체 6,471장, val 전체 548장, 120 epochs, virtual batch32 /
microbatch1, float32 AdamW lr1e-3/weight_decay1e-4다. empty graph는 별도 기록한다.
매 epoch 전체 nonempty graph를 한 번씩 방문하며 sample/update 제한을 두지 않는다.
loss는 microbatch별 평균의 평균이 아니라 virtual batch의 전체 weight 합으로 정규화한다.

모든 완료 epoch의 model/optimizer/RNG/history를 보존한다. val-loss 최저 3개
checkpoint의 epoch/SHA/loss를 고정한 뒤 그 3개만 COCO 평가한다.
정식 결과가 없을 때 AP=0, 추정 score plot 또는 placeholder metric을 생성하지 않는다.

## 남은 범위와 위험

2026-09-15 13:29 UTC 실행 확인: 전체 train/val 입력 준비 및 SHA 검증이 끝났고,
SignNet seed42/aligned의 full-input quantile GPU 검사를 통과한 뒤 정식 학습 첫
32-graph optimizer update가 완료됐다. 실행 PID는 시작 당시 `1209617`이다.
설정은 120 epochs이며, 완료를 의미하지 않는다. 최신 상태는
`experiment_1/runs/graph_pe_production_v1/10/signnet/aligned/seed_42/progress.json`을 따른다.
전체 입력의 train N/E 최대값은 1,626/19,362, val N/E 최대값은 1,608/19,052다.
학습 후보 1,378,319개와 검증 후보 156,763개를 보존했고 empty graph는 없다.

- 실제 학습·평가 진행과 완료 여부는 run artifact가 기준이다. CPU/GPU 진단 통과가
  120 epochs 완료 또는 좋은 AP를 의미하지 않는다.
- 실험 2 production runner는 아직 구현하지 않았고, .05 cache의 실제 view metadata가
  없어 입력도 막혀 있다. cache 재생성을 임의로 실행하지 않았다.
- 명시된 작은 capacity 진단 profiles는 설계만 있고 아직 구현/실행하지 않았다.
- SignNet의 반복 고유값 basis rotation invariance는 제공하지 않는다.
- R-PEARL M120에는 Monte Carlo 변동이 있다. 16개 synthetic graph × 8개 bank에서
  M120/240/480 변동을 측정했고, 이 진단을 detection 성능이나 안정성 보증으로 확대하지 않는다.
- 과거 실험 1 02-04는 `unverified_historical`, 05는 `invalid_control_comparison`이다.
  새 결과와 혼합하지 않으며 원본 역사 산출물은 보존한다.
