# 실험2 production PE 구현 검증 범위

2026-09-23 실험1·2 재학습 요청에 따라 기존 production 설계의 실험2 구현을 준비한다.
새 architecture를 제안하지 않으며 `../GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`의
실험2 H256/L6, 4-head, dropout .10 및 PE H128/L8 계약을 그대로 구현한다.
기준 모델은 공개 06, SignNet 09, R-PEARL 10이다. 별도 baseline 번호는 없다.

`production_pe_model.py`는 기존 `sparse_ppr_sage.Experiment2GraphRefiner`의
class-relation-aware homogeneous backbone을 사용한다. Input40/edge27, class-pair
key/value/bias, 이웃 class context, 각 층 residual과 raw-feature head concat을 유지한다.
Class-context 뒤, 첫 message layer 앞에만 선택된 PE를 projection/gate로 더한다.
기존 그래프 feature/edge/target 생성 함수를 변경하지 않는다. GT 입력이나 full-class
probability 합성은 금지한다. 기존 hard class one-hot만 사용한다.

PPR attention은 기존 연산을 edge chunk4096으로 나눠 계산하되, 모든 edge의 score를
모은 뒤 destination 전체 softmax를 한다. Chunk별 softmax를 하지 않는다.
6개 독립 layer, attention4heads, PE8개 독립 layer, k32/M120/K12는 축소하지 않는다.
Activation checkpointing은 dropout RNG를 보존한다. PE encoder는 실험1의 승인된
SignNetK32/RPEARLM120을 그대로 재사용한다.

공통 초기화는 experiment_2/seed42 backbone substream, PE는 별도 mode substream이다.
Gate는 0, loss·optimizer·후보·scoring·evaluation 계약은 원래 production 설계를 따른다.
Parameter count는 constructed model에서 baseline 2,488,187 / SignNet 2,903,172 /
R-PEARL 2,838,916을 확인해야 한다. 등록 개수는 활성 gradient 경로의 증명이 아니다.

이번 검증은 CPU synthetic에 한정한다. 전체 입력 cache의 실제 crop/view metadata,
공통 tensor identity, loss accumulation, GPU 실데이터 quantile 검사 및 runner 검증이
완료되기 전 실험2 학습·평가를 실행하지 않는다. Detector cache 재생성 승인은 별도다.
모델만 구현되었다고 실험2 전체 production runner가 구현되었거나 학습되었다고 쓰지 않는다.
