# Experiment 3: Sequential Pre-GOIS Graph Prior + Sparse Rescue

**2026-09-23: `blocked_stage_design_unconfirmed`. 아래는 보존된 기존 구현 설명이며,
사용자가 거부한 연결 방향을 승인된 현재 설계로 취급하지 않는다.**
`run_experiment_3.sh`와 Python main/run_variant/train_epoch는 실행 플래그와 무관하게
차단된다. 새 단계 순서·GOIS 위치·전달 tensor·중간 cutoff 경계는
[`COMPARISON_REPAIR_SPEC.md`](../COMPARISON_REPAIR_SPEC.md)에 따라 확인해야 한다.
기존 00/01은 active 공개 목록에서 제외하고 source 및 CPU 회귀 test만 보존한다.

실험 3은 실험 1과 실험 2의 최종 prediction을 합치지 않는다. 하나의 raw
low-confidence candidate pool 안에서 실험 1의 **NMS 이전 graph score**를 만들고,
그 값을 실험 2의 sparse PPR 후처리 입력으로 전달한다.

## Pipeline

```text
raw coarse+fine candidates at confidence 0.05
                    │
                    ├─ core candidates (detector score >= 0.25)
                    │        │
                    │        ▼
                    │  frozen Experiment 1 / 02 typed relation GNN
                    │        │
                    │        └─ object/small/large/ROI graph probabilities
                    │
                    ▼
full low-confidence pool + five Experiment-1 prior channels
                    │
                    ▼
Experiment 2 / 06 class-relation PPR-GATv2-SAGE
                    │
                    ▼
score threshold -> one final class-wise NMS
```

실험 1 checkpoint는 `02_gnn_no_cluster/epoch_116.pt`를 frozen 상태로 사용한다.
이는 실험 1의 `05_gnn_prune_same_pool`이 사용한 scorer다. checkpoint가 학습된
고정 후보 조건을 지키기 위해 shared `0.05` pool 중 detector score가 `0.25`
이상인 core만 실험 1 graph에 넣는다. 낮은 confidence 후보는 실험 2 graph에서
core node와 spatial/PPR relation으로 연결되어 rescue될 수 있다.

실험 1 단계에서는 후보 제거, COCO prediction 생성, NMS를 하지 않는다.

## Controlled variants

| Local ID | Variant | Experiment-1 prior |
|---|---|---|
| 00 | `00_zero_prior_control` | 다섯 prior channel을 모두 0으로 고정 |
| 01 | `01_exp1_graph_prior_exp2_ppr` | `[core mask, object, small, large, ROI]` frozen probabilities |

두 variant는 다음을 동일하게 고정한다.

- raw `0.05` coarse+fine candidate pool과 node 순서
- sparse local/PPR edges와 edge features
- target, sample weight, BCE loss
- stage-2 architecture와 parameter count
- score threshold와 최종 class-wise NMS
- random seed와 checkpoint selection 절차

오직 다섯 prior channel의 **값**만 `zero`와 `experiment-1 output`으로 달라진다.
따라서 `01 - 00`이 실험 1 graph prior의 통제된 효과다.

## Safety and execution

학습·평가는 실행하지 않았다. 아래 두 조건은 과거 가드이며 이제 이것만으로
실행할 수 없다. 먼저 새 설계 승인과 구현 검증이 필요하다.

1. `ALLOW_MODEL_TRAINING=1`의 명시적 승인
2. 기존 detector coarse/fine cache의 존재

cache가 없을 때 detector inference로 대체하지 않고 즉시 실패한다.

```bash
# 승인 전: 아무 실험도 실행하지 않고 종료
final/experiment_3/run_experiment_3.sh

# 역사 명령: 현재는 환경변수를 설정해도 설계 불일치로 차단됨
ALLOW_MODEL_TRAINING=1 final/experiment_3/run_experiment_3.sh
```

## Files

- `sequential_exp1_exp2.py`: shared pool, prior 전달, sequential tensor/model 계약
- `run_experiment_3.py`: cache-only training/evaluation runner
- `run_experiment_3.sh`: 이중 training guard가 있는 entrypoint
- `test_sequential_exp1_exp2.py`: 실제 데이터가 필요 없는 합성 계약 검증

## Invalid earlier implementation

이전에 생성한 post-NMS late-fusion 코드와 결과는 의도한 실험 3이 아니다. 삭제하지
않고 `archive/invalid_post_nms_late_fusion_20260813/`에 보존했으며, active 결과나
실험 3 성능으로 인용하지 않는다.
