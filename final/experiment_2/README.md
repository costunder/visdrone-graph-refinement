# Experiment 2: Low-confidence Candidate Rescue

공개 ID는 실험 1과 독립적으로 `00`부터 시작하며
[`current_experiments.json`](../manifests/current_experiments.json)을 따른다.
기존 폴더/실행기 번호는 provenance로 보존한다. [비교 복구 계약](../COMPARISON_REPAIR_SPEC.md)
을 따른다. 2026-09-23 후속 실행 요청으로 `production_pe_model.py`에 대형 모델을
구현하고 CPU 검증했다. 실제 view 입력과 production 학습 실행기 연결은 미완료이며
GPU 학습을 시작하지 않았다. 상세 범위는 `PRODUCTION_PE_CONFORMANCE_SPEC.md`를 따른다.

| Local ID | Variant | Internal legacy name | Status |
|---|---|---|---|
| 00 | `00_gnn_conf_rescue` | `04_gnn_conf_rescue` | completed |
| 01 | `01_size_refinement_conf_rescue` | `06_size_refinement_conf_rescue` | completed |
| 02 | `02_low_conf_cluster_token_hgnn_refinement` | `07_low_conf_cluster_token_hgnn_refinement` | completed |
| 03 | `03_sparse_edge_sage` | `09_exp2_sparse_edge_sage` | completed, best AP 0.1758 |
| 04 | `04_ppr_gatv2_sage` | `10_exp2_ppr_gatv2_sage` | completed, best AP 0.1771 |
| 05 | `05_ppr_gatv2_sage_no_gois` | `11_exp2_ppr_gatv2_sage_no_gois` | completed, best AP 0.1166 |
| 06 | `06_class_relation_ppr_gatv2_sage` | new | completed, best AP 0.1772; relational homogeneous |
| 07 | `07_hetero_detection_class_view_gatv2` | new | completed, primary AP 0.152326; non-controlled heterogeneous |
| 08 | `08_controlled_hetero_ppr_gatv2_sage` | new | completed, best AP 0.175064; controlled heterogeneous |
| 09 | `class_relation_ppr_signnet_pe` | legacy PE ID `10` | new model CPU verified; input/runner pending; not trained |
| 10 | `class_relation_ppr_rpearl_pe` | legacy PE ID `11` | new model CPU verified; input/runner pending; not trained |

\[
\mathcal{B}_{\mathrm{low\text{-}conf}}
\xrightarrow{\text{sparse relation graph}}
\widehat{\mathcal{B}}_{\mathrm{rescued}}
\xrightarrow{\mathrm{NMS}}
\mathcal{Y}
\]

`06` adds a directed class-pair embedding and the observed neighbor-class distribution to `04`:

\[
r_{ij}=E_{c_i,c_j},\qquad
q_i(k)=\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}\mathbf 1[c_j=k].
\]

It is a **class-relation-aware homogeneous GNN**, not a formal heterogeneous graph. All graph nodes are still detection candidates in one tensor.

`07` is the strict heterogeneous extension. It uses separate `detection`, `class`, and `view` node stores and relation-specific GATv2 message passing:

\[
\mathcal V=\mathcal V_{det}\cup\mathcal V_{class}\cup\mathcal V_{view},\qquad
(\hat o_i,\hat{\mathbf p}_i)=f_{det}(h_i),\quad
\hat{\mathbf p}_i\in\mathbb R^{K+1}.
\]

The detector cache stores only a hard `category_id` and score. Accordingly, `predicted_as` uses the observed hard class while `supports` connects every class node to every detection so the class head can still correct a label. This is a real heterogeneous graph, but retaining full detector class probabilities remains a stronger future input design. See `SPARSE_FOLLOWUP_SPEC.md`.

## Historical masked-PE configuration (blocked, not current public enumeration)

이하 09/10/11은 구형 실행기 ID다. 공개 목록에는 no-PE를 따로 넣지 않으며
09=SignNet, 10=R-PEARL이다. 아래 설명을 새 production 구현 완료로 해석하지 않는다.

현재 아래 구형 96×3 구성은 CPU 회귀 검사용이며 production 실행은 차단했다.
학습·평가 플래그로도 우회할 수 없다. 256×6 backbone과 128×8 PE의 승인 대기안은
[`GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`](../GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md)에
있으며, 아직 새 모델을 구현하거나 실행하지 않았다.

`09/10/11`은 `06`의 low-confidence candidate pool, sparse local/PPR edge,
ordered class-relation backbone, target/weight/loss, seed, score cutoff와 final
NMS를 고정한다. 세 모델 모두 SignNet과 R-PEARL module을 instantiate하며
activation mask만 `[0,0]`, `[1,0]`, `[0,1]`로 다르다. 따라서 각 model은
base 237,087 + PE 179,623 = `416,710`개의 등록 trainable parameters와 동일한 17회 PE
sparse operator call을 갖는다. ReZero gate는 0에서 시작한다. gate가 열린 뒤 loss에
연결 가능한 PE parameter는 각각 `0 / 84,676 / 94,948`이므로 active capacity는
같지 않다. PE 구조 정보와 추가 학습 용량의 효과를 분리한 비교로 주장하지 않는다.

CPU contract test만 통과했으며 training, detector inference, COCO evaluation,
checkpoint, AP/AR 및 score plot은 생성하지 않았다. 상세 계약은
`../GRAPH_POSITIONAL_ENCODING_SPEC.md`에 있다.

## Variant 08: Controlled Heterogeneous Test

`07` changed the graph backbone, target construction, auxiliary class loss, and
inference score gate at the same time. It therefore cannot isolate the effect of
heterogeneous message passing. `08` fixes the `06` protocol:

\[
(X_{det},E_{det\to det},Y,W,\mathcal L,\mathcal S,\mathrm{NMS})_{08}
=
(X_{det},E_{det\to det},Y,W,\mathcal L,\mathcal S,\mathrm{NMS})_{06}.
\]

Only the backbone changes. The model adds active class nodes and observed view
nodes, then learns relation-specific messages along

\[
det\leftrightarrow class,\qquad det\leftrightarrow view,\qquad
class\to class,
\]

while retaining the exact sparse spatial/PPR detection graph from `06`.
Class support is one membership edge per detection, so it is \(O(N)\), not the
`07` all-class \(O(KN)\) pseudo-prior. Only classes present in the image are
materialized. A synthetic two-graph test and a real VisDrone image with 330
candidates verified bitwise-identical detection features, targets, weights,
and detection-to-detection edges. The class and view paths also receive finite
gradients.

To reduce capacity as a confound, `08` uses hidden width 56 (224,021 trainable
parameters) against `06` width 96 (237,087 parameters), a -5.5% difference.
The runner writes the exact audit to `model_capacity.json`.

## Variant 08 Result

The controlled heterogeneous model completed 40 epochs using the existing
detector cache. Following the predeclared selection procedure, epochs 40, 36,
and 34 were the three lowest-validation-loss checkpoints evaluated with COCO
metrics. Epoch 34 had the highest AP:

| Epoch | Predictions | Validation loss | AP | AP50 | AP75 | AP-small |
|---:|---:|---:|---:|---:|---:|---:|
| 34 | 93,045 | 2.259671 | 0.175064 | 0.324153 | 0.166801 | 0.127815 |
| 36 | 88,068 | 2.250943 | 0.174872 | 0.322422 | 0.166827 | 0.127023 |
| 40 | 87,675 | 2.246036 | 0.172481 | 0.319836 | 0.163643 | 0.125629 |

The best `08` result is 0.002086 AP below `06` and 0.022738 AP above the
non-controlled `07`. Thus this run does not show that heterogeneous message
passing improves on the relational homogeneous baseline, but it does show that
the severe `07` loss was largely caused by the combined target, loss, and
inference-policy changes rather than heterogeneous structure alone.

For messages targeting detection nodes, the learned spatial-PPR relation weight
was 0.678, 0.657, and 0.640 across the three layers. Class support and view
context together retained 0.322, 0.343, and 0.360. These mixture coefficients
show that the typed paths were used, but they are not causal feature-importance
scores.

## Variant 06 Result

The 40-epoch run selected epoch 40 and obtained AP 0.177150, AP50 0.327180,
AP75 0.168583, and AP-small 0.130209. Against `04`, AP changed by only
+0.000084: this is practical parity, not evidence of a meaningful overall
improvement. AP50 and AP-small improved slightly, while AP75 and AP-large decreased.

## Variant 07 Result

The true heterogeneous model completed 40 epochs and selected epoch 40 as its
best evaluated checkpoint. Its primary policy obtained AP 0.152326, AP50
0.276638, AP75 0.146536, and AP-small 0.102563. This is 0.024824 AP below
`06`; the result does not support an overall improvement claim.

An inference-only policy ablation reused the same checkpoint and detector cache:

| Policy | Predictions | Corrections | AP | Delta vs primary |
|---|---:|---:|---:|---:|
| foreground gate 1.0 + correction (primary) | 56,403 | 2,342 | 0.152326 | 0.000000 |
| no gate, no correction | 86,530 | 0 | 0.143382 | -0.008944 |
| correction only | 85,670 | 2,342 | 0.155569 | +0.003243 |
| foreground gate 0.25 + correction | 79,807 | 2,342 | 0.155987 | +0.003661 |
| foreground gate 0.50 + correction | 72,406 | 2,342 | 0.155126 | +0.002800 |

At the same zero-gate setting, class correction improves AP by 0.012187 over
no correction despite changing only 0.885% of detection nodes. The class branch
contains useful correction signal, but the heterogeneous size/object score is
not calibrated strongly enough to match `04/06`. The post-hoc 0.25 gate was
selected on the evaluation split and is reported as an ablation, not as the
primary result.

- Existing runner (`00-02`): `run_experiment_2.sh`
- Sparse follow-up runner (`03-08`): `run_sparse_followups.sh`
- Legacy PE runner: `run_graph_pe_ablation.sh` (blocked; public `09,10` map to legacy names)
- PE CPU model contract: `/home/kim-sanghwa/miniconda3/envs/GNN/bin/python final/test_graph_pe_contract.py`
- PE CPU runner contract: `/home/kim-sanghwa/miniconda3/envs/GNN/bin/python final/test_graph_pe_runner_contract.py`
- 2026-09-15 PE 정정: rho-before-sum, top-3-only selection, 원본 precision, 실행 guard와 cache-only 경로; 실제 실험 2 parser의 `graph_radius_ratio` 누락도 수정.
- Homogeneous sparse model (`03-06`): `sparse_ppr_sage.py`
- Non-controlled heterogeneous model/training (`07`): `hetero_detection_class_view.py`, `hetero_training.py`
- Controlled heterogeneous model/training (`08`): `controlled_hetero_graph.py`, `controlled_hetero_training.py`
- Detailed specification: `SPARSE_FOLLOWUP_SPEC.md`
- Best-AP table: `reports/publication_results_best_ap.csv`

Existing training never starts unless `--allow_training` is passed explicitly. Variants
`06`, `07`, and `08` completed 40 epochs with `ALLOW_MODEL_TRAINING=1`. All
three reused the existing detector caches; the completed `08` run did not rerun
detector inference. New PE variants `09-11` additionally require both
`ALLOW_MODEL_TRAINING=1` and `ALLOW_COCO_EVALUATION=1`; neither authorization
was given or used while adding them.
