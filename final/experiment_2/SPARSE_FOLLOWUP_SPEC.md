# Experiment 2 Sparse Inductive Follow-up Specification

2026-09-23: 이 문서의 기존 ID 표는 역사 실행기/artifact 기록이다. 현재 공개 번호는
`../manifests/current_experiments.json`이며 별도 zero-PE 번호 없이 09=SignNet,
10=R-PEARL이다. 새 production PE는 미구현·미실행이고 비교 복구 계약을 먼저 확정한다.

## Local Order

| Local ID | Candidate source | Graph | PPR | Class relation | GOIS dependency | Status |
|---|---|---|---|---|---|---|
| 00 | low-conf coarse+fine | existing pairwise GNN | no | implicit | yes | completed |
| 01 | low-conf coarse+fine | 00 + size refinement | no | implicit | yes | completed |
| 02 | lower-conf coarse+fine | cluster-token HGNN | no | implicit | yes | completed |
| 03 | low-conf coarse+fine | sparse EdgeSAGE | no | same/cross flag | yes | completed |
| 04 | low-conf coarse+fine | sparse GraphSAGE | yes | same/cross flag | yes | completed |
| 05 | full-image only | sparse GraphSAGE | yes | same/cross flag | no | completed |
| 06 | low-conf coarse+fine | sparse GraphSAGE | yes | ordered class-pair embedding | yes | completed |
| 07 | low-conf coarse+fine | dense-support detection/class/view HeteroConv-GATv2 | yes | explicit class nodes + correction head | yes | completed |
| 08 | low-conf coarse+fine | controlled sparse detection/class/view GATv2-SAGE | yes | active class nodes, unchanged 06 score head | yes | completed; best AP 0.175064 |
| 09 | low-conf coarse+fine | 06 + masked zero-PE adapter (registered-count matched only) | yes | unchanged 06 ordered class-pair relation | yes | configured, not run |
| 10 | low-conf coarse+fine | 06 + SignNet adaptation | yes | unchanged 06 ordered class-pair relation | yes | configured, not run |
| 11 | low-conf coarse+fine | 06 + R-PEARL adaptation | yes | unchanged 06 ordered class-pair relation | yes | configured, not run |

## Typed Nodes and Relations

Each detector candidate is one node:

\[
h_i^{(0)}=
[s_i,g(b_i),\operatorname{onehot}(c_i),
\operatorname{onehot}(\tau_i),\operatorname{onehot}(q_i)].
\]

The restored experiment-2 schema has 40 node features and 27 edge features for the VisDrone 10-class setup. Existing `03-05` checkpoint shapes remain unchanged.

Edges contain relative position, scale ratio, IoU, containment, same/cross-class indicators, candidate-source relations, view relations, and PPR score. A spatial hash retains at most the configured same-class and cross-class neighbors instead of constructing an \(N^2\) graph.

## Sparse Neighborhood

\[
|E_{\mathrm{local}}|\lesssim N(K_s+K_c)+N.
\]

The current defaults are \(K_s=12\), \(K_c=4\), plus self-loops. Symmetrization can increase incoming degree but does not allocate a dense relation matrix.

## High-order PPR Relation

\[
\pi_i \approx \alpha\sum_{t=0}^{T}(1-\alpha)^t e_iP^t.
\]

Each target keeps only its top-\(K_p\) PPR neighbors. With the current defaults, \(T=8\) and \(K_p=8\).

## Variant 06: Directed Class Relations

`06` assigns an ordered relation ID to every message:

\[
r_{ij}=c_iK+c_j,
\]

so `person -> car` and `car -> person` have different parameters. Every PPR-GATv2 layer learns a class-pair key, value, and per-head attention bias:

\[
\ell_{ij}^{(m)}=
\frac{a_m^\top\operatorname{LeakyReLU}
(W_sh_i+W_dh_j+W_ee_{ij}+E^{\mathrm{key}}_{c_i,c_j})}
{\sqrt{d_m}}
+B^{(m)}_{c_i,c_j}
+\operatorname{softplus}(\beta_m)
\log\left(1+\frac{\pi_i(j)}{\epsilon}\right).
\]

The message value also receives \(E^{\mathrm{value}}_{c_i,c_j}\). Before message passing, the model forms the full neighbor-class histogram

\[
q_i(k)=\frac{1}{|\mathcal N(i)|}
\sum_{j\in\mathcal N(i)}\mathbf 1[c_j=k]
\]

and injects \([\operatorname{onehot}(c_i)\Vert q_i]\) through a gated MLP. Therefore a zero entry can represent an expected class relation that is absent. This is supervised indirectly through the object/small/large refinement loss; explicit non-edge relation supervision is not included yet.

`06` still outputs object, small, and large logits. It can suppress a contextually inconsistent detection but does not yet change `car` into `truck`.

## Is Variant 06 Heterogeneous?

No, not in the strict data-model sense. All vertices still satisfy

\[
\phi(v)=\text{detection candidate}
\]

and are packed into one node tensor and one `edge_index`. Class-pair-specific parameters make `06` a relational or typed homogeneous GNN, but there are no separate entity node sets.

Treating the detector's hard predicted class as the node type is also undesirable: a misclassified candidate would be locked into the wrong type and become difficult to reclassify.

## Variant 07: True Heterogeneous Graph

`07` implements three explicit PyG entity types:

\[
\mathcal V=\mathcal V_{det}\cup\mathcal V_{class}\cup\mathcal V_{view}.
\]

- `detection`: one node per candidate box
- `class`: ten semantic class nodes with learned projections
- `view`: full/coarse/fine tile nodes

Implemented directed relations:

| Source | Relation | Destination | Meaning |
|---|---|---|---|
| detection | spatial / overlap / containment / PPR | detection | geometric and high-order context |
| detection | predicted-as | class | hard predicted class edge in the current cache |
| class | supports | detection | class-conditioned message back to candidates |
| class | co-occurs | class | learned semantic compatibility |
| detection | observed-in | view | candidate provenance |
| view | contains | detection | shared-view context |

`HeteroConv` applies a separate edge-aware `GATv2Conv` to every relation and produces both size-aware objectness and corrected class logits on detection nodes:

\[
(\hat o_i,\hat{\mathbf p}_i)
=f_{det}(h_i^{HGT}),\qquad
\hat{\mathbf p}_i\in\mathbb R^{K+1}.
\]

Current detector caches retain only `category_id` and score, not top-k/full class probabilities. Therefore `detection -> predicted_as -> class` uses one observed hard-class edge, while `class -> supports -> detection` connects all \(K\) class nodes to every detection with a softened score prior. A class-agnostic one-to-one IoU match supplies \(K+1\) correction targets, including background, so a geometrically matched wrong-class candidate can be relabeled. Full detector class probabilities remain the preferred future replacement for this pseudo-prior.

## Variant 08: Single-factor Controlled Heterogeneous Backbone

`08` is the valid test of the heterogeneous-backbone hypothesis. Its controlled
variables are

\[
\begin{aligned}
X_{det}^{08}&=X_{det}^{06}, & E_{det\to det}^{08}&=E_{det\to det}^{06},\\
Y^{08}&=Y^{06}, & W^{08}&=W^{06},\\
\mathcal L^{08}&=\mathcal L^{06}, &
\mathcal S^{08}&=\mathcal S^{06}.
\end{aligned}
\]

Here \(\mathcal S\) includes graph-score construction, confidence threshold,
and class-wise NMS. There is no auxiliary class CE, background gate, or class
correction in the primary `08` protocol. The only intervention is

\[
f_{\mathrm{homogeneous}}^{06}
\longrightarrow
f_{\mathrm{heterogeneous}}^{08}.
\]

For an image with \(N\) detections, \(A\le K\) active classes, and \(V\) views,
`08` creates \(N+A+V\) nodes. Membership edges are sparse:

\[
|E_{det\leftrightarrow class}|=2N,\qquad
|E_{det\leftrightarrow view}|=2N,\qquad
|E_{class\to class}|\le A^2.
\]

A two-layer-or-deeper path such as
\(det_i\to class_c\to det_j\) learns same-class context, while
\(det_i\to det_k\to det_j\) retains the original local/PPR high-order context.
Relation-specific GATv2 messages are mixed by a learned softmax for each target
node type before a residual GraphSAGE-style update.

The default width is capacity matched: `08` uses 56 hidden channels and 224,021
trainable parameters; `06` uses 96 channels and 237,087 parameters. The ratio is
0.9449. Exact counts are emitted per run rather than inferred from checkpoint
file size.

## Variants 09-11: Controlled Graph PE

Execution hold (2026-09-15): the 96x3 masked-branch configuration below is
retained for CPU regressions only. Runner version 3 blocks main, run_variant
and train_epoch even with execution flags set. The unimplemented replacement
proposal is `../GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`; its status is
`design_pending_approval`, not an active architecture contract.

The PE comparison freezes the complete `06` data and inference protocol. Each
model contains the same 96-wide, three-layer SignNet adaptation, the same
96-wide, three-layer R-PEARL adaptation, a bias-free 192-to-96 fusion, and a
zero-initialized ReZero scalar. Only the non-trainable activation mask differs:

\[
m_{09}=[0,0],\qquad m_{10}=[1,0],\qquad m_{11}=[0,1].
\]

All three models have 416,710 registered trainable parameters and execute both
PE branches on every forward pass. This does not match active learning capacity:
after the gate opens, the PE loss paths contain 0 / 84,676 / 94,948 parameter
elements in 09 / 10 / 11. Structural PE information is not isolated from the added
active learning capacity. The 32 raw channels are eight normalized-
Laplacian eigenvectors, their eight validity masks, and sixteen deterministic
random probes. They are appended without changing the 40 base features and are
injected as a hidden residual only after the existing class-context block.
Detection edges, including local and PPR relations, remain unchanged.

The complete eigensolver, node-order, initialization, gradient-path, operator,
cache-only execution and claim contract is fixed in
`../GRAPH_POSITIONAL_ENCODING_SPEC.md`. The variants are not trained or
evaluated and therefore have no AP/AR values.

## Verification State

- `03-05`: trained and evaluated; results are under `runs/` and `reports/`.
- `06`: synthetic forward verification plus a full 40-epoch training and COCO evaluation completed; it remains relational homogeneous.
- `07`: true `HeteroData` construction, two-graph batch forward/backward smoke test, 40-epoch training, and COCO evaluation completed; it is not a single-factor comparison to `06`.
- `08`: controlled `HeteroData` construction passed synthetic two-graph forward/backward and a real 330-candidate VisDrone image audit; 40-epoch training and COCO evaluation completed.
- `09-11`: 2026-09-15 CPU model and isolated-runner contracts passed, including rho-before-pool, raw-PE reconstruction limitations, real operator hooks, active gradients, strict PE optimizer resume, top-three-only selection and one final NMS. Status remains `configured_not_run` with no experiment checkpoint or metric.
- Existing `04/epoch_039.pt`: strict-loaded against the restored 40-node/27-edge schema.
- `06`, `07`, and `08` reused the existing low-confidence detector caches. Detector inference was not rerun.

## Completed Variant Evaluation

| Variant | Epoch | AP | AP50 | AP75 | AP-small | AP-medium | AP-large |
|---|---:|---:|---:|---:|---:|---:|---:|
| `04_ppr_gatv2_sage` | 39 | 0.177066 | 0.326150 | 0.169085 | 0.128963 | 0.236865 | 0.246917 |
| `06_class_relation_ppr_gatv2_sage` | 40 | 0.177150 | 0.327180 | 0.168583 | 0.130209 | 0.236234 | 0.240657 |
| `07_hetero_detection_class_view_gatv2` | 40 | 0.152326 | 0.276638 | 0.146536 | 0.102563 | 0.210700 | 0.230325 |
| `08_controlled_hetero_ppr_gatv2_sage` | 34 | 0.175064 | 0.324153 | 0.166801 | 0.127815 | 0.233630 | 0.233929 |

The AP delta over `04` is \(+0.000084\), which should be reported as parity
for this single-seed run. The directed class-relation module slightly improves
AP50 and small-object AP, but does not provide a consistent gain across IoU and
object-size slices. A publication claim of improvement requires repeated seeds
or a stronger relation-supervision/class-correction design.

For `08`, the predeclared selector evaluated the three lowest-validation-loss
checkpoints (epochs 40, 36, and 34) and selected epoch 34 by AP. Its AP is
\(-0.002086\) relative to `06` and \(+0.022738\) relative to `07`. The
controlled result therefore rejects an improvement claim for the current
heterogeneous backbone, while also showing that heterogeneous structure alone
does not explain the large degradation in `07`. Detection-target relation
mixtures retained 32.2-36.0% combined weight on class and view messages across
layers; these learned mixture values are descriptive, not causal importance.

## Variant 07 Inference-policy Ablation

All rows below reuse epoch 40; no model training or detector inference was run.

| Foreground gate | Class correction | Predictions | AP | Delta vs `06` |
|---:|---|---:|---:|---:|
| 1.00 | yes | 56,403 | 0.152326 | -0.024824 |
| 0.00 | no | 86,530 | 0.143382 | -0.033768 |
| 0.00 | yes | 85,670 | 0.155569 | -0.021581 |
| 0.25 | yes | 79,807 | 0.155987 | -0.021163 |
| 0.50 | yes | 72,406 | 0.155126 | -0.022024 |

With gate fixed to zero, enabling class correction raises AP by
\(0.155569-0.143382=0.012187\). Thus the correction head learns useful class
signal, but the full heterogeneous object/size scoring pipeline underperforms the
relational homogeneous `06`. The 0.25 gate is a post-hoc evaluation-split
ablation and is not substituted for the primary predeclared policy.
