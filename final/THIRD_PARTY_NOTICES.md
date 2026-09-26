# Third-party Research and License Notices

검토일: 2026-09-05

이 문서는 연구 인용과 소프트웨어 사용 권한을 구분하기 위한 provenance 기록이다. 법률 자문을 대신하지 않는다.

## GOIS Paper

- Title: Enhancing Tiny Object Detection Using Guided Object Inference Slicing (GOIS): An efficient dynamic adaptive framework for fine-tuned and non-fine-tuned deep learning models
- Authors: Muhammad Muzammul, Xuewei Li, Xi Li
- Journal: Neurocomputing, Volume 640, Article 130327, 2025
- DOI: https://doi.org/10.1016/j.neucom.2025.130327
- Official repository: https://github.com/MMUZAMMUL/GOIS

```bibtex
@article{MUZAMMUL2025130327,
  title   = {Enhancing Tiny Object Detection Using Guided Object Inference Slicing (GOIS): An efficient dynamic adaptive framework for fine-tuned and non-fine-tuned deep learning models},
  author  = {Muhammad Muzammul and Xuewei Li and Xi Li},
  journal = {Neurocomputing},
  volume  = {640},
  pages   = {130327},
  year    = {2025},
  doi     = {10.1016/j.neucom.2025.130327}
}
```

논문 인용과 baseline 비교는 허용 여부를 코드 라이선스로 판단하지 않는다. GOIS를 선행 연구 또는 비교 기준으로 사용한 모든 문서에는 위 논문을 인용한다.

## GOIS Repository License Review

2026-07-23에 확인한 공식 저장소의 `LICENSE` 파일은 제목이 `Proprietary License for GOIS`이며 다음 범위를 명시한다.

- educational purpose 또는 personal study만 허용
- commercial use, redistribution, modification, publication은 사전 서면 허가 없이 금지
- reverse engineering, reproduction 및 derivative work 생성 금지

저장소 README 일부에는 `MIT License`라는 표현이 있지만, 실제 `LICENSE` 본문은 표준 MIT 조건과 일치하지 않는다. 권리자로부터 서면 확인을 받기 전까지 제한적인 독점 라이선스로 취급한다.

License URL:
https://github.com/MMUZAMMUL/GOIS/blob/main/LICENSE

## SignNet / BasisNet

- Paper: *Sign and Basis Invariant Networks for Spectral Graph Representation Learning*, ICLR 2023
- Paper URL: https://openreview.net/forum?id=Q-UHqMorzil
- Reference repository: https://github.com/cptq/SignNet-BasisNet
- Repository license reviewed for this project: MIT

`final/graph_positional_encoding.py`의 SignNet branch는 eigenvector sign
invariance `phi(u) + phi(-u)` 원리를 사용한 project-specific adaptation이다.
공식 repository의 SetTransformer 구성이나 source file을 복사·vendoring하지
않았으며, 이 구현을 공식 SignNet 실행 또는 exact reproduction으로 표시하지 않는다.
특히 repeated-eigenvalue eigenspace의 basis invariance는 현재 구현 범위가 아니다.

## PEARL

- Paper: *Learning Efficient Positional Encodings with Graph Neural Networks*, ICLR 2025
- Paper URL: https://cs.stanford.edu/people/jure/pubs/efficient-positional-encodings-iclr25.pdf
- Reference repository: https://github.com/ehejin/Pearl-PE
- Repository license reviewed for this project: MIT

`final/graph_positional_encoding.py`의 R-PEARL branch는 random probe,
normalized-adjacency polynomial filtering 및 sample aggregation 원리를 사용한
project-specific adaptation이다. 공식 source file이나 checkpoint를 포함하지 않으며
공식 PEARL implementation의 exact run으로 주장하지 않는다.
고정 16개 probe를 사용한 단일 실행에 정확한 topology-only 순열 등변성이 있다고
주장하지 않는다. 원 논문의 통계적 pooling 성질과 node/probe를 함께 순열할 때의
조건부 일관성은 구분한다. local adaptation의 rho-before-sum 순서는
`GRAPH_POSITIONAL_ENCODING_SPEC.md`에 명시한다.

## This Repository Policy

### Production PE revision (2026-09-15)

`graph_pe_production.py`는 승인된 `pe_production_v1` adaptation이다. SignNet의
shared phi 부호 대칭, node별 signal-axis SetTransformer와 R-PEARL의 K12 random
signal filtering / rho-before-mean을 구현한다. backbone H256/L6, PE H128/L8,
k32/M120은 이 프로젝트에서 승인한 조합이며 논문 전체 설정의 exact reproduction이 아니다.
구형 `graph_positional_encoding.py`의 masked H96/L3, k8/M16 설명과 구분한다.
공식 코드 파일을 복사하거나 vendoring하지 않았다. 참고한 공식 commit 및 구성 차이는
`GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md`의 출처 절에 고정되어 있다.
SignNet의 basis-rotation invariance나 유한 M120 R-PEARL의 exact topology invariance를
주장하지 않는다. 새로운 BasisNet 구현을 포함하지 않는다.

- `01_gois_reimplementation`은 공식 GOIS inference entrypoint 실행 결과로 표시하지 않는다.
- `01_gois_original_repo`는 과거 산출물의 legacy ID일 뿐이며 publication label로 사용하지 않는다.
- `original repo`, `official GOIS implementation`, `exact Table 6 reproduction`이라는 표현을 사용하지 않는다.
- 공식 GOIS 소스, 수정본 또는 파생 패키지를 이 폴더에 추가하거나 공개 저장소로 배포하지 않는다.
- 공식 코드 사용이 필요한 연구 출판, 코드 공개, checkpoint 공개 전에는 GOIS 저자에게 서면 허가를 요청한다.
- 허가를 받지 못한 경우 논문에 공개된 방법을 참고한 로컬 구현과 논문 보고 수치를 명확히 분리한다.
- 특허 또는 출원 여부는 이 문서에서 확인하지 않았다. 출판 또는 상용화 전 별도 확인이 필요하다.

권한 요청에는 최소한 다음 범위를 명시한다.

```text
non-commercial academic research use
execution and modification of the official GOIS source
publication of experimental results and a paper
release of derivative code, patches, configurations, and checkpoints
```

## Artifact Scope

`GOIS_00_03_colab_ablation*.ipynb`는 프로젝트 제출 스냅샷이며 현재 연구 코드의 배포 가능성 판단에서 제외한다. 해당 notebook을 외부에 다시 배포할 때는 notebook 자체의 upstream dependency와 포함 코드를 별도로 재검토해야 한다.
