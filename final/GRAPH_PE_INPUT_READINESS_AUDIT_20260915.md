# Graph PE 입력·GPU 준비 상태 감사

확인일: 2026-09-15. 읽기 전용 입력 조사이며 새 모델 구현·GPU 학습·detector
inference·COCO 평가 또는 실행 권한 변경을 수행하지 않았다.

## 자원

- RTX 3090: VRAM 24,576 MiB, 확인 당시 107 MiB 사용, GPU utilization 0%.
- 시스템 RAM 약 16 GiB, 확인 당시 available 약 11 GiB.
- workspace filesystem 여유 약 22 GiB. 여러 run의 전체 checkpoint를 보존할 경우
  실행 전에 저장 공간 계획을 검증해야 한다. 외부 저장소를 자동으로 사용하지 않는다.

## 실험 1: 실제 view cache 발견

기존 PE runner가 고정한 shared `final/runs/.../common_cache`에는 실제 tile 정보가
없는 legacy cache가 있었다. 반면 아래 실험 1 로컬 directory에는 실제 view
metadata를 담은 cache가 존재한다.

`final/experiment_1/runs/table6_yolo11_10class_extra_ablation/common_cache`

현재 10-class mapping, detector/data 경로·파일 signature, slice/confidence 설정으로
기존 `prediction_cache_path`와 `fine_prediction_cache_path`를 계산하여 아래 경로가
정확히 존재함을 확인했다. 각 image ID가 존재하고 모든 candidate에
`_view_type`, `_view_id`, `_view_bbox` key가 있으며 값이 null이 아님을 전수 검사했다.
geometry의 모든 수치, crop membership 및 전체 graph tensor conformance까지
검증했다는 뜻은 아니다.

| Split/source | Images | Candidates | View metadata 누락 | 파일 |
|---|---:|---:|---:|---|
| train/coarse | 6,471 | 642,788 | 0 | `train_coarse_conf0.250_s640_o0.20_n6471_d90ceabde7f1.json` |
| train/fine | 6,471 | 735,531 | 0 | `train_fine_conf0.250_s256_o0.20_n6471_e1f27eaddb3d.json` |
| val/coarse | 548 | 77,927 | 0 | `eval_coarse_conf0.250_s640_o0.20_n548_bd594f17ddeb.json` |
| val/fine | 548 | 78,836 | 0 | `eval_fine_conf0.250_s256_o0.20_n548_68b19c9c7f53.json` |

train은 총 1,378,319 candidates, val은 156,763 candidates다. 후보 병합·제거 또는
detector 재추론은 하지 않았다. 위 cache의 SHA256:

```text
train/coarse 839f0d7d136650221054a697f1e2b6ab257092c5e6e740959c119d8e28b6bc4c
train/fine   43749b5538ebc88869571eac3efd9bedb6e98cf6ef3da236c5560b67e748cbb8
val/coarse   7068ab981838222d587ed26cebba89d4e65a75e559d9894148709406948c4a19
val/fine     02d74ea2a66ba0f3e263529d98416c547b29800e9f57fd964621bcc5ad8273b2
```

새 study에서 사용하려면 새 설계 승인과 함께 입력 경로를 명시적으로 등록해야 한다.
shared directory에 파일을 덮어쓰거나 alias를 추가하지 않았다.

## 실험 2: 저신뢰도 cache의 실제 view metadata 누락

shared common_cache의 아래 validation 파일을 전수 확인했다.

| 파일 | Candidates | View metadata 누락 |
|---|---:|---:|
| `eval_coarse_conf0.050_s640_o0.20_n548_ad5f0265a779.json` | 114,284 | 114,284 |
| `eval_fine_conf0.050_s256_o0.20_n548_78d88c3494c2.json` | 150,265 | 150,265 |

현재 `sparse_ppr_sage.py`는 누락된 view bbox에 full-image geometry, view ID에
legacy source ID를 사용하는 fallback을 갖는다. 이 값은 실제 crop/view의 복원이
아니므로 production 승인안의 실제 view contract를 충족한 것으로 처리할 수 없다.
실험 1 로컬 cache의 0.05 사본도 조사했으며 별도의 실제 view cache를 찾지 못했다.
실험 2를 진행하려면 provenance를 갖는 기존 입력을 추가로 제공하거나, 별도 cache
재생성을 승인받아야 한다. 이 감사에서 재생성·후보 교체·입력 합성은 하지 않았다.

## 과거 02 checkpoint와 승인 상태

- `02_gnn_no_cluster/checkpoints/epoch_116.pt`를 CPU에서 `weights_only=True`로
  읽었다. 입력 weight shape은 `[96,37]`, 저장 parameter element 수는 449,187이다.
- 기존 run config의 graph source SHA는 현재 source와 같지만 runner SHA는 다르다.
  exact source snapshot 부재 문제는 그대로 남으며 과거 성능을 승격하지 않는다.
- 새 설계·학습·GPU 검증·평가 승인을 영구 기록하려던 변경은 자동 안전 검토에서
  사용자 요청의 승인 범위가 모호하다는 이유로 거부되었고 적용되지 않았다.
  기존 guard와 proposal의 승인 대기 상태를 변경하지 않았다.
- 사용자에게 구현/GPU 검증/학습/평가 범위, 실험 2 cache 재생성 범위를 구분하여
  확인 요청했다. 명시적인 응답 전 실행 가드를 해제하거나 우회하지 않는다.
