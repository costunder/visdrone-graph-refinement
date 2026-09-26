# VisDrone Graph Refinement

실험 1·2·3의 연구 코드, 설계 계약, CPU 테스트 및 공통 데이터셋 재현 도구를 보관한다.
공식 GOIS 구현 저장소가 아니며, 논문 수치의 exact reproduction이나 SOTA를 주장하지 않는다.

## 구성

- `final/experiment_1/`: 동일 후보 pool의 graph refinement 및 PE 비교
- `final/experiment_2/`: low-confidence candidate rescue 및 sparse/PPR graph
- `final/experiment_3/`: 기존 결합 구현 보존. **단계 연결 미확정으로 실행 차단**
- `final/manifests/current_experiments.json`: 공개 실험 번호의 기준
- `tools/reproduce_dataset.py`: 세 실험이 공유하는 VisDrone train/val 다운로드·변환·검증
- `datasets/visdrone.lock.json`: 고정 원본 revision과 7,019장 이미지/라벨 SHA-256

실험 상태와 승인 범위는 `final/COMPARISON_REPAIR_SPEC.md`,
`final/MATCHED_PE_EXECUTION_20260923.md` 및 각 manifest를 함께 확인한다.
기존 문서의 날짜별 상태는 역사 기록이며 최신 학습 상태나 완료 증거가 아니다.
체크포인트와 원본 평가 산출물이 포함되지 않으므로 이 코드 스냅샷만으로 성능을 검증할 수 없다.

## 환경

기록된 로컬 환경: Linux, Python 3.13.9. 라이브러리 버전은 `requirements.txt`에 기록했다.
이 버전 목록은 관측 환경이며 새 환경에서 전체 학습 재현까지 검증했다는 의미는 아니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 환경에 맞는 PyTorch wheel/driver 호환성을 별도로 확인한다.
데이터 다운로드 도구와 해당 CPU 테스트는 Python 표준 라이브러리만 사용한다.

## 데이터셋 재현

데이터 사용 조건을 확인한 후 저장소 루트에서 실행한다. 원본은 Git에 올리지 않는다.

```bash
python tools/reproduce_dataset.py --download
python tools/reproduce_dataset.py --verify-existing Full/data/visdrone_det_yolo_10class
```

이 명령은 train 6,471장과 val 548장, YOLO 10-class 라벨, COCO validation GT,
현재 경로에 맞는 `dataset.yaml`을 만든다. 상세 절차와 검증 범위는
[데이터 재현 문서](datasets/README.md)를 따른다.
이미 있는 데이터는 덮어쓰지 않는다. 학습·detector inference·COCO 평가는 실행하지 않는다.

## 연구 실행 전 필요한 별도 산출물

데이터셋 복원과 실험 결과 재현은 다르다. 다음 파일은 용량·배포 범위 때문에 포함하지 않았다.

- 학습된 공통 detector: `final/runs/table6_yolo11_10class_extra_ablation/detector/weights/best.pt`
- 원본 detection/graph cache, graph checkpoint, run config·평가 산출물
- `GOIS/`, `Full/`, `CP/` 참조 트리, 노트북 제출 스냅샷, 생성 보고서·archive

현재 production PE 입력 코드는 승인된 detector cache의 checksum을 요구한다.
위 산출물 없이 바로 학습할 수 없으며, 다운로드 도구가 checkpoint/cache를 대체 생성하지 않는다.
cache 생성과 detector 재학습은 별도 승인·provenance 확인이 필요하다.
기존 shell entrypoint에는 로컬 Python 기본 경로가 남아 있다. 지원되는 script는
`PYTHON_BIN`으로 지정하고, 고정 경로 entrypoint는 대응 Python runner를 직접 사용한다.
업로드 작업에서 연구 runner나 architecture, 실행 가드를 수정하지 않았다.

```bash
# 표준 라이브러리만 사용하는 새 도구 테스트
python -B -m unittest discover -s tools -p 'test_*.py' -v
```

설치/데이터 준비는 실험 실행 승인이 아니다. `AGENTS.md`와 각 실행기의 fail-closed 가드를 유지한다.
외부 코드 및 인용 범위는 [THIRD_PARTY_NOTICES](final/THIRD_PARTY_NOTICES.md)를 따른다.
저장소를 공개 전환하기 전 배포 범위를 다시 검토해야 한다.
