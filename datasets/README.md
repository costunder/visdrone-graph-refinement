# 공통 VisDrone 데이터셋 복원

실험 1·2·3의 데이터 경로 계약은 동일한 `Full/data/visdrone_det_yolo_10class`다.
실험별 차이는 candidate/graph 처리이며 별도의 train/val split을 만들지 않는다.
실험 3은 데이터 복원 여부와 관계없이 실행 차단 상태다.

## 고정 원본과 변환

- 원본 mirror: https://huggingface.co/datasets/Voxel51/VisDrone2019-DET
- revision: `3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2`
- `samples.json` SHA-256: `45777883b6b95fe504dfa59e963955190ad07c2714677bd3dd31b730036f4d82`
- train 6,471장 / val 548장. mirror의 test 1,610장은 사용·다운로드하지 않는다.
- YOLO class 0–9 ↔ COCO category 1–10. ignore_regions와 others 제외.
- train: 기존 `prepare_visdrone_10class_train.py`의 clipping과 소수점 8자리 사용.
- val: 정수 pixel box 복원, 소수점 6자리 YOLO 라벨. COCO는 원본 pixel box 및
  truncation/occlusion을 보존하며 파일명 정렬로 image ID를 1부터 부여한다.
- 기존 데이터와 동일한 줄바꿈까지 라벨 SHA-256으로 검증한다.
- COCO는 key 정렬/compact JSON의 SHA-256으로 전체 구조와 값을 검증한다.

`visdrone.lock.json`의 이미지/라벨 hash는 2026-09-27 로컬 공통 데이터셋을
읽기 전용으로 해시한 것이다. 원본 이미지와 annotation 내용은 이 저장소에 배포하지 않는다.
원본 사용 조건과 mirror의 재배포 조건을 확인하고 다운로드한다.

## 새 다운로드

```bash
python tools/reproduce_dataset.py --download
```

`downloads/visdrone`에 metadata와 train/val 이미지를 다운로드한다. 다운로드된 파일은
각각 SHA-256을 검사하며, 재시도 시 검증된 cache만 재사용한다. 손상된 기존 cache는
덮어쓰지 않고 경로를 보고한다. 해당 파일을 별도로 옮긴 후 다시 시도한다.
동시 요청 기본값은 4이며 `--workers 1`로 낮출 수 있다. 충분한 디스크와 네트워크가 필요하다.
완성 데이터는 이미지 cache를 가리키는 상대 symlink를 사용하므로 cache를 지우면 안 된다.
네트워크는 고정 Hugging Face revision을 이용하며 인증정보를 요구하거나 저장하지 않는다.

다른 출력 위치를 쓰려면 `--output /absolute/path/to/dataset`을 지정한다.
새 폴더를 준비한 뒤 검증을 통과한 경우에만 결과 경로로 이동한다.
기존 output은 빈 폴더라도 거부한다. 출력·cache 경로는 실행마다 전용으로 사용한다.

## 다운로드 없이 기존 원본으로 복원

```bash
python tools/reproduce_dataset.py \
  --samples-json /path/to/VisDrone2019-DET-samples.json \
  --source-images /path/to/train_images \
  --source-images /path/to/val_images \
  --output /path/to/new_dataset
```

offline 모드도 동일 checksum과 파일 목록을 요구한다. 다른 버전으로 자동 fallback하지 않는다.
검증만 필요하면 다음을 사용한다. 파일 변경과 모델 실행은 없다.

```bash
python tools/reproduce_dataset.py --verify-existing /path/to/dataset
```

## 검증 한계

전체 로컬 이미지/라벨 inventory와 hash, 변환된 train/val 라벨 및 COCO GT를 대조한다.
인터넷에서 7,019장을 전부 다시 받는 작업은 업로드 준비 과정에서 실행하지 않았다.
현재 dataset과 일치하지 않는 mirror 이미지가 있으면 다운로드가 실패하며 다른 파일로 대체하지 않는다.
이 도구는 detector weights/cache 또는 graph checkpoint를 복원하지 않고 GPU를 호출하지 않는다.
기존 학습 성능의 exact reproduction까지 보장하는 도구가 아니다.
