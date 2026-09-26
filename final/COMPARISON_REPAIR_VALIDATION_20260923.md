# 비교 복구 검증 기록 — 2026-09-23

검증 범위: 공개 번호, 저장 config 비교, 실행 차단, 기존 CPU 모델 계약.
새 학습 결과나 새 architecture의 승인 기록이 아니다.

## 실행한 검사

- GPU를 숨기고 Python `-B -m unittest -v test_comparison_repair test_graph_pe_production_contract test_graph_pe_production_runner` 실행: **21개 통과**.
- 추가 7개 검사는 연속 공개 번호/no-PE 별도 번호 제외, PE ID 매핑, config 변경 거부,
  구형·누락 config 거부, 완료 기록과 비교 완료 구분, production 실행 보류,
  실험3 shell/main/run_variant/train_epoch 차단을 검증한다.
- 기존 모델 CPU 계약 10개와 실행기 CPU 계약 4개도 통과했다.
- `check_pe_comparison.py`에 실제 기존 실험1/02와 완료된 SignNet run_config를 입력:
  `config_comparable=false`, 종료 코드 **2**. 누락/형식/값 불일치 진단은 총 38개이며,
  이 개수는 독립적인 실험 요인 차이 38개라는 뜻이 아니다.
- 수정한 shell entrypoint 4개 `bash -n`: 종료 코드 **0**.

## 보존 및 미실행

기존 checkpoint, detector/graph cache, run_config, prediction, metric은 이동·삭제·수정하지 않았다.
새 GPU 작업, detector inference, COCO evaluation, 데이터셋 학습은 실행하지 않았다.
CPU optimizer 테스트는 synthetic fixture만 사용했다.

비교 config 검사는 실제 후보 tensor, gradient, 학습 완료, 원본 코드 및 checkpoint
checksum의 전체 감사를 대신하지 않는다. 기존 SignNet run의 재사용은 자동 승인하지 않는다.
실행기 테스트 코드가 변경되었으므로 과거 run의 code SHA를 새 코드로 덮어쓰지 않는다.

남은 작업: matched baseline revision의 실행 전 검증/승인, 실험2 production 구현 및
actual-view cache 확보 승인, 실험3의 처리 순서·전달 경계 확인과 재설계.
