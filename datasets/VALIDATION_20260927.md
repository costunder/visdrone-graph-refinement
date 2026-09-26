# 데이터 재현·코드 업로드 검증 기록

검증일: 2026-09-27. 이 기록은 모델 학습·평가 결과가 아니다.

## 수행한 검증

- 기존 원본 `samples.json`의 SHA-256과 고정 mirror revision의 LFS SHA-256 일치.
- 원본 train/val 이미지 7,019장과 라벨 7,019개를 읽기 전용으로 해시해 lock 생성.
- 임시 디렉터리에서 전체 offline 복원: train 6,471장 / val 548장.
- 모든 복원 라벨의 파일 내용 SHA-256이 기존 라벨과 일치.
- 모든 복원 이미지의 파일 내용 SHA-256이 기존 이미지와 일치.
- COCO validation GT 전체 구조와 값 일치: images 548개, annotations 38,759개.
- 고정 원본에서 train 1장, val 1장 HTTP 다운로드 후 기존 SHA-256과 일치.
- 다운로드·변환 CPU 단위 테스트 9개 통과: clipping/precision, COCO ID,
  무효 class/box/path 및 split 중복 거부, cache 검증, mock download/checksum 실패,
  기존 output 보존, JSON semantic hash.
- 업로드 후보 Python 50개 AST parsing, shell 9개 `bash -n` 통과.
- 업로드 후보 파일의 일반적인 GitHub/Hugging Face/AWS key, private key 및
  하드코딩 credential 패턴 검사에서 일치 없음. 모든 비밀정보를 탐지한다는 보장은 아니다.

## 출처와 배포 범위

`final/scripts/my_package/visdrone_categories.py`는 참조 트리에도 동일 복사본이 있으나,
참조 저장소 HEAD `132c81dda39b0b83c4a639846ef22df89dfcea93`의 추적 파일이 아니다.
로컬 Git history에도 해당 경로가 없고 upstream 현재 contents 조회 역시 해당 파일이 없다.
따라서 로컬 클래스 매핑 도우미로 포함하며, 이것이 전체 프로젝트의 법률 검토를 대신하지는 않는다.
공식 GOIS 트리 및 제출 notebook은 제외한다. `THIRD_PARTY_NOTICES.md`의 제한은 유지한다.

## 변경하지 않은 항목과 한계

기존 연구 runner·model·manifest·설정·실행 가드 및 데이터 원본은 변경하지 않았다.
추가한 것은 Git 제외 규칙, 환경 버전 목록, 저장소 안내, 데이터 다운로드/변환 도구와
그 테스트 및 checksum lock이다. 새 GPU 학습·detector inference·COCO 평가는 실행하지 않았다.
7,019장 전체 인터넷 재다운로드, 새 Python 환경 설치, 새 환경에서 전체 연구 실행은 수행하지 않았다.
체크포인트와 cache는 업로드하지 않으며 데이터 복원 성공을 성능 재현 성공으로 취급하지 않는다.
