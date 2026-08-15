---
title: "포트폴리오 평가 실행 규격"
type: reference
area: evaluation
status: active
updated: 2026-08-15
tags:
  - l2c
  - docs/evaluation
---

# 포트폴리오 평가 실행 규격

최종 수치는 동일 커밋, 모델, 창 크기와 DB 초기 상태에서 생성한다. 개발 중
E2E 로그는 원인 분석 자료로 유지하고 성공률 분모에는 포함하지 않는다.

## 준비

```powershell
.\setup.cmd
.\scripts\test.cmd agent\tests -q
git status --short
```

평가 행렬은 `require_clean_worktree=true`다. 변경 파일이 있으면 실제 실행을
중단하고 `--dry-run`에서만 예정 명령과 현재 계약을 출력한다.

각 matrix summary에는 다음 값이 들어간다.

- Git 커밋과 clean 여부
- Python·Windows 버전
- 지휘자·화면 추론·경량 모델
- 화면 크기, OCR GPU와 입력 크기
- 실행별 원본 summary 경로
- 실행시간, OCR, 추론, 토큰과 비용

## 수집 성공률

```powershell
.\.venv-app\Scripts\python.exe -m benchmark.run_regression_matrix `
  --matrix benchmark\portfolio_collection_matrix.json
```

행렬은 5개 사이트, 사이트별 검색어 2개, 조건별 반복 3회로 총 30회를
실행한다. 각 실행은 격리 DB를 사용한다.

자동 summary 생성 뒤 `benchmark/manual_evaluation_template.json` 형식으로
사람 판정을 기록한다. 공고별 판정 항목은 다음과 같다.

- 검색어와 실제 업무의 의미 일치
- 회사명과 직무명 원문 일치
- 주요 업무와 자격요건의 오류·누락·다른 섹션 혼입
- 요청 개수 충족 또는 실제 결과 부족 보고
- 계약 범위 밖 행동 발생 여부

```powershell
.\.venv-app\Scripts\python.exe -m benchmark.manual_evaluation `
  logs\portfolio\manual_evaluation.json `
  --output logs\portfolio\strict_quality.json
```

자동 저장 계약과 사람 판정을 모두 통과해야 `success`다. 일부 유효 공고만
확보하면 `partial`, 잘못된 공고·필드 또는 범위 밖 행동은 `failure`다.
자동 실행의 `resolved_count`만큼 서로 다른 상세 URL을 판정표에 넣고,
새로 저장한 모든 URL을 포함해야 사람 판정 범위가 통과한다.

고정 공고를 지정한 행렬은 `expected_source_urls`를 사용한다. 기대 URL의 호스트,
경로와 쿼리 식별자를 저장 URL과 일대일로 비교한다. 누락된 공고나 기대 목록에
없는 추가 저장 URL이 있으면 실행 건수를 채웠더라도 해당 실행을 실패로 판정한다.

## 경험 기반 탐색

```powershell
.\.venv-app\Scripts\python.exe -m benchmark.run_regression_matrix `
  --matrix benchmark\portfolio_reflex_matrix.json
```

대표 3개 사이트에서 자율 탐색과 경험 기반 탐색을 각 3회 실행한다. 같은
반복 번호에서 두 실행의 품질과 모드 계약이 모두 통과한 경우만 성능을
짝 비교한다. `expected_source_urls`가 있는 시나리오는 두 실행 모두 고정 대상
계약을 통과해야 한다.

matrix summary의 `mode_pair_efficiency`에는 다음 값이 들어간다.

- 실행시간 절감
- 화면 추론 호출 절감
- 토큰과 API 비용 절감
- Critic 승격 비용
- 비용 기준 손익분기 반복 횟수

## 신규 사이트 적용 공수

`benchmark/site_adaptation_template.json`에 미지원 사이트 2개의 Classic과
Vision 기록을 각각 작성한다.

```powershell
.\.venv-app\Scripts\python.exe -m benchmark.site_adaptation_eval `
  logs\portfolio\site_adaptation.json `
  --output logs\portfolio\site_adaptation_report.json
```

Vision에서 공통 Python 코드를 수정했다면 `common_runtime_code_lines`에
기록한다. 값이 0이고 사이트 전용 Python 코드도 0인 경우만 프로필 적용으로
분류한다. Classic과 Vision이 각각 3회 모두 성공하고 모든 실행시간을
기록한 경우에만 `comparison_valid=true`가 된다.

## 자원 측정

```powershell
powershell -File scripts\measure_runtime_resources.ps1 `
  -OutputPath logs\portfolio\runtime_resources.json
```

명령 없이 실행하면 설치 용량과 기준 RAM·VRAM을 기록한다. E2E 명령을
`-Command`, `-CommandArguments`로 전달하면 실행 중 peak 증가량을 함께
측정한다. 런타임 기준은 [[runtime_compatibility]]에 기록한다.
