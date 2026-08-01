---
title: "검색 의미 사전"
type: reference
area: search
status: active
updated: 2026-07-23
tags:
  - l2c
  - docs/search
---

# 검색 의미 사전

## 목적

사용자 표현과 공고 표현을 검토된 개념 키로 연결한다. 직무는 `업무 영역 → 직무군 → 세부 직무` 계층으로 관리하고, 기술은 공고의 `자격요건`, `우대사항`, `주요업무`, `기술스택` 중 어디에서 확인됐는지 함께 저장한다.

업무 영역은 회사의 산업 분류가 아니라 실제 수행하는 업무 기준이다. 제조 회사의 소프트웨어 개발자는 `생산·제조·정비`가 아니라 `IT·데이터`에 속한다.

## 데이터 출처

- `l2c_ko_core`: 6개 업무 영역, SOC 대분류 직무군, 국내 공고 직무 계층과 최신 AI·모바일 기술을 검토해 관리하는 버전 파일
- `onet_30_3`: O*NET 30.3의 세부 직업명과 소프트웨어 기술 어휘
- `l2c_user_curated`: 수집 중 발견된 후보를 사용자가 검토해 승인한 로컬 개념과 별칭

O*NET 세부 직업은 SOC 코드의 대분류를 검토된 로컬 직무군에 연결한다. 사용자가 영역을 고르면 의미 판정 모델에는 그 하위 직무만 전달된다. O*NET 원본 ZIP은 `data/taxonomy/source/onet_30_3_csv.zip`에 보관하며 운영 DB와 원본 ZIP은 Git에 포함하지 않는다.

## 적재와 재색인

```powershell
.\.venv-app\Scripts\python.exe scripts\import_search_taxonomy.py --download-onet
```

원본 ZIP이 이미 있으면 `--download-onet`을 생략한다. 적재가 끝나면 기존 공고도 현재 사전으로 다시 연결한다.

## 실행 계약

- `occupation_scope_required`: 일반 공고 요청에서 업무 범위를 먼저 정해야 하는지 여부
- `occupation_domain_query`: 사용자가 말한 업무 기능 기준 영역 표현
- `occupation_domain_concept_keys`: 사전에서 확정한 업무 영역
- `occupation_query`: 사용자가 말한 직무 표현
- `occupation_concept_keys`: 사전에서 확정한 직무 범위
- `occupation_resolution`: 정확 별칭, 사용자 선택, 사용자 확인 별칭 등 직무 확정 경로
- `skill_queries`: 사용자가 명시한 기술 표현
- `skill_concept_keys`: 사전에서 확정한 기술 조건
- `collection_search_term`: 채용 사이트 검색창에 입력할 표현
- `exact_text_groups`: 사용자가 문자열 자체를 조건으로 요구했을 때만 쓰는 검색 조건

사이트 검색어는 DB 후보 판정에 재사용하지 않는다. 사전에서 확정된 직무·기술 조건은 SQLite가 처리하고, 미등록 직무나 지역·경력처럼 아직 구조화되지 않은 의미 조건만 LLM이 후보를 검토한다.

색인은 `jobs.tech_stack`이나 OCR 원문을 수정하지 않는다. 직무·기술 정규화 결과는 `job_concept_links`에만 저장하므로 같은 사전을 반복 적재해도 원본 필드와 링크 수가 달라지지 않아야 한다.

## 단계형 범위 질문

1. `채용공고 찾아줘`처럼 범위가 없는 요청에는 6개 업무 영역과 직접 입력을 제시한다.
2. 영역을 고르면 그 아래의 직무군을 제시한다. 저장 공고가 0건인 직무군도 새 수집을 위해 숨기지 않는다.
3. 각 선택지에는 현재 DB의 `matching_count`와 사전 하위 직무의 `concept_count`를 별도로 제공한다.
4. 정확 별칭이 있으면 결정론적으로 개념 키를 확정한다.
5. 직접 입력한 표현이 미등록이고 영역이 확정돼 있으면 그 영역 하위 직무만 LLM에 제공한다.
6. LLM의 대응 결과는 확인 질문으로 제시하고 사용자가 고른 경우에만 검토 별칭으로 승격한다.
7. 넓은 직무를 선택한 뒤 현재 DB에 둘 이상의 하위 직무가 있으면 기존 카디널리티 질문으로 더 좁힐 수 있다.

복합 직무 공고는 여러 하위 범위에 포함될 수 있으므로 하위 항목 수의 합이 전체 고유 공고 수보다 클 수 있다.

## 미등록 용어 검토

수집 공고의 명시적 `tech_stack`이 활성 별칭과 일치하지 않으면 후보로만 기록한다. 영역 안에서도 대응할 직무를 찾지 못한 사용자 표현 역시 직무 후보로 기록한다. 둘 다 자동 승격하지 않는다.

```powershell
# 검토 대기 후보
.\.venv-app\Scripts\python.exe scripts\review_search_taxonomy.py list

# 기존 개념의 별칭으로 승인
.\.venv-app\Scripts\python.exe scripts\review_search_taxonomy.py alias 12 l2c:skill:swiftui

# 새 개념으로 승인
.\.venv-app\Scripts\python.exe scripts\review_search_taxonomy.py new 13 "새 기술명" --alias "다른 표기"

# 검색 단위가 아닌 노이즈 거절
.\.venv-app\Scripts\python.exe scripts\review_search_taxonomy.py reject 14 --note "제품 설명 문장"
```

기술 후보를 승인하면 해당 후보가 관찰된 공고만 즉시 다시 색인한다. 의미 질문에서 사용자가 직무를 확인하면 그 표현을 별칭으로 승인하고 기존 공고 전체를 다시 색인한다. 새 사전 버전을 적재할 때 기존 후보가 활성 별칭 하나와 정확히 일치하면 자동으로 해소한다. 둘 이상의 개념과 겹치거나 사전에 없는 후보는 계속 검토 대상으로 남는다. 거절된 표현은 관찰 횟수만 누적되고 활성 검색 사전에 들어가지 않는다.

## 테이블 책임

- `taxonomy_sources`: 사전 출처와 버전
- `search_concepts`: 업무 영역·직무·기술의 대표 개념
- `search_aliases`: 검토된 동일 표현
- `search_concept_relations`: 직무 상하위 관계
- `search_external_mappings`: 내부 개념과 외부 코드 연결
- `job_concept_links`: 공고별 직무·기술, 근거 섹션, 필수·우대 구분
- `search_term_candidates`: 미등록 표현과 검토 결과
- `search_term_candidate_observations`: 후보가 관찰된 공고

`scripts/import_search_taxonomy.py` 실행 결과의 `taxonomy_counts`로 적재 건수를 확인한다. 업무 영역 질문 계약은 실제 Chat API 질문 보완 흐름에서 검증한다.
