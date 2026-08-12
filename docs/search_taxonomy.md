---
title: "검색 의미 사전"
type: reference
area: search
status: active
updated: 2026-08-12
tags:
  - l2c
  - docs/search
---

# 검색 의미 사전

## 목적

사용자와 공고가 같은 직무·기술을 다르게 표현할 때 검토된 개념 키로 연결한다. 운영 데이터는 `data/samples/search_taxonomy_ko.json` 한 파일에서 관리한다.

사전에는 표준 직무·기술명과 별칭이 들어 있다. 공고 원문은 수정하지 않고 정규화 결과와 근거 필드만 `job_concept_links`에 저장한다.

## 적재

```powershell
.\.venv-app\Scripts\python.exe scripts\import_search_taxonomy.py
```

애플리케이션 시작 시 로컬 사전 버전을 확인한다. 버전이 바뀌면 사전을 다시 적재하고 기존 공고의 개념 링크를 갱신한다. 버전이 같으면 아직 색인되지 않은 공고만 처리한다.

## 요청 처리

1. 정확한 별칭은 코드가 개념 키로 변환한다.
2. 직무가 없는 채용공고 요청은 직무명을 한 번 질문한다.
3. 사전에서 확정할 수 없는 표현은 원문을 유지하고 공고 본문 의미 판정으로 넘긴다.
4. 미등록 표현을 DB 사전에 자동 추가하지 않는다. 반복해서 필요한 표현은 JSON 사전을 수정하고 버전을 올린다.

사이트 검색창에 입력하는 `collection_search_term`은 수집 진입용이다. 저장된 공고의 포함 여부는 개념 키, 구조화 필드와 필요한 경우 본문 의미 검토로 판정한다.

## 주요 상태

- `occupation_query`: 사용자가 말한 직무 표현
- `occupation_concept_keys`: 별칭 사전에서 찾은 표준 직무
- `skill_queries`: 사용자가 말한 기술 표현
- `skill_concept_keys`: 확정된 기술 조건
- `collection_search_term`: 웹사이트 검색창에 입력할 표현

## 테이블

- `taxonomy_sources`: 로컬 사전 버전
- `search_concepts`: 직무·기술 개념
- `search_aliases`: 개념별 별칭
- `job_concept_links`: 공고별 직무·기술과 근거 필드

사전 적재 결과는 `scripts/import_search_taxonomy.py`의 `database` 항목에서 확인한다.
