# 테스트 실행 흐름

제품 요청 한 건은 다음 순서로 검증한다.

1. `test_backend_boundaries.py`: FastAPI가 질문과 식별자를 바꾸지 않고 조사 그래프에 전달하는지 확인한다.
2. `test_investigation_graph.py`: 요청 이해, 확인 질문, DB 근거 검사, 수집 계획, 원문 수집, 후처리, 저장, 재검사, 답변 순서를 검증한다.
3. `test_investigation_collection_flow.py`: 작업자 관찰과 DB 저장 결과가 하나의 `CollectionResult`로 합쳐지는 규칙을 검증한다.
4. `test_collection_postprocessing.py`, `test_collection_storage.py`, `test_collection_experience.py`: 원문 구조화, 공고 저장, 작업자 제출물과 레시피 후보 기록을 각각 확인한다.
5. `test_db_persistence.py`: 공고 UPSERT, 근거 저장과 버전 기록을 확인한다.
6. `test_worker_graph_boundaries.py` 이하 작업자 테스트: 캡처, OCR, Reflex, 추론, 물리 행동과 화면 전환을 검증한다.

전체 단위·통합 테스트는 다음 명령으로 실행한다.

```powershell
.\scripts\test.cmd agent\tests -q
```

## 유지 기준

기본 테스트는 다음 제품 계약만 검증한다.

- 사용자 요청의 질문 보완, DB 조회, 웹 수집 경로
- 저장 데이터와 답변 출처의 무결성
- 화면 캡처, OCR, 물리 입력의 안전 경계
- ROI Reflex 기록, 검증, 승격, 실패 폴백
- 사이트 등록 정보와 검색 의미 사전
- 자율 탐색/경험 기반 탐색 실행기의 입력·요약·품질 판정 계약

다음 항목은 기본 단위 테스트에 추가하지 않는다.

- 비공개 헬퍼의 구현 순서. 벤치마크 계산식처럼 결과 자체가 계약인 함수는 제외한다.
- 화면 탐색 품질을 위한 프롬프트의 정확한 문장. 안전 경계 문구는 중앙 계약에서 검증한다.
- 로그 메시지와 보고서 서식
- 이미 상위 계약 테스트가 검증하는 동일 분기
- 삭제된 레거시 스키마의 반복 마이그레이션
- 외부 모델의 응답 속도나 실제 화면 성공 여부

현재 `agent/tests`에는 외부 모델이나 실제 브라우저를 실행하는 테스트가 없다.
`external`, `e2e` 표식은 해당 테스트가 추가될 때 기본 실행에서 제외하기 위해
예약해 두며, 실제 화면 검증은 `benchmark` 명령으로 실행한다.
