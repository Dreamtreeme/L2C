# 테스트 실행 흐름

제품 요청 한 건은 다음 순서로 검증한다.

1. `test_backend_boundaries.py`: FastAPI가 질문과 식별자를 바꾸지 않고 조사 그래프에 전달하는지 확인한다.
2. `test_investigation_graph.py`: 요청 이해, 확인 질문, DB 근거 검사, 수집 계획, 수집, 저장, 재검사, 답변 순서를 검증한다.
3. `test_investigation_collection_flow.py`: 작업자 관찰과 DB 저장 결과가 하나의 `CollectionResult`로 합쳐지는 규칙을 검증한다.
4. `test_collection_persistence.py`, `test_db_persistence.py`: 정제된 공고, 작업자 제출물과 레시피 후보가 저장되는지 확인한다.
5. `test_worker_graph_boundaries.py` 이하 작업자 테스트: 캡처, OCR, Reflex, 추론, 물리 행동과 화면 전환을 검증한다.

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
- 자율 탐색/경험 기반 탐색 E2E 실행 계약과 품질 지표

다음 항목은 기본 단위 테스트에 추가하지 않는다.

- 비공개 헬퍼의 구현 순서
- 프롬프트의 정확한 문장
- 로그 메시지와 보고서 서식
- 이미 상위 계약 테스트가 검증하는 동일 분기
- 삭제된 레거시 스키마의 반복 마이그레이션
- 외부 모델의 응답 속도나 실제 화면 성공 여부

외부 모델과 실제 브라우저 검증은 `external`, `e2e` 표식 또는
`benchmark` 실행으로 분리한다.
