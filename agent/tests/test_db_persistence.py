"""
데이터 전처리 및 DB 적재 신뢰성 검증 테스트
- 비전 수집 JSON 로드 및 Preprocessor 정제 검증
- 신규 MVP 스키마 필드 적재 검증 (source_platform, content_hash 등)
- 중복 적재 방지(Deduplication) 검증
"""

import json
import os
import sqlite3
from pathlib import Path
from agent.utils.preprocessor import Preprocessor
from shared.db.database import Database
from shared.schema.jd_schema import JobPosting

# 테스트 DB 경로 지정
TEST_DB_PATH = Path("data/test_jobs.db")
AGENT_JSON_PATH = Path("data/agent_extracted_multi_jds_decoded.json")


def test_persistence_pipeline():
    print("=== [테스트 시작] 전처리 및 DB 적재 파이프라인 검증 ===")
    
    # 0. 이전 테스트 DB 제거
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)
        print(f"이전 테스트 DB 제거 완료: {TEST_DB_PATH}")

    # 1. 원천 비전 수집 JSON 파일 로드
    assert AGENT_JSON_PATH.exists(), f"테스트 데이터 파일 없음: {AGENT_JSON_PATH}"
    with open(AGENT_JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    jds = raw_data.get("공고목록", [])
    print(f"로드 완료: 총 {len(jds)}건의 원천 비전 공고 데이터")
    assert len(jds) > 0, "공고 목록이 비어있습니다."

    # 2. Database 객체 초기화 (마이그레이션 자동 수행)
    db = Database(TEST_DB_PATH)
    print("Database 초기화 및 스키마 검증 완료")

    # 3. 각 공고 데이터 전처리 및 적재 수행
    for idx, raw_jd in enumerate(jds):
        print(f"\n--- [{idx+1}/{len(jds)}] {raw_jd.get('회사명')} - {raw_jd.get('직무명')} ---")
        
        # [A] 전처리 수행
        # 테스트용 SoM 마커 노이즈를 자격요건 텍스트에 강제 삽입하여 제거 기능 확인
        if raw_jd.get("자격요건"):
            raw_jd["자격요건"].append("[0] RxSwift 실무 능숙자")
            raw_jd["자격요건"].append("[id: 102] SwiftUI 기반 아키텍처 리팩토링 경험")
        
        job_posting = Preprocessor.process_raw_jd(raw_jd)
        
        # [B] Pydantic 검증 상태 확인
        assert isinstance(job_posting, JobPosting)
        print(f"✓ Pydantic 검증 완료")
        print(f"  - 소스 플랫폼: {job_posting.source_platform}")
        print(f"  - 최소 경력: {job_posting.experience_min}년, 최대: {job_posting.experience_max}년 (원문: '{job_posting.experience_text}')")
        print(f"  - 추출된 기술 스택: {job_posting.tech_stack}")
        print(f"  - 컨텐츠 해시: {job_posting.content_hash}")

        # [C] 마커 제거 확인
        for req in job_posting.requirements:
            assert "[0]" not in req, f"마커 제거 실패: {req}"
            assert "[id: 102]" not in req, f"마커 제거 실패: {req}"
        print(f"✓ SoM 마커 제거 성공")

        # [D] 기술 스택 동의어 치환 확인
        if "RxSwift" in job_posting.tech_stack:
            assert "rxswift" not in job_posting.tech_stack
            assert "rx swift" not in job_posting.tech_stack
        if "SwiftUI" in job_posting.tech_stack:
            assert "swift ui" not in job_posting.tech_stack
            assert "swiftui" not in job_posting.tech_stack
        print(f"✓ 기술 스택 동의어 정규화 성공")

        # [E] DB UPSERT 실행
        job_dict = job_posting.model_dump()
        row_id = db.upsert(job_posting.url, job_dict)
        assert row_id > 0, "DB 적재 오류: row_id가 0 이하입니다."
        print(f"✓ DB 적재(UPSERT) 성공: row_id={row_id}")

        # [F] DB 상세 조회 및 데이터 정합성 검사
        saved = db.get(row_id)
        assert saved is not None
        assert saved["company_name"] == job_posting.company_name
        assert saved["position"] == job_posting.position
        assert saved["source_platform"] == job_posting.source_platform
        assert saved["content_hash"] == job_posting.content_hash
        assert saved["experience_min"] == job_posting.experience_min
        assert saved["experience_max"] == job_posting.experience_max
        assert saved["experience_text"] == job_posting.experience_text
        assert isinstance(saved["tech_stack"], list)
        print(f"✓ DB 조회 및 매핑 필드 정합성 일치율 100%")

    # 4. 중복 적재 방지(Deduplication) 검증
    print("\n--- [중복 적재 방지 및 UPSERT 검증] ---")
    first_jd = jds[0]
    jp_1 = Preprocessor.process_raw_jd(first_jd)
    
    # 동일 URL로 한번 더 UPSERT 수행
    id_1 = db.upsert(jp_1.url, jp_1.model_dump())
    id_2 = db.upsert(jp_1.url, jp_1.model_dump())
    assert id_1 == id_2, "동일 URL 적재 시 새로운 row가 생성되었습니다. (UPSERT 오작동)"
    print("✓ URL 중복 충돌 시 정상 UPDATE 확인")

    # URL은 다르지만 content_hash가 동일한 경우 중복 차단 검증
    jp_2 = Preprocessor.process_raw_jd(first_jd)
    jp_2.url = "https://www.wanted.co.kr/wd/9999999999" # 임의의 새 URL
    assert jp_1.content_hash == jp_2.content_hash, "동일 데이터에 대한 hash가 불일치합니다."

    id_3 = db.upsert(jp_2.url, jp_2.model_dump())
    assert id_1 == id_3, f"동일 content_hash인데 신규 row가 적재되었습니다. (Deduplication 실패) id_1={id_1}, id_3={id_3}"
    print("✓ content_hash 중복 충돌 시 정상 UPDATE 및 중복 차단 확인")

    # DB의 최종 레코드 수 확인
    conn = sqlite3.connect(TEST_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM jobs")
        cnt = cursor.fetchone()["cnt"]
        print(f"\n✓ 최종 DB 레코드 수: {cnt}건 (정적 검증 기대치: {len(jds)}건)")
        assert cnt == len(jds), f"최종 적재 개수 오류: 기대={len(jds)}건, 실제={cnt}건"
    finally:
        conn.close()

    print("\n=== [테스트 성공] 모든 전처리 및 DB 영속화 적재 파이프라인의 안전성을 검증 완료했습니다! ===")

    # 5. 테스트 DB 정리
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)


if __name__ == "__main__":
    test_persistence_pipeline()
