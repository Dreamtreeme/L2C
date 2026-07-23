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

AGENT_JSON_PATH = Path("data/samples/agent_extracted_multi_jds_decoded.json")


def test_preprocessor_accepts_korean_benefits_alias():
    job_posting = Preprocessor.process_raw_jd({
        "회사명": "테스트컴퍼니",
        "직무명": "데이터 엔지니어",
        "url": "https://www.wanted.co.kr/wd/123",
        "혜택": ["식대지원", "장비지원"],
    })

    assert job_posting.benefits == ["식대지원", "장비지원"]


def test_preprocessor_preserves_canonical_llm_fields():
    job_posting = Preprocessor.process_raw_jd({
        "company_name": "Acme",
        "position": "iOS Engineer",
        "url": "https://www.wanted.co.kr/wd/123",
        "tech_stack": ["SwiftUI", "UIKit"],
        "requirements": ["Swift experience"],
        "experience_min": 3,
        "experience_max": 99,
        "experience_text": "3+ years",
    })

    assert job_posting.tech_stack == ["SwiftUI", "UIKit"]
    assert job_posting.experience_min == 3
    assert job_posting.experience_max == 99
    assert job_posting.experience_text == "3+ years"


def test_preprocessor_does_not_treat_skill_experience_as_total_experience():
    job_posting = Preprocessor.process_raw_jd({
        "company_name": "Acme",
        "position": "AI Engineer",
        "url": "https://example.com/jobs/skill-years",
        "requirements": ["Python 개발 경험 2년 이상"],
    })

    assert job_posting.experience_min is None
    assert job_posting.experience_max is None
    assert job_posting.experience_text == ""


def test_preprocessor_does_not_invent_years_for_junior_label():
    job_posting = Preprocessor.process_raw_jd({
        "company_name": "Acme",
        "position": "Junior Backend Engineer",
        "url": "https://example.com/jobs/junior",
    })

    assert job_posting.experience_min is None
    assert job_posting.experience_max is None
    assert job_posting.experience_text == ""


def test_preprocessor_preserves_verified_posted_date_and_source_text():
    job_posting = Preprocessor.process_raw_jd({
        "company_name": "Acme",
        "position": "Backend Engineer",
        "url": "https://example.com/jobs/1",
        "posted_at": "2026-07-10",
        "posted_at_text": "2026.07.10 등록",
    })

    assert job_posting.posted_at == "2026-07-10"
    assert job_posting.posted_at_text == "2026.07.10 등록"


def test_job_posting_rejects_relative_date_from_standard_field():
    job_posting = JobPosting(posted_at="3일 전", posted_at_text="3일 전")

    assert job_posting.posted_at is None
    assert job_posting.posted_at_text == "3일 전"


def test_database_migrates_posted_date_columns_without_rebuilding_jobs(tmp_path):
    db_path = tmp_path / "legacy_jobs.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jobs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "url TEXT NOT NULL UNIQUE, "
            "company_name TEXT, "
            "content_hash TEXT, "
            "created_at TEXT NOT NULL"
            ")"
        )

    Database(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(jobs)")}

    assert {"posted_at", "posted_at_text"} <= columns
    assert "idx_jobs_posted_at" in indexes


def test_database_upsert_and_recent_list_include_posted_date(tmp_path):
    db = Database(tmp_path / "jobs.db")
    row_id = db.upsert(
        "https://example.com/jobs/posted",
        {
            "company_name": "Acme",
            "position": "Data Engineer",
            "posted_at": "2026-07-11",
            "posted_at_text": "2026.07.11 게시",
            "content_hash": "posted-date-hash",
        },
    )

    saved = db.get(row_id)
    recent = db.list_recent(limit=1)

    assert saved["posted_at"] == "2026-07-11"
    assert saved["posted_at_text"] == "2026.07.11 게시"
    assert recent[0]["posted_at"] == "2026-07-11"


def test_database_records_only_meaningful_job_versions(tmp_path):
    db = Database(tmp_path / "versioned_jobs.db")
    url = "https://example.com/jobs/versioned"
    first = {
        "company_name": "Acme",
        "position": "Data Engineer",
        "requirements": ["Python"],
        "benefits": ["장비 지원"],
        "raw_ocr_text": "첫 번째 공고 원문",
        "content_hash": "stable-semantic-hash",
        "source_platform": "Example",
    }

    job_id = db.upsert(url, first)
    db.upsert(url, dict(first))

    changed = {**first, "benefits": ["장비 지원", "재택근무"], "raw_ocr_text": "변경된 공고 원문"}
    db.upsert(url, changed)
    versions = db.list_versions(job_id)

    assert [item["version_number"] for item in versions] == [2, 1]
    assert versions[0]["evidence_hash"] != versions[1]["evidence_hash"]
    assert {"benefits", "raw_ocr_text"} <= set(versions[0]["changed_fields"])
    assert versions[0]["content"]["benefits"] == ["장비 지원", "재택근무"]
    saved = db.get(job_id)
    assert saved["evidence_hash"] == versions[0]["evidence_hash"]
    assert saved["raw_json"]["evidence_hash"] == saved["evidence_hash"]


def test_persistence_keeps_collected_ocr_and_screenshot_evidence(monkeypatch, tmp_path):
    import shared.config as cfg
    from agent.application.job_persistence_service import persist_collected_data_with_report

    db_path = tmp_path / "evidence_jobs.db"
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    screenshot_path = str(tmp_path / "detail.png")
    result = persist_collected_data_with_report(
        {
            "jobs": [
                {
                    "company_name": "증거회사",
                    "position": "iOS 개발자",
                    "url": "https://www.jobkorea.co.kr/Recruit/GI_Read/50000001",
                    "requirements": ["Swift"],
                    "raw_ocr_text": "실제 누적 OCR 원문",
                    "_evidence_screenshot_path": screenshot_path,
                }
            ]
        },
        "iOS 개발자",
    )

    saved = Database(db_path).get(result["persisted_items"][0]["job_id"])
    assert saved["raw_ocr_text"] == "실제 누적 OCR 원문"
    assert saved["screenshot_path"] == screenshot_path
    assert saved["source_platform"] == "JobKorea"


def test_persistence_separates_embedded_detail_cards_sharing_search_url(monkeypatch, tmp_path):
    import shared.config as cfg
    from agent.application.job_persistence_service import persist_collected_data_with_report
    from agent.runtime.job_identity import source_card_key

    db_path = tmp_path / "embedded-details.db"
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    search_url = "https://www.saramin.co.kr/zf_user/search?searchword=ml"

    first = {
        "jobs": [
            {
                "company_name": "에너자이",
                "position": "ML Engineer/Researcher",
                "url": search_url,
                "requirements": ["Python"],
                "_source_card_key": source_card_key(
                    search_url,
                    "(주)에너자이",
                    "ML Engineer/Researcher",
                ),
            }
        ]
    }
    second = {
        "jobs": [
            {
                "company_name": "로민",
                "position": "ML 머신러닝 엔지니어 전문연구요원 신규/전직 채용",
                "url": search_url,
                "requirements": ["머신러닝"],
                "_source_card_key": source_card_key(
                    search_url,
                    "(주)로민",
                    "ML(머신러닝) 엔지니어 (전문연구요원 신규/전직 채용)",
                ),
            }
        ]
    }

    first_result = persist_collected_data_with_report(first, "머신러닝 엔지니어")
    second_result = persist_collected_data_with_report(second, "머신러닝 엔지니어")

    assert first_result["created_count"] == 1
    assert second_result["created_count"] == 1
    assert first_result["persisted_items"][0]["job_id"] != second_result["persisted_items"][0]["job_id"]
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT url, company_name FROM jobs ORDER BY id").fetchall()
    assert len(rows) == 2
    assert all("#l2c-card=" in row[0] for row in rows)
    assert {row[1] for row in rows} == {"에너자이", "로민"}


def test_pre_persistence_validation_rejects_missing_identity_and_date_mismatch(monkeypatch, tmp_path):
    import shared.config as cfg
    from agent.application.job_persistence_service import persist_collected_data_with_report

    monkeypatch.setattr(cfg, "DB_PATH", tmp_path / "validated_jobs.db")
    result = persist_collected_data_with_report(
        {
            "jobs": [
                {
                    "company_name": "",
                    "position": "Data Engineer",
                    "url": "https://example.com/jobs/missing-company",
                    "posted_at": "2026-06-01",
                },
                {
                    "company_name": "Acme",
                    "position": "Data Engineer",
                    "url": "https://example.com/jobs/out-of-range",
                    "posted_at": "2026-06-01",
                },
            ]
        },
        "Data Engineer",
        collection_intent={
            "filters": {"posted_from": "2026-07-01", "posted_to": "2026-07-31"}
        },
    )

    assert result["persisted_count"] == 0
    assert result["rejected_count"] == 2
    assert "required_field_missing:company_name" in result["rejected_items"][0]["issues"]
    assert "requested_filter_mismatch:posted_at_before_range" in result["rejected_items"][1]["issues"]


def test_pre_persistence_validation_rejects_job_without_actual_content(monkeypatch, tmp_path):
    import shared.config as cfg
    from agent.application.job_persistence_service import persist_collected_data_with_report

    monkeypatch.setattr(cfg, "DB_PATH", tmp_path / "content_required_jobs.db")
    result = persist_collected_data_with_report(
        {
            "jobs": [
                {
                    "company_name": "중계회사",
                    "position": "백엔드 개발자",
                    "url": "https://example.com/jobs/intermediary",
                }
            ]
        },
        "백엔드 개발자",
        collection_intent={"require_job_content": True},
    )

    assert result["persisted_count"] == 0
    assert result["rejected_count"] == 1
    assert (
        "required_content_missing:main_tasks_or_requirements"
        in result["rejected_items"][0]["issues"]
    )


def test_persistence_report_distinguishes_created_and_updated_jobs(monkeypatch, tmp_path):
    from agent.application.job_persistence_service import persist_collected_data_with_report

    monkeypatch.setattr("shared.config.DB_PATH", tmp_path / "jobs.db")
    payload = {
        "공고목록": [
            {
                "url": "https://example.com/jobs/scope-1",
                "company_name": "범위회사",
                "position": "백엔드 개발자",
            }
        ]
    }

    created = persist_collected_data_with_report(payload, "백엔드 개발자")
    updated = persist_collected_data_with_report(payload, "백엔드 개발자")

    assert created["created_count"] == 1
    assert created["updated_count"] == 0
    assert created["persisted_items"][0]["operation"] == "created"
    assert updated["created_count"] == 0
    assert updated["updated_count"] == 1
    assert updated["persisted_items"][0]["operation"] == "updated"
    assert created["persisted_items"][0]["job_id"] == updated["persisted_items"][0]["job_id"]


def test_persistence_pipeline(tmp_path):
    print("=== [테스트 시작] 전처리 및 DB 적재 파이프라인 검증 ===")
    test_db_path = tmp_path / "test_jobs.db"
    
    # 0. 이전 테스트 DB 제거
    if test_db_path.exists():
        os.remove(test_db_path)
        print(f"이전 테스트 DB 제거 완료: {test_db_path}")

    # 1. 원천 비전 수집 JSON 파일 로드
    assert AGENT_JSON_PATH.exists(), f"테스트 데이터 파일 없음: {AGENT_JSON_PATH}"
    with open(AGENT_JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    jds = raw_data.get("공고목록", [])
    print(f"로드 완료: 총 {len(jds)}건의 원천 비전 공고 데이터")
    assert len(jds) > 0, "공고 목록이 비어있습니다."

    # 2. Database 객체 초기화 (마이그레이션 자동 수행)
    db = Database(test_db_path)
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
    conn = sqlite3.connect(test_db_path)
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
    if test_db_path.exists():
        os.remove(test_db_path)


if __name__ == "__main__":
    test_persistence_pipeline()
