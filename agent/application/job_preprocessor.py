"""
수집한 채용공고를 DB 저장 규격으로 전처리한다.
- OCR 마커 노이즈 제거
- 경력 구조화 추출 (experience_min, experience_max, experience_text)
- 중복 후보 그룹용 content_hash 생성
- Pydantic JobPosting 스키마 유효성 검사 및 정형화
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from shared.schema.jd_schema import JobPosting


class Preprocessor:
    @staticmethod
    def clean_text(text: str | None) -> str:
        """OCR 마커 노이즈([0], [id: 10]) 및 특수문자 클렌징"""
        if not text:
            return ""
        text = str(text)
        # 1. SoM 숫자 라벨 제거 ([0], [1], [id: 2])
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[id:\s*\d+\]", "", text)
        # 2. 앞뒤 공백 및 마커 기호 트리밍
        text = text.strip()
        text = re.sub(r"^[-•*#\s]+", "", text)
        return text.strip()

    @classmethod
    def clean_list(cls, items: list[str] | str | None) -> list[str]:
        """텍스트 리스트 전처리 및 빈 값 제거"""
        if not items:
            return []
        if isinstance(items, str):
            raw = items.strip()
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [raw]
            except json.JSONDecodeError:
                items = [line for line in re.split(r"\r?\n", raw) if line.strip()] or [raw]

        cleaned = []
        for item in items:
            c = cls.clean_text(item)
            if c:
                cleaned.append(c)
        return cleaned

    @staticmethod
    def parse_source_platform(url: str | None) -> str:
        """사이트 레지스트리를 우선 사용해 수집 출처 플랫폼을 분류한다."""
        if not url:
            return "Unknown"
        try:
            from agent.runtime.site_context import site_profile_for_url

            profile = site_profile_for_url(url)
            source_platform = str(profile.source_platform if profile else "").strip()
            if source_platform:
                return source_platform
        except Exception:
            pass
        return "Unknown"

    @classmethod
    def parse_experience(
        cls,
        position: str | None,
        requirements: list[str] | None,
    ) -> tuple[int | None, int | None, str]:
        """명시된 경력 조건만 구조화하고 미확인 값은 비워 둔다."""

        exp_text = ""
        exp_min: int | None = None
        exp_max: int | None = None

        # 1. 직무명 내 괄호 형식 파싱 (예: "iOS 개발자 (3년 이상)")
        if position:
            match = re.search(r"\(([^)]+)\)", position)
            if match:
                inner_text = match.group(1).strip()
                if "년" in inner_text or "신입" in inner_text or "경력" in inner_text:
                    exp_text = inner_text

        # 2. 자격요건에서 전체 경력임이 명시된 표현만 수집
        if not exp_text and requirements:
            for req in requirements:
                if "경력" in req:
                    if len(req) < 50:
                        exp_text = req.strip()
                        break

        # 3. 경력 텍스트를 구조화 수치(min/max)로 파싱
        cleaned_exp = cls.clean_text(exp_text)
        full_text = f"{position or ''} {' '.join(requirements or [])}"
        if not cleaned_exp and "경력 무관" in full_text:
            return 0, None, "경력 무관"
        
        # 패턴 A: "3~7년", "3년~7년", "3년 이상 ~ 7년 이하"
        range_match = re.search(
            r"(\d+)\s*년?\s*(?:이상)?\s*[~\-]\s*(\d+)\s*년?\s*(?:이하)?",
            cleaned_exp,
        )
        if range_match:
            exp_min = int(range_match.group(1))
            exp_max = int(range_match.group(2))
        else:
            # 패턴 B: "3년 이상", "3년차 이상"
            above_match = re.search(r"(\d+)년\s*(이상|차|년차)?", cleaned_exp)
            if above_match and "이하" not in cleaned_exp and "미만" not in cleaned_exp:
                exp_min = int(above_match.group(1))
                exp_max = None
            else:
                # 패턴 C: "3년 이하", "3년 미만"
                below_match = re.search(r"(\d+)년\s*(이하|미만)", cleaned_exp)
                if below_match:
                    exp_min = 0
                    exp_max = int(below_match.group(1))
                elif "신입" in cleaned_exp:
                    exp_min = 0
                    exp_max = 0

        return exp_min, exp_max, exp_text

    @staticmethod
    def generate_content_hash(company_name: str | None, position: str | None, requirements: list[str]) -> str:
        """출처를 보존한 채 중복 후보를 묶기 위한 컨텐츠 해시를 생성한다."""
        def normalize(s: str | None) -> str:
            if not s:
                return ""
            # 공백 제거 및 소문자화
            return re.sub(r"\s+", "", s).lower()

        req_str = "".join(normalize(r) for r in requirements)
        comp_str = normalize(company_name)
        pos_str = normalize(position)

        combined = f"{comp_str}|{pos_str}|{req_str}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @classmethod
    def process_raw_jd(cls, raw_data: dict[str, Any], raw_ocr_text: str | None = None) -> JobPosting:
        """
        비전 수집 raw 공고 데이터를 입력받아 
        정제, 파싱 및 정규화를 거쳐 Pydantic JobPosting 객체로 반환합니다.
        """
        company_name = cls.clean_text(raw_data.get("company_name"))
        position = cls.clean_text(raw_data.get("position"))
        url = str(raw_data.get("url") or "").strip()

        main_tasks = cls.clean_list(raw_data.get("main_tasks"))
        requirements = cls.clean_list(raw_data.get("requirements"))
        preferred = cls.clean_list(raw_data.get("preferred"))
        benefits = cls.clean_list(raw_data.get("benefits"))

        # 플랫폼 분류
        source_platform = cls.parse_source_platform(url)

        # 경력 파싱
        if raw_data.get("experience_min") is not None or raw_data.get("experience_max") is not None or raw_data.get("experience_text"):
            def optional_int(value: Any) -> int | None:
                if value in (None, ""):
                    return None
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

            exp_min = optional_int(raw_data.get("experience_min"))
            exp_max = optional_int(raw_data.get("experience_max"))
            exp_text = cls.clean_text(
                raw_data.get("experience_text") or raw_data.get("experience_level")
            )
        else:
            exp_min, exp_max, exp_text = cls.parse_experience(position, requirements)

        # 기술 사전 색인은 저장 후 SearchTaxonomyService가 섹션별로 수행한다.
        provided_tech_stack = cls.clean_list(raw_data.get("tech_stack"))
        tech_stack = list(dict.fromkeys(provided_tech_stack))

        # 컨텐츠 해시 생성
        content_hash = cls.generate_content_hash(company_name, position, requirements)

        # raw_ocr_text가 없으면 payload 조합으로 복원 시도
        if not raw_ocr_text:
            raw_ocr_text = f"회사명: {company_name}\n직무명: {position}\nURL: {url}\n" \
                           f"주요업무: {' | '.join(main_tasks)}\n자격요건: {' | '.join(requirements)}\n" \
                           f"우대사항: {' | '.join(preferred)}\n혜택정보: {' | '.join(benefits)}"

        # Pydantic 스키마 생성 및 반환
        # 수집되지 않은 필드는 None으로 둡니다.
        # 추측값(예: "서울", "정규직")을 기본값으로 채우면 잘못된 데이터가 DB에 저장됩니다.
        job_category = cls.clean_text(raw_data.get("job_category")) or None
        education = cls.clean_text(raw_data.get("education")) or None
        employment_type = cls.clean_text(raw_data.get("employment_type")) or None
        location = cls.clean_text(raw_data.get("location")) or None
        posted_at = cls.clean_text(raw_data.get("posted_at")) or None
        posted_at_text = cls.clean_text(raw_data.get("posted_at_text")) or posted_at
        deadline = cls.clean_text(raw_data.get("deadline")) or None
        salary = cls.clean_text(raw_data.get("salary")) or None

        from shared.integrity import source_evidence_hash

        evidence_hash = source_evidence_hash(url, raw_data)
        return JobPosting(
            company_name=company_name,
            position=position,
            url=url,
            job_category=job_category,
            experience_level=exp_text,
            education=education,
            employment_type=employment_type,
            location=location,
            posted_at=posted_at,
            posted_at_text=posted_at_text,
            deadline=deadline,
            salary=salary,
            tech_stack=tech_stack,
            main_tasks=main_tasks,
            requirements=requirements,
            preferred=preferred,
            benefits=benefits,
            source_platform=source_platform,
            raw_ocr_text=raw_ocr_text,
            content_hash=content_hash,
            evidence_hash=evidence_hash,
            experience_min=exp_min,
            experience_max=exp_max,
            experience_text=exp_text,
        )
