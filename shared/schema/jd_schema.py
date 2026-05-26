"""
채용공고 JSON 스키마 정의
VLM 출력 결과를 검증하고 구조화합니다.
"""

from __future__ import annotations
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class JobPosting(BaseModel):
    """원티드 채용공고 데이터 스키마"""

    company_name: Optional[str] = Field(None, description="회사명")
    position: Optional[str] = Field(None, description="포지션/직무명")
    url: Optional[str] = Field(None, description="공고 상세 URL")
    job_category: Optional[str] = Field(None, description="직군/카테고리")
    experience_level: Optional[str] = Field(None, description="경력 요건")
    education: Optional[str] = Field(None, description="학력 요건")
    employment_type: Optional[str] = Field(None, description="고용 형태")
    location: Optional[str] = Field(None, description="근무 위치")
    deadline: Optional[str] = Field(None, description="마감일")
    tech_stack: Optional[list[str]] = Field(default_factory=list, description="기술스택 목록")
    main_tasks: Optional[list[str]] = Field(default_factory=list, description="주요 업무")
    requirements: Optional[list[str]] = Field(default_factory=list, description="자격 요건")
    preferred: Optional[list[str]] = Field(default_factory=list, description="우대 사항")
    benefits: Optional[list[str]] = Field(default_factory=list, description="복지 및 혜택")
    salary: Optional[str] = Field(None, description="연봉 정보")
    source_platform: Optional[str] = Field(None, description="수집 출처 플랫폼")
    raw_ocr_text: Optional[str] = Field(None, description="전처리 전 원천 OCR/SoM 텍스트 전체 백업")
    content_hash: Optional[str] = Field(None, description="회사명+직무명+자격요건 해시값 (SHA256)")
    experience_min: Optional[int] = Field(0, description="최소 경력 년수")
    experience_max: Optional[int] = Field(99, description="최대 경력 년수")
    experience_text: Optional[str] = Field("경력 무관", description="경력 정보 원문")

    # LLM이 list/string 타입을 혼동하는 케이스를 스키마 레벨에서 흡수합니다.
    # classic(llm_engine) + vision(preprocessor) 두 파이프라인 공통 적용.

    @field_validator("tech_stack", "main_tasks", "requirements", "preferred", "benefits", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        """string으로 온 list 필드를 쉼표/세미콜론으로 split합니다."""
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in re.split(r"[,;]\s*", v) if s.strip()]
        return v

    @field_validator(
        "company_name", "position", "job_category", "experience_level",
        "education", "employment_type", "location", "deadline", "salary",
        mode="before",
    )
    @classmethod
    def coerce_to_str(cls, v):
        """list로 온 string 필드를 ', '.join합니다."""
        if isinstance(v, list):
            joined = ", ".join(str(x).strip() for x in v if str(x).strip())
            return joined or None
        return v
