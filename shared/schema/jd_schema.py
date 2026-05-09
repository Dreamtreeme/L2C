"""
채용공고 JSON 스키마 정의
VLM 출력 결과를 검증하고 구조화합니다.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class JobPosting(BaseModel):
    """원티드 채용공고 데이터 스키마"""

    company_name: Optional[str] = Field(
        None, description="회사명"
    )
    position: Optional[str] = Field(
        None, description="포지션/직무명"
    )
    job_category: Optional[str] = Field(
        None, description="직군/카테고리"
    )
    experience_level: Optional[str] = Field(
        None, description="경력 요건 (예: '신입', '3년 이상')"
    )
    education: Optional[str] = Field(
        None, description="학력 요건"
    )
    employment_type: Optional[str] = Field(
        None, description="고용 형태 (예: '정규직', '계약직')"
    )
    location: Optional[str] = Field(
        None, description="근무 위치"
    )
    deadline: Optional[str] = Field(
        None, description="마감일"
    )
    tech_stack: Optional[list[str]] = Field(
        default_factory=list, description="기술스택 목록"
    )
    main_tasks: Optional[list[str]] = Field(
        default_factory=list, description="주요 업무"
    )
    requirements: Optional[list[str]] = Field(
        default_factory=list, description="자격 요건"
    )
    preferred: Optional[list[str]] = Field(
        default_factory=list, description="우대 사항"
    )
    benefits: Optional[list[str]] = Field(
        default_factory=list, description="복지 및 혜택"
    )
    salary: Optional[str] = Field(
        None, description="연봉 정보"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "토스",
                "position": "Backend Engineer",
                "job_category": "개발",
                "experience_level": "3년 이상",
                "education": "학력 무관",
                "employment_type": "정규직",
                "location": "서울특별시 강남구",
                "deadline": "2026-06-30",
                "tech_stack": ["Java", "Spring Boot", "Kotlin", "Kafka"],
                "main_tasks": ["결제 시스템 API 개발", "대용량 트래픽 처리"],
                "requirements": ["Java/Kotlin 3년 이상", "REST API 설계 경험"],
                "preferred": ["금융 도메인 경험", "MSA 경험"],
                "benefits": ["스톡옵션", "자기개발비 지원", "유연근무"],
                "salary": "회사 내규에 따름 (협의)"
            }
        }
