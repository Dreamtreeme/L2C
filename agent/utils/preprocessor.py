"""
비전 에이전트 수집 채용공고 데이터 전처리 유틸리티
- OCR 마커 노이즈 제거
- 경력 구조화 추출 (experience_min, experience_max, experience_text)
- 기술 스택 동의어 정규화 (Synonym Normalization)
- 중복 방지용 content_hash 생성
- Pydantic JobPosting 스키마 유효성 검사 및 정형화
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse
from typing import Any
from shared.schema.jd_schema import JobPosting

# 1. 기술 스택 동의어 사전 정의
TECH_SYNONYMS = {
    "swift ui": "SwiftUI",
    "swiftui": "SwiftUI",
    "rxswift": "RxSwift",
    "rx swift": "RxSwift",
    "objective c": "Objective-C",
    "objective-c": "Objective-C",
    "objc": "Objective-C",
    "cicd": "CI/CD",
    "ci/cd": "CI/CD",
    "ci cd": "CI/CD",
    "reactorkit": "ReactorKit",
    "reactor kit": "ReactorKit",
    "fastlane": "Fastlane",
    "fast lane": "Fastlane",
    "xcodecloud": "XcodeCloud",
    "xcode cloud": "XcodeCloud",
    "tca": "TCA",
    "uikit": "UIKit",
    "ui kit": "UIKit",
    "combine": "Combine",
    "cocoapods": "CocoaPods",
    "carthage": "Carthage",
    "git": "Git",
    "github": "GitHub",
    "git hub": "GitHub",
    "flutter": "Flutter",
    "react native": "React Native",
    "reactnative": "React Native",
}

# 정형 기술 스택 명칭 세트
VALID_TECH_STACKS = set(TECH_SYNONYMS.values())


class Preprocessor:
    @staticmethod
    def clean_text(text: str | None) -> str:
        """OCR 마커 노이즈([0], [id: 10]) 및 특수문자 클렌징"""
        if not text:
            return ""
        # 1. SoM 숫자 라벨 제거 ([0], [1], [id: 2])
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[id:\s*\d+\]", "", text)
        # 2. 앞뒤 공백 및 마커 기호 트리밍
        text = text.strip()
        text = re.sub(r"^[-•*#\s]+", "", text)
        return text.strip()

    @classmethod
    def clean_list(cls, items: list[str] | None) -> list[str]:
        """텍스트 리스트 전처리 및 빈 값 제거"""
        if not items:
            return []
        cleaned = []
        for item in items:
            c = cls.clean_text(item)
            if c:
                cleaned.append(c)
        return cleaned

    @staticmethod
    def parse_source_platform(url: str | None) -> str:
        """URL을 바탕으로 수집 출처 플랫폼 분류"""
        if not url:
            return "Unknown"
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if "wanted.co.kr" in domain:
            return "Wanted"
        elif "rememberapp" in domain or "remember" in domain:
            return "Remember"
        elif "jobplanet" in domain:
            return "Jobplanet"
        elif "saramin" in domain:
            return "Saramin"
        elif "linkedin" in domain:
            return "Linkedin"
        return "Unknown"

    @classmethod
    def parse_experience(cls, position: str | None, requirements: list[str] | None) -> tuple[int, int, str]:
        """직무명 및 자격요건 텍스트에서 경력 조건 파싱"""
        exp_text = "경력 무관"
        exp_min = 0
        exp_max = 99

        # 1. 직무명 내 괄호 형식 파싱 (예: "iOS 개발자 (3년 이상)")
        if position:
            match = re.search(r"\(([^)]+)\)", position)
            if match:
                inner_text = match.group(1).strip()
                if "년" in inner_text or "신입" in inner_text or "경력" in inner_text:
                    exp_text = inner_text

        # 2. 자격요건 리스트 내 경력 키워드 수색
        if exp_text == "경력 무관" and requirements:
            for req in requirements:
                # 경력/년 관련 표현 수집
                if "경력" in req or "년" in req:
                    # 너무 긴 자격요건 문장은 제외하고 짧은 핵심 구문만 캡처
                    if len(req) < 50:
                        exp_text = req.strip()
                        break

        # 3. 경력 텍스트를 구조화 수치(min/max)로 파싱
        cleaned_exp = cls.clean_text(exp_text)
        
        # 패턴 A: "3~7년", "3년~7년", "신입~3년"
        range_match = re.search(r"(\d+)\s*~\s*(\d+)년", cleaned_exp)
        if range_match:
            exp_min = int(range_match.group(1))
            exp_max = int(range_match.group(2))
        else:
            # 패턴 B: "3년 이상", "3년차 이상"
            above_match = re.search(r"(\d+)년\s*(이상|차|년차)?", cleaned_exp)
            if above_match and "이하" not in cleaned_exp and "미만" not in cleaned_exp:
                exp_min = int(above_match.group(1))
                exp_max = 99
            else:
                # 패턴 C: "3년 이하", "3년 미만"
                below_match = re.search(r"(\d+)년\s*(이하|미만)", cleaned_exp)
                if below_match:
                    exp_min = 0
                    exp_max = int(below_match.group(1))
                elif "신입" in cleaned_exp:
                    exp_min = 0
                    exp_max = 0

        # 추가 보정: "Junior", "주니어" 명시된 경우 (경력 무관일 때)
        if exp_min == 0 and exp_max == 99:
            full_text = f"{position or ''} {' '.join(requirements or [])}".lower()
            if "junior" in full_text or "주니어" in full_text:
                exp_min = 1
                exp_max = 3
                exp_text = "주니어 (1~3년)"

        return exp_min, exp_max, exp_text

    @staticmethod
    def extract_tech_stacks(texts: list[str]) -> list[str]:
        """텍스트 리스트로부터 기술 스택 단어를 사전 및 정규식 매칭을 통해 중복 없이 추출"""
        found_stacks = set()
        combined_text = " ".join(texts).lower()

        # 동의어 및 표준 단어 탐색 (word boundary '\b' 적용하여 단어 단위로 매칭)
        for term, std_name in TECH_SYNONYMS.items():
            escaped_term = re.escape(term)
            pattern = rf"\b{escaped_term}\b"
            if re.search(pattern, combined_text):
                found_stacks.add(std_name)

        return sorted(list(found_stacks))

    @staticmethod
    def generate_content_hash(company_name: str | None, position: str | None, requirements: list[str]) -> str:
        """중복 방지용 컨텐츠 해시(SHA256) 생성"""
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
        # 한글/영문 호환 처리
        company_name = cls.clean_text(raw_data.get("회사명") or raw_data.get("company_name"))
        position = cls.clean_text(raw_data.get("직무명") or raw_data.get("position"))
        url = (raw_data.get("공고url") or raw_data.get("url") or "").strip()

        # 리스트 클렌징
        main_tasks = cls.clean_list(raw_data.get("주요업무") or raw_data.get("main_tasks"))
        requirements = cls.clean_list(raw_data.get("자격요건") or raw_data.get("requirements"))
        preferred = cls.clean_list(raw_data.get("우대사항") or raw_data.get("preferred"))
        benefits = cls.clean_list(raw_data.get("혜택정보") or raw_data.get("benefits"))

        # 플랫폼 분류
        source_platform = cls.parse_source_platform(url)

        # 경력 파싱
        exp_min, exp_max, exp_text = cls.parse_experience(position, requirements)

        # 기술 스택 동의어 역파싱 추출
        all_texts_for_tech = main_tasks + requirements + preferred
        tech_stack = cls.extract_tech_stacks(all_texts_for_tech)

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
        job_category = cls.clean_text(raw_data.get("직군") or raw_data.get("job_category")) or None
        education = cls.clean_text(raw_data.get("학력") or raw_data.get("education")) or None
        employment_type = cls.clean_text(raw_data.get("고용형태") or raw_data.get("employment_type")) or None
        location = cls.clean_text(raw_data.get("근무지") or raw_data.get("location")) or None
        deadline = cls.clean_text(raw_data.get("마감일") or raw_data.get("deadline")) or None
        salary = cls.clean_text(raw_data.get("연봉") or raw_data.get("salary")) or None

        return JobPosting(
            company_name=company_name,
            position=position,
            url=url,
            job_category=job_category,
            experience_level=exp_text,
            education=education,
            employment_type=employment_type,
            location=location,
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
            experience_min=exp_min,
            experience_max=exp_max,
            experience_text=exp_text,
        )
