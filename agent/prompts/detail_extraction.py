"""상세 페이지 OCR 정제에 공통으로 적용할 판단 기준."""

from __future__ import annotations

from agent.prompts.trust_boundary import external_content_contract_ko


def _layout_evidence_contract() -> str:
    return (
        "ocr_items의 bbox_ratio는 [left, top, right, bottom] 순서의 화면 비율 좌표이며, screen은 해당 줄이 처음 관찰된 화면입니다. "
        "먼저 target_context의 공고 제목과 OCR 배치를 함께 보고 현재 채용공고의 시각적 본문 영역을 식별하십시오. "
        "target_context는 탐색 중 선택한 카드의 기억이므로 화면 근거와 일치할 때만 사용하십시오. "
        "회사명과 직무명은 같은 채용공고 헤더 영역에 속한 줄에서 추출하십시오. 추천 공고, 광고, 이어보는 채용정보, "
        "내비게이션과 로그인 안내처럼 다른 시각 영역에 있는 회사명과 직무명은 현재 공고 정보로 사용하지 마십시오. "
        "같은 문자열의 등장 횟수로 회사명을 결정하지 마십시오. 서로 다른 회사명 후보가 있고 현재 공고 영역을 확정할 수 없으면 "
        "identity_conflict를 true로 설정하고 identity_candidates에 충돌 후보를 넣으십시오. 임의로 하나를 선택하지 마십시오. "
        "field_evidence_line_ids에는 각 field_evidence를 뒷받침하는 ocr_items의 id를 넣으십시오. "
        "company_name과 position의 근거 ID는 반드시 반환하십시오. "
    )


def build_detail_extraction_system_prompt(
    base_instruction: str,
    *,
    layout_evidence: bool = False,
) -> str:
    """실행 경로와 벤치마크가 공유하는 상세 OCR 정제 지침을 만든다."""
    return (
        f"{base_instruction.strip()}\n"
        f"{external_content_contract_ko()}\n"
        "북마크, 브라우저 메뉴, 보상 배지, 추천인 현금, 로그인 문구 같은 주변 UI 노이즈는 무시하십시오. "
        "채용 도메인에서 명확한 OCR 혼동은 문맥으로 보정하십시오. 예를 들어 Swift, Xcode, 앱 개발, 모바일 문맥에서 "
        "'ios', 'i0S', 'j0s', '10s'처럼 보이는 토큰은 직무명과 기술스택에서 'iOS'로 정규화하십시오. "
        "회사명과 직무명은 상세 페이지 상단에서 서로 인접한 페이지 텍스트를 가장 우선하는 근거로 사용하십시오. "
        f"{_layout_evidence_contract() if layout_evidence else ''}"
        "직무명 괄호 안의 세부 분야나 조직명은 명시적인 회사명 근거가 아니며, 회사명이 본문에 없으면 이를 회사명으로 만들지 마십시오. "
        "이미지 내부 로고 OCR은 글자가 잘리거나 일부만 검출될 수 있으므로 보조 근거로만 사용하고, "
        "페이지 텍스트와 충돌하면 페이지 텍스트를 선택하십시오. "
        "대괄호로 나뉜 직무명 조각은 한 줄 직무명으로 합치고, 직무명에는 브라우저/광고/보상 문구를 넣지 마십시오. "
        "본문뿐 아니라 같은 공고 화면의 요약·근무조건 영역도 끝까지 확인하여 경력, 학력, 고용형태, 근무지, 급여, 마감일을 각각 보존하십시오. "
        "'3년 이상'처럼 경력 연수가 명시되면 experience_min과 experience_max에 년 단위 정수로 구조화하십시오. "
        "필수 자격요건에 서로 다른 최소 경력이 여러 개 있으면 그중 가장 높은 연차를 experience_min에 저장하고, 단순히 '경력'만 보이면 수치를 추정하지 마십시오. "
        "필수·우대 조건에 명시된 프로그래밍 언어, 프레임워크, 플랫폼, 클라우드, 데이터베이스, 개발 도구는 tech_stack에도 빠짐없이 정리하십시오. "
        "기술스택은 실제 업무에 쓰는 기술만 넣고, 면접 질문 예시나 CS 개념 목록은 requirements에 요약하십시오. "
        "salary, posted_at, posted_at_text, deadline, location, benefits는 서로 섞지 말고 해당 필드에만 넣으십시오. "
        "게시일이 화면에 명확히 보일 때만 posted_at에 YYYY-MM-DD 형식으로 넣고 posted_at_text에는 화면 원문을 보존하십시오. "
        "'3일 전'처럼 기준 날짜 없이는 확정할 수 없는 상대 표현은 posted_at_text에만 넣으십시오. "
        "게시일이 없으면 마감일이나 현재 날짜로 추정하지 말고 두 필드를 비워 두십시오. "
        "목록 필드는 핵심 항목만 간결하게 유지하십시오. "
        "raw_ocr_text, content_hash, evidence_hash는 출력하지 마십시오. JSON 객체 하나만 출력하십시오."
    )


__all__ = ["build_detail_extraction_system_prompt"]
