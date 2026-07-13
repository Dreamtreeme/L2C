"""상세 페이지 OCR 정제에 공통으로 적용할 판단 기준."""

from __future__ import annotations


def build_detail_extraction_system_prompt(base_instruction: str) -> str:
    """실행 경로와 벤치마크가 공유하는 상세 OCR 정제 지침을 만든다."""
    return (
        f"{base_instruction.strip()} "
        "북마크, 브라우저 메뉴, 보상 배지, 추천인 현금, 로그인 문구 같은 주변 UI 노이즈는 무시하십시오. "
        "채용 도메인에서 명확한 OCR 혼동은 문맥으로 보정하십시오. 예를 들어 Swift, Xcode, 앱 개발, 모바일 문맥에서 "
        "'ios', 'i0S', 'j0s', '10s'처럼 보이는 토큰은 직무명과 기술스택에서 'iOS'로 정규화하십시오. "
        "회사명과 직무명은 상세 페이지 상단에서 서로 인접한 페이지 텍스트를 가장 우선하는 근거로 사용하십시오. "
        "이미지 내부 로고 OCR은 글자가 잘리거나 일부만 검출될 수 있으므로 보조 근거로만 사용하고, "
        "페이지 텍스트와 충돌하면 페이지 텍스트를 선택하십시오. "
        "대괄호로 나뉜 직무명 조각은 한 줄 직무명으로 합치고, 직무명에는 브라우저/광고/보상 문구를 넣지 마십시오. "
        "기술스택은 실제 업무에 쓰는 기술만 넣고, 면접 질문 예시나 CS 개념 목록은 requirements에 요약하십시오. "
        "salary, posted_at, posted_at_text, deadline, location, benefits는 서로 섞지 말고 해당 필드에만 넣으십시오. "
        "게시일이 화면에 명확히 보일 때만 posted_at에 YYYY-MM-DD 형식으로 넣고 posted_at_text에는 화면 원문을 보존하십시오. "
        "'3일 전'처럼 기준 날짜 없이는 확정할 수 없는 상대 표현은 posted_at_text에만 넣으십시오. "
        "게시일이 없으면 마감일이나 현재 날짜로 추정하지 말고 두 필드를 비워 두십시오. "
        "목록 필드는 핵심 항목만 간결하게 유지하십시오. "
        "raw_ocr_text와 content_hash는 출력하지 마십시오. JSON 객체 하나만 출력하십시오."
    )


__all__ = ["build_detail_extraction_system_prompt"]
