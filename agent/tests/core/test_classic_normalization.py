from classic.extractor.normalization import normalize_dom_posting


class _Engine:
    def extract_from_text(self, _text):
        return {
            "company_name": "LLM 추론 회사",
            "position": "설명에서 확장한 직무명",
            "main_tasks": ["서비스 개발"],
            "requirements": ["개발 경험"],
        }


def test_dom_identity_fields_override_llm_inference():
    posting = normalize_dom_posting(
        {
            "company_name": "DOM 확인 회사",
            "position": "DOM 확인 제목",
            "full_text": "상세 공고 본문",
        },
        url="https://example.test/jobs/dynamic",
        source_platform="example",
        engine=_Engine(),
    )

    assert posting.company_name == "DOM 확인 회사"
    assert posting.position == "DOM 확인 제목"
