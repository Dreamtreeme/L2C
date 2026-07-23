"""프로젝트 문서 계약 회귀 테스트."""

from scripts.check_docs import main


def test_managed_document_contracts() -> None:
    assert main() == 0
