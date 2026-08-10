"""검색 의미 사전 서비스가 공유하는 작은 결정론적 유틸리티."""

from __future__ import annotations

from datetime import datetime


CORE_SOURCE_KEY = "l2c_ko_core"
OCCUPATION_DOMAIN_ROOT_KEY = "l2c:domain:occupation"


def taxonomy_timestamp() -> str:
    """현재 시각을 사전 DB 기록 형식으로 반환한다."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def contains_taxonomy_alias(text: str, alias: str) -> bool:
    """영문 별칭이 더 긴 토큰 일부로 잘못 일치하지 않게 검사한다."""

    if not text or not alias:
        return False
    start = 0
    while True:
        index = text.find(alias, start)
        if index < 0:
            return False
        end = index + len(alias)
        left_ok = index == 0 or not (
            text[index - 1].isalnum() and alias[0].isalnum()
        )
        right_ok = end == len(text) or not (
            text[end].isalnum() and alias[-1].isalnum()
        )
        if left_ok and right_ok:
            return True
        start = index + 1


__all__ = [
    "CORE_SOURCE_KEY",
    "OCCUPATION_DOMAIN_ROOT_KEY",
    "contains_taxonomy_alias",
    "taxonomy_timestamp",
]
