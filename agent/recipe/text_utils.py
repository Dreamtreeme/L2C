"""Reflex 레시피에서 공통으로 쓰는 텍스트/URL 정규화 유틸."""

from __future__ import annotations

import re
from urllib.parse import urlparse

_MARKER_NOISE = re.compile(r"\[(?:id:\s*)?\d+\]")
_MULTISPACE = re.compile(r"\s+")
_NUM_SEG = re.compile(r"/\d+")
_UUID_SEG = re.compile(r"/[0-9a-f]{8,}(?=/|$)", re.IGNORECASE)


def normalize_text(text) -> str:
    """OCR 마커 노이즈([0], [id: 2]) 제거 + 공백 정규화."""
    if not text:
        return ""
    raw = str(text)
    try:
        from agent.utils.preprocessor import Preprocessor

        raw = Preprocessor.clean_text(raw)
    except Exception:
        pass
    t = _MARKER_NOISE.sub("", raw)
    t = _MULTISPACE.sub(" ", t).strip()
    return t


def site_of(url: str) -> str:
    try:
        net = (urlparse(url or "").netloc or "").lower()
    except Exception:
        return ""
    return net[4:] if net.startswith("www.") else net


def url_template(url: str) -> str:
    """URL에서 숫자/UUID 같은 가변 경로 성분을 제거한 템플릿을 만든다."""
    if not url:
        return ""
    try:
        p = urlparse(url)
    except Exception:
        return ""
    net = site_of(url)
    path = _UUID_SEG.sub("/{id}", p.path or "")
    path = _NUM_SEG.sub("/{id}", path)
    keys = sorted({kv.split("=")[0] for kv in (p.query or "").split("&") if kv})
    q = ("?" + ",".join(keys)) if keys else ""
    return f"{net}{path}{q}"


def recipe_url_scope_matches(saved_template: str, current_url: str) -> bool:
    """저장 시점과 현재 URL이 같은 호스트·경로·쿼리 키 범위인지 확인한다."""

    saved = str(saved_template or "").strip().casefold()
    current = url_template(current_url).strip().casefold()
    if not saved or not current:
        return True

    def split_template(value: str) -> tuple[str, tuple[str, ...]]:
        base, separator, query_keys = value.partition("?")
        normalized_base = base.rstrip("/") or base
        keys = tuple(sorted(key for key in query_keys.split(",") if key)) if separator else ()
        return normalized_base, keys

    return split_template(saved) == split_template(current)
