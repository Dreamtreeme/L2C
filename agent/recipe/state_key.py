"""
화면-상태 키 계산.
state_key = URL 템플릿 + 앵커 마커 텍스트 집합의 해시.
공고 ID 같은 가변 경로는 {id}로 템플릿화하므로, 같은 화면-상태는 같은 키가 된다.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

_MARKER_NOISE = re.compile(r"\[(?:id:\s*)?\d+\]")
_MULTISPACE = re.compile(r"\s+")
_NUM_SEG = re.compile(r"/\d+")
_UUID_SEG = re.compile(r"/[0-9a-f]{8,}(?=/|$)", re.IGNORECASE)
_VOLATILE_TEXT = re.compile(r"^\d+([,.]\d+)?\s*(명|개|건|원|%|일|시간|분)?$")
_NUMBER_TOKEN = re.compile(r"(?<![A-Za-z0-9가-힣])\d+([,.]\d+)?(?![A-Za-z0-9가-힣])")
_NUMBER_WITH_UNIT = re.compile(r"\d+([,.]\d+)?\s*(명|개|건|원|%|일|시간|분)")


def normalize_text(text) -> str:
    """OCR 마커 노이즈([0],[id: 2]) 제거 + 공백 정규화."""
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


def canonical_anchor_text(text) -> str:
    """상태 판별용 텍스트. 추천수/조회수 같은 동적 숫자는 값 대신 자리표시자로 둔다."""
    t = normalize_text(text)
    if not t or _VOLATILE_TEXT.match(t):
        return ""
    t = _NUMBER_WITH_UNIT.sub("{n}\\2", t)
    t = _NUMBER_TOKEN.sub("{n}", t)
    t = _MULTISPACE.sub(" ", t).strip()
    return t


def site_of(url: str) -> str:
    try:
        net = (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    return net[4:] if net.startswith("www.") else net


def url_template(url: str) -> str:
    """URL을 가변 성분 제거한 템플릿으로. 숫자 경로 -> /{id}, 쿼리는 키만 정렬."""
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


def anchor_signature(markers, top: int = 8) -> str:
    """현재 화면 마커 텍스트 중 안정적인 상위 N개를 정렬·해시한 시그니처."""
    texts = []
    for m in markers or []:
        if isinstance(m, dict):
            t = canonical_anchor_text(m.get("text"))
            if len(t) >= 2:
                texts.append(t)
    uniq = sorted(set(texts), key=lambda s: (-len(s), s))[:top]
    uniq = sorted(uniq)
    if not uniq:
        return "0"
    return hashlib.sha1("|".join(uniq).encode("utf-8")).hexdigest()[:10]


def compute_state_key(url: str, markers) -> str:
    return f"{url_template(url)}#{anchor_signature(markers)}"
