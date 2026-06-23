"""
화면 상태 키(state_key)를 OCR 앵커 마커 텍스트만으로 계산한다.

URL은 기록 메타데이터나 최종 공고 출처에는 쓸 수 있지만, Reflex 화면 상태 판단에는
사용하지 않는다.
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
    texts = state_anchor_texts(markers)
    uniq = sorted(set(texts), key=lambda s: (-len(s), s))[:top]
    uniq = sorted(uniq)
    if not uniq:
        return "0"
    return hashlib.sha1("|".join(uniq).encode("utf-8")).hexdigest()[:10]


def anchor_texts_from_values(values) -> list[str]:
    """저장된 OCR 문자열을 화면 유사도 비교용 앵커 집합으로 정규화한다."""
    anchors: set[str] = set()
    for value in values or []:
        text = canonical_anchor_text(value)
        if len(text) < 2 or text.startswith("상호작용 가능한 요소"):
            continue
        anchors.add(text.casefold().replace(" ", ""))
    return sorted(anchors)


def state_anchor_texts(markers) -> list[str]:
    """현재 OCR 마커에서 화면 유사도 비교용 앵커 텍스트를 만든다."""
    values = [
        marker.get("text")
        for marker in markers or []
        if isinstance(marker, dict)
    ]
    return anchor_texts_from_values(values)


def anchor_similarity(saved_anchors, markers) -> float:
    """기록 화면과 현재 화면의 OCR 앵커 자카드 유사도(Jaccard similarity)를 계산한다."""
    saved = set(anchor_texts_from_values(saved_anchors))
    current = set(state_anchor_texts(markers))
    if not saved or not current:
        return 0.0
    return len(saved & current) / len(saved | current)


def compute_state_key(url: str, markers) -> str:
    return f"ocr#{anchor_signature(markers)}"
