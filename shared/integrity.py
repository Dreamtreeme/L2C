"""수집 데이터의 출처 증거를 재현 가능한 해시로 계산한다."""

from __future__ import annotations

import hashlib
import json
from typing import Any


_VOLATILE_FIELDS = {
    "_normalization_error",
    "content_hash",
    "evidence_hash",
    "created_at",
    "updated_at",
}


def source_evidence_hash(url: str, data: dict[str, Any]) -> str:
    """같은 URL과 수집 내용이 같은지 확인할 SHA-256 해시를 만든다."""

    canonical = {
        "url": str(url or data.get("url") or "").strip(),
        "data": {
            str(key): value
            for key, value in data.items()
            if str(key) not in _VOLATILE_FIELDS
        },
    }
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["source_evidence_hash"]
