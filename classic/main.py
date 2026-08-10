"""
Wanted JD Text Extractor — CLI 엔트리포인트 (Playwright 기반).

서브커맨드:
  extract <url>   채용공고 URL을 받아 텍스트 추출 실행
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from agent.config import get_settings
from shared.db import Database
from zoneinfo import ZoneInfo

logger = logging.getLogger("l2c.classic")
_PATHS = get_settings().paths

class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("Asia/Seoul"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def _setup_logging(verbose: bool, log_file: Path | None = None) -> Path:
    """
    콘솔 + 파일 동시 로깅.
    파일은 항상 DEBUG 레벨로 남기고, 콘솔만 verbose 플래그 따라감.
    """
    if log_file is None:
        kst_now = datetime.now(ZoneInfo("Asia/Seoul"))
        log_file = _PATHS.log_dir / f"run_{kst_now.strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = KSTFormatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    for noisy in ("playwright", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(f"로그 파일: {log_file}")
    return log_file


def _slug_from_url(url: str) -> str:
    """URL에서 파일명용 슬러그를 만든다.

    형식: <adapter_name>_<job_id>_<YYYYMMDD_HHMMSS>
      - adapter_name: URL을 매칭하는 사이트 어댑터 이름 (없으면 'unknown')
      - job_id: URL에서 가장 긴 5자리 이상 숫자열. 없으면 'unknown'
    """
    import re
    from classic.automation.sites import resolve_adapter

    try:
        site_name = resolve_adapter(url).name
    except ValueError:
        site_name = "unknown"

    # URL 안의 5자리 이상 숫자열 중 가장 긴 것을 잡 ID로 추정.
    # 잡코리아 ?Oem_Code=C1&...&listno=2&sc=630&...49105168 같은 케이스에서
    # 가장 긴 49105168이 잡 ID일 확률이 가장 큼.
    candidates = re.findall(r"\d{5,}", url)
    job_id = max(candidates, key=len) if candidates else "unknown"

    kst_now = datetime.now(ZoneInfo("Asia/Seoul"))
    return f"{site_name}_{job_id}_{kst_now.strftime('%Y%m%d_%H%M%S')}"


def _phase(name: str):
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        t0 = time.time()
        logger.info(f"━━━ {name} 시작 ━━━")
        try:
            yield
        finally:
            logger.info(f"━━━ {name} 끝 ({time.time()-t0:.2f}s) ━━━")
    return _cm()


def cmd_extract(args: argparse.Namespace) -> int:
    """Playwright 브라우저를 통한 DOM 텍스트 추출."""
    from classic.automation.capture import capture_and_extract_dom

    logger.info(f"▶ extract URL={args.url}")
    db = Database(_PATHS.db_path)

    if not args.force and db.exists(args.url):
        existing = db.get_by_url(args.url)
        logger.warning(
            f"이미 DB에 존재 (id={existing['id']}, company={existing['company_name']}). --force로 재추출 가능."
        )
        print(json.dumps(existing.get("raw_json"), ensure_ascii=False, indent=2))
        return 0

    slug = _slug_from_url(args.url)
    t0 = time.time()

    try:
        with _phase("[1/2] Playwright DOM 추출"):
            dom_raw = capture_and_extract_dom(url=args.url)

        with _phase(f"[2/2] LLM 텍스트 정제 ({args.model or '기본 경량 모델'})"):
            from agent.application.job_normalization_service import (
                complete_extracted_job,
            )
            from classic.extractor.llm_engine import LLMEngine
            from shared.schema.jd_schema import JobPosting

            # DOM에서 가져온 텍스트 전문을 LLM에 전달
            full_text = dom_raw.get("full_text", "")
            extracted = LLMEngine(args.model).extract_from_text(full_text)
            
            # 메타데이터 보완 (LLM이 놓쳤을 경우 대비)
            posting = JobPosting.model_validate(extracted)
            posting = posting.model_copy(
                update={
                    "company_name": posting.company_name
                    or dom_raw.get("company_name"),
                    "position": posting.position or dom_raw.get("position"),
                }
            )
            posting = complete_extracted_job(
                posting,
                current_url=args.url,
                raw_ocr_text=full_text,
            )
            data = posting.model_dump(mode="json")
            
        logger.info(f"데이터 정제 완료: {data.get('company_name')} - {data.get('position')}")

        _PATHS.json_dir.mkdir(parents=True, exist_ok=True)
        json_path = _PATHS.json_dir / f"{slug}.json"
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"JSON 백업 → {json_path}")

        job_id = db.upsert(posting)

        elapsed = time.time() - t0
        logger.info(f"✅ 완료 (db.id={job_id}, 총 {elapsed:.1f}s)")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    except KeyboardInterrupt:
        logger.warning("사용자 중단 (Ctrl+C)")
        return 130
    except Exception as e:
        logger.exception(f"파이프라인 실패: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="콘솔에 DEBUG 로그까지 출력")

    parser = argparse.ArgumentParser(
        prog="l2c-classic",
        description=(
            "L2C Classic — 3개 채용 사이트(원티드·잡코리아·로켓펀치)"
            "에서 공고를 Playwright로 추출하고 LLM으로 정형화"
        ),
        parents=[common],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="URL에서 공고 추출", parents=[common])
    p_ext.add_argument(
        "url",
        help=(
            "채용공고 URL "
            "(지원: 원티드 · 잡코리아 · 로켓펀치)"
        ),
    )
    p_ext.add_argument("--force", action="store_true", help="DB에 있어도 재추출")
    p_ext.add_argument("--model", help="이번 실행에만 사용할 Gemini 모델명")
    p_ext.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
