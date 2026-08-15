"""
L2C Classic CLI 엔트리포인트 (Playwright 기반).

서브커맨드:
  extract <url>   채용공고 URL을 받아 텍스트 추출 실행
  collect <url>   홈페이지에서 검색해 공고 여러 건 수집
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.config import get_settings
from shared.db import Database
from shared.schema.jd_schema import JobField

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
            logger.info(f"━━━ {name} 끝 ({time.time() - t0:.2f}s) ━━━")

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
            from classic.automation.sites import resolve_adapter
            from classic.extractor.llm_engine import LLMEngine
            from classic.extractor.normalization import normalize_dom_posting

            posting = normalize_dom_posting(
                dom_raw,
                url=args.url,
                source_platform=resolve_adapter(args.url).name,
                engine=LLMEngine(args.model),
            )
            data = posting.model_dump(mode="json")

        logger.info(
            f"데이터 정제 완료: {data.get('company_name')} - {data.get('position')}"
        )

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


def cmd_collect(args: argparse.Namespace) -> int:
    """홈페이지에서 검색해 여러 상세 공고를 공통 스키마로 저장한다."""

    from classic.automation.collection import ClassicCollectionRunner
    from classic.extractor.normalization import LLMDomJobNormalizer
    from shared.schema.agent_contract import DEFAULT_JOB_COLLECTION_FIELDS
    from shared.schema.collection_intent import CollectionIntent

    required_fields = [
        JobField(value)
        for value in (
            args.required_field
            or [field.value for field in DEFAULT_JOB_COLLECTION_FIELDS]
        )
    ]
    intent = CollectionIntent(
        original_query=args.query,
        search_keyword=args.query,
        target_count=args.count,
        required_fields=required_fields,
    )
    runner = ClassicCollectionRunner(
        db_path=args.db_path,
        normalizer=LLMDomJobNormalizer(args.model),
    )
    result = runner.run(args.homepage, intent)
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", action="store_true", help="콘솔에 DEBUG 로그까지 출력"
    )

    parser = argparse.ArgumentParser(
        prog="l2c-classic",
        description=(
            "L2C Classic — 등록된 채용 사이트에서 공고를 Playwright로 "
            "추출하고 LLM으로 정형화"
        ),
        parents=[common],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="URL에서 공고 추출", parents=[common])
    p_ext.add_argument(
        "url",
        help=("채용공고 URL (지원: 원티드 · 잡코리아 · 로켓펀치)"),
    )
    p_ext.add_argument("--force", action="store_true", help="DB에 있어도 재추출")
    p_ext.add_argument("--model", help="이번 실행에만 사용할 Gemini 모델명")
    p_ext.set_defaults(func=cmd_extract)

    p_collect = sub.add_parser(
        "collect",
        help="홈페이지에서 검색해 공고 여러 건 수집",
        parents=[common],
    )
    p_collect.add_argument("homepage", help="수집을 시작할 공식 홈페이지")
    p_collect.add_argument("--query", required=True, help="채용공고 검색어")
    p_collect.add_argument("--count", type=int, default=2, help="수집할 공고 수")
    p_collect.add_argument("--db-path", type=Path, help="격리 SQLite DB 경로")
    p_collect.add_argument("--model", help="이번 실행에 사용할 경량 모델명")
    p_collect.add_argument(
        "--required-field",
        action="append",
        choices=[field.value for field in JobField],
        help="필수 공고 필드. 여러 번 지정할 수 있음",
    )
    p_collect.set_defaults(func=cmd_collect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
