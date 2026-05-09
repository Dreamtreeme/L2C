"""
Wanted JD to JSON — CLI 엔트리포인트 (Playwright + Vision LLM).

서브커맨드:
  extract <url>   원티드 채용공고 URL을 받아 풀 파이프라인 실행
  list            DB에 저장된 최근 추출 이력
  show <id|url>   특정 공고 상세
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from shared.config import DB_PATH, JSON_DIR, LOGS_DIR
from shared.db import Database
from zoneinfo import ZoneInfo

logger = logging.getLogger("autoserch")

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
        log_file = LOGS_DIR / f"run_{kst_now.strftime('%Y%m%d_%H%M%S')}.log"

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
    import re
    m = re.search(r"/wd/(\d+)", url)
    job_id = m.group(1) if m else "unknown"
    kst_now = datetime.now(ZoneInfo("Asia/Seoul"))
    return f"wanted_{job_id}_{kst_now.strftime('%Y%m%d_%H%M%S')}"


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
    """풀 파이프라인: Playwright 브라우저 캡처 → Vision LLM → DB."""
    from classic.automation.capture import capture_and_extract_dom

    logger.info(f"▶ extract URL={args.url} model={args.model or 'default'}")
    db = Database(DB_PATH)

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
        with _phase("[1/1] Playwright DOM 추출 및 캡처"):
            screenshot_path, data = capture_and_extract_dom(url=args.url, save_name=slug)

        # VLM 대신 DOM 데이터를 그대로 사용 (필요시 정제 로직 추가 가능)
        logger.info(f"DOM에서 데이터 추출 완료: {data.get('company_name')} - {data.get('position')}")

        json_path = JSON_DIR / f"{slug}.json"
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"JSON 백업 → {json_path}")

        job_id = db.upsert(
            url=args.url,
            data=data,
            screenshot_path=str(screenshot_path),
            ocr_text_path=None,
        )

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


def cmd_list(args: argparse.Namespace) -> int:
    db = Database(DB_PATH)
    rows = db.list_recent(limit=args.limit)
    logger.info(f"DB에서 {len(rows)}건 조회 (limit={args.limit})")
    if not rows:
        print("(저장된 공고 없음)")
        return 0
    print(f"{'id':>4}  {'created_at':19}  {'company':20}  position")
    print("-" * 80)
    for r in rows:
        company = (r.get("company_name") or "-")[:20]
        position = (r.get("position") or "-")[:60]
        print(f"{r['id']:>4}  {r['created_at']:19}  {company:20}  {position}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    db = Database(DB_PATH)
    target = args.target
    logger.debug(f"show target={target}")

    if target.isdigit():
        record = db.get(int(target))
    elif target.startswith("http"):
        record = db.get_by_url(target)
    else:
        logger.error("target은 숫자 id 또는 http URL이어야 합니다.")
        return 2

    if not record:
        print("(찾을 수 없음)")
        return 1

    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="콘솔에 DEBUG 로그까지 출력")

    parser = argparse.ArgumentParser(
        prog="autoserch",
        description="원티드 채용공고를 Vision LLM으로 JSON 추출",
        parents=[common],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="URL에서 공고 추출", parents=[common])
    p_ext.add_argument("url", help="원티드 채용공고 URL (https://www.wanted.co.kr/wd/...)")
    p_ext.add_argument("--force", action="store_true", help="DB에 있어도 재추출")
    p_ext.add_argument("--model", help="이번 실행에만 사용할 Ollama 모델명")
    p_ext.set_defaults(func=cmd_extract)

    p_list = sub.add_parser("list", help="추출 이력 조회", parents=[common])
    p_list.add_argument("-n", "--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="특정 공고 조회 (id 또는 URL)", parents=[common])
    p_show.add_argument("target", help="DB id 또는 원본 URL")
    p_show.set_defaults(func=cmd_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
