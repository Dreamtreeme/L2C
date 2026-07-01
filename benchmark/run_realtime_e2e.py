"""Realtime worker E2E runner with stable file logging."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TextIO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class _Tee:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run realtime_scraping E2E and tee stdout/stderr to a log file.")
    parser.add_argument("--site", default="wanted")
    parser.add_argument("--query", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(original_stderr, log_file)  # type: ignore[assignment]
        try:
            from agent.tools.realtime_scraping import realtime_scraping

            start = time.time()
            result = realtime_scraping.invoke({"site": args.site, "query": args.query})
            elapsed = time.time() - start
            if isinstance(result, str):
                print(result)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"WALL_TIME_SEC={elapsed:.3f}")
            print(f"LOG_TARGET={log_path}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
