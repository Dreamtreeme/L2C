import logging
import os
import sys
from pathlib import Path

import sentry_sdk
import structlog
from dotenv import load_dotenv

load_dotenv()


def _sample_rate(name: str, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


# Sentry 초기화
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=_sample_rate("SENTRY_TRACES_SAMPLE_RATE"),
        profiles_sample_rate=_sample_rate("SENTRY_PROFILES_SAMPLE_RATE"),
        environment=os.getenv("APP_ENV", "development"),
    )

def setup_agent_logger():
    """
    Agent용 structlog 로거를 초기화합니다.
    퍼포먼스 벤치마크 및 JSON 포맷 로깅에 최적화되어 있습니다.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=bool(sys.stdout.isatty()))
            if os.getenv("LOG_FORMAT", "").strip().lower() != "json"
            and os.getenv("APP_ENV") != "production"
            else structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 표준 logging 모듈과 통합
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    log_target = os.getenv("LOG_TARGET")
    if log_target:
        log_path = Path(log_target)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        resolved = str(log_path.resolve())
        has_handler = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", "") == resolved
            for handler in root.handlers
        )
        if not has_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            root.addHandler(file_handler)

setup_agent_logger()
logger = structlog.get_logger("l2c.agent")
