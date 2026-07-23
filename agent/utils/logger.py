import logging
import sys
from pathlib import Path

import sentry_sdk
import structlog
from agent.config import get_settings


# Sentry 초기화
_OBSERVABILITY = get_settings().observability
SENTRY_DSN = (
    _OBSERVABILITY.sentry_dsn.get_secret_value()
    if _OBSERVABILITY.sentry_dsn is not None
    else ""
)
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=_OBSERVABILITY.sentry_traces_sample_rate,
        profiles_sample_rate=_OBSERVABILITY.sentry_profiles_sample_rate,
        environment=_OBSERVABILITY.app_env,
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
            if _OBSERVABILITY.log_format.strip().lower() != "json"
            and _OBSERVABILITY.app_env != "production"
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
    log_target = _OBSERVABILITY.log_target
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
