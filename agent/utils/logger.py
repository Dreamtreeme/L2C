import logging
import os
import sys

import sentry_sdk
import structlog
from dotenv import load_dotenv

load_dotenv()

# Sentry 초기화
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        profiles_sample_rate=1.0,
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
            # 콘솔 출력을 위한 ConsoleRenderer (터미널에서 가독성 좋음)
            # 운영 시에는 JSONRenderer로 변경 가능
            structlog.dev.ConsoleRenderer(colors=True)
            if os.getenv("APP_ENV") != "production"
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

setup_agent_logger()
logger = structlog.get_logger("l2c.agent")
