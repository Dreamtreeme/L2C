"""환경 변수 기반 설정 캐시가 테스트 사이에 새지 않게 한다."""

import pytest


@pytest.fixture(autouse=True)
def reset_typed_settings_cache():
    from agent.config import clear_settings_cache

    clear_settings_cache()
    yield
    clear_settings_cache()
