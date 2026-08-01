"""역할별 외부 모델 기본값과 환경변수 우선순위를 관리한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.config import get_settings


DEFAULT_COMMANDER_MODEL = "gemini-3.6-flash"
DEFAULT_LIGHTWEIGHT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_WORKER_REASONING_THINKING_LEVEL = "low"
ModelExecutionRole = Literal[
    "commander",
    "worker_reasoning",
    "lightweight",
    "detail",
    "critic",
]


@dataclass(frozen=True)
class ModelExecutionPolicy:
    role: ModelExecutionRole
    request_timeout_sec: float
    retries: int


def model_execution_policy(
    role: ModelExecutionRole,
) -> ModelExecutionPolicy:
    """모델 역할별 timeout과 공급자 재시도 횟수를 반환한다."""

    settings = get_settings()
    timeout_by_role = {
        "commander": settings.execution.commander_request_timeout_sec,
        "worker_reasoning": (
            settings.execution.worker_reasoning_request_timeout_sec
        ),
        "lightweight": settings.execution.lightweight_request_timeout_sec,
        "detail": settings.execution.detail_request_timeout_sec,
        "critic": settings.recipe.critic_timeout_sec,
    }
    return ModelExecutionPolicy(
        role=role,
        request_timeout_sec=float(timeout_by_role[role]),
        retries=int(settings.execution.transient_retries),
    )


def _first_configured_model(env_names: tuple[str, ...], default: str) -> str:
    settings = get_settings().models
    for env_name in env_names:
        value = settings.model_override(env_name) or ""
        if value:
            return value
    return default


def commander_model_name(*override_env_names: str) -> str:
    """복잡한 계획·화면 행동·검토에 사용할 지휘자 모델을 반환한다."""

    return _first_configured_model(
        (*override_env_names, "COMMANDER_MODEL"),
        DEFAULT_COMMANDER_MODEL,
    )


def worker_reasoning_model_name() -> str:
    """화면을 보고 다음 물리 행동을 결정할 모델을 반환한다."""

    return commander_model_name("VISION_WORKER_REASONING_MODEL")


def worker_reasoning_thinking_level() -> str:
    """화면 행동 추론에 사용할 사고 수준을 반환한다."""

    return get_settings().models.worker_reasoning_thinking_level


def lightweight_model_name(*override_env_names: str) -> str:
    """구조화·요약·짧은 분류에 사용할 경량 모델을 반환한다."""

    return _first_configured_model(
        (*override_env_names, "VISION_LIGHTWEIGHT_MODEL"),
        DEFAULT_LIGHTWEIGHT_MODEL,
    )


__all__ = [
    "DEFAULT_COMMANDER_MODEL",
    "DEFAULT_LIGHTWEIGHT_MODEL",
    "DEFAULT_WORKER_REASONING_THINKING_LEVEL",
    "ModelExecutionPolicy",
    "ModelExecutionRole",
    "commander_model_name",
    "lightweight_model_name",
    "model_execution_policy",
    "worker_reasoning_model_name",
    "worker_reasoning_thinking_level",
]
