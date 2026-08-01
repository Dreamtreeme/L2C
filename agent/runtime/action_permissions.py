"""공개 채용공고 수집 작업의 결정론적 실행 권한 계약."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from agent.graph.state import GraphState
from agent.sites.profile import SiteProfile


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _collect_input_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            values.update(_collect_input_values(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            values.update(_collect_input_values(item))
    elif isinstance(value, str):
        normalized = _normalized_text(value)
        if normalized:
            values.add(normalized)
    return values


def build_public_collection_permission_contract(
    site_profile: SiteProfile,
    recipe_params: dict[str, Any],
) -> dict[str, Any]:
    """사이트 프로필과 확정된 요청 인자로 실행 권한을 만든다."""

    return {
        "site": site_profile.slug,
        "allowed_domains": list(site_profile.domains),
        "allowed_input_values": sorted(_collect_input_values(recipe_params)),
    }


def _url_host_allowed(url: str, domains: set[str]) -> bool:
    host = str(urlsplit(str(url or "")).hostname or "").casefold()
    return bool(
        host
        and any(
            host == domain or host.endswith("." + domain)
            for domain in domains
        )
    )


def task_permission_reason(
    state: GraphState,
    action_name: str,
    args: dict[str, Any],
    *,
    source: str,
) -> str:
    """현재 행동이 작업 계약을 벗어나면 승인 사유를 반환한다."""

    contract = (
        dict(state.get("action_permission_contract") or {})
        if isinstance(state.get("action_permission_contract"), dict)
        else {}
    )
    if not contract:
        return ""

    if action_name == "open_browser":
        domains = {
            str(domain).strip().casefold()
            for domain in contract.get("allowed_domains", []) or []
            if str(domain).strip()
        }
        if domains and not _url_host_allowed(
            str(args.get("url") or ""),
            domains,
        ):
            return "task_contract_external_domain"

    if action_name == "type_in_marker":
        allowed_values = {
            _normalized_text(value)
            for value in contract.get("allowed_input_values", []) or []
            if _normalized_text(value)
        }
        if (
            allowed_values
            and _normalized_text(args.get("text")) not in allowed_values
        ):
            return "task_contract_input_not_authorized"

    if (
        source == "llm"
        and action_name
        in {
            "click_marker",
            "type_in_marker",
            "press_key",
            "scroll",
            "open_browser",
            "go_back",
            "close_current_tab",
            "switch_tab",
        }
    ):
        risk_level = str(args.get("risk_level") or "").strip().casefold()
        if risk_level not in {"safe_read", "safe_navigation", "sensitive"}:
            return "task_contract_risk_not_declared"
        if args.get("needs_user_confirmation") not in {True, False}:
            return "task_contract_confirmation_not_declared"
    return ""


__all__ = [
    "build_public_collection_permission_contract",
    "task_permission_reason",
]
