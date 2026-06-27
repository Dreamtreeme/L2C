"""Pre-execution triage and public-web research helpers."""

from __future__ import annotations

import json
import os
import re
from html import unescape
from typing import Any
from urllib.parse import quote_plus, urlparse

from shared.schema.task_triage_schema import PublicResearchReport, ResearchPath, TaskTriage


_JOB_TERMS = (
    "job",
    "jobs",
    "career",
    "careers",
    "engineer",
    "developer",
    "hiring",
    "recruit",
    "채용",
    "공고",
    "직무",
    "구인",
    "개발자",
    "엔지니어",
)
_FINANCE_TERMS = ("은행", "적금", "예금", "대출", "보험", "투자", "계좌", "카드", "송금", "결제", "financial", "bank")
_SENSITIVE_TERMS = (
    "로그인",
    "비밀번호",
    "인증",
    "주민등록",
    "계좌",
    "카드",
    "결제",
    "신청",
    "가입",
    "제출",
    "동의",
    "약관",
    "송금",
    "대출",
    "투자",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(term.lower() in lowered for term in terms)


def deterministic_task_triage(user_query: str) -> TaskTriage:
    """Conservative fallback triage that never permits unknown sensitive work directly."""

    query = str(user_query or "").strip()
    reasons: list[str] = []
    sensitive_steps = [term for term in _SENSITIVE_TERMS if term in query]
    if _contains_any(query, _JOB_TERMS):
        reasons.append("job-domain terms matched configured collection workflow")
        return TaskTriage(
            goal_type="job_collection",
            known_or_unknown="known",
            risk_level="safe_navigation",
            requires_research=False,
            sensitive_steps=sensitive_steps,
            reasons=reasons,
        )
    if _contains_any(query, _FINANCE_TERMS) or sensitive_steps:
        reasons.append("financial or sensitive action terms require public research and human confirmation")
        return TaskTriage(
            goal_type="financial_action" if _contains_any(query, _FINANCE_TERMS) else "account_action",
            known_or_unknown="unknown",
            risk_level="sensitive",
            requires_research=True,
            sensitive_steps=sensitive_steps or ["login", "personal information", "agreement", "application"],
            reasons=reasons,
        )
    return TaskTriage(
        goal_type="information_lookup",
        known_or_unknown="unknown",
        risk_level="safe_read",
        requires_research=True,
        sensitive_steps=[],
        reasons=["no configured domain route matched"],
    )


def _triage_mode() -> str:
    mode = os.getenv("COMMANDER_TASK_TRIAGE_MODE", "deterministic").strip().lower()
    return mode if mode in {"deterministic", "llm"} else "deterministic"


def triage_user_task(user_query: str) -> TaskTriage:
    """Classify the user task before launching a browser worker."""

    if _triage_mode() != "llm":
        return deterministic_task_triage(user_query)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = os.getenv("COMMANDER_TASK_TRIAGE_MODEL", os.getenv("VISION_WORKER_REVIEW_MODEL", "gemini-3.5-flash"))
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0).with_structured_output(TaskTriage)
        result = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Classify whether an autonomous browser task can run immediately. "
                        "Unknown or new features must be researched before execution. "
                        "Login, personal data, payment, agreement, submission, account, finance, legal, or application "
                        "steps are sensitive and require human confirmation."
                    )
                ),
                HumanMessage(content=json.dumps({"user_query": user_query}, ensure_ascii=False)),
            ]
        )
        return result if isinstance(result, TaskTriage) else TaskTriage(**result)
    except Exception as exc:  # pragma: no cover - provider failures use conservative fallback
        fallback = deterministic_task_triage(user_query)
        fallback.reasons.append(f"llm_triage_failed: {str(exc)[:160]}")
        return fallback


def _research_mode() -> str:
    mode = os.getenv("COMMANDER_PUBLIC_RESEARCH_MODE", "requests").strip().lower()
    return mode if mode in {"off", "requests"} else "requests"


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _official_hint(domain: str, title: str, snippet: str) -> bool:
    joined = f"{domain} {title} {snippet}".lower()
    return any(token in joined for token in ("official", "공식", "go.kr", "or.kr", "bank", "은행", "금융"))


def _parse_duckduckgo_results(html: str, limit: int = 6) -> list[ResearchPath]:
    paths: list[ResearchPath] = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for result in soup.select(".result"):
            link = result.select_one(".result__a")
            if not link:
                continue
            url = str(link.get("href") or "").strip()
            title = unescape(link.get_text(" ", strip=True))
            snippet_el = result.select_one(".result__snippet")
            snippet = unescape(snippet_el.get_text(" ", strip=True)) if snippet_el else ""
            if not url or not title:
                continue
            domain = _domain(url)
            paths.append(
                ResearchPath(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=domain,
                    official_hint=_official_hint(domain, title, snippet),
                )
            )
            if len(paths) >= limit:
                break
    except Exception:
        for match in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.I | re.S):
            url = unescape(match.group(1))
            title = re.sub(r"<[^>]+>", "", match.group(2))
            domain = _domain(url)
            paths.append(ResearchPath(title=unescape(title).strip(), url=url, domain=domain, official_hint=_official_hint(domain, title, "")))
            if len(paths) >= limit:
                break
    return paths


def research_public_web(user_query: str, triage: TaskTriage | dict[str, Any] | None = None) -> PublicResearchReport:
    """Research unknown tasks before autonomous execution.

    The function is intentionally read-only: it collects public search result metadata
    and never proceeds into login, agreement, purchase, or application flows.
    """

    triage_obj = triage if isinstance(triage, TaskTriage) else TaskTriage(**(triage or {}))
    query = str(user_query or "").strip()
    if not triage_obj.requires_research:
        return PublicResearchReport(status="skipped", query=query)
    if _research_mode() == "off":
        return PublicResearchReport(
            status="failed",
            query=query,
            needs_user_confirmation=True,
            sensitive_steps=list(triage_obj.sensitive_steps),
            error="public_research_disabled",
        )
    try:
        import requests

        search_query = f"{query} 공식 신청 방법 조건"
        url = "https://duckduckgo.com/html/?q=" + quote_plus(search_query)
        response = requests.get(url, timeout=float(os.getenv("COMMANDER_PUBLIC_RESEARCH_TIMEOUT", "8")), headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        paths = _parse_duckduckgo_results(response.text)
        official_paths = [path for path in paths if path.official_hint]
        domains = {path.domain for path in official_paths or paths if path.domain}
        return PublicResearchReport(
            status="completed",
            query=search_query,
            meaning="Public search metadata collected before autonomous execution.",
            possible_sites=paths,
            official_paths=official_paths,
            requirements=[],
            sensitive_steps=list(triage_obj.sensitive_steps),
            needs_user_choice=len(domains) > 1,
            needs_user_confirmation=triage_obj.risk_level == "sensitive" or bool(triage_obj.sensitive_steps),
            evidence={"result_count": len(paths), "official_result_count": len(official_paths)},
        )
    except Exception as exc:
        return PublicResearchReport(
            status="failed",
            query=query,
            possible_sites=[],
            official_paths=[],
            sensitive_steps=list(triage_obj.sensitive_steps),
            needs_user_confirmation=True,
            error=str(exc)[:240],
        )
