"""Site profile loader for commander-driven multi-site collection.

This module only reads the architecture files under agent/sites. It does not
start browsers; realtime tools may load these profiles to build site-specific
vision collection goals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SITES_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SITES_DIR / "registry.json"


class SiteProfileError(ValueError):
    """Raised when a requested site profile is missing or malformed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SiteProfileError(f"Site profile file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SiteProfileError(f"Invalid JSON in site profile file: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SiteProfileError(f"Site profile JSON must be an object: {path}")
    return data


def load_registry() -> dict[str, Any]:
    registry = _read_json(REGISTRY_PATH)
    sites = registry.get("sites")
    if not isinstance(sites, list):
        raise SiteProfileError("registry.json must contain a 'sites' list")
    return registry


def list_supported_sites(enabled_only: bool = True) -> list[dict[str, Any]]:
    """Return registry entries for known sites."""
    sites = list(load_registry()["sites"])
    if enabled_only:
        sites = [site for site in sites if site.get("enabled", True)]
    return sites


def get_site_entry(site: str) -> dict[str, Any]:
    """Resolve a site by slug, display name, or domain substring."""
    needle = (site or "").strip().lower()
    if not needle:
        raise SiteProfileError("site is required")

    for entry in list_supported_sites(enabled_only=False):
        slug = str(entry.get("slug", "")).lower()
        display_name = str(entry.get("display_name", "")).lower()
        domains = [str(domain).lower() for domain in entry.get("domains", [])]
        if needle == slug or needle == display_name or needle in domains:
            return entry
        if any(needle in domain or domain in needle for domain in domains):
            return entry
    raise SiteProfileError(f"Unsupported site: {site}")


def _profile_path(entry: dict[str, Any], key: str) -> Path:
    rel = entry.get(key)
    if not isinstance(rel, str) or not rel:
        raise SiteProfileError(f"Registry entry for {entry.get('slug')} missing {key}")
    path = (SITES_DIR / rel).resolve()
    try:
        path.relative_to(SITES_DIR.resolve())
    except ValueError as exc:
        raise SiteProfileError(f"Site profile path escapes sites dir: {path}") from exc
    return path


def load_site_profile(site: str) -> dict[str, Any]:
    """Load registry entry, structured manual, prompt text, and tool policy."""
    entry = get_site_entry(site)
    manual = _read_json(_profile_path(entry, "manual_path"))
    tools = _read_json(_profile_path(entry, "tools_path"))
    prompt_path = _profile_path(entry, "prompt_path")
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SiteProfileError(f"Site prompt file not found: {prompt_path}") from exc

    manual_site = str(manual.get("site", ""))
    if manual_site and manual_site != entry.get("slug"):
        raise SiteProfileError(
            f"Manual site mismatch: registry={entry.get('slug')} manual={manual_site}"
        )

    return {
        "entry": entry,
        "manual": manual,
        "prompt": prompt,
        "tools": tools,
    }


__all__ = [
    "SiteProfileError",
    "load_registry",
    "list_supported_sites",
    "get_site_entry",
    "load_site_profile",
]