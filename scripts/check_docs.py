"""사람이 관리하는 Markdown 속성과 내부 링크를 결정론적으로 검사한다."""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import yaml


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
REQUIRED_PROPERTIES = {"title", "type", "area", "status", "updated", "tags"}
ALLOWED_TYPES = {
    "hub",
    "architecture",
    "decision",
    "guide",
    "plan",
    "reference",
    "retrospective",
}
ALLOWED_AREAS = {
    "project",
    "architecture",
    "documentation",
    "runtime",
    "observability",
    "reflex",
    "search",
    "sites",
}
ALLOWED_STATUSES = {"active", "planned", "historical", "deprecated"}
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class UniqueKeyLoader(yaml.SafeLoader):
    """중복 YAML 키를 허용하지 않는 검사 전용 로더."""


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.YAMLError(f"중복 YAML 키: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml(text: str, source: Path) -> dict:
    try:
        payload = yaml.load(text, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source.relative_to(ROOT)}: YAML 오류: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source.relative_to(ROOT)}: YAML 최상위 값은 객체여야 합니다.")
    return payload


def _split_frontmatter(text: str, source: Path) -> tuple[dict, str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"{source.relative_to(ROOT)}: frontmatter가 없습니다.")
    return _load_yaml(match.group(1), source), text[match.end() :]


def _without_fenced_code(text: str) -> str:
    """코드 예시 안의 제목과 링크가 문서 구조로 집계되지 않게 한다."""

    output: list[str] = []
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else ""
        if marker:
            if not fence:
                fence = marker
            elif fence == marker:
                fence = ""
            continue
        if not fence:
            output.append(line)
    without_fences = "\n".join(output)
    return re.sub(r"`[^`\n]+`", "", without_fences)


def _validate_metadata(path: Path, metadata: dict, body: str) -> list[str]:
    errors: list[str] = []
    label = str(path.relative_to(ROOT))
    missing = sorted(REQUIRED_PROPERTIES - set(metadata))
    if missing:
        errors.append(f"{label}: 필수 속성 누락: {', '.join(missing)}")
    if not str(metadata.get("title") or "").strip():
        errors.append(f"{label}: title이 비어 있습니다.")
    if metadata.get("type") not in ALLOWED_TYPES:
        errors.append(f"{label}: 허용되지 않은 type: {metadata.get('type')}")
    if metadata.get("area") not in ALLOWED_AREAS:
        errors.append(f"{label}: 허용되지 않은 area: {metadata.get('area')}")
    if metadata.get("status") not in ALLOWED_STATUSES:
        errors.append(f"{label}: 허용되지 않은 status: {metadata.get('status')}")
    if not isinstance(metadata.get("updated"), (date, str)):
        errors.append(f"{label}: updated는 YYYY-MM-DD 값이어야 합니다.")
    tags = metadata.get("tags")
    if not isinstance(tags, list) or "l2c" not in tags:
        errors.append(f"{label}: tags 목록에 l2c가 필요합니다.")
    h1_count = len(re.findall(r"^# ", _without_fenced_code(body), flags=re.MULTILINE))
    if h1_count != 1:
        errors.append(f"{label}: 본문 H1 개수는 1이어야 합니다. 현재 {h1_count}개입니다.")
    return errors


def _markdown_files() -> list[Path]:
    ignored = {
        ".git",
        ".venv",
        ".venv-app",
        ".venv-ocr",
        "venv",
        "node_modules",
    }
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(part in ignored for part in path.relative_to(ROOT).parts)
    )


def _validate_links(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        text = _without_fenced_code(path.read_text(encoding="utf-8"))
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>")
            if re.match(r"^(?:https?://|mailto:|file:|#)", target, flags=re.IGNORECASE):
                continue
            path_part = unquote(target.split("#", 1)[0])
            if not path_part.lower().endswith((".md", ".base")):
                continue
            resolved = (path.parent / path_part).resolve()
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(ROOT)}: 존재하지 않는 내부 링크: {target}"
                )
    return errors


def _validate_base(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = _load_yaml(path.read_text(encoding="utf-8"), path)
    except ValueError as exc:
        return [str(exc)]
    views = payload.get("views")
    if not isinstance(views, list) or not views:
        return [f"{path.relative_to(ROOT)}: views 목록이 필요합니다."]
    names: set[str] = set()
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            errors.append(f"{path.relative_to(ROOT)}: views[{index}]는 객체여야 합니다.")
            continue
        name = str(view.get("name") or "").strip()
        if not name:
            errors.append(f"{path.relative_to(ROOT)}: views[{index}] name이 없습니다.")
        elif name in names:
            errors.append(f"{path.relative_to(ROOT)}: 중복 view 이름: {name}")
        names.add(name)
        if view.get("type") not in {"table", "cards", "list", "map"}:
            errors.append(f"{path.relative_to(ROOT)}: {name}의 type이 잘못됐습니다.")
        if not isinstance(view.get("order"), list):
            errors.append(f"{path.relative_to(ROOT)}: {name}의 order 목록이 필요합니다.")
    return errors


def main() -> int:
    errors: list[str] = []
    docs = sorted(DOCS_DIR.rglob("*.md"))
    for path in docs:
        try:
            metadata, body = _split_frontmatter(path.read_text(encoding="utf-8"), path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        errors.extend(_validate_metadata(path, metadata, body))
    markdown_files = _markdown_files()
    errors.extend(_validate_links(markdown_files))
    bases = sorted(DOCS_DIR.rglob("*.base"))
    for path in bases:
        errors.extend(_validate_base(path))

    if errors:
        for error in errors:
            print(error)
        return 1
    print(
        "문서 검사 통과: "
        f"관리 문서 {len(docs)}개, 링크 대상 {len(markdown_files)}개, Base {len(bases)}개"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
