"""저장소 산출물에 기록할 파일 경로를 이식 가능한 형태로 만든다."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def portable_repo_path(
    value: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> str:
    """저장소 내부 경로는 POSIX 형식의 루트 상대경로로 직렬화한다."""

    raw = str(value).strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    resolved = path if path.is_absolute() else Path(base_dir or Path.cwd()) / path
    resolved = resolved.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


__all__ = ["REPOSITORY_ROOT", "portable_repo_path"]
