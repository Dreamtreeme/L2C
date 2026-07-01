"""Hugging Face 모델 파일을 Range resume 방식으로 단순 다운로드합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from huggingface_hub import HfApi


def _resolve_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _should_download(filename: str) -> bool:
    lowered = filename.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".svg", ".md")):
        return False
    return True


def _download_file(repo_id: str, filename: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    current_size = out_path.stat().st_size if out_path.exists() else 0
    headers = {"Range": f"bytes={current_size}-"} if current_size else {}
    with requests.get(
        _resolve_url(repo_id, filename),
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        mode = "ab" if current_size and response.status_code == 206 else "wb"
        if mode == "wb":
            current_size = 0
        total_header = response.headers.get("content-length")
        remaining = int(total_header) if total_header and total_header.isdigit() else 0
        expected_total = current_size + remaining if remaining else 0
        print(
            f"DOWNLOAD {filename} start={current_size} "
            f"expected_total={expected_total or 'unknown'}"
        )
        downloaded = current_size
        next_report = downloaded + 256 * 1024 * 1024
        with out_path.open(mode + "") as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"PROGRESS {filename} {downloaded / 1024 / 1024:.1f} MB")
                    next_report = downloaded + 256 * 1024 * 1024
    print(f"DONE {filename} {out_path.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--local-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.local_dir)
    files = [name for name in HfApi().list_repo_files(args.repo_id) if _should_download(name)]
    for filename in files:
        _download_file(args.repo_id, filename, out_dir / filename)


if __name__ == "__main__":
    main()
