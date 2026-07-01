"""NuExtract 계열 모델로 상세 OCR 텍스트 정제만 단독 벤치합니다."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError
from transformers import AutoModelForMultimodalLM, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.schema.jd_schema import JobPosting


DEFAULT_SUBMISSION_ID = "worker-20260701010743-25ecd359:0"
DEFAULT_MODEL = "numind/NuExtract3"


def _dedupe_marker_texts(submission_id: str, db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT payload_json FROM worker_submissions WHERE submission_id=?",
        (submission_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"worker submission not found: {submission_id}")

    payload = json.loads(row["payload_json"])
    lines_by_url: dict[str, list[str]] = defaultdict(list)
    seen_by_url: dict[str, set[str]] = defaultdict(set)
    card_by_url: dict[str, dict[str, str]] = {}

    for episode in payload.get("feedback_episodes") or []:
        before = ((episode.get("observation") or {}).get("before") or {})
        url = str(before.get("url") or "")
        if "/wd/" not in url:
            continue

        proposal = episode.get("proposal") or {}
        active_card = proposal.get("active_result_card") or {}
        if active_card and url not in card_by_url:
            card_by_url[url] = {
                "title": active_card.get("title") or active_card.get("target_label") or "",
                "company": active_card.get("company") or "",
            }

        for raw_text in before.get("marker_texts") or []:
            text = str(raw_text or "").strip()
            if not text:
                continue
            if "상호작용 가능한 요소" in text:
                continue
            if text in seen_by_url[url]:
                continue
            seen_by_url[url].add(text)
            lines_by_url[url].append(text)

    documents = []
    for url, lines in lines_by_url.items():
        documents.append(
            {
                "url": url,
                "active_result_card": card_by_url.get(url, {}),
                "ocr_text": "\n".join(lines),
                "line_count": len(lines),
            }
        )
    return documents


def _job_template() -> dict[str, Any]:
    return {
        "company_name": "string",
        "position": "string",
        "url": "string",
        "job_category": "string",
        "experience_level": "string",
        "education": "string",
        "employment_type": "string",
        "location": "string",
        "deadline": "string",
        "tech_stack": ["string"],
        "main_tasks": ["string"],
        "requirements": ["string"],
        "preferred": ["string"],
        "benefits": ["string"],
        "salary": "string",
        "source_platform": "string",
        "experience_min": "integer",
        "experience_max": "integer",
        "experience_text": "string",
    }


def _build_prompt(document: dict[str, Any]) -> list[dict[str, Any]]:
    instructions = (
        "원티드 채용공고 OCR 텍스트에서 채용공고 1건만 추출하십시오. "
        "북마크, 브라우저 메뉴, 보상 배지, 추천인 현금, 로그인 문구 같은 주변 UI 노이즈는 무시하십시오. "
        "OCR에 없는 사실은 만들지 말고, 알 수 없으면 null 또는 빈 배열을 사용하십시오. "
        "url은 입력 current_url을 그대로 보존하십시오. "
        "active_result_card의 title/company가 OCR 노이즈보다 명확하면 회사명/직무명 보정에 사용하십시오."
    )
    context = {
        "current_url": document["url"],
        "active_result_card": document.get("active_result_card") or {},
        "ocr_text": document["ocr_text"],
    }
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(context, ensure_ascii=False, indent=2),
                }
            ],
        }
    ], instructions


def _parse_output(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        job = JobPosting.model_validate(parsed)
        return job.model_dump(exclude_none=True), ""
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--submission-id", default=DEFAULT_SUBMISSION_ID)
    parser.add_argument("--db", default="data/jobs.db")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-docs", type=int, default=2)
    args = parser.parse_args()

    documents = _dedupe_marker_texts(args.submission_id, Path(args.db))[: args.max_docs]
    if not documents:
        raise SystemExit("no detail documents reconstructed")

    print(
        json.dumps(
            {
                "model": args.model,
                "submission_id": args.submission_id,
                "doc_count": len(documents),
                "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            },
            ensure_ascii=False,
        )
    )

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        max_memory={0: "8.5GiB", "cpu": "22GiB"} if torch.cuda.is_available() else None,
    ).eval()
    print(json.dumps({"load_sec": round(time.perf_counter() - load_started, 3)}, ensure_ascii=False))

    template = json.dumps(_job_template(), ensure_ascii=False)
    results = []
    for index, document in enumerate(documents):
        messages, instructions = _build_prompt(document)
        text = processor.tokenizer.apply_chat_template(
            messages,
            template=template,
            instructions=instructions,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=[text],
            images=None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        started = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[-1] :]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        duration = time.perf_counter() - started
        parsed, error = _parse_output(output_text)
        result = {
            "index": index,
            "url": document["url"],
            "line_count": document["line_count"],
            "char_count": len(document["ocr_text"]),
            "duration_sec": round(duration, 3),
            "parse_ok": parsed is not None,
            "error": error[:500],
            "raw_output": output_text[:3000],
            "parsed": parsed,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "total_infer_sec": round(sum(item["duration_sec"] for item in results), 3),
                "parse_ok_count": sum(1 for item in results if item["parse_ok"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
