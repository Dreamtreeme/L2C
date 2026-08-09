"""DOM으로 추출한 채용공고 본문을 공통 스키마로 정제한다."""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm.clients import get_structured_google_model
from agent.llm.policy import lightweight_model_name
from agent.observability.run_context import invoke_with_metrics
from agent.prompts.detail_extraction import build_detail_extraction_system_prompt
from agent.utils.model_conversion import parse_model_payload
from shared.schema.jd_schema import JobPosting

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or lightweight_model_name()
        self.model = get_structured_google_model(
            self.model_name,
            JobPosting,
            temperature=0.0,
            execution_role="lightweight",
        )

    def extract_from_text(self, text: str) -> dict:
        """DOM 본문 하나를 채용공고 JSON으로 변환한다."""

        if not text.strip():
            return {}
        messages = [
            SystemMessage(
                content=build_detail_extraction_system_prompt(
                    "채용공고 DOM 본문 한 건을 JobPosting 스키마로 정리하십시오. "
                    "본문에 없는 사실은 만들지 말고 알 수 없는 필드는 비우십시오."
                )
            ),
            HumanMessage(content=text),
        ]
        started = time.perf_counter()
        response = invoke_with_metrics(
            self.model,
            messages,
            "classic_extraction",
            stream=True,
        )
        logger.info(
            "[LLMEngine] 텍스트 정제 완료 (모델=%s, %.1fs)",
            self.model_name,
            time.perf_counter() - started,
        )
        return parse_model_payload(response, JobPosting).model_dump(mode="json")
