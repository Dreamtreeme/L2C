from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class CommanderState(TypedDict, total=False):
    """State for the top-level commander orchestration graph."""

    user_query: str
    site_queue: List[str]
    current_site: str
    current_site_index: int
    current_run_id: str
    review_attempt: int
    review_feedback: str
    max_review_retries: int
    current_worker_result: Dict[str, Any]
    current_submission: Dict[str, Any]
    current_review: Dict[str, Any]
    current_submission_id: str
    task_triage: Dict[str, Any]
    research_report: Dict[str, Any]
    task_context: Dict[str, Any]
    worker_submissions: Annotated[List[Dict[str, Any]], operator.add]
    reviews: Annotated[List[Dict[str, Any]], operator.add]
    accepted_sites: Annotated[List[str], operator.add]
    failed_sites: Annotated[List[Dict[str, Any]], operator.add]
    pending_human_approval: bool
    human_approval_reason: str
    human_approval_request: Dict[str, Any]
    intermediate_report: Dict[str, Any]
    db_results: str
    final_answer: str
    done: bool
