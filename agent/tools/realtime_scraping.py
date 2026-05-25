import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def realtime_scraping(company: str = None, tech_stack: str = None) -> str:
    """
    비전 기반 자율 에이전트를 구동하여 특정 기업(company)이나 기술 스택(tech_stack)에 맞는
    최신 채용 공고를 실시간으로 수집하고 데이터베이스에 적재하는 도구입니다.
    내부적으로 SoM(Set-of-Mark) 마커 기반 화면 인식 → LLM 추론 → 물리 조작의
    자율 수집 LangGraph 워크플로우를 기동하며, 모든 도구 호출 궤적이
    LangSmith에 저장되어 Playwright 스크립트 자동 생성의 학습 데이터로 환원됩니다.
    데이터베이스에 정보가 없거나 부족할 때 호출되어 RAG 지식베이스를 동적으로 보강합니다.
    """
    # 검색 키워드 조합
    search_keyword = ""
    if company and tech_stack:
        search_keyword = f"{company} {tech_stack}"
    elif company:
        search_keyword = company
    elif tech_stack:
        search_keyword = tech_stack
    else:
        return "수집 실패: 검색 키워드(company, tech_stack)가 모두 누락되었습니다."

    logger.info(f"[realtime_scraping] Invoking vision agent graph for keyword: '{search_keyword}'")

    try:
        # 비전 자율 수집 LangGraph 워크플로우 빌드 및 기동
        from agent.graph.workflow import build_graph

        app = build_graph()

        # 검색 키워드를 기반으로 자율 수집 목표(goal) 구성
        goal = (
            f"원티드(https://www.wanted.co.kr)에서 '{search_keyword}'를 검색하여 "
            f"검색 결과에 노출된 채용 공고들의 상세 페이지를 순회하며 "
            f"회사명, 직무명, 주요업무, 자격요건, 우대사항, 혜택 등의 "
            f"모든 정보를 빠짐없이 수집하세요."
        )

        # 초기 상태 구성 (GraphState 스키마에 맞춤)
        initial_state = {
            "goal": goal,
            "ui_context": "",
            "current_markers": [],
            "action_history": [],
            "recent_images": [],
            "marked_image": "",
            "error_count": 0,
            "is_finished": False,
            "collected_data": [],
            "extracted_jd": {},
            "last_action_result": None,
            "plan": [],
            "current_plan_step": 0,
        }

        logger.info(f"[realtime_scraping] Starting autonomous vision collection graph with goal: {goal}")

        # LangGraph 앱 실행 (동기 invoke)
        # 모든 perception → reasoning → action 루프가 자율적으로 순환하며
        # 도구 호출 궤적이 LangSmith 트레이스에 자동 기록됨
        final_state = app.invoke(initial_state)

        # 수집 결과 분석
        collected = final_state.get("collected_data", [])
        extracted = final_state.get("extracted_jd", {})
        is_finished = final_state.get("is_finished", False)

        if is_finished and (collected or extracted):
            # 수집된 데이터를 DB에 전처리 및 적재
            _persist_collected_data(extracted, search_keyword)

            item_count = len(collected) if collected else (1 if extracted else 0)
            logger.info(f"[realtime_scraping] Vision agent collection completed. Items: {item_count}")
            return (
                f"실시간 비전 자율 수집 및 적재 완료: '{search_keyword}' 키워드로 "
                f"총 {item_count}건의 채용 공고 정보가 데이터베이스에 성공적으로 동적 업데이트되었습니다."
            )
        else:
            logger.warning(f"[realtime_scraping] Vision agent finished but no data collected for '{search_keyword}'")
            return f"실시간 수집 완료: '{search_keyword}'에 매칭되는 유효한 채용 정보를 찾지 못했습니다."

    except Exception as e:
        logger.error(f"[realtime_scraping] Vision agent execution error: {e}", exc_info=True)
        return f"실시간 수집 오류: {e}"


def _persist_collected_data(extracted_jd: dict, keyword: str) -> None:
    """
    비전 에이전트가 수집한 extracted_jd 데이터를 전처리 후 DB에 UPSERT합니다.
    extracted_jd는 단일 공고 dict이거나, '공고목록' 키 아래 리스트를 담고 있을 수 있습니다.
    """
    from shared.config import DB_PATH
    from shared.db.database import Database
    from agent.utils.preprocessor import Preprocessor

    db = Database(DB_PATH)

    # 공고 목록 추출 (리스트 or 단건)
    if "공고목록" in extracted_jd:
        job_list = extracted_jd["공고목록"]
        if not isinstance(job_list, list):
            job_list = [job_list]
    else:
        job_list = [extracted_jd] if extracted_jd else []

    for idx, job in enumerate(job_list):
        if not isinstance(job, dict) or not job:
            continue

        try:
            company_name = job.get("회사명", job.get("company_name", ""))
            position = job.get("직무명", job.get("position", ""))
            full_text = _build_full_text(job)

            if not full_text:
                logger.warning(f"[_persist] Skipping job #{idx}: empty text body")
                continue

            # 전처리 파이프라인 적용
            cleaned_text = Preprocessor.clean_text(full_text)
            text_parts = [position, cleaned_text] if position else [cleaned_text]
            normalized_stack = Preprocessor.extract_tech_stacks(text_parts)
            requirements_list = [cleaned_text]  # parse_experience는 list[str]를 기대
            exp_min, exp_max, exp_text = Preprocessor.parse_experience(position, requirements_list)
            content_hash = Preprocessor.generate_content_hash(company_name, position, requirements_list)

            # 원본 URL 복원 (에이전트가 수집했을 수 있음)
            url = job.get("url", job.get("URL", f"https://www.wanted.co.kr/search?query={keyword}&idx={idx}"))

            db_payload = {
                "company_name": company_name,
                "position": position,
                "tech_stack": normalized_stack,
                "raw_ocr_text": cleaned_text,
                "source_platform": "Wanted",
                "experience_min": exp_min,
                "experience_max": exp_max,
                "experience_text": exp_text,
                "content_hash": content_hash,
            }
            db.upsert(url=url, data=db_payload, embedding=None)
            logger.info(f"[_persist] Successfully upserted job #{idx}: {company_name} - {position}")

        except Exception as e:
            logger.error(f"[_persist] Failed to persist job #{idx}: {e}")
            continue


def _build_full_text(job: dict) -> str:
    """job dict의 주요 필드들을 하나의 문자열로 조합합니다."""
    parts = []
    for key in ["주요업무", "자격요건", "우대사항", "혜택", "복지",
                 "description", "requirements", "preferred", "benefits",
                 "raw_ocr_text"]:
        val = job.get(key)
        if val:
            if isinstance(val, list):
                parts.append(f"{key}: " + ", ".join(str(v) for v in val))
            else:
                parts.append(f"{key}: {val}")

    # 직접 텍스트 필드가 없으면 전체 dict를 문자열화
    if not parts:
        import json
        parts.append(json.dumps(job, ensure_ascii=False))

    return "\n".join(parts)
