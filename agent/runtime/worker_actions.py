"""작업자 행동 종류와 화면 전환 특성의 공통 계약."""

UI_ACTIONS = frozenset(
    {
        "click_marker",
        "type_in_marker",
        "scroll",
        "press_key",
        "open_browser",
        "close_browser",
        "close_current_tab",
        "switch_tab",
        "go_back",
    }
)

STATE_UPDATE_ACTIONS = frozenset(
    {
        "update_extracted_info",
        "finish_detail_reading",
        "set_job_card_queue",
    }
)

TERMINAL_ACTIONS = frozenset({"finish_task"})

RETURN_ACTIONS = frozenset(
    {
        "go_back",
        "close_current_tab",
        "switch_tab",
    }
)

URL_STALE_ACTIONS = frozenset(
    {
        "click_marker",
        "press_key",
        "open_browser",
        "close_browser",
        "close_current_tab",
        "switch_tab",
        "go_back",
    }
)

DIRECT_SCREEN_ACTION_SOURCES = frozenset(
    {
        "reflex",
        "job_card_queue",
        "page_policy",
        "duplicate_job_policy",
    }
)


__all__ = [
    "DIRECT_SCREEN_ACTION_SOURCES",
    "RETURN_ACTIONS",
    "STATE_UPDATE_ACTIONS",
    "TERMINAL_ACTIONS",
    "UI_ACTIONS",
    "URL_STALE_ACTIONS",
]
