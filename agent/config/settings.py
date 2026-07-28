"""환경 변수를 시작 시 검증하는 애플리케이션 설정 계약."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
_ENV_FILE = BASE_DIR / ".env"


class SectionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class PathSettings(SectionSettings):
    db_path: Path = Field(Path("data/jobs.db"), validation_alias="DB_PATH")
    json_dir: Path = Field(Path("data/json"), validation_alias="JSON_OUTPUT_DIR")
    log_dir: Path = Field(Path("logs"), validation_alias="LOG_DIR")
    screenshot_dir: Path = Field(Path("data/screenshots"), validation_alias="SCREENSHOT_DIR")
    browser_profile_dir: Path = Field(
        Path("data/browser_profile"),
        validation_alias="VISION_BROWSER_PROFILE_DIR",
    )
    llm_pricing_file: Path | None = Field(None, validation_alias="LLM_PRICING_FILE")

    @model_validator(mode="after")
    def resolve_paths(self) -> "PathSettings":
        for name in (
            "db_path",
            "json_dir",
            "log_dir",
            "screenshot_dir",
            "browser_profile_dir",
            "llm_pricing_file",
        ):
            value = getattr(self, name)
            if value is not None and not value.is_absolute():
                setattr(self, name, (BASE_DIR / value).resolve())
        return self


class ModelSettings(SectionSettings):
    commander_model: str = Field("gemini-3.6-flash", validation_alias="COMMANDER_MODEL")
    worker_reasoning_model: str | None = Field(None, validation_alias="VISION_WORKER_REASONING_MODEL")
    worker_reasoning_thinking_level: str = Field(
        "low",
        validation_alias="VISION_WORKER_REASONING_THINKING_LEVEL",
    )
    lightweight_model: str = Field(
        "gemini-3.5-flash-lite",
        validation_alias="VISION_LIGHTWEIGHT_MODEL",
    )
    lightweight_max_output_tokens: int = Field(
        1536,
        ge=0,
        le=65536,
        validation_alias="VISION_LIGHTWEIGHT_MAX_OUTPUT_TOKENS",
    )
    detail_final_extraction_model: str | None = Field(
        None,
        validation_alias="VISION_DETAIL_FINAL_EXTRACTION_MODEL",
    )
    jd_normalization_model: str | None = Field(None, validation_alias="VISION_JD_NORMALIZATION_MODEL")
    worker_summary_model: str | None = Field(None, validation_alias="VISION_WORKER_SUMMARY_MODEL")
    worker_review_model: str | None = Field(None, validation_alias="VISION_WORKER_REVIEW_MODEL")
    search_intent_model: str | None = Field(None, validation_alias="VISION_SEARCH_INTENT_MODEL")
    result_card_selector_model: str | None = Field(
        None,
        validation_alias="VISION_RESULT_CARD_SELECTOR_MODEL",
    )
    recipe_critic_model: str | None = Field(None, validation_alias="VISION_RECIPE_CRITIC_MODEL")
    ollama_host: str = Field("http://localhost:11434", validation_alias="OLLAMA_HOST")
    ollama_model: str = Field("qwen3:8b", validation_alias="OLLAMA_MODEL")
    ollama_num_predict: int = Field(2048, ge=1, le=65536, validation_alias="LLM_NUM_PREDICT")
    detail_ollama_num_predict: int = Field(
        2048,
        ge=1,
        le=65536,
        validation_alias="VISION_DETAIL_OLLAMA_NUM_PREDICT",
    )
    detail_openai_max_output_tokens: int = Field(
        2048,
        ge=1,
        le=65536,
        validation_alias="VISION_DETAIL_OPENAI_MAX_OUTPUT_TOKENS",
    )
    detail_openai_timeout_sec: float = Field(
        120.0,
        gt=0,
        le=600,
        validation_alias="VISION_DETAIL_OPENAI_TIMEOUT",
    )
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0, validation_alias="LLM_TEMPERATURE")
    gemini_api_key: SecretStr | None = Field(None, validation_alias="GEMINI_API_KEY")
    openai_api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")
    worknet_api_key: SecretStr | None = Field(None, validation_alias="WORKNET_API_KEY")

    @field_validator("worker_reasoning_thinking_level")
    @classmethod
    def validate_thinking_level(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"minimal", "low", "medium", "high"}:
            raise ValueError("사고 수준은 minimal, low, medium, high 중 하나여야 합니다.")
        return normalized

    def model_override(self, env_name: str) -> str | None:
        field_by_env = {
            "VISION_DETAIL_FINAL_EXTRACTION_MODEL": "detail_final_extraction_model",
            "VISION_JD_NORMALIZATION_MODEL": "jd_normalization_model",
            "VISION_WORKER_SUMMARY_MODEL": "worker_summary_model",
            "VISION_WORKER_REVIEW_MODEL": "worker_review_model",
            "VISION_SEARCH_INTENT_MODEL": "search_intent_model",
            "VISION_RESULT_CARD_SELECTOR_MODEL": "result_card_selector_model",
            "VISION_RECIPE_CRITIC_MODEL": "recipe_critic_model",
            "VISION_WORKER_REASONING_MODEL": "worker_reasoning_model",
            "COMMANDER_MODEL": "commander_model",
            "VISION_LIGHTWEIGHT_MODEL": "lightweight_model",
        }
        field_name = field_by_env.get(env_name)
        return str(getattr(self, field_name) or "").strip() if field_name else None


class BrowserSettings(SectionSettings):
    playwright_headless: bool = Field(True, validation_alias="PLAYWRIGHT_HEADLESS")
    playwright_timeout_ms: int = Field(30000, gt=0, le=300000, validation_alias="PLAYWRIGHT_TIMEOUT_MS")
    chrome_window_width: int = Field(1024, ge=640, le=7680, validation_alias="CHROME_WINDOW_WIDTH")
    chrome_window_height: int = Field(768, ge=480, le=4320, validation_alias="CHROME_WINDOW_HEIGHT")
    page_load_wait_sec: float = Field(4.0, ge=0, le=120, validation_alias="PAGE_LOAD_WAIT_SEC")
    worker_preopen_browser: bool = Field(True, validation_alias="VISION_WORKER_PREOPEN_BROWSER")
    close_browser_after_run: bool = Field(True, validation_alias="VISION_CLOSE_BROWSER_AFTER_RUN")
    input_capture_initial_wait_sec: float = Field(
        0.7,
        ge=0,
        le=10,
        validation_alias="VISION_INPUT_CAPTURE_INITIAL_WAIT_SEC",
    )
    use_configured_window_size: bool = Field(True, validation_alias="VISION_BROWSER_WINDOW_SIZE")
    reset_zoom: bool = Field(True, validation_alias="VISION_BROWSER_RESET_ZOOM")
    executable: str | None = Field(None, validation_alias="VISION_BROWSER_EXECUTABLE")
    vision_window_width: int = Field(1976, ge=640, le=7680, validation_alias="VISION_BROWSER_WINDOW_WIDTH")
    vision_window_height: int = Field(2129, ge=480, le=4320, validation_alias="VISION_BROWSER_WINDOW_HEIGHT")
    resize_wait_sec: float = Field(0.12, ge=0, le=10, validation_alias="VISION_BROWSER_RESIZE_WAIT_SEC")
    zoom_reset_wait_sec: float = Field(0.08, ge=0, le=10, validation_alias="VISION_BROWSER_ZOOM_RESET_WAIT_SEC")
    open_wait_sec: float = Field(0.8, ge=0, le=30, validation_alias="VISION_BROWSER_OPEN_WAIT_SEC")
    action_pause_sec: float = Field(0.03, ge=0, le=5, validation_alias="VISION_ACTION_PAUSE_SEC")
    action_move_duration_sec: float = Field(0.05, ge=0, le=5, validation_alias="VISION_ACTION_MOVE_DURATION_SEC")
    action_input_delay_sec: float = Field(0.02, ge=0, le=5, validation_alias="VISION_ACTION_INPUT_DELAY_SEC")
    action_clipboard_delay_sec: float = Field(0.02, ge=0, le=5, validation_alias="VISION_ACTION_CLIPBOARD_DELAY_SEC")
    scroll_page_steps: int = Field(8, ge=1, le=100, validation_alias="VISION_SCROLL_PAGE_STEPS")
    scroll_small_steps: int = Field(3, ge=1, le=100, validation_alias="VISION_SCROLL_SMALL_STEPS")
    cors_allow_origins: Annotated[tuple[str, ...], NoDecode] = Field(
        ("http://127.0.0.1:8000", "http://localhost:8000"),
        validation_alias="CORS_ALLOW_ORIGINS",
    )
    local_api_allowed_hosts: Annotated[tuple[str, ...], NoDecode] = Field(
        ("127.0.0.1", "localhost", "testserver", "[::1]"),
        validation_alias="LOCAL_API_ALLOWED_HOSTS",
    )

    @field_validator("cors_allow_origins", "local_api_allowed_hosts", mode="before")
    @classmethod
    def split_csv(cls, value):
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value


class VisionSettings(SectionSettings):
    recursion_limit: int = Field(180, ge=10, le=500, validation_alias="VISION_AGENT_RECURSION_LIMIT")
    recursion_limit_increment: int = Field(90, ge=1, le=500, validation_alias="VISION_AGENT_RECURSION_LIMIT_INCREMENT")
    hitl_on_recursion_limit: bool = Field(True, validation_alias="VISION_HITL_ON_RECURSION_LIMIT")
    auto_finish_on_target: bool = Field(True, validation_alias="VISION_AUTO_FINISH_ON_TARGET")
    ui_text_marker_limit: int = Field(90, ge=1, le=1000, validation_alias="VISION_UI_TEXT_MARKER_LIMIT")
    ui_icon_marker_limit: int = Field(45, ge=0, le=1000, validation_alias="VISION_UI_ICON_MARKER_LIMIT")
    reasoning_action_history_limit: int = Field(
        2,
        ge=1,
        le=20,
        validation_alias="VISION_REASONING_ACTION_HISTORY_LIMIT",
    )
    reasoning_image_max_dim: int = Field(
        768,
        ge=256,
        le=4096,
        validation_alias="VISION_REASONING_IMAGE_MAX_DIM",
    )
    reasoning_image_quality: int = Field(
        60,
        ge=20,
        le=100,
        validation_alias="VISION_REASONING_IMAGE_QUALITY",
    )
    reasoning_screen_guard: bool = Field(True, validation_alias="VISION_REASONING_SCREEN_GUARD")
    reasoning_stale_phash_max_distance: int = Field(
        10,
        ge=0,
        le=64,
        validation_alias="VISION_REASONING_STALE_PHASH_MAX_DISTANCE",
    )
    detail_action_marker_limit: int = Field(35, ge=1, le=500, validation_alias="VISION_DETAIL_ACTION_MARKER_LIMIT")
    detail_section_min_text_markers: int = Field(120, ge=1, le=2000, validation_alias="VISION_DETAIL_SECTION_MIN_TEXT_MARKERS")
    detail_ocr_max_lines: int = Field(90, ge=1, le=2000, validation_alias="VISION_DETAIL_OCR_MAX_LINES")
    detail_section_max_line_chars: int = Field(180, ge=20, le=2000, validation_alias="VISION_DETAIL_SECTION_MAX_LINE_CHARS")
    detail_buffer_max_lines: int = Field(260, ge=1, le=5000, validation_alias="VISION_DETAIL_OCR_BUFFER_MAX_LINES")
    detail_buffer_max_line_chars: int = Field(220, ge=20, le=4000, validation_alias="VISION_DETAIL_OCR_BUFFER_MAX_LINE_CHARS")
    detail_final_ocr_max_chars: int = Field(16000, ge=1000, le=200000, validation_alias="VISION_DETAIL_FINAL_OCR_MAX_CHARS")
    detail_section_context_enabled: bool = Field(True, validation_alias="VISION_DETAIL_SECTION_CONTEXT_ENABLED")
    detail_lightweight_marked_image_enabled: bool = Field(True, validation_alias="VISION_DETAIL_LIGHTWEIGHT_MARKED_IMAGE_ENABLED")
    detail_ocr_buffer_enabled: bool = Field(True, validation_alias="VISION_DETAIL_OCR_BUFFER_ENABLED")
    search_intent_mode: str = Field("llm", validation_alias="VISION_SEARCH_INTENT_MODE")
    ui_analysis_cache_limit: int = Field(8, ge=0, le=128, validation_alias="VISION_UI_ANALYSIS_CACHE_LIMIT")
    page_content_top_px: str = Field("auto", validation_alias="VISION_PAGE_CONTENT_TOP_PX")
    som_crop_top: str = Field("auto", validation_alias="VISION_SOM_CROP_TOP")
    capture_initial_wait_sec: float = Field(0.16, ge=0, le=10, validation_alias="VISION_CAPTURE_INITIAL_WAIT_SEC")
    page_content_bottom_ignore_px: int = Field(80, ge=0, le=2000, validation_alias="VISION_PAGE_CONTENT_BOTTOM_IGNORE_PX")
    page_quality_sample_width: int = Field(240, ge=32, le=4096, validation_alias="VISION_PAGE_QUALITY_SAMPLE_WIDTH")
    page_blank_max_stddev: float = Field(6.0, ge=0, le=255, validation_alias="VISION_PAGE_BLANK_MAX_STDDEV")
    page_blank_max_edge_mean: float = Field(1.2, ge=0, le=255, validation_alias="VISION_PAGE_BLANK_MAX_EDGE_MEAN")
    page_blank_min_dominant_ratio: float = Field(0.98, ge=0, le=1, validation_alias="VISION_PAGE_BLANK_MIN_DOMINANT_RATIO")
    page_capture_retry_sec: float = Field(0.4, ge=0, le=30, validation_alias="VISION_PAGE_CAPTURE_RETRY_SEC")
    page_ready_timeout_sec: float = Field(15.0, gt=0, le=300, validation_alias="VISION_PAGE_READY_TIMEOUT_SEC")
    low_information_max_capture_cycles: int = Field(
        2,
        ge=1,
        le=10,
        validation_alias="VISION_LOW_INFORMATION_MAX_CAPTURE_CYCLES",
    )
    url_key_pause_sec: float = Field(0.015, ge=0, le=2, validation_alias="VISION_URL_KEY_PAUSE_SEC")
    url_copy_wait_sec: float = Field(0.015, ge=0, le=2, validation_alias="VISION_URL_COPY_WAIT_SEC")
    url_copy_timeout_sec: float = Field(0.25, gt=0, le=10, validation_alias="VISION_URL_COPY_TIMEOUT_SEC")
    som_crop_fallback_top_px: int = Field(140, ge=0, le=2000, validation_alias="VISION_SOM_CROP_FALLBACK_TOP_PX")
    som_crop_scan_min_y: int = Field(80, ge=0, le=2000, validation_alias="VISION_SOM_CROP_SCAN_MIN_Y")
    som_crop_scan_max_y: int = Field(320, ge=1, le=4000, validation_alias="VISION_SOM_CROP_SCAN_MAX_Y")
    som_crop_sample_width: int = Field(256, ge=32, le=4096, validation_alias="VISION_SOM_CROP_SAMPLE_WIDTH")
    som_crop_min_row_delta: float = Field(60.0, ge=0, le=765, validation_alias="VISION_SOM_CROP_MIN_ROW_DELTA")
    transition_change_max_wait_sec: float = Field(1.2, ge=0, le=30, validation_alias="VISION_TRANSITION_CHANGE_MAX_WAIT_SEC")
    transition_change_check_sec: float = Field(0.08, gt=0, le=5, validation_alias="VISION_TRANSITION_CHANGE_CHECK_SEC")
    stable_max_wait_sec: float = Field(2.0, ge=0, le=60, validation_alias="VISION_STABLE_MAX_WAIT_SEC")
    stable_check_interval_sec: float = Field(0.04, gt=0, le=5, validation_alias="VISION_STABLE_CHECK_INTERVAL_SEC")
    stable_threshold_percent: float = Field(1.0, ge=0, le=100, validation_alias="VISION_STABLE_THRESHOLD_PERCENT")
    stable_sample_width: int = Field(360, ge=32, le=4096, validation_alias="VISION_STABLE_SAMPLE_WIDTH")


class OcrSettings(SectionSettings):
    python_executable: Path = Field(
        Path(".venv-ocr/Scripts/python.exe"),
        validation_alias="PADDLE_OCR_PYTHON",
    )
    yolo_config_dir: Path = Field(
        Path(".cache/ultralytics"),
        validation_alias="YOLO_CONFIG_DIR",
    )
    paddlex_cache_dir: Path | None = Field(None, validation_alias="PADDLE_PDX_CACHE_HOME")
    model_source: str = Field("BOS", validation_alias="PADDLE_PDX_MODEL_SOURCE")
    language: str = Field("korean", validation_alias="PADDLEOCR_LANG")
    ocr_version: str = Field("PP-OCRv5", validation_alias="PADDLEOCR_VERSION")
    use_gpu: bool | None = Field(None, validation_alias="PADDLEOCR_USE_GPU")
    cuda_bin_dir: Path | None = Field(None, validation_alias="PADDLE_CUDA_BIN_DIR")
    cudnn_bin_dir: Path | None = Field(None, validation_alias="PADDLE_CUDNN_BIN_DIR")
    worker_start_timeout_sec: float = Field(
        45.0,
        ge=1,
        le=300,
        validation_alias="SOM_OCR_WORKER_START_TIMEOUT_SEC",
    )
    request_timeout_sec: float = Field(
        20.0,
        ge=1,
        le=300,
        validation_alias="SOM_OCR_REQUEST_TIMEOUT_SEC",
    )
    worker_max_attempts: int = Field(
        2,
        ge=1,
        le=10,
        validation_alias="SOM_OCR_WORKER_MAX_ATTEMPTS",
    )
    resize_enabled: bool = Field(True, validation_alias="SOM_OCR_RESIZE")
    max_image_dim: int = Field(1152, ge=0, le=8192, validation_alias="SOM_OCR_MAX_DIM")
    inference_max_dim: int = Field(
        1024,
        ge=256,
        le=8192,
        validation_alias="SOM_INFERENCE_MAX_DIM",
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> "OcrSettings":
        for name in (
            "python_executable",
            "yolo_config_dir",
            "paddlex_cache_dir",
            "cuda_bin_dir",
            "cudnn_bin_dir",
        ):
            value = getattr(self, name)
            if value is not None and not value.is_absolute():
                setattr(self, name, (BASE_DIR / value).resolve())
        return self


class ReflexSettings(SectionSettings):
    enabled: bool = Field(True, validation_alias="REFLEX_ENABLED")
    roi_phash_max_distance: int = Field(22, ge=0, le=64, validation_alias="REFLEX_ROI_PHASH_MAX_DISTANCE")
    target_center_max_distance: float = Field(0.065, ge=0, le=1, validation_alias="REFLEX_TARGET_CENTER_MAX_DISTANCE")
    no_effect_phash_max_distance: int = Field(2, ge=0, le=64, validation_alias="REFLEX_NO_EFFECT_PHASH_MAX_DISTANCE")
    visual_change_pixel_threshold: int = Field(8, ge=0, le=255, validation_alias="REFLEX_VISUAL_CHANGE_PIXEL_THRESHOLD")
    visual_change_min_ratio: float = Field(0.03, ge=0, le=1, validation_alias="REFLEX_VISUAL_CHANGE_MIN_RATIO")
    card_queue_return_phash_max_distance: int = Field(16, ge=0, le=64, validation_alias="VISION_CARD_QUEUE_RETURN_PHASH_MAX_DISTANCE")
    card_queue_return_min_anchor_overlap: float = Field(0.20, ge=0, le=1, validation_alias="VISION_CARD_QUEUE_RETURN_MIN_ANCHOR_OVERLAP")
    result_card_queue_enabled: bool = Field(True, validation_alias="VISION_RESULT_CARD_QUEUE_ENABLED")
    result_card_selector_enabled: bool = Field(True, validation_alias="VISION_RESULT_CARD_SELECTOR_ENABLED")
    skip_existing_job_details: bool = Field(True, validation_alias="VISION_SKIP_EXISTING_JOB_DETAILS")
    transition_cycle_phash_max_distance: int = Field(
        4,
        ge=0,
        le=64,
        validation_alias="VISION_TRANSITION_CYCLE_PHASH_MAX_DISTANCE",
    )
    capture_width_tolerance_px: int = Field(32, ge=0, le=1000, validation_alias="REFLEX_CAPTURE_WIDTH_TOLERANCE_PX")
    capture_height_tolerance_px: int = Field(48, ge=0, le=1000, validation_alias="REFLEX_CAPTURE_HEIGHT_TOLERANCE_PX")
    interactive_content_top_px: int = Field(180, ge=0, le=4000, validation_alias="VISION_INTERACTIVE_CONTENT_TOP_PX")
    visual_change_sufficient_components: Annotated[tuple[str, ...], NoDecode] = Field(
        ("tab_button", "search_button", "expand_detail_button", "reveal_button", "details_toggle"),
        validation_alias="REFLEX_VISUAL_CHANGE_SUFFICIENT_COMPONENTS",
    )
    idempotent_control_components: Annotated[tuple[str, ...], NoDecode] = Field(
        (
            "tab_button",
            "search_button",
            "expand_detail_button",
            "reveal_button",
            "details_toggle",
            "result_filter",
            "result_filter_input",
        ),
        validation_alias="REFLEX_IDEMPOTENT_CONTROL_COMPONENTS",
    )
    idempotent_scope_ignored_query_keys: Annotated[tuple[str, ...], NoDecode] = Field(
        ("tab",),
        validation_alias="REFLEX_IDEMPOTENT_SCOPE_IGNORED_QUERY_KEYS",
    )

    @field_validator(
        "visual_change_sufficient_components",
        "idempotent_control_components",
        "idempotent_scope_ignored_query_keys",
        mode="before",
    )
    @classmethod
    def split_csv(cls, value):
        if isinstance(value, str):
            return tuple(item.strip().casefold() for item in value.split(",") if item.strip())
        return value


class RecipeSettings(SectionSettings):
    learning_mode: str = Field("record", validation_alias="VISION_RECIPE_LEARNING_MODE")
    auto_promote: bool = Field(True, validation_alias="VISION_RECIPE_AUTO_PROMOTE")
    critic_evidence_text_limit: int = Field(60, ge=1, le=500, validation_alias="VISION_RECIPE_CRITIC_EVIDENCE_TEXT_LIMIT")
    critic_timeout_sec: float = Field(30.0, gt=0, le=300, validation_alias="VISION_RECIPE_CRITIC_TIMEOUT_SEC")
    promotion_poll_sec: float = Field(1.0, ge=0.1, le=300, validation_alias="VISION_RECIPE_PROMOTION_POLL_SEC")
    promotion_retry_delay_sec: float = Field(30.0, ge=0, le=3600, validation_alias="VISION_RECIPE_PROMOTION_RETRY_DELAY_SEC")
    promotion_max_attempts: int = Field(3, ge=1, le=100, validation_alias="VISION_RECIPE_PROMOTION_MAX_ATTEMPTS")
    worker_review_retries: int = Field(0, ge=0, le=20, validation_alias="VISION_WORKER_REVIEW_RETRIES")
    worker_summary_mode: str = Field("deterministic", validation_alias="VISION_WORKER_SUMMARY_MODE")
    worker_review_mode: str = Field("shape", validation_alias="VISION_WORKER_REVIEW_MODE")
    jd_normalization_mode: str = Field("deterministic", validation_alias="VISION_JD_NORMALIZATION_MODE")


class RetentionSettings(SectionSettings):
    log_days: int = Field(30, ge=1, le=3650, validation_alias="RETENTION_LOG_DAYS")
    artifact_days: int = Field(90, ge=1, le=3650, validation_alias="RETENTION_ARTIFACT_DAYS")
    audit_days: int = Field(90, ge=1, le=3650, validation_alias="RETENTION_AUDIT_DAYS")
    job_version_days: int = Field(180, ge=1, le=3650, validation_alias="RETENTION_JOB_VERSION_DAYS")
    keep_job_versions: int = Field(5, ge=1, le=100, validation_alias="RETENTION_KEEP_JOB_VERSIONS")


class ObservabilitySettings(SectionSettings):
    app_env: str = Field("development", validation_alias="APP_ENV")
    log_format: str = Field("console", validation_alias="LOG_FORMAT")
    log_target: str | None = Field(None, validation_alias="LOG_TARGET")
    sentry_dsn: SecretStr | None = Field(None, validation_alias="SENTRY_DSN")
    sentry_traces_sample_rate: float = Field(0.0, ge=0, le=1, validation_alias="SENTRY_TRACES_SAMPLE_RATE")
    sentry_profiles_sample_rate: float = Field(0.0, ge=0, le=1, validation_alias="SENTRY_PROFILES_SAMPLE_RATE")
    langsmith_project: str = Field("l2c-local", validation_alias="LANGSMITH_PROJECT")
    langsmith_e2e_feedback: bool = Field(True, validation_alias="L2C_LANGSMITH_E2E_FEEDBACK")
    langsmith_flush_timeout_sec: float = Field(5.0, gt=0, le=120, validation_alias="LANGSMITH_FLUSH_TIMEOUT_SEC")


class AppSettings:
    """섹션별 BaseSettings를 한 번만 생성해 애플리케이션에 주입한다."""

    def __init__(self) -> None:
        self.paths = PathSettings()
        self.models = ModelSettings()
        self.browser = BrowserSettings()
        self.vision = VisionSettings()
        self.ocr = OcrSettings()
        self.reflex = ReflexSettings()
        self.recipe = RecipeSettings()
        self.retention = RetentionSettings()
        self.observability = ObservabilitySettings()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def clear_settings_cache() -> None:
    """테스트와 명시적 설정 재로딩에서만 사용한다."""

    get_settings.cache_clear()


__all__ = [
    "AppSettings",
    "BASE_DIR",
    "BrowserSettings",
    "ModelSettings",
    "OcrSettings",
    "ObservabilitySettings",
    "PathSettings",
    "RecipeSettings",
    "RetentionSettings",
    "ReflexSettings",
    "VisionSettings",
    "clear_settings_cache",
    "get_settings",
]
