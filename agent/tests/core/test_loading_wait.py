import numpy as np
from PIL import Image, ImageDraw

from agent.vision.frame_compare import changed_pixel_ratio, mean_difference_percent
from agent.vision.loading_wait import (
    LoadingWait,
    detect_browser_content_top,
    frame_quality,
)


def test_frame_compare_separates_changed_area_from_motion_strength():
    before = np.zeros((200, 200), dtype=np.uint8)
    after = before.copy()
    after[50:150, 50:150] = 255

    assert changed_pixel_ratio(before, after, intensity_threshold=20) > 0.2
    assert mean_difference_percent(before, before) == 0.0
    assert mean_difference_percent(before, after) > 20.0


def test_frame_quality_rejects_blank_body_and_keeps_content(monkeypatch):
    blank = Image.new("RGB", (800, 900), "white")
    ImageDraw.Draw(blank).rectangle((0, 0, 800, 184), fill=(45, 58, 65))
    content = blank.copy()
    draw = ImageDraw.Draw(content)
    for row in range(8):
        y = 220 + row * 65
        draw.rectangle((60, y, 340, y + 35), fill=(40 + row * 15, 80, 160))
        draw.rectangle((400, y, 730, y + 20), fill=(20, 20, 20))

    monkeypatch.delenv("VISION_CONTENT_TOP", raising=False)

    assert frame_quality(blank)["low_information"] is True
    assert frame_quality(content)["low_information"] is False


def test_frame_quality_keeps_sparse_search_controls(monkeypatch):
    image = Image.new("RGB", (800, 900), "white")
    image.paste((45, 58, 65), (0, 0, 800, 185))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((290, 220, 510, 260), radius=8, fill=(248, 248, 248))
    draw.rectangle((300, 232, 400, 240), fill=(180, 180, 180))
    for row in range(4):
        y = 300 + row * 35
        draw.rectangle((300, y, 360, y + 7), fill=(170, 170, 170))
        draw.rectangle((440, y, 500, y + 7), fill=(170, 170, 170))

    monkeypatch.delenv("VISION_CONTENT_TOP", raising=False)

    quality = frame_quality(image)

    assert quality["stddev"] <= 12.0
    assert quality["edge_mean"] <= 3.0
    assert quality["dominant_ratio"] >= 0.97
    assert quality["compact_component_count"] >= 4
    assert quality["low_information"] is False


def test_content_boundary_excludes_browser_chrome():
    frame = np.full((900, 800), 255, dtype=np.uint8)
    frame[:185] = 45

    assert detect_browser_content_top(frame) == 185


def test_frame_quality_rejects_low_contrast_loading_shell(monkeypatch):
    image = Image.new("RGB", (800, 900), "white")
    image.paste((45, 58, 65), (0, 0, 800, 185))
    draw = ImageDraw.Draw(image)
    for row in range(2):
        y = 225 + row * 42
        draw.rounded_rectangle(
            (250, y, 550, y + 18),
            radius=7,
            fill=(180, 180, 180),
        )

    monkeypatch.delenv("VISION_CONTENT_TOP", raising=False)

    quality = frame_quality(image)

    assert quality["stddev"] <= 12.0
    assert quality["compact_component_count"] < 4
    assert quality["low_information"] is True


def test_wait_until_ready_detects_change_and_stability_in_one_loop(
    monkeypatch,
    tmp_path,
):
    reference_path = tmp_path / "before.png"
    Image.new("RGB", (600, 600), "white").save(reference_path)
    before = np.full((600, 600), 255, dtype=np.uint8)
    content = before.copy()
    content[100:500, 100:500] = 0
    frames = iter([before, content, content, content])
    loading_wait = object.__new__(LoadingWait)
    loading_wait.perception = object()
    loading_wait.last_result = {}
    monkeypatch.setattr(
        loading_wait,
        "_capture_memory_frame",
        lambda **_kwargs: next(frames),
    )
    monkeypatch.setattr("agent.vision.loading_wait.time.sleep", lambda _seconds: None)

    result = loading_wait.wait_until_ready(
        reference_image_path=reference_path,
        max_wait_sec=1,
        change_grace_sec=0.5,
        check_interval_sec=0,
        quiet_period_sec=0,
        threshold_percent=1,
        required_stable_frames=2,
        region={"top": 0, "left": 0, "width": 600, "height": 600},
    )

    assert result["ready"] is True
    assert result["visual_change_detected"] is True
    assert result["visual_change_ratio"] > 0.4
    assert result["probe_count"] == 3


def test_wait_until_ready_does_not_accept_a_short_intermediate_plateau(
    monkeypatch,
    tmp_path,
):
    reference_path = tmp_path / "before.png"
    before = np.full((600, 600), 255, dtype=np.uint8)
    Image.fromarray(before).save(reference_path)
    intermediate = before.copy()
    intermediate[100:300, 100:500] = 0
    final = before.copy()
    final[100:500, 100:500] = 0
    frames = iter(
        [intermediate, intermediate, intermediate, final, final, final]
    )
    clock = iter([0.0, 0.1, 0.2, 0.5, 0.8, 1.4])
    loading_wait = object.__new__(LoadingWait)
    loading_wait.perception = object()
    loading_wait.last_result = {}
    monkeypatch.setattr(
        loading_wait,
        "_capture_memory_frame",
        lambda **_kwargs: next(frames),
    )
    monkeypatch.setattr("agent.vision.loading_wait.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "agent.vision.loading_wait.time.perf_counter",
        lambda: next(clock),
    )

    result = loading_wait.wait_until_ready(
        reference_image_path=reference_path,
        max_wait_sec=2,
        check_interval_sec=0,
        quiet_period_sec=0.8,
        threshold_percent=1,
        required_stable_frames=2,
        region={"top": 0, "left": 0, "width": 600, "height": 600},
    )

    assert result["ready"] is True
    assert result["probe_count"] == 5
    assert result["quiet_elapsed_sec"] == 0.9


def test_wait_until_ready_releases_unchanged_action_after_grace(
    monkeypatch,
    tmp_path,
):
    reference_path = tmp_path / "before.png"
    content = np.full((600, 600), 255, dtype=np.uint8)
    content[100:500, 100:500] = 0
    Image.fromarray(content).save(reference_path)
    clock = iter([0.0, 0.2, 0.7, 1.21])
    loading_wait = object.__new__(LoadingWait)
    loading_wait.perception = object()
    loading_wait.last_result = {}
    monkeypatch.setattr(
        loading_wait,
        "_capture_memory_frame",
        lambda **_kwargs: content,
    )
    monkeypatch.setattr("agent.vision.loading_wait.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "agent.vision.loading_wait.time.perf_counter",
        lambda: next(clock),
    )

    result = loading_wait.wait_until_ready(
        reference_image_path=reference_path,
        max_wait_sec=2,
        change_grace_sec=1.2,
        check_interval_sec=0,
        threshold_percent=1,
        required_stable_frames=2,
        region={"top": 0, "left": 0, "width": 600, "height": 600},
    )

    assert result["ready"] is True
    assert result["visual_change_detected"] is False
    assert result["visual_change_ratio"] == 0.0
    assert result["probe_count"] == 3


def test_wait_until_ready_requires_stable_informative_frames(monkeypatch):
    blank = np.full((600, 600), 255, dtype=np.uint8)
    content = blank.copy()
    content[100:500:20, 100:500] = 0
    frames = iter([blank, content, content, content])
    loading_wait = object.__new__(LoadingWait)
    loading_wait.perception = object()
    loading_wait.last_result = {}
    monkeypatch.setattr(
        loading_wait,
        "_capture_memory_frame",
        lambda **_kwargs: next(frames),
    )
    monkeypatch.setattr("agent.vision.loading_wait.time.sleep", lambda _seconds: None)

    result = loading_wait.wait_until_ready(
        max_wait_sec=1,
        check_interval_sec=0,
        threshold_percent=1,
        required_stable_frames=2,
        region={"top": 0, "left": 0, "width": 600, "height": 600},
    )

    assert result["ready"] is True
    assert result["low_information"] is False
    assert result["probe_count"] == 3
    assert result["stable_frames"] == 2


def test_wait_until_ready_reports_low_information_timeout(monkeypatch):
    blank = np.full((600, 600), 255, dtype=np.uint8)
    frames = iter([blank, blank])
    clock = iter([0.0, 1.1])
    loading_wait = object.__new__(LoadingWait)
    loading_wait.perception = object()
    loading_wait.last_result = {}
    monkeypatch.setattr(
        loading_wait,
        "_capture_memory_frame",
        lambda **_kwargs: next(frames),
    )
    monkeypatch.setattr("agent.vision.loading_wait.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "agent.vision.loading_wait.time.perf_counter",
        lambda: next(clock),
    )

    result = loading_wait.wait_until_ready(
        max_wait_sec=1,
        check_interval_sec=0,
        threshold_percent=1,
        required_stable_frames=2,
        region={"top": 0, "left": 0, "width": 600, "height": 600},
    )

    assert result["ready"] is False
    assert result["wait_reason"] == "low_information_timeout"
    assert result["probe_count"] == 1
