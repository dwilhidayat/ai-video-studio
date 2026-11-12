"""
Tests for Video Generation Engine logic.
"""

from pathlib import Path
import pytest
from src.config import BackendID
from src.engine import VideoEngine, VideoEngineError


def test_extract_video_file_from_dict():
    engine = VideoEngine(default_output_dir=Path("tests/scratch"))
    raw = {"video": "C:/tmp/sample.mp4", "subtitles": None}
    extracted = engine._extract_video_file(raw)
    assert extracted == "C:/tmp/sample.mp4"


def test_extract_video_file_from_tuple():
    engine = VideoEngine(default_output_dir=Path("tests/scratch"))
    raw = ({"video": "C:/tmp/sample.mp4"}, 12345)
    extracted = engine._extract_video_file(raw)
    assert extracted == "C:/tmp/sample.mp4"


def test_extract_video_file_invalid():
    engine = VideoEngine(default_output_dir=Path("tests/scratch"))
    with pytest.raises(VideoEngineError):
        engine._extract_video_file({"invalid_key": 123})


def test_check_space_health():
    engine = VideoEngine(default_output_dir=Path("tests/scratch"))
    health = engine.check_space_health(BackendID.LTX_VIDEO_DISTILLED)
    assert "stage" in health
    assert "hardware" in health
    assert health["stage"] == "RUNNING"
    assert health["ready"] is True
