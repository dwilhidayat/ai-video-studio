"""
Tests for Configuration Models and Resolution Presets.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import (
    BackendID,
    RESOLUTION_PRESETS,
    SUPPORTED_BACKENDS,
    VideoGenerationRequest,
    VideoGenerationResult,
)


def test_supported_backends():
    assert BackendID.LTX_VIDEO_DISTILLED in SUPPORTED_BACKENDS
    assert BackendID.COGVIDEOX_5B in SUPPORTED_BACKENDS
    assert BackendID.COGVIDEOX_2B in SUPPORTED_BACKENDS
    assert BackendID.LTX_VIDEO_OPTIMIZED in SUPPORTED_BACKENDS

    primary = SUPPORTED_BACKENDS[BackendID.LTX_VIDEO_DISTILLED]
    assert primary.is_primary is True
    assert "Lightricks" in primary.space_repo


def test_resolution_presets():
    assert "16:9 Landscape (704x512)" in RESOLUTION_PRESETS
    assert "9:16 Vertical (512x704)" in RESOLUTION_PRESETS
    res = RESOLUTION_PRESETS["16:9 Landscape (704x512)"]
    assert res.width == 704
    assert res.height == 512
    assert res.aspect_ratio == "16:9"


def test_video_generation_request_validation():
    # Valid request
    req = VideoGenerationRequest(prompt="A futuristic flying car")
    assert req.prompt == "A futuristic flying car"
    assert req.width == 704
    assert req.height == 512

    # Empty prompt fails validation
    with pytest.raises(ValidationError):
        VideoGenerationRequest(prompt="")
