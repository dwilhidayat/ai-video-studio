"""
Tests for Camera Motion Controls and Prompt Compilation.
"""

import pytest
from src.camera import CAMERA_MOTIONS, compose_prompt, get_camera_description, list_camera_motions


def test_camera_motions_list():
    motions = list_camera_motions()
    assert len(motions) >= 5
    assert "Slow Push In (Dolly In)" in motions
    assert "360 Orbit" in motions
    assert "Drone Sweeping Pan" in motions


def test_compose_prompt_cinematic_drone():
    user_prompt = "snowy alpine peaks at dawn"
    compiled, neg, guidance = compose_prompt(
        user_prompt=user_prompt,
        preset_name="Cinematic Drone",
        camera_motion="Drone Sweeping Pan",
    )
    # Check that camera motion directive is present
    assert "sweeping aerial drone pan" in compiled.lower()
    # Check that user prompt is present
    assert "snowy alpine peaks at dawn" in compiled
    # Check that preset aesthetic tokens are present
    assert "8k resolution" in compiled or "cinematic" in compiled
    # Check that negative prompt contains base negative
    assert "worst quality" in neg
    assert guidance >= 3.0


def test_compose_prompt_user_negative_stacking():
    user_prompt = "neon sports car"
    compiled, neg, _ = compose_prompt(
        user_prompt=user_prompt,
        preset_name="Cyberpunk Alley",
        camera_motion="Low Angle Tracking",
        user_negative="no humans, no animals",
    )
    assert "no humans, no animals" in neg
    assert "worst quality" in neg or "daylight" in neg
