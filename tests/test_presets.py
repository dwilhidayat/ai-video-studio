"""
Tests for Prompt Engineering Presets.
Verifies all required presets (Cinematic Drone, Cyberpunk Alley, Macro Nature, Studio Product).
"""

import pytest
from src.presets import PROMPT_PRESETS, get_preset, list_preset_names


def test_required_presets_present():
    required = ["Cinematic Drone", "Cyberpunk Alley", "Macro Nature", "Studio Product"]
    available = list_preset_names()
    for name in required:
        assert name in available, f"Missing required preset: {name}"


def test_preset_attributes():
    for name, preset in PROMPT_PRESETS.items():
        assert preset.name == name
        assert isinstance(preset.description, str) and len(preset.description) > 0
        assert isinstance(preset.default_negative, str) and len(preset.default_negative) > 0
        assert preset.recommended_guidance > 0.0
        assert preset.recommended_duration > 0.0


def test_get_preset_fallback():
    # Non-existent preset returns None / Raw Prompt fallback
    preset = get_preset("NonExistentPreset")
    assert preset.name == "None / Raw Prompt"
