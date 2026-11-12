"""
Tests for Command-Line Interface (generate.py) argument parsing.
"""

import sys
import pytest
from generate import parse_args


def test_cli_presets_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--list-presets"])
    args = parse_args()
    assert args.list_presets is True


def test_cli_cameras_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--list-cameras"])
    args = parse_args()
    assert args.list_cameras is True


def test_cli_models_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--list-models"])
    args = parse_args()
    assert args.list_models is True


def test_cli_generation_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--prompt", "Cyberpunk alley in the rain",
            "--preset", "Cyberpunk Alley",
            "--camera", "Low Angle Tracking",
            "--model", "ltx-video-distilled",
            "--output", "my_video.mp4",
            "--duration", "3.0",
            "--resolution", "512x704",
        ],
    )
    args = parse_args()
    assert args.prompt == "Cyberpunk alley in the rain"
    assert args.preset == "Cyberpunk Alley"
    assert args.camera == "Low Angle Tracking"
    assert args.model == "ltx-video-distilled"
    assert args.output == "my_video.mp4"
    assert args.duration == 3.0
    assert args.resolution == "512x704"
