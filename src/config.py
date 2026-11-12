"""
Configuration module for AI Video Generation Suite.
Defines supported backend spaces, generation schemas, and resolution presets.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BackendID(str, Enum):
    LTX_VIDEO_DISTILLED = "ltx-video-distilled"
    LTX_VIDEO_OPTIMIZED = "ltx-video-optimized"
    COGVIDEOX_5B = "cogvideox-5b"
    COGVIDEOX_2B = "cogvideox-2b"


@dataclass
class BackendInfo:
    id: BackendID
    name: str
    space_repo: str
    description: str
    hardware: str
    default_endpoint: str
    is_primary: bool = False
    requires_zero_gpu_token: bool = False


# Supported Free Cloud GPU Backends
SUPPORTED_BACKENDS: Dict[BackendID, BackendInfo] = {
    BackendID.LTX_VIDEO_DISTILLED: BackendInfo(
        id=BackendID.LTX_VIDEO_DISTILLED,
        name="LTX-Video Distilled (Fast)",
        space_repo="Lightricks/ltx-video-distilled",
        description="Lightricks SOTA distilled video model on Zero-A10G GPU (~15-20s generation).",
        hardware="zero-a10g",
        default_endpoint="/text_to_video",
        is_primary=True,
        requires_zero_gpu_token=False,
    ),
    BackendID.LTX_VIDEO_OPTIMIZED: BackendInfo(
        id=BackendID.LTX_VIDEO_OPTIMIZED,
        name="LTX-Video ZeroGPU-Optimized",
        space_repo="DeepRat/LTX-Video-ZeroGPU-Optimized",
        description="Community optimized LTX-Video deployment on Zero-A10G GPU.",
        hardware="zero-a10g",
        default_endpoint="/text_to_video",
        is_primary=False,
        requires_zero_gpu_token=False,
    ),
    BackendID.COGVIDEOX_5B: BackendInfo(
        id=BackendID.COGVIDEOX_5B,
        name="CogVideoX-5B (Cloud GPU)",
        space_repo="zai-org/CogVideoX-5B-Space",
        description="Zhipu AI / THUDM 5-Billion parameter transformer video model.",
        hardware="zero-a10g",
        default_endpoint="/generate",
        is_primary=False,
        requires_zero_gpu_token=True,
    ),
    BackendID.COGVIDEOX_2B: BackendInfo(
        id=BackendID.COGVIDEOX_2B,
        name="CogVideoX-2B (Cloud GPU)",
        space_repo="zai-org/CogVideoX-2B-Space",
        description="Zhipu AI / THUDM 2-Billion parameter transformer video model.",
        hardware="zero-a10g",
        default_endpoint="/generate",
        is_primary=False,
        requires_zero_gpu_token=True,
    ),
}


@dataclass
class ResolutionPreset:
    label: str
    width: int
    height: int
    aspect_ratio: str
    description: str


RESOLUTION_PRESETS: Dict[str, ResolutionPreset] = {
    "16:9 Landscape (704x512)": ResolutionPreset(
        label="16:9 Landscape (704x512)",
        width=704,
        height=512,
        aspect_ratio="16:9",
        description="Standard widescreen HD for YouTube, desktop, and cinema.",
    ),
    "16:9 Cinema Widescreen (768x432)": ResolutionPreset(
        label="16:9 Cinema Widescreen (768x432)",
        width=768,
        height=432,
        aspect_ratio="16:9",
        description="Ultra-wide cinematic aspect ratio.",
    ),
    "9:16 Vertical (512x704)": ResolutionPreset(
        label="9:16 Vertical (512x704)",
        width=512,
        height=704,
        aspect_ratio="9:16",
        description="Vertical video optimized for TikTok, Reels, and YouTube Shorts.",
    ),
    "1:1 Square (512x512)": ResolutionPreset(
        label="1:1 Square (512x512)",
        width=512,
        height=512,
        aspect_ratio="1:1",
        description="Square format for Instagram posts and feeds.",
    ),
}


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt describing the desired video scene")
    negative_prompt: Optional[str] = Field(
        default="worst quality, blurry, jittery, distorted, pixelated, noisy, bad anatomy, artifacts",
        description="Negative prompt to suppress undesirable video artifacts",
    )
    preset_name: str = Field(default="Cinematic Drone", description="Built-in prompt engineering preset")
    camera_motion: str = Field(default="Slow Push In (Dolly In)", description="Camera movement dynamic")
    backend_id: BackendID = Field(default=BackendID.LTX_VIDEO_DISTILLED, description="Cloud GPU backend")
    width: int = Field(default=704, ge=256, le=1280)
    height: int = Field(default=512, ge=256, le=1280)
    duration_seconds: float = Field(default=2.0, ge=1.0, le=10.0)
    guidance_scale: float = Field(default=3.0, ge=1.0, le=15.0)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    randomize_seed: bool = Field(default=True)
    hf_token: Optional[str] = Field(default=None, description="Optional Hugging Face token for extended quotas")
    output_dir: Path = Field(default_factory=lambda: Path("output_videos"))


class VideoGenerationResult(BaseModel):
    video_path: str = Field(..., description="Absolute local path to the generated MP4 video")
    backend_id: str
    space_repo: str
    seed_used: int
    duration_seconds: float
    resolution: str
    generation_time_seconds: float
    compiled_prompt: str
    timestamp: str
