"""
Video Generation Engine for AI Generative Video Agent.
Interfaces with free cloud GPU backends hosted on Hugging Face Spaces via gradio_client.
Includes automatic failover, output caching, robust error diagnosis, and timing benchmarks.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable, Dict, Optional, Tuple

import httpx
from gradio_client import Client

from src.camera import compose_prompt
from src.config import (
    BackendID,
    BackendInfo,
    SUPPORTED_BACKENDS,
    VideoGenerationRequest,
    VideoGenerationResult,
)


class VideoEngineError(Exception):
    """Custom exception raised when video generation fails across backends."""
    pass


class VideoEngine:
    def __init__(self, default_output_dir: Path = Path("output_videos")):
        self.output_dir = default_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clients: Dict[str, Client] = {}

    def get_client(self, space_repo: str, hf_token: Optional[str] = None) -> Client:
        """
        Retrieve a cached gradio_client.Client or instantiate a new one.
        Handles optional HF_TOKEN for private spaces or extended ZeroGPU quotas.
        """
        token = hf_token or os.environ.get("HF_TOKEN") or None
        cache_key = f"{space_repo}_{bool(token)}"
        if cache_key not in self._clients:
            try:
                client = Client(
                    space_repo,
                    token=token,
                    httpx_kwargs={"timeout": 600.0},
                    verbose=False,
                )
                self._clients[cache_key] = client
            except Exception as exc:
                raise VideoEngineError(
                    f"Failed to connect to backend space '{space_repo}': {exc}"
                ) from exc
        return self._clients[cache_key]

    def check_space_health(self, backend_id: BackendID) -> Dict[str, Any]:
        """
        Query the Hugging Face API directly to determine current space stage,
        hardware allocation (e.g. zero-a10g), and connectivity.
        """
        info = SUPPORTED_BACKENDS[backend_id]
        url = f"https://huggingface.co/api/spaces/{info.space_repo}"
        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    runtime = data.get("runtime", {})
                    stage = runtime.get("stage", "UNKNOWN")
                    hw = runtime.get("hardware", {}).get("current", "UNKNOWN")
                    likes = data.get("likes", 0)
                    is_ready = stage == "RUNNING"
                    return {
                        "backend_id": backend_id.value,
                        "name": info.name,
                        "space_repo": info.space_repo,
                        "stage": stage,
                        "hardware": hw,
                        "likes": likes,
                        "ready": is_ready,
                        "message": f"Space is {stage} on {hw} GPU." if is_ready else f"Space is in {stage} state."
                    }
                else:
                    return {
                        "backend_id": backend_id.value,
                        "name": info.name,
                        "space_repo": info.space_repo,
                        "stage": "HTTP_ERROR",
                        "hardware": "NONE",
                        "likes": 0,
                        "ready": False,
                        "message": f"Hugging Face API returned status {res.status_code}."
                    }
        except Exception as e:
            return {
                "backend_id": backend_id.value,
                "name": info.name,
                "space_repo": info.space_repo,
                "stage": "UNREACHABLE",
                "hardware": "NONE",
                "likes": 0,
                "ready": False,
                "message": f"Connection error checking space health: {e}"
            }

    def _extract_video_file(self, raw_result: Any) -> str:
        """
        Safely extract the generated MP4 file path from various Gradio return signatures.
        """
        if isinstance(raw_result, tuple) or isinstance(raw_result, list):
            # Check items in tuple (first item or file item)
            for item in raw_result:
                if isinstance(item, dict) and "video" in item and item["video"]:
                    return str(item["video"])
                if isinstance(item, str) and (item.endswith(".mp4") or item.endswith(".webm")):
                    return item
            # If still not found, check if first element is dict
            if raw_result and isinstance(raw_result[0], dict):
                first_dict = raw_result[0]
                if "video" in first_dict and first_dict["video"]:
                    return str(first_dict["video"])
                if "path" in first_dict and first_dict["path"]:
                    return str(first_dict["path"])
        elif isinstance(raw_result, dict):
            if "video" in raw_result and raw_result["video"]:
                return str(raw_result["video"])
            if "path" in raw_result and raw_result["path"]:
                return str(raw_result["path"])
        elif isinstance(raw_result, str) and os.path.exists(raw_result):
            return raw_result

        raise VideoEngineError(f"Could not parse valid video file path from backend response: {raw_result}")

    def _generate_ltx(
        self,
        request: VideoGenerationRequest,
        backend_info: BackendInfo,
        compiled_prompt: str,
        compiled_negative: str,
        seed: int,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, int]:
        """
        Execute generation against LTX-Video distilled or zero-gpu optimized spaces.
        """
        client = self.get_client(backend_info.space_repo, request.hf_token)
        if progress_cb:
            progress_cb(f"Submitting job to {backend_info.name} cloud GPU...")

        # Exact parameter signature for Lightricks / DeepRat / LTX endpoints
        result = client.predict(
            prompt=compiled_prompt,
            negative_prompt=compiled_negative,
            input_image_filepath=None,
            input_video_filepath=None,
            height_ui=float(request.height),
            width_ui=float(request.width),
            mode="text-to-video",
            duration_ui=float(request.duration_seconds),
            ui_frames_to_use=9.0,
            seed_ui=int(seed),
            randomize_seed=bool(request.randomize_seed),
            ui_guidance_scale=float(request.guidance_scale),
            improve_texture_flag=True,
            api_name="/text_to_video",
        )

        video_path = self._extract_video_file(result)
        # Extract returned seed if present in tuple
        used_seed = seed
        if isinstance(result, (tuple, list)) and len(result) > 1 and isinstance(result[1], (int, float)):
            used_seed = int(result[1])

        return video_path, used_seed

    def _generate_cogvideox(
        self,
        request: VideoGenerationRequest,
        backend_info: BackendInfo,
        compiled_prompt: str,
        compiled_negative: str,
        seed: int,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, int]:
        """
        Execute generation against CogVideoX spaces.
        """
        client = self.get_client(backend_info.space_repo, request.hf_token)
        if progress_cb:
            progress_cb(f"Submitting job to {backend_info.name} cloud GPU...")

        if backend_info.id == BackendID.COGVIDEOX_2B:
            # CogVideoX-2B endpoint: /generate(prompt, num_inference_steps, guidance_scale)
            result = client.predict(
                prompt=compiled_prompt,
                num_inference_steps=30.0,
                guidance_scale=float(request.guidance_scale),
                api_name="/generate",
            )
        else:
            # CogVideoX-5B endpoint: /generate(prompt, image_input, video_input, video_strength, seed_value, scale_status, rife_status)
            result = client.predict(
                prompt=compiled_prompt,
                image_input=None,
                video_input=None,
                video_strength=0.8,
                seed_value=float(seed),
                scale_status=False,
                rife_status=False,
                api_name="/generate",
            )

        video_path = self._extract_video_file(result)
        return video_path, seed

    def generate(
        self,
        request: VideoGenerationRequest,
        progress_cb: Optional[Callable[[str], None]] = None,
        auto_fallback: bool = True,
    ) -> VideoGenerationResult:
        """
        Main generation entry point.
        Compiles prompts, resolves camera motions, executes on cloud GPU,
        copies the generated MP4 to output directory, and returns complete result.
        """
        t_start = time.time()
        
        # Compile prompt, camera motion, and presets
        compiled_prompt, compiled_negative, rec_guidance = compose_prompt(
            user_prompt=request.prompt,
            preset_name=request.preset_name,
            camera_motion=request.camera_motion,
            user_negative=request.negative_prompt or "",
        )

        # Determine seed
        if request.randomize_seed or request.seed == -1:
            active_seed = random.randint(1, 2147483647)
        else:
            active_seed = request.seed

        primary_info = SUPPORTED_BACKENDS.get(request.backend_id)
        if not primary_info:
            raise VideoEngineError(f"Unsupported backend ID: {request.backend_id}")

        backends_to_try = [primary_info]
        if auto_fallback:
            # If primary is LTX distilled, fallback to LTX optimized
            if primary_info.id == BackendID.LTX_VIDEO_DISTILLED:
                backends_to_try.append(SUPPORTED_BACKENDS[BackendID.LTX_VIDEO_OPTIMIZED])
            elif primary_info.id != BackendID.LTX_VIDEO_DISTILLED:
                # If CogVideo or other fails, fallback to fast LTX-Video distilled
                backends_to_try.append(SUPPORTED_BACKENDS[BackendID.LTX_VIDEO_DISTILLED])

        last_error: Optional[Exception] = None
        source_video_path: Optional[str] = None
        used_backend_info: Optional[BackendInfo] = None
        final_seed = active_seed

        for backend in backends_to_try:
            try:
                used_backend_info = backend
                if progress_cb:
                    progress_cb(f"Connecting to {backend.name} [{backend.space_repo}]...")

                if backend.id in (BackendID.LTX_VIDEO_DISTILLED, BackendID.LTX_VIDEO_OPTIMIZED):
                    source_video_path, final_seed = self._generate_ltx(
                        request=request,
                        backend_info=backend,
                        compiled_prompt=compiled_prompt,
                        compiled_negative=compiled_negative,
                        seed=active_seed,
                        progress_cb=progress_cb,
                    )
                else:
                    source_video_path, final_seed = self._generate_cogvideox(
                        request=request,
                        backend_info=backend,
                        compiled_prompt=compiled_prompt,
                        compiled_negative=compiled_negative,
                        seed=active_seed,
                        progress_cb=progress_cb,
                    )

                if source_video_path and os.path.exists(source_video_path):
                    break

            except Exception as e:
                last_error = e
                err_msg = str(e)
                # Check for ZeroGPU duration restriction
                if "larger than the maximum allowed" in err_msg or "Subscribe to Hugging Face PRO" in err_msg:
                    quota_note = (
                        f"Backend '{backend.name}' currently requires Hugging Face PRO or a smaller GPU allocation. "
                        "Attempting automatic fallback to fast free LTX-Video distilled backend..."
                    )
                    if progress_cb:
                        progress_cb(quota_note)
                else:
                    if progress_cb:
                        progress_cb(f"Backend '{backend.name}' encountered error: {e}. Trying fallback...")
                continue

        if not source_video_path or not os.path.exists(source_video_path):
            error_details = (
                f"Video generation failed across backends. Last error: {last_error}.\n"
                "Tip: Ensure internet connectivity to huggingface.co and that the public spaces are running."
            )
            raise VideoEngineError(error_details)

        # Copy the video to the persistent output directory
        target_dir = request.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_backend = used_backend_info.id.value if used_backend_info else "backend"
        dest_filename = f"video_{timestamp_str}_{safe_backend}_{final_seed}.mp4"
        dest_path = target_dir / dest_filename
        
        shutil.copyfile(source_video_path, dest_path)
        gen_duration = round(time.time() - t_start, 2)

        if progress_cb:
            progress_cb(f"Video saved successfully to {dest_path.name} ({gen_duration}s)")

        return VideoGenerationResult(
            video_path=str(dest_path.resolve()),
            backend_id=used_backend_info.id.value if used_backend_info else "unknown",
            space_repo=used_backend_info.space_repo if used_backend_info else "unknown",
            seed_used=final_seed,
            duration_seconds=request.duration_seconds,
            resolution=f"{request.width}x{request.height}",
            generation_time_seconds=gen_duration,
            compiled_prompt=compiled_prompt,
            timestamp=datetime.now().isoformat(),
        )
