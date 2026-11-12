"""
Modern, Clean Local Web UI for AI Generative Video Agent.
Built with Gradio featuring prompt presets, camera motion controls,
direct MP4 downloads, live compilation inspection, and cloud GPU health monitoring.
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.camera import CAMERA_MOTIONS, compose_prompt, list_camera_motions
from src.config import (
    BackendID,
    RESOLUTION_PRESETS,
    SUPPORTED_BACKENDS,
    VideoGenerationRequest,
    VideoGenerationResult,
)
from src.engine import VideoEngine, VideoEngineError
from src.presets import PROMPT_PRESETS, get_preset, list_preset_names


CUSTOM_CSS = """
/* Modern Dark Studio Styling */
:root {
    --primary-hue: 215;
    --accent-color: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.25);
    --bg-dark: #0f172a;
    --card-dark: #1e293b;
    --border-color: #334155;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

.header-container {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
}

.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 10px;
    flex-wrap: wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid #3b82f6; }
.badge-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid #22c55e; }
.badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid #a855f7; }

.generate-btn {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
}

.generate-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.6) !important;
}

.metadata-box {
    background: #1e293b;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    font-size: 0.9rem;
}

/* Mobile-Specific Responsive Styling */
@media (max-width: 768px) {
    .header-container {
        padding: 16px !important;
        margin-bottom: 12px !important;
    }
    .header-container h1 {
        font-size: 1.5rem !important;
    }
    .badge-row {
        gap: 6px !important;
    }
    .badge {
        font-size: 0.72rem !important;
        padding: 3px 8px !important;
    }
    .generate-btn {
        width: 100% !important;
        font-size: 1rem !important;
        padding: 14px 16px !important;
    }
    .gradio-container {
        padding: 8px !important;
    }
}
"""


def create_ui(output_dir: Path = Path("output_videos")) -> gr.Blocks:
    engine = VideoEngine(default_output_dir=output_dir)

    def on_preset_change(preset_name: str) -> str:
        preset = get_preset(preset_name)
        return (
            f"**Preset Description:** {preset.description}\n\n"
            f"*Style Infusion:* `{preset.prefix}... {preset.suffix}`\n\n"
            f"*Recommended Guidance:* `{preset.recommended_guidance}` | *Duration:* `{preset.recommended_duration}s`"
        )

    def load_recent_videos() -> List[str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        videos = sorted(
            [str(p.resolve()) for p in output_dir.glob("*.mp4")],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        return videos[:6]

    def check_backend_status() -> str:
        report = ["### ⚡ Cloud GPU Backend Health Report", ""]
        for b_id, info in SUPPORTED_BACKENDS.items():
            health = engine.check_space_health(b_id)
            icon = "🟢" if health["ready"] else "🔴"
            token_note = " *(Requires HF token or Pro)*" if info.requires_zero_gpu_token else " *(100% Free Public ZeroGPU)*"
            report.append(
                f"- **{icon} {info.name}** (`{info.space_repo}`): "
                f"Status: **{health['stage']}** on `{health['hardware']}` GPU{token_note}"
            )
        report.append("\n*Note: Primary LTX-Video distilled backend executes fully free without accounts or subscriptions.*")
        return "\n".join(report)

    def handle_generation(
        user_prompt: str,
        preset_name: str,
        camera_motion: str,
        backend_choice: str,
        resolution_choice: str,
        duration_seconds: float,
        guidance_scale: float,
        seed: int,
        randomize_seed: bool,
        negative_prompt: str,
        hf_token: str,
        progress=gr.Progress(track_tqdm=True),
    ) -> Tuple[Optional[str], Optional[str], str, str]:
        if not user_prompt.strip():
            raise gr.Error("Prompt cannot be empty! Please describe your video scene.")

        # Map backend
        backend_mapping = {
            "LTX-Video Distilled (Fast Cloud GPU)": BackendID.LTX_VIDEO_DISTILLED,
            "LTX-Video ZeroGPU-Optimized": BackendID.LTX_VIDEO_OPTIMIZED,
            "CogVideoX-5B (Cloud GPU)": BackendID.COGVIDEOX_5B,
            "CogVideoX-2B (Cloud GPU)": BackendID.COGVIDEOX_2B,
        }
        b_id = backend_mapping.get(backend_choice, BackendID.LTX_VIDEO_DISTILLED)

        # Map resolution
        preset_res = RESOLUTION_PRESETS.get(resolution_choice, RESOLUTION_PRESETS["16:9 Landscape (704x512)"])
        width = preset_res.width
        height = preset_res.height

        progress(0.1, desc="Compiling cinematic prompt and camera motion...")
        compiled_prompt, compiled_neg, _ = compose_prompt(
            user_prompt=user_prompt,
            preset_name=preset_name,
            camera_motion=camera_motion,
            user_negative=negative_prompt,
        )

        request = VideoGenerationRequest(
            prompt=user_prompt,
            negative_prompt=negative_prompt,
            preset_name=preset_name,
            camera_motion=camera_motion,
            backend_id=b_id,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            guidance_scale=guidance_scale,
            seed=int(seed),
            randomize_seed=randomize_seed,
            hf_token=hf_token.strip() if hf_token else None,
            output_dir=output_dir,
        )

        def progress_callback(msg: str) -> None:
            progress(0.5, desc=msg)

        try:
            progress(0.3, desc="Connecting to cloud GPU backend...")
            result: VideoGenerationResult = engine.generate(
                request=request,
                progress_cb=progress_callback,
                auto_fallback=True,
            )
            progress(1.0, desc="Video generated and downloaded successfully!")

            meta_markdown = f"""
### 🎬 Video Generation Complete
- **Saved File:** `{Path(result.video_path).name}`
- **Generation Time:** `{result.generation_time_seconds}s`
- **Backend Model:** `{result.space_repo}` (`{result.backend_id}`)
- **Resolution:** `{result.resolution}` ({preset_res.aspect_ratio})
- **Seed Used:** `{result.seed_used}`
- **Storage Location:** `{result.video_path}`

#### 📝 Compiled Prompt:
> *{result.compiled_prompt}*
"""
            status_banner = f"✔ Generated in {result.generation_time_seconds}s on {result.space_repo}!"
            return result.video_path, result.video_path, meta_markdown, status_banner

        except Exception as e:
            err_msg = f"❌ Generation Error: {str(e)}"
            meta_err = f"**Error encountered during generation:**\n\n```\n{e}\n```"
            return None, None, meta_err, err_msg

    with gr.Blocks(title="AI Generative Video Studio Free", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
        with gr.Row(elem_classes=["header-container"]):
            with gr.Column():
                gr.Markdown(
                    """
                    # 🎬 AI Generative Video Studio
                    ### Free Cloud GPU High-Definition Video Generator • Powered by `gradio_client`
                    """
                )
                with gr.Row(elem_classes=["badge-row"]):
                    gr.HTML('<span class="badge badge-green">⚡ 100% Free Cloud GPU</span>')
                    gr.HTML('<span class="badge badge-blue">💎 SOTA LTX-Video & CogVideoX</span>')
                    gr.HTML('<span class="badge badge-purple">🔓 Zero Tokens or Subscriptions</span>')
                    gr.HTML('<span class="badge badge-blue">🎥 Cinematic Camera Controls</span>')

        with gr.Row():
            # LEFT COLUMN: CONTROLS
            with gr.Column(scale=5):
                gr.Markdown("### 1. Scene & Prompt Formulation")
                prompt_input = gr.Textbox(
                    label="Core Scene Description (Prompt)",
                    placeholder="Describe the action, subject, and atmosphere... (e.g. A futuristic cyberpunk sports car accelerating down a neon highway at dusk)",
                    lines=3,
                    max_lines=6,
                )

                # Prompt examples
                gr.Examples(
                    examples=[
                        ["A majestic golden eagle soaring above snow-capped mountain peaks at sunrise", "Cinematic Drone", "Drone Sweeping Pan"],
                        ["A neon-lit ramen stall in a rainy Tokyo backstreet with reflections on wet pavement", "Cyberpunk Alley", "Low Angle Tracking"],
                        ["Dewdrops glistening on a delicate spiderweb amidst lush green morning moss", "Macro Nature", "360 Orbit"],
                        ["A sleek black chronograph watch rotating smoothly on a polished marble pedestal", "Studio Product", "Slow Push In (Dolly In)"],
                    ],
                    inputs=[prompt_input],
                    label="Quick Prompt Inspiration",
                )

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list_preset_names(),
                        value="Cinematic Drone",
                        label="Built-in Prompt Preset",
                        info="Injects professional cinematographic lighting and aesthetics",
                    )
                    camera_dropdown = gr.Dropdown(
                        choices=list_camera_motions(),
                        value="Slow Push In (Dolly In)",
                        label="Camera Motion Control",
                        info="Guides the 3D diffusion camera trajectory",
                    )

                preset_info_box = gr.Markdown(
                    value=on_preset_change("Cinematic Drone"),
                )
                preset_dropdown.change(
                    fn=on_preset_change,
                    inputs=[preset_dropdown],
                    outputs=[preset_info_box],
                )

                gr.Markdown("### 2. Backend & Video Configuration")
                with gr.Row():
                    backend_dropdown = gr.Dropdown(
                        choices=[
                            "LTX-Video Distilled (Fast Cloud GPU)",
                            "LTX-Video ZeroGPU-Optimized",
                            "CogVideoX-5B (Cloud GPU)",
                            "CogVideoX-2B (Cloud GPU)",
                        ],
                        value="LTX-Video Distilled (Fast Cloud GPU)",
                        label="Cloud GPU Model Backend",
                        info="Default uses Lightricks Zero-A10G GPU for ~18s generation",
                    )
                    resolution_dropdown = gr.Dropdown(
                        choices=list(RESOLUTION_PRESETS.keys()),
                        value="16:9 Landscape (704x512)",
                        label="Resolution & Aspect Ratio",
                        info="Output video dimensions",
                    )

                with gr.Accordion("⚙ Advanced Generation Parameters", open=False):
                    with gr.Row():
                        duration_slider = gr.Slider(
                            minimum=1.0,
                            maximum=5.0,
                            value=2.0,
                            step=0.5,
                            label="Duration (seconds)",
                            info="Video clip length",
                        )
                        guidance_slider = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.1,
                            label="Guidance Scale",
                            info="Prompt adherence strength",
                        )
                    negative_input = gr.Textbox(
                        label="Additional Negative Prompt",
                        placeholder="Artifacts to avoid (optional)",
                        value="worst quality, blurry, jittery, distorted, pixelated, noisy, bad anatomy",
                        lines=2,
                    )
                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            info="-1 for random seed",
                        )
                        randomize_checkbox = gr.Checkbox(
                            label="Randomize Seed Each Time",
                            value=True,
                        )
                    hf_token_input = gr.Textbox(
                        label="Optional Hugging Face Token",
                        placeholder="hf_... (optional, leave blank for free public zero-gpu usage)",
                        type="password",
                        info="Allows higher priority queues or authenticated access",
                    )

                generate_btn = gr.Button(
                    "🎬 Generate Video Free (Cloud GPU)",
                    variant="primary",
                    elem_classes=["generate-btn"],
                )
                status_box = gr.Markdown("Ready to generate. Select your prompt and click Generate.")

            # RIGHT COLUMN: SCREEN & EXPORT
            with gr.Column(scale=5):
                gr.Markdown("### 3. Video Player & Direct MP4 Download")
                video_player = gr.Video(
                    label="Generated High-Definition Video",
                    interactive=False,
                    autoplay=True,
                    loop=True,
                )
                with gr.Row():
                    download_file = gr.File(
                        label="Direct MP4 File Download",
                        interactive=False,
                    )

                metadata_display = gr.Markdown(
                    label="Generation Metadata",
                    value="*Video generation metadata and compiled prompt tokens will appear here after generation.*",
                    elem_classes=["metadata-box"],
                )

                with gr.Accordion("🌐 Cloud GPU Backend Status & Health Monitor", open=False):
                    backend_status_md = gr.Markdown(value="Click 'Check Backend Health' to query Hugging Face cloud GPU endpoints.")
                    check_status_btn = gr.Button("🔄 Check Cloud GPU Status", size="sm")
                    check_status_btn.click(
                        fn=check_backend_status,
                        outputs=[backend_status_md],
                    )

                with gr.Accordion("📂 Output Video Gallery", open=False):
                    gr.Markdown(f"Videos are automatically saved locally in `{output_dir.resolve()}`")
                    gallery = gr.Gallery(
                        label="Recent Generations",
                        value=load_recent_videos(),
                        columns=3,
                        height="auto",
                    )
                    refresh_gallery_btn = gr.Button("🔄 Refresh Recent Gallery", size="sm")
                    refresh_gallery_btn.click(
                        fn=load_recent_videos,
                        outputs=[gallery],
                    )

        # Wire generate button
        generate_btn.click(
            fn=handle_generation,
            inputs=[
                prompt_input,
                preset_dropdown,
                camera_dropdown,
                backend_dropdown,
                resolution_dropdown,
                duration_slider,
                guidance_slider,
                seed_input,
                randomize_checkbox,
                negative_input,
                hf_token_input,
            ],
            outputs=[
                video_player,
                download_file,
                metadata_display,
                status_box,
            ],
        )

    return demo
