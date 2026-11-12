#!/usr/bin/env python
"""
Command-Line Interface (CLI) for AI Generative Video Agent.
Enables fast, free, tokenless video generation from the terminal.

Usage Examples:
    python generate.py --prompt "A majestic eagle soaring over snow-capped mountain peaks"
    python generate.py --prompt "Rain pouring down on neon city streets" --preset "Cyberpunk Alley" --camera "Low Angle Tracking"
    python generate.py --prompt "Exotic tropical flower blooming" --preset "Macro Nature" --camera "360 Orbit" --output flower.mp4
    python generate.py --list-presets
    python generate.py --list-cameras
    python generate.py --list-models
"""

import argparse
from pathlib import Path
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.camera import CAMERA_MOTIONS, list_camera_motions
from src.config import (
    BackendID,
    RESOLUTION_PRESETS,
    SUPPORTED_BACKENDS,
    VideoGenerationRequest,
)
from src.engine import VideoEngine, VideoEngineError
from src.presets import PROMPT_PRESETS, list_preset_names


console = Console()


def show_presets_table() -> None:
    table = Table(title="🎬 Built-in Prompt Engineering Presets", border_style="cyan")
    table.add_column("Preset Name", style="bold green", width=22)
    table.add_column("Guidance", style="yellow", width=10)
    table.add_column("Description", style="white")
    for name, p in PROMPT_PRESETS.items():
        table.add_row(name, f"{p.recommended_guidance}", p.description)
    console.print(table)


def show_cameras_table() -> None:
    table = Table(title="🎥 Camera Motion Controls", border_style="magenta")
    table.add_column("Camera Motion", style="bold cyan", width=28)
    table.add_column("Motion Description", style="white")
    for name, desc in CAMERA_MOTIONS.items():
        table.add_row(name, desc)
    console.print(table)


def show_models_table(engine: VideoEngine) -> None:
    table = Table(title="⚡ Free Cloud GPU Backends (Hugging Face Spaces)", border_style="blue")
    table.add_column("Backend Key", style="bold yellow", width=22)
    table.add_column("Model Name", style="bold green", width=30)
    table.add_column("Status / Stage", style="cyan", width=18)
    table.add_column("Hardware", style="magenta", width=14)
    table.add_column("Default", style="bold", width=10)

    for b_id, info in SUPPORTED_BACKENDS.items():
        health = engine.check_space_health(b_id)
        stage_display = f"[green]{health['stage']}[/green]" if health['ready'] else f"[red]{health['stage']}[/red]"
        table.add_row(
            b_id.value,
            info.name,
            stage_display,
            health['hardware'],
            "⭐ YES" if info.is_primary else "NO",
        )
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast, free, tokenless AI video generator powered by cloud GPU backends via gradio_client.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --prompt "Futuristic sports car speeding across salt flats"
  python generate.py --prompt "Raindrops falling on a lotus leaf" --preset "Macro Nature" --camera "360 Orbit"
  python generate.py --prompt "Cyberpunk street with ramen stall" --preset "Cyberpunk Alley" --output ./street.mp4
        """,
    )

    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Text prompt describing the video scene",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="Cinematic Drone",
        choices=list_preset_names(),
        help="Prompt engineering preset (default: 'Cinematic Drone')",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="Slow Push In (Dolly In)",
        choices=list_camera_motions(),
        help="Camera motion dynamic (default: 'Slow Push In (Dolly In)')",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=BackendID.LTX_VIDEO_DISTILLED.value,
        choices=[b.value for b in BackendID],
        help="Target video model backend (default: ltx-video-distilled)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_videos",
        help="Output destination path or directory (default: output_videos/)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="704x512",
        help="Video resolution WxH, e.g. 704x512, 512x704, 768x432 (default: 704x512)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=2.0,
        help="Video duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "-g", "--guidance",
        type=float,
        default=3.0,
        help="Guidance scale (default: 3.0)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=-1,
        help="Random seed for generation (-1 for randomized seed, default: -1)",
    )
    parser.add_argument(
        "-n", "--negative",
        type=str,
        default="",
        help="Additional negative prompt to suppress artifacts",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face access token for extended GPU quotas",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable automatic fallback to alternative backends upon failure",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Display all available prompt engineering presets and exit",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Display all supported camera motion controls and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Display all supported free cloud GPU backends and health status",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = VideoEngine()

    if args.list_presets:
        show_presets_table()
        return 0

    if args.list_cameras:
        show_cameras_table()
        return 0

    if args.list_models:
        show_models_table(engine)
        return 0

    if not args.prompt:
        console.print("[bold red]Error:[/bold red] Prompt argument '--prompt' or '-p' is required.")
        console.print("Run [cyan]python generate.py --help[/cyan] for usage instructions.")
        return 1

    # Parse resolution WxH
    width, height = 704, 512
    if "x" in args.resolution:
        try:
            w_str, h_str = args.resolution.lower().split("x")
            width, height = int(w_str), int(h_str)
        except ValueError:
            console.print(f"[yellow]Warning: Invalid resolution '{args.resolution}'. Falling back to 704x512.[/yellow]")

    # Resolve output directory/file
    out_path = Path(args.output)
    if out_path.suffix.lower() == ".mp4":
        target_dir = out_path.parent
        custom_target_file = out_path
    else:
        target_dir = out_path
        custom_target_file = None

    target_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold cyan]AI Generative Video CLI[/bold cyan]\n"
            f"[bold white]Prompt:[/bold white] {args.prompt}\n"
            f"[bold yellow]Preset:[/bold yellow] {args.preset} | [bold magenta]Camera:[/bold magenta] {args.camera}\n"
            f"[bold green]Model:[/bold green] {args.model} | [bold blue]Resolution:[/bold blue] {width}x{height} | [bold white]Duration:[/bold white] {args.duration}s",
            title="🎬 Generation Request",
            border_style="cyan",
        )
    )

    request = VideoGenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative,
        preset_name=args.preset,
        camera_motion=args.camera,
        backend_id=BackendID(args.model),
        width=width,
        height=height,
        duration_seconds=args.duration,
        guidance_scale=args.guidance,
        seed=args.seed,
        randomize_seed=(args.seed == -1),
        hf_token=args.hf_token,
        output_dir=target_dir,
    )

    def on_progress(msg: str) -> None:
        console.print(f"[dim]» {msg}[/dim]")

    with console.status("[bold green]Generating video on free cloud GPU...[/bold green]", spinner="dots"):
        try:
            result = engine.generate(
                request=request,
                progress_cb=on_progress,
                auto_fallback=not args.no_fallback,
            )
        except VideoEngineError as err:
            console.print(f"\n[bold red]Generation Failed:[/bold red] {err}")
            return 1
        except Exception as exc:
            console.print(f"\n[bold red]Unexpected Error:[/bold red] {exc}")
            return 1

    # If the user specified a custom .mp4 file path, move the generated video to that exact location
    final_video_path = result.video_path
    if custom_target_file:
        try:
            custom_target_file.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(result.video_path, custom_target_file)
            final_video_path = str(custom_target_file.resolve())
        except Exception as move_err:
            console.print(f"[yellow]Could not rename to custom path: {move_err}[/yellow]")

    console.print(
        Panel(
            f"[bold green]✔ Video successfully generated and saved![/bold green]\n\n"
            f"[bold white]File Location:[/bold white] [underline cyan]{final_video_path}[/underline cyan]\n"
            f"[bold white]Backend Space:[/bold white] {result.space_repo} ({result.backend_id})\n"
            f"[bold white]Elapsed Time:[/bold white] {result.generation_time_seconds}s\n"
            f"[bold white]Seed Used:[/bold white] {result.seed_used}\n"
            f"[bold white]Resolution:[/bold white] {result.resolution}\n\n"
            f"[dim]Compiled Prompt:[/dim]\n[italic]{result.compiled_prompt}[/italic]",
            title="✨ Video Ready",
            border_style="green",
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
