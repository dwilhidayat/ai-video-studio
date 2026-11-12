#!/usr/bin/env python
"""
Verification Test Script for AI Generative Video Agent.
Probes and verifies connectivity to free cloud GPU backends on Hugging Face Spaces.

Usage:
    python verify_backends.py
    python verify_backends.py --benchmark
    python verify_backends.py --model ltx-video-distilled
"""

import argparse
import sys
import time
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import (
    BackendID,
    SUPPORTED_BACKENDS,
    VideoGenerationRequest,
)
from src.engine import VideoEngine


console = Console()


def verify_all_backends(benchmark: bool = False, target_backend: str = None) -> bool:
    console.print(
        Panel(
            "[bold cyan]AI Generative Video - Cloud GPU Connectivity Diagnostic[/bold cyan]\n"
            "Verifying connectivity to public Gradio endpoints and ZeroGPU resources...",
            title="🔍 Backend Verification",
            border_style="cyan",
        )
    )

    engine = VideoEngine()
    table = Table(title="Backend Health & API Status", border_style="blue")
    table.add_column("Backend ID", style="bold yellow", width=22)
    table.add_column("Model Name", style="bold white", width=28)
    table.add_column("HuggingFace Space", style="cyan", width=34)
    table.add_column("Runtime Stage", style="bold", width=16)
    table.add_column("Hardware", style="magenta", width=12)
    table.add_column("Client API", style="bold", width=14)
    table.add_column("Free ZeroGPU", style="bold green", width=14)

    all_ok = True
    primary_ok = False

    backends = [
        (b_id, info) for b_id, info in SUPPORTED_BACKENDS.items()
        if not target_backend or b_id.value == target_backend
    ]

    for b_id, info in backends:
        console.print(f"[dim]Checking {info.name} ({info.space_repo})...[/dim]")
        health = engine.check_space_health(b_id)
        stage = health.get("stage", "UNKNOWN")
        hw = health.get("hardware", "UNKNOWN")

        # Test gradio_client connection
        api_status = "[red]FAILED[/red]"
        is_client_ok = False
        try:
            client = engine.get_client(info.space_repo)
            api_info = client.view_api(return_format="dict")
            named = api_info.get("named_endpoints", {})
            if info.default_endpoint in named or len(named) > 0:
                api_status = "[green]ONLINE[/green]"
                is_client_ok = True
        except Exception as e:
            api_status = f"[red]ERR: {str(e)[:15]}[/red]"

        is_stage_running = (stage == "RUNNING")
        stage_str = f"[green]{stage}[/green]" if is_stage_running else f"[red]{stage}[/red]"
        hw_str = f"[cyan]{hw}[/cyan]"

        # ZeroGPU free status
        if info.requires_zero_gpu_token:
            zerogpu_status = "[yellow]Token/Pro[/yellow]"
        elif is_stage_running and "a10" in str(hw).lower():
            zerogpu_status = "[bold green]100% Free[/bold green]"
        else:
            zerogpu_status = "[dim]Variable[/dim]"

        table.add_row(
            b_id.value,
            info.name,
            info.space_repo,
            stage_str,
            hw_str,
            api_status,
            zerogpu_status,
        )

        if info.is_primary and is_stage_running and is_client_ok:
            primary_ok = True

    console.print(table)

    if not primary_ok:
        console.print("[bold red]⚠ Warning: Primary backend (LTX-Video Distilled) was not verified as ready.[/bold red]")
        all_ok = False
    else:
        console.print("[bold green]✔ Primary free cloud GPU backend is healthy, verified, and operational![/bold green]")

    # Run quick benchmark generation if requested
    if benchmark:
        console.print("\n[bold yellow]Running active benchmark generation test on primary backend...[/bold yellow]")
        req = VideoGenerationRequest(
            prompt="A calm ocean sunset with gentle rolling waves, 4k cinematic",
            preset_name="Cinematic Drone",
            camera_motion="Slow Push In (Dolly In)",
            backend_id=BackendID.LTX_VIDEO_DISTILLED,
            width=704,
            height=512,
            duration_seconds=2.0,
            seed=42,
            randomize_seed=False,
        )

        t0 = time.time()
        try:
            result = engine.generate(req, progress_cb=lambda m: console.print(f"[dim]» {m}[/dim]"))
            elapsed = round(time.time() - t0, 2)
            console.print(
                Panel(
                    f"[bold green]✔ Benchmark generation succeeded in {elapsed}s![/bold green]\n"
                    f"Video file saved at: [cyan]{result.video_path}[/cyan]\n"
                    f"File size: [white]{round(Path(result.video_path).stat().st_size / 1024, 1)} KB[/white]",
                    title="⚡ Benchmark Test Passed",
                    border_style="green",
                )
            )
        except Exception as exc:
            console.print(f"[bold red]Benchmark generation failed: {exc}[/bold red]")
            all_ok = False

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify connectivity to free cloud GPU video generation backends.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Perform an active end-to-end benchmark generation on the primary backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[b.value for b in BackendID],
        help="Check a specific model backend only",
    )
    args = parser.parse_args()

    success = verify_all_backends(benchmark=args.benchmark, target_backend=args.model)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
