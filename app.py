#!/usr/bin/env python
"""
Desktop Web UI Launcher for AI Generative Video Agent.
Launches the modern local web interface with browser auto-open,
prompt presets, camera motion controls, and direct MP4 downloads.

Usage:
    python app.py
    python app.py --port 7860 --browser
    python app.py --share
"""

import argparse
import os
from pathlib import Path
import sys
import webbrowser

from rich.console import Console
from rich.panel import Panel

from src.ui import create_ui


console = Console()


def parse_args() -> argparse.Namespace:
    default_port = int(os.environ.get("PORT", 7860))
    default_host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"

    parser = argparse.ArgumentParser(description="Launch AI Generative Video Studio Web UI.")
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help=f"Server host IP (default: {default_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Server port (default: {default_port})",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Automatically open the Web UI in the default web browser",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Password protect the UI with username:password (e.g. --auth admin:secret123)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_videos",
        help="Directory to save generated MP4 videos (default: output_videos/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve authentication credentials
    auth_tuple = None
    auth_str = args.auth or os.environ.get("APP_AUTH")
    if auth_str and ":" in auth_str:
        u, p = auth_str.split(":", 1)
        auth_tuple = (u.strip(), p.strip())
    elif os.environ.get("APP_USER") and os.environ.get("APP_PASSWORD"):
        auth_tuple = (os.environ["APP_USER"].strip(), os.environ["APP_PASSWORD"].strip())

    auth_status = f"[green]Enabled ({auth_tuple[0]})[/green]" if auth_tuple else "[yellow]None (Public)[/yellow]"

    console.print(
        Panel(
            f"[bold cyan]AI Generative Video Studio Desktop UI[/bold cyan]\n"
            f"[bold green]Local Server:[/bold green] [underline white]http://{args.host}:{args.port}[/underline white]\n"
            f"[bold yellow]Output Directory:[/bold yellow] {output_path.resolve()}\n"
            f"[bold magenta]Backends:[/bold magenta] Free Cloud GPU (LTX-Video, CogVideoX)\n"
            f"[bold blue]Tokens Required:[/bold blue] None (Completely Free)\n"
            f"[bold cyan]Authentication:[/bold cyan] {auth_status}",
            title="🎬 Starting Studio",
            border_style="cyan",
        )
    )

    demo = create_ui(output_dir=output_path)

    if args.browser:
        webbrowser.open(f"http://{args.host}:{args.port}")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.browser,
        auth=auth_tuple,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
