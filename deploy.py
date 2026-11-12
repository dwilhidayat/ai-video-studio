#!/usr/bin/env python
"""
Deployment Manager for AI Generative Video Agent.
Automates deployment across 3 targets:
1. Instant Public Web Link (Gradio Share Tunnel - 0 configuration)
2. Hugging Face Spaces (24/7 Free Cloud Hosting on CPU tier)
3. Docker Container (Self-hosted VPS, Render, Fly.io, Cloud Run)

Usage:
    python deploy.py --target share
    python deploy.py --target hf --space-name my-video-studio
    python deploy.py --target docker
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def deploy_share(auth: str = None) -> None:
    console.print(
        Panel(
            "[bold green]Starting Private Mobile-Accessible Tunnel via Gradio...[/bold green]\n"
            "This will launch the studio and create an encrypted public HTTPS URL (e.g., https://xxxx.gradio.live)\n"
            "accessible from your phone anywhere in the world.\n"
            f"[bold cyan]Authentication:[/bold cyan] {auth or 'None (Add --auth user:pass for password protection)'}",
            title="📱 Secure Phone / Remote Tunnel",
            border_style="green",
        )
    )
    cmd = [sys.executable, "app.py", "--share"]
    if auth:
        cmd.extend(["--auth", auth])
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Share tunnel stopped.[/yellow]")


def deploy_huggingface(space_repo: str, private: bool = True) -> None:
    visibility_str = "[bold magenta]PRIVATE (Only You Can Access)[/bold magenta]" if private else "[bold green]PUBLIC[/bold green]"
    console.print(
        Panel(
            f"[bold cyan]Deploying to Hugging Face Spaces: [white]{space_repo}[/white][/bold cyan]\n"
            f"Visibility: {visibility_str}\n"
            "Hugging Face Spaces provides permanent 24/7 free hosting on their CPU tier.\n"
            "When Private, you simply log in with your Hugging Face account on your phone browser.\n"
            "Nobody else can see your space or submit generation requests.",
            title="🤗 Personal Cloud Space Deploy",
            border_style="cyan",
        )
    )

    try:
        from huggingface_hub import HfApi
    except ImportError:
        console.print("[bold red]Error:[/bold red] huggingface_hub is required. Install with: pip install huggingface_hub")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[yellow]Note: HF_TOKEN environment variable not set. Prompting for Hugging Face authentication...[/yellow]")
        from huggingface_hub import login
        login()

    api = HfApi()
    try:
        console.print(f"Checking/Creating Space: [bold]{space_repo}[/bold] (Private={private})...")
        api.create_repo(
            repo_id=space_repo,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=private,
        )

        files_to_upload = [
            "app.py",
            "requirements.txt",
            "README.md",
        ]

        console.print("Uploading application files...")
        for file in files_to_upload:
            if Path(file).exists():
                console.print(f" » Uploading {file}...")
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=space_repo,
                    repo_type="space",
                )

        # Upload src folder
        console.print(" » Uploading src/ directory...")
        api.upload_folder(
            folder_path="src",
            path_in_repo="src",
            repo_id=space_repo,
            repo_type="space",
        )

        space_url = f"https://huggingface.co/spaces/{space_repo}"
        console.print(
            Panel(
                f"[bold green]✔ Deployment triggered successfully![/bold green]\n\n"
                f"Your free 24/7 web studio is building at:\n"
                f"[bold underline cyan]{space_url}[/bold underline cyan]\n\n"
                "The build takes ~1-2 minutes to install dependencies and will remain live 24/7.",
                title="✨ Live Space Ready",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Deployment failed:[/bold red] {e}")


def deploy_docker_instructions() -> None:
    console.print(
        Panel(
            """[bold cyan]Docker Production Deployment Options:[/bold cyan]

[bold green]Option A: Local / VPS Docker Compose[/bold green]
  1. Build and run in detached mode:
     [cyan]docker compose up -d --build[/cyan]
  2. View logs:
     [cyan]docker compose logs -f[/cyan]
  3. Access the Studio at: [white]http://localhost:7860[/white]

[bold green]Option B: Cloud Platforms (Render, Railway, Fly.io, Cloud Run)[/bold green]
  - [bold]Render / Railway[/bold]: Connect your Git repo; it will automatically detect the Dockerfile and expose port 7860.
  - [bold]Fly.io[/bold]: Run [cyan]fly launch[/cyan] and select default 7860 internal port.
  - [bold]GCP Cloud Run[/bold]:
     [cyan]gcloud run deploy ai-video-studio --source . --port 7860 --allow-unauthenticated[/cyan]
""",
            title="🐳 Docker Deployment",
            border_style="blue",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy AI Generative Video Agent to Cloud or Docker.")
    parser.add_argument(
        "--target",
        type=str,
        default="share",
        choices=["share", "hf", "docker"],
        help="Deployment target: 'share' (instant link), 'hf' (Hugging Face Space), 'docker' (container)",
    )
    parser.add_argument(
        "--space-name",
        type=str,
        default=None,
        help="Hugging Face Space repository name (e.g. username/ai-video-studio)",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Password protect with username:password (for share tunnel)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make Hugging Face Space public instead of private (default is private)",
    )
    args = parser.parse_args()

    if args.target == "share":
        deploy_share(auth=args.auth)
    elif args.target == "hf":
        if not args.space_name:
            console.print("[bold red]Error:[/bold red] --space-name is required when target is 'hf' (e.g. --space-name username/video-studio)")
            return 1
        deploy_huggingface(args.space_name, private=not args.public)
    elif args.target == "docker":
        deploy_docker_instructions()

    return 0


if __name__ == "__main__":
    sys.exit(main())
