---
title: AI Generative Video Studio Free
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎬 AI Generative Video Agent (Desktop Studio & CLI)

A production-grade, tokenless desktop tool for generating high-definition AI videos completely free without API tokens, credits, or paid subscriptions. It leverages public cloud GPU backends (ZeroGPU A10G instances on Hugging Face Spaces) via `gradio_client` to deliver State-of-the-Art (SOTA) video generation directly to your local workstation.

---

## 🌟 Key Capabilities

- **100% Free Cloud GPU Computation**: Generates high-definition MP4 videos without requiring local GPUs, paid cloud credits, or API tokens.
- **State-of-the-Art Model Support**:
  - **LTX-Video Distilled** (`Lightricks/ltx-video-distilled`): Ultra-fast ~15–20s generation on Zero-A10G GPU.
  - **LTX-Video ZeroGPU-Optimized** (`DeepRat/LTX-Video-ZeroGPU-Optimized`): Robust community-optimized deployment.
  - **CogVideoX-5B & 2B** (`zai-org/CogVideoX-5B-Space`, `zai-org/CogVideoX-2B-Space`): 5-Billion and 2-Billion parameter transformer architectures.
- **Dual Interface Modes**:
  - **Modern Web UI**: Clean, responsive Gradio interface with live prompt compilation previews, camera controls, video player, direct MP4 downloads, and a local output gallery.
  - **High-Speed CLI**: Rapid single-command terminal execution (`python generate.py --prompt "..."`).
- **Cinematic Prompt Engineering Presets**: Built-in visual enhancers (*Cinematic Drone*, *Cyberpunk Alley*, *Macro Nature*, *Studio Product*).
- **Camera Motion Controls**: 3D trajectory steering (*Slow Push In*, *Slow Pull Back*, *360 Orbit*, *Drone Sweeping Pan*, *Low Angle Tracking*, *Tilt Up Reveal*, *Crane Down*, *Handheld Kinetic*).
- **Automatic Backend Failover**: Seamlessly fails over to secondary healthy endpoints if a primary cloud space encounters traffic or queue limits.
- **Direct MP4 Export**: Automatically archives generated clips locally with timestamped, metadata-rich filenames.

---

## 🏗 Architecture

```mermaid
flowchart TD
    subgraph User Interfaces
        A["Modern Web UI (app.py)"]
        B["Fast CLI (generate.py)"]
    end

    subgraph Core Engine
        C["Prompt & Camera Compiler (src/camera.py + src/presets.py)"]
        D["Video Engine (src/engine.py)"]
        E["Configuration & Presets (src/config.py)"]
    end

    subgraph Cloud GPU Backends (gradio_client)
        F["Primary: Lightricks/ltx-video-distilled (Zero-A10G)"]
        G["Fallback: DeepRat/LTX-Video-ZeroGPU-Optimized (Zero-A10G)"]
        H["CogVideoX: zai-org/CogVideoX-5B-Space & 2B-Space"]
    end

    subgraph Local Storage
        I["Local Output Archive (output_videos/*.mp4)"]
        J["Direct Browser MP4 Download"]
    end

    A --> C
    B --> C
    C --> D
    E --> D
    D -->|Predict /text_to_video| F
    D -.->|Auto Fallback| G
    D -.->|Predict /generate| H
    F -->|Return MP4| D
    G -->|Return MP4| D
    H -->|Return MP4| D
    D --> I
    I --> J
```

---

## 🛠 1. Virtual Environment Setup & Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your system.
- Active internet connection (to connect to free cloud GPU spaces).

### Windows (PowerShell)
```powershell
# Navigate to the project directory
cd c:\Users\User\Desktop\ai_generative_agent

# Create a clean virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Command Prompt)
```cmd
cd c:\Users\User\Desktop\ai_generative_agent
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux (bash or zsh)
```bash
cd ai_generative_agent
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔍 2. Backend Connectivity Verification

Before generating videos, you can run the included verification diagnostic script to confirm connectivity to the free cloud GPU endpoints:

```bash
# Check status and API readiness of all backends
python verify_backends.py

# Run verification with an active benchmark video generation
python verify_backends.py --benchmark

# Verify a specific backend only
python verify_backends.py --model ltx-video-distilled
```

Output displays a formatted status table reporting runtime stage (`RUNNING`), hardware allocation (`zero-a10g`), and client API readiness.

---

## 🖥 3. Web UI Desktop Studio

Launch the local web studio:

```bash
# Standard launch (accessible at http://127.0.0.1:7860)
python app.py

# Launch and automatically open your default browser
python app.py --browser

# Launch on custom port or enable a public share link
python app.py --port 8080 --share
```

### Web Studio Features
1. **Scene Input**: Multi-line prompt box with one-click inspiration pills.
2. **Style Presets**: Live interactive style descriptions for *Cinematic Drone*, *Cyberpunk Alley*, *Macro Nature*, and *Studio Product*.
3. **Camera Trajectory**: Dropdown to select 3D motion dynamics (*360 Orbit*, *Sweeping Pan*, *Dolly In*, etc.).
4. **Resolution Presets**: One-click switching between `16:9 Landscape (704x512)`, `16:9 Cinema (768x432)`, `9:16 Vertical (512x704)`, and `1:1 Square (512x512)`.
5. **Video Player**: High-definition HTML5 video player with seamless looping and playback controls.
6. **Direct MP4 Download**: Dedicated download widget to save the MP4 straight to your machine.
7. **Cloud GPU Health Monitor**: Real-time status query of Hugging Face Spaces.
8. **Recent Generation Gallery**: Interactive gallery showing all recently created videos.

---

## ⚡ 4. Fast CLI Generation (`generate.py`)

The CLI offers instant, scriptable video generation from your terminal.

### Quick Start
```bash
python generate.py --prompt "A majestic eagle soaring over snow-capped mountain peaks"
```

### Using Presets & Camera Motion
```bash
# Cyberpunk rain scene with low-angle tracking camera
python generate.py \
  --prompt "Rain pouring down on neon city streets with reflections" \
  --preset "Cyberpunk Alley" \
  --camera "Low Angle Tracking" \
  --output ./cyberpunk_alley.mp4

# Macro nature shot with 360-degree orbiting camera
python generate.py \
  --prompt "An exotic violet orchid blooming with dewdrops" \
  --preset "Macro Nature" \
  --camera "360 Orbit" \
  --resolution 512x512

# Luxury product showcase
python generate.py \
  --prompt "A luxury mechanical watch with exposed golden gears" \
  --preset "Studio Product" \
  --camera "Slow Push In (Dolly In)" \
  --duration 2.5
```

### Inspecting CLI Catalogs
```bash
# List all built-in presets
python generate.py --list-presets

# List all supported camera motion controls
python generate.py --list-cameras

# List all free cloud GPU backends and live health
python generate.py --list-models
```

### All CLI Options
| Flag | Short | Default | Description |
|---|---|---|---|
| `--prompt` | `-p` | *Required* | Description of the video scene |
| `--preset` | | `Cinematic Drone` | Preset style (*Cinematic Drone*, *Cyberpunk Alley*, *Macro Nature*, *Studio Product*, *None*) |
| `--camera` | | `Slow Push In (Dolly In)` | Camera motion directive |
| `--model` | `-m` | `ltx-video-distilled` | Backend model (`ltx-video-distilled`, `ltx-video-optimized`, `cogvideox-5b`, `cogvideox-2b`) |
| `--output` | `-o` | `output_videos/` | Destination output directory or specific `.mp4` file path |
| `--resolution` | | `704x512` | Video resolution `WxH` (e.g. `704x512`, `512x704`, `768x432`, `512x512`) |
| `--duration` | `-d` | `2.0` | Video duration in seconds (1.0 to 5.0) |
| `--guidance` | `-g` | `3.0` | CFG prompt adherence guidance scale |
| `--seed` | `-s` | `-1` | Random seed (`-1` for randomized seed) |
| `--negative` | `-n` | `""` | Additional negative prompt to suppress artifacts |
| `--hf-token` | | `None` | Optional Hugging Face token for private spaces or higher quotas |
| `--no-fallback` | | `False` | Disables automatic fallback to secondary backend on error |

---

## 🎨 5. Prompt Engineering Presets Breakdown

| Preset | Aesthetic & Lighting Focus | Recommended Guidance | Recommended Motion |
|---|---|---|---|
| **Cinematic Drone** | Sweeping panoramic aerial shots, 8k resolution, IMAX film look, volumetric atmospheric haze, natural photorealistic grading. | `3.0` | Drone Sweeping Pan |
| **Cyberpunk Alley** | Rain-slicked asphalt, neon sign reflections in magenta/cyan, rising steam, Blade Runner aesthetic, Octane render depth. | `3.5` | Low Angle Tracking |
| **Macro Nature** | Razor-sharp micro-textures, glistening translucent dewdrops, shallow depth of field, creamy bokeh, BBC Planet Earth rim light. | `3.0` | 360 Orbit |
| **Studio Product** | Three-point softbox studio illumination, sleek glossy reflections, clean minimalist backdrop, commercial advertising grade. | `2.8` | Slow Push In (Dolly In) |

---

## 🧪 6. Automated Test Suite

Run the full pytest suite:

```bash
pytest -v
```

All 17 unit and integration tests cover:
- Built-in prompt engineering presets and fallbacks.
- Camera motion directives and negative prompt stacking.
- Resolution mapping and Pydantic schema validation.
- Video file extraction from various Gradio return signatures.
- Space health probing and Hugging Face API checks.
- CLI argument parsing and catalog flags.

---

## 📁 7. Project Structure

```
ai_generative_agent/
├── app.py                   # Desktop Web UI launcher (Gradio blocks)
├── generate.py              # Fast CLI video generation entrypoint
├── verify_backends.py       # Cloud GPU connectivity verification tool
├── requirements.txt         # Tested, pinned dependency requirements
├── pytest.ini               # Pytest configuration
├── README.md                # Full documentation & setup guide
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Backend definitions, Pydantic models, resolutions
│   ├── presets.py           # Cinematic prompt engineering presets
│   ├── camera.py            # 3D camera trajectory controls & prompt compiler
│   ├── engine.py            # Video generation engine via gradio_client with failover
│   └── ui.py                # Modern Gradio desktop studio interface
├── tests/
│   ├── test_presets.py      # Preset unit tests
│   ├── test_camera.py       # Camera motion tests
│   ├── test_config.py       # Configuration and schema tests
│   ├── test_engine.py       # Engine logic & output extraction tests
│   └── test_cli.py          # CLI flag and parser tests
└── output_videos/           # Automatic persistent MP4 output archive
```

---

## 🛡 License & Terms
This tool interfaces exclusively with public, open-source model checkpoints hosted on Hugging Face Spaces. It uses documented, sanctioned Gradio client endpoints and complies with open-weights model usage terms.
