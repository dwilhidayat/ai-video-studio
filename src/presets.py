"""
Prompt Engineering Presets for High-Definition Video Generation.
Provides professionally tuned style templates, negative prompts, and visual dynamics.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PromptPreset:
    name: str
    description: str
    prefix: str
    suffix: str
    default_negative: str
    recommended_guidance: float
    recommended_duration: float


PROMPT_PRESETS: Dict[str, PromptPreset] = {
    "Cinematic Drone": PromptPreset(
        name="Cinematic Drone",
        description="Epic sweeping aerial landscape cinematography, photorealistic atmospheric lighting, 8k IMAX look.",
        prefix="Breathtaking ultra-high-definition cinematic aerial drone shot of",
        suffix=(
            "sweeping panoramic landscape, 8k resolution, cinematic film look, photorealistic lighting, "
            "natural color grading, hyper-detailed terrain, volumetric atmosphere, National Geographic cinematography"
        ),
        default_negative=(
            "worst quality, blurry, jittery, distorted, pixelated, amateur footage, noisy, artifacts, "
            "oversaturated, foggy lens, low contrast, compression artifacts"
        ),
        recommended_guidance=3.0,
        recommended_duration=2.5,
    ),
    "Cyberpunk Alley": PromptPreset(
        name="Cyberpunk Alley",
        description="Atmospheric rain-slicked futuristic cityscape with vibrant neon reflections and volumetric fog.",
        prefix="Moody atmospheric cyberpunk alleyway at night featuring",
        suffix=(
            "gleaming wet asphalt reflections, glowing neon signs in vivid magenta and cyan, dense rising steam, "
            "cinematic shallow depth of field, Blade Runner aesthetic, intricate architectural cables, realistic rain drops, octane render"
        ),
        default_negative=(
            "daylight, sunny, washed out colors, low contrast, cartoonish, low resolution, flat lighting, "
            "jittery frames, muddy textures, plain"
        ),
        recommended_guidance=3.5,
        recommended_duration=2.0,
    ),
    "Macro Nature": PromptPreset(
        name="Macro Nature",
        description="Razor-sharp close-up footage revealing intricate biological micro-textures, water droplets, and creamy bokeh.",
        prefix="Extreme macro cinematic close-up of",
        suffix=(
            "razor-sharp focus, shallow depth of field with creamy bokeh, glistening translucent dewdrops, "
            "vibrant micro-textures, delicate organic details, golden hour rim lighting, BBC Planet Earth style, crystal clear 4k"
        ),
        default_negative=(
            "blurry foreground, out of focus subject, flat lighting, artificial CGI look, compression artifacts, "
            "harsh noise, overexposed, low detail"
        ),
        recommended_guidance=3.0,
        recommended_duration=2.0,
    ),
    "Studio Product": PromptPreset(
        name="Studio Product",
        description="Commercial-grade luxury product showcase with three-point studio softbox illumination and immaculate reflections.",
        prefix="Commercial high-end luxury studio product cinematography showcasing",
        suffix=(
            "sleek glossy surfaces, perfectly diffused studio softbox three-point lighting, clean minimalist backdrop, "
            "crisp specular highlights, commercial advertising grade, pristine 4k clarity, smooth elegant motion"
        ),
        default_negative=(
            "dust, scratches, harsh shadows, amateur lighting, cluttered background, low resolution, "
            "motion blur, flickering, noisy, dull finish"
        ),
        recommended_guidance=2.8,
        recommended_duration=2.0,
    ),
    "None / Raw Prompt": PromptPreset(
        name="None / Raw Prompt",
        description="Direct user prompt without stylistic alterations, only standard high-fidelity video enhancement.",
        prefix="",
        suffix="high quality, detailed, smooth motion, photorealistic, 4k",
        default_negative="worst quality, inconsistent motion, blurry, jittery, distorted, low resolution",
        recommended_guidance=3.0,
        recommended_duration=2.0,
    ),
}


def get_preset(name: str) -> PromptPreset:
    """Retrieve a preset by name or return the raw prompt default."""
    return PROMPT_PRESETS.get(name, PROMPT_PRESETS["None / Raw Prompt"])


def list_preset_names() -> list[str]:
    """Return all available preset names."""
    return list(PROMPT_PRESETS.keys())
