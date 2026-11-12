"""
Camera Motion Controls and Prompt Compilation Module.
Translates cinematic camera directives into diffusion video motion tokens.
"""

from typing import Dict, Tuple
from src.presets import get_preset


CAMERA_MOTIONS: Dict[str, str] = {
    "Slow Push In (Dolly In)": "slow smooth cinematic push-in dolly shot moving steadily forward towards the subject",
    "Slow Pull Back (Dolly Out)": "slow cinematic pull-back camera shot gliding backwards to reveal the expansive surrounding environment",
    "360 Orbit": "smooth 360-degree orbiting camera rotation circling around the central subject with fluid parallax",
    "Drone Sweeping Pan": "wide sweeping aerial drone pan flying smoothly across the horizon with panoramic stability",
    "Low Angle Tracking": "dramatic low-angle tracking shot moving forward fast with ground proximity and dynamic perspective",
    "Tilt Up Reveal": "smooth vertical camera tilt upward starting low at the base and slowly revealing the grand scale",
    "Crane Down": "slow majestic crane shot descending from high vantage point down to eye-level focus",
    "Handheld Kinetic": "subtle natural handheld camera motion with organic micro-movements, cinematic realism",
    "Static Locked-Off": "locked-off static tripod camera, pure natural motion within the frame without camera drift",
}


def list_camera_motions() -> list[str]:
    """Return all available camera motion keys."""
    return list(CAMERA_MOTIONS.keys())


def get_camera_description(motion_name: str) -> str:
    """Get the motion descriptor for a given camera movement key."""
    return CAMERA_MOTIONS.get(motion_name, CAMERA_MOTIONS["Slow Push In (Dolly In)"])


def compose_prompt(
    user_prompt: str,
    preset_name: str = "Cinematic Drone",
    camera_motion: str = "Slow Push In (Dolly In)",
    user_negative: str = "",
    hd_boost: bool = True,
) -> Tuple[str, str, float]:
    """
    Compile the user prompt, style preset, and camera motion into an optimal video prompt.
    
    Returns:
        (compiled_positive_prompt, compiled_negative_prompt, recommended_guidance_scale)
    """
    clean_user = user_prompt.strip()
    preset = get_preset(preset_name)
    camera_directive = get_camera_description(camera_motion)

    # Build positive prompt with strict token hierarchy:
    # [Camera directive] + [Preset prefix] + [User core prompt] + [HD Master tokens] + [Preset aesthetic suffix]
    parts: list[str] = []
    if camera_directive:
        parts.append(camera_directive)

    if preset.prefix:
        parts.append(f"{preset.prefix} {clean_user}")
    else:
        parts.append(clean_user)

    if hd_boost:
        parts.append(
            "shot on 35mm Arri Alexa 8k, master cinematography, photorealistic raytraced reflections, "
            "razor sharp deep focus, fine organic micro-textures, pristine cinema quality, hyper-detailed"
        )

    if preset.suffix:
        parts.append(preset.suffix)

    compiled_positive = ", ".join(part.strip() for part in parts if part.strip())

    # Build negative prompt with heavy suppression of blur and lowres artifacts
    base_negative = preset.default_negative
    if hd_boost:
        base_negative = (
            f"{base_negative}, 480p, low resolution, blurry, pixelated, jpeg artifacts, "
            "plastic skin, amateur footage, noisy, distorted, stuttering, flickering, muddy details"
        )

    if user_negative.strip():
        compiled_negative = f"{base_negative}, {user_negative.strip()}"
    else:
        compiled_negative = base_negative

    recommended_guidance = max(preset.recommended_guidance, 4.0) if hd_boost else preset.recommended_guidance
    return compiled_positive, compiled_negative, recommended_guidance
