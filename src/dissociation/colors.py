"""Colour palette of the figure.

The whole figure is built from three chromatic hues plus a neutral grey
axis. Every shade is specified in OKLCH (perceptually uniform lightness,
chroma, hue) and derived from a base by moving only lightness and/or
chroma at a fixed hue, so tints and muted variants stay perceptually
consistent across hues.

Functions:
---------

* :func:`oklch_to_hex`: Convert an OKLCH colour to an sRGB hex string.
* :func:`grey`: Neutral grey at a given OKLCH lightness.
* :func:`blue_for_measure`: Blue tint encoding the similarity measure.
* :func:`family_linestyle`: Line style encoding the plotted metric.
"""

__all__ = [
    "BASE_C",
    "BASE_L",
    "COLOR_BLUE",
    "COLOR_BLUE_L1",
    "COLOR_BLUE_L2",
    "COLOR_GREEN",
    "COLOR_GREY_DARK",
    "COLOR_GREY_LIGHT",
    "COLOR_GREY_MID",
    "COLOR_RED",
    "HUE_BLUE",
    "HUE_GREEN",
    "HUE_RED",
    "blue_for_measure",
    "family_linestyle",
    "grey",
    "oklch_to_hex",
]

from math import cos
from math import radians
from math import sin
from typing import Literal

# sRGB transfer function: linear values below this threshold use the
# linear segment of the piecewise encoding.
_SRGB_LINEAR_THRESHOLD = 0.0031308


def oklch_to_hex(lightness: float, chroma: float, hue: float) -> str:
    """Convert an OKLCH colour to an sRGB hex string (clamped to gamut).

    Args:
        lightness: Perceptual lightness in [0, 1].
        chroma: Chroma (colourfulness); 0 gives a neutral grey.
        hue: Hue angle in degrees.

    Returns:
        The colour as ``#rrggbb``.
    """
    hue_rad = radians(hue)
    a = chroma * cos(hue_rad)
    b = chroma * sin(hue_rad)
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    red = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    green = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    blue = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def encode(channel: float) -> int:
        channel = min(1.0, max(0.0, channel))
        srgb = (
            1.055 * channel ** (1 / 2.4) - 0.055
            if channel > _SRGB_LINEAR_THRESHOLD
            else 12.92 * channel
        )
        return round(255 * srgb)

    return f"#{encode(red):02x}{encode(green):02x}{encode(blue):02x}"


def grey(lightness: float) -> str:
    """Return a neutral grey at the given OKLCH lightness.

    Args:
        lightness: Perceptual lightness in [0, 1].

    Returns:
        The grey as ``#rrggbb``.
    """
    return oklch_to_hex(lightness, 0.0, 0.0)


# Base hues (degrees), one per model family. Chosen to stay distinguishable
# under red-green and blue-yellow colour-vision deficiency.
HUE_BLUE = 250.0  # ResNet (panels A, B)
HUE_GREEN = 150.0  # AlexNet / CIFAR-10 (panel C)
HUE_RED = 30.0  # Macaque IT (panel A)

# Base colours use the same OKLCH lightness and chroma and differ only in hue.
# Blue and green require slight clipping to the sRGB gamut at these values.
BASE_L = 0.5
BASE_C = 0.16
COLOR_BLUE = oklch_to_hex(BASE_L, BASE_C, HUE_BLUE)  # ResNet
COLOR_GREEN = oklch_to_hex(BASE_L, BASE_C, HUE_GREEN)  # AlexNet / CIFAR-10
COLOR_RED = oklch_to_hex(BASE_L, BASE_C, HUE_RED)  # Macaque IT

# Lighter blue tints (raised lightness, chroma tapered toward white). Blue
# lightness encodes the similarity measure: RSA-Spearman (COLOR_BLUE),
# RSA-Pearson (L1), linear CKA (L2), via blue_for_measure().
COLOR_BLUE_L1 = oklch_to_hex(0.65, 0.13, HUE_BLUE)
COLOR_BLUE_L2 = oklch_to_hex(0.8, 0.10, HUE_BLUE)

# Neutral grey ramp: exactly three well-separated shades.
COLOR_GREY_LIGHT = grey(0.85)  # panel-B violin edges
COLOR_GREY_MID = grey(0.6)  # legend swatches, violin fills, baseline markers
COLOR_GREY_DARK = grey(0.5)  # spines and ticks


def blue_for_measure(measure_family: str) -> str:
    """Return the blue tint for a similarity measure (panels A and B).

    Args:
        measure_family: ``"rsa"`` (Spearman), ``"rsa2"`` (Pearson) or
            ``"cka"``.

    Returns:
        The tint as ``#rrggbb``; unknown families fall back to the base
        blue.
    """
    if measure_family == "cka":
        return COLOR_BLUE_L2
    if measure_family == "rsa2":
        return COLOR_BLUE_L1
    return COLOR_BLUE


def family_linestyle(family: str) -> Literal["-", "--", ":"]:
    """Return the matplotlib line style for a plotted metric.

    Args:
        family: ``"RS"``, ``"MCA"`` (accuracy) or ``"EC"``.

    Returns:
        A matplotlib line style string.
    """
    if family == "EC":
        return ":"
    if family == "MCA":
        return "--"
    return "-"
