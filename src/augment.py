"""Bridge to the shared `cellaug` package (ObliqueSection).

VoCell slices *geometrically*: it builds a 64³ volume and cuts it with an
arbitrary plane. `ObliqueSection` — the augmentation used to train SimCLR in
cellf-supervised — approximates that same operation directly in 2D, without
ever building a volume. Both are therefore two answers to the same question,
and this module makes them comparable by expressing the app's cutting plane
(azimuth / elevation / offset) in ObliqueSection's own parameters.

Pure numpy: no Streamlit here. The augmentation itself lives in `cellaug`
(https://github.com/Arthur-Chiron/cellaug) and must never be copied in.
"""

import numpy as np
from cellaug import ObliqueSection, estimate_nucleus_cov
from typing import Dict, Optional, Tuple

# Below this elevation the cutting plane is nearly vertical: ObliqueSection
# models a *section* of a flattened nucleus and stops meaning anything there.
MIN_ELEVATION_DEG = 20.0


def plane_to_oblique(azimuth: float, elevation: float, slice_offset: float,
                     img: np.ndarray, aspect: float) -> Dict[str, float]:
    """Express the app's cutting plane in ObliqueSection's parameters.

    The app's plane has normal n = (cos el cos az, cos el sin az, sin el) and
    satisfies n · (p - c) = slice_offset. Solving for z on that plane gives

        z - cz = slice_offset / nz - cot(el) * (dx cos az + dy sin az)

    which is exactly ObliqueSection's affine depth map with

        tilt = 90° - el,  phi = az + 180°,  offset = slice_offset / sin(el)

    Offset is returned as a fraction of the nucleus half-height (ObliqueSection's
    own unit), so it stays independent of nucleus size.

    Returns a dict with `tilt_deg`, `phi`, `offset`, plus `h_half_px` and
    `clamped` (True when the plane was too steep to be representable).
    """
    az, el, off = float(azimuth), float(elevation), float(slice_offset)

    # A plane and its opposite normal are the same plane: fold to el > 0.
    if el < 0:
        az, el, off = az + 180.0, -el, -off

    clamped = el < MIN_ELEVATION_DEG
    el = max(el, MIN_ELEVATION_DEG)

    h_half = max(aspect * nucleus_radius(img), 1e-3)
    return {
        "tilt_deg": 90.0 - el,
        "phi": np.radians(az + 180.0),
        "offset": (off / np.sin(np.radians(el))) / h_half,
        "h_half_px": h_half,
        "clamped": clamped,
    }


def nucleus_radius(img: np.ndarray) -> float:
    """Equivalent lateral radius of the nucleus, in pixels (cellaug's estimate)."""
    return estimate_nucleus_cov(np.asarray(img, dtype=np.float32))[3]


def aspect_from_volume(volume: np.ndarray) -> float:
    """Measure the actual aspect ratio (half-height / lateral radius) of a volume.

    Lets the 2D approximation be calibrated on the very volume the geometric
    slice cuts, instead of the default 0.4 of cultured cells. Same moment-based
    convention as `cellaug.estimate_nucleus`: for a uniform segment of
    half-length H, var = H²/3.
    """
    vol = np.asarray(volume, dtype=np.float32)
    R = nucleus_radius(vol.max(axis=0))

    w = vol.sum(axis=(1, 2))
    total = w.sum()
    if total <= 0 or R <= 0:
        return 0.4
    z = np.arange(vol.shape[0], dtype=np.float32)
    cz = float((w * z).sum() / total)
    var = float((w * (z - cz) ** 2).sum() / total)
    return float(np.clip(np.sqrt(3.0 * var) / R, 0.05, 3.0))


def make_augmentation(physical: Dict, rng: Optional[np.random.Generator] = None) -> ObliqueSection:
    """Build an ObliqueSection from the sidebar's physical parameters."""
    return ObliqueSection(rng=rng, **physical)


def apply_forced(img: np.ndarray, plane: Dict[str, float], physical: Dict) -> Tuple[np.ndarray, Dict]:
    """Apply the plane imposed by the app's sliders — deterministic."""
    aug = make_augmentation(physical)
    return aug.apply(img, tilt_deg=plane["tilt_deg"], phi=plane["phi"],
                     offset=plane["offset"])


def apply_random(img: np.ndarray, seed: int, physical: Dict,
                 max_tilt_deg: float, max_offset: float) -> Tuple[np.ndarray, Dict]:
    """Draw a plane the way training does, but from a fixed seed.

    The seed is what keeps a Streamlit re-render from silently redrawing a
    different plane every time a slider moves.
    """
    aug = make_augmentation(
        {**physical, "max_tilt_deg": max_tilt_deg, "max_offset": max_offset, "p": 1.0},
        rng=np.random.default_rng(seed),
    )
    return aug(img)
