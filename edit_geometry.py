"""Geometry helpers shared by edit precaching and inference.

The ``square`` geometry stretches each reference to the target canvas. The ``fit`` geometry keeps
the reference aspect ratio, fits it inside the target token grid, and minimally crops before the
final resize so snapping to the model's pixel alignment does not introduce anisotropic squash.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


EDIT_REF_GEOMETRIES = ("square", "fit")


@dataclass(frozen=True)
class ReferenceTransform:
    """A source crop followed by an aligned resize."""

    crop: tuple[int, int, int, int]
    size: tuple[int, int]  # (width, height)


def validate_edit_ref_geometry(value: str) -> str:
    value = str(value or "square").lower()
    if value not in EDIT_REF_GEOMETRIES:
        raise ValueError(
            f"data.edit_ref_geometry must be one of {EDIT_REF_GEOMETRIES}, got {value!r}")
    return value


def reference_transform(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    align: int = 16,
    crop_tolerance: float = 0.08,
) -> ReferenceTransform:
    """Return the aspect-preserving fit transform for one reference.

    Near-matching aspect ratios are minimally center-cropped to fill the target.  Otherwise the
    reference is fitted inside the target.  The source is cropped before resize so both fitted
    dimensions land exactly on the aligned grid without stretching.
    """
    sw, sh = map(int, source_size)
    tw, th = map(int, target_size)
    if min(sw, sh, tw, th, align) <= 0:
        raise ValueError(
            f"image sizes and alignment must be positive, got source={source_size}, "
            f"target={target_size}, align={align}")
    if not 0.0 <= crop_tolerance < 1.0:
        raise ValueError(f"crop_tolerance must be in [0,1), got {crop_tolerance}")

    fit_scale = min(th / sh, tw / sw)
    close = (
        sh * fit_scale >= th * (1.0 - crop_tolerance)
        and sw * fit_scale >= tw * (1.0 - crop_tolerance)
    )
    if close:
        fill_scale = max(th / sh, tw / sw)
        crop_h = min(sh, max(1, round(th / fill_scale)))
        crop_w = min(sw, max(1, round(tw / fill_scale)))
        out_w, out_h = tw, th
    else:
        out_h = min(max(align, int(sh * fit_scale) // align * align),
                    max(align, th // align * align))
        out_w = min(max(align, int(sw * fit_scale) // align * align),
                    max(align, tw // align * align))
        crop_h = min(sh, max(1, round(out_h / fit_scale)))
        crop_w = min(sw, max(1, round(out_w / fit_scale)))

    left = (sw - crop_w) // 2
    top = (sh - crop_h) // 2
    return ReferenceTransform(
        crop=(left, top, left + crop_w, top + crop_h),
        size=(out_w, out_h),
    )


def prepare_reference_image(
    image: Image.Image,
    target_size: tuple[int, int],
    *,
    geometry: str,
    align: int = 16,
    crop_tolerance: float = 0.08,
) -> Image.Image:
    """Convert a reference to RGB and apply the selected edit geometry."""
    geometry = validate_edit_ref_geometry(geometry)
    image = image.convert("RGB")
    if geometry == "square":
        return image.resize(target_size, Image.Resampling.BILINEAR)
    transform = reference_transform(
        image.size, target_size, align=align, crop_tolerance=crop_tolerance)
    if transform.crop != (0, 0, image.width, image.height):
        image = image.crop(transform.crop)
    if image.size != transform.size:
        image = image.resize(transform.size, Image.Resampling.BILINEAR)
    return image


def centered_grid_offsets(
    target_grid: tuple[int, int],
    ref_grids: list[tuple[int, int]],
) -> list[tuple[float, float]]:
    """Fractionally center reference grids inside the target grid."""
    th, tw = target_grid
    return [
        (max(0.0, (th - rh) / 2.0), max(0.0, (tw - rw) / 2.0))
        for rh, rw in ref_grids
    ]
