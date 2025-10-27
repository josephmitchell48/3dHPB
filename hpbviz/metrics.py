from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class ComponentVolume:
    """Volume information for a single connected component."""

    label_id: int
    voxel_count: int
    volume_mm3: float
    volume_ml: float


def _as_binary_mask(mask: sitk.Image, label_value: Optional[int]) -> sitk.Image:
    if label_value is None:
        binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    else:
        binary = sitk.Cast(sitk.Equal(mask, int(label_value)), sitk.sitkUInt8)
    return binary


def component_volumes(
    mask: sitk.Image,
    *,
    label_value: Optional[int] = None,
    connectivity: int = 26,
) -> List[ComponentVolume]:
    """
    Compute per-component volumes (mm^3 and mL) for a binary or labeled mask.

    Parameters
    ----------
    mask:
        SimpleITK image representing the segmentation mask.
    label_value:
        Specific label to isolate; if omitted the mask is treated as binary (>0).
    connectivity:
        Neighbor definition for connected components (6, 18, or 26).
    """
    binary = _as_binary_mask(mask, label_value)

    arr = sitk.GetArrayViewFromImage(binary)
    if not np.any(arr):
        return []

    fully_connected = connectivity == 26
    components = sitk.ConnectedComponent(binary, fully_connected)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(components)

    spacing = mask.GetSpacing()
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])

    volumes: List[ComponentVolume] = []
    for label in stats.GetLabels():
        voxels = int(stats.GetNumberOfPixels(label))
        volume_mm3 = voxels * voxel_volume
        volumes.append(
            ComponentVolume(
                label_id=int(label),
                voxel_count=voxels,
                volume_mm3=volume_mm3,
                volume_ml=volume_mm3 / 1000.0,
            )
        )
    return volumes


def tumor_volume_summary(
    mask: sitk.Image,
    *,
    label_value: Optional[int] = None,
    connectivity: int = 26,
) -> Tuple[List[ComponentVolume], float, float]:
    """
    Convenience helper returning per-component stats and aggregate volumes.

    Returns
    -------
    components:
        List of ComponentVolume entries for each connected tumour.
    total_mm3:
        Aggregate tumour volume in cubic millimetres.
    total_ml:
        Aggregate tumour volume in millilitres.
    """
    components = component_volumes(mask, label_value=label_value, connectivity=connectivity)
    total_mm3 = float(sum(comp.volume_mm3 for comp in components))
    total_ml = total_mm3 / 1000.0
    return components, total_mm3, total_ml
