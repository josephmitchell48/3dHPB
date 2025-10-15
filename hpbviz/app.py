import argparse
import SimpleITK as sitk
import numpy as np

from .segment import AutoLiver, AutoLiverResult
from .mesh import MeshBuilder
from .io import load_dicom_series, save_mesh, list_dicom_series


def _resample_mask_to_image(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    same_geometry = (
        mask.GetSize() == reference.GetSize()
        and mask.GetSpacing() == reference.GetSpacing()
        and mask.GetOrigin() == reference.GetOrigin()
        and mask.GetDirection() == reference.GetDirection()
    )
    if same_geometry:
        return mask

    resampled = sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )
    resampled.CopyInformation(reference)
    return resampled


def run_pipeline(
    input_path: str,
    export_path: str | None = None,
    series_uid: str | None = None,
    liver_mask_path: str | None = None,
    save_mask_path: str | None = None,
    task008_mask_path: str | None = None,
):
    """Run segmentation + meshing pipeline or load precomputed masks."""
    img = load_dicom_series(input_path, series_uid=series_uid)  # DICOM dir, single DICOM file, or NIfTI

    if liver_mask_path:
        mask_img = sitk.Cast(sitk.ReadImage(liver_mask_path), sitk.sitkUInt8)
        mask_img = _resample_mask_to_image(mask_img, img)
        result = AutoLiverResult(mask=mask_img, phase_hint="unknown", used="precomputed")
    else:
        result = AutoLiver().run(img)
        mask_img = result.mask

    if save_mask_path:
        sitk.WriteImage(mask_img, save_mask_path)
        print(f"[pipeline] saved liver mask to {save_mask_path}")

    mesh_builder = MeshBuilder()
    surfaces: dict[str, dict[str, np.ndarray]] = {}

    mask_np = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)  # z,y,x
    sx, sy, sz = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()
    liver_surface = mesh_builder.mask_to_surface(mask_np, spacing=(sx, sy, sz), origin=origin)
    liver_surface["color"] = (0.1, 0.8, 0.2, 1.0)
    surfaces["liver"] = liver_surface

    if task008_mask_path:
        t8_img = sitk.Cast(sitk.ReadImage(task008_mask_path), sitk.sitkUInt8)
        t8_img = _resample_mask_to_image(t8_img, img)
        t8_np = sitk.GetArrayFromImage(t8_img).astype(np.uint8)
        t8_spacing = t8_img.GetSpacing()
        t8_origin = t8_img.GetOrigin()
        label_map = {
            1: ("hepatic_vessels", (0.9, 0.3, 0.3, 1.0)),
            2: ("liver_tumors", (0.8, 0.6, 0.1, 1.0)),
        }
        for label, (name, color) in label_map.items():
            if np.any(t8_np == label):
                surface = mesh_builder.mask_to_surface(
                    t8_np,
                    spacing=tuple(t8_spacing),
                    origin=tuple(t8_origin),
                    label=label,
                )
                surface["color"] = color
                surfaces[name] = surface

    if export_path and "liver" in surfaces:
        save_mesh(surfaces["liver"], export_path)

    return img, surfaces, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to DICOM folder, single DICOM file, or NIfTI (.nii/.nii.gz)")
    p.add_argument("--export", help="Mesh output path (e.g., liver.obj)", default=None)
    p.add_argument("--no-gui", action="store_true", help="Do not open viewer")
    p.add_argument("--series", help="SeriesInstanceUID within a DICOM folder", default=None)
    p.add_argument(
        "--liver-mask",
        help="Path to precomputed liver mask (NIfTI). Skips TotalSegmentator.",
        default=None,
    )
    p.add_argument(
        "--save-mask",
        help="Optional path to save the liver mask (NIfTI). Useful for dev mode.",
        default=None,
    )
    p.add_argument(
        "--task008-mask",
        help="Path to nnU-Net v1 Task008 output (NIfTI).",
        default=None,
    )
    p.add_argument("--list-series", action="store_true", help="List DICOM series in the given folder and exit")
    args = p.parse_args()

    if args.list_series:
        series = list_dicom_series(args.input)
        if not series:
            print("No DICOM series found.")
            return
        print("Found DICOM series:")
        for s in series:
            desc = f" — {s['description']}" if s['description'] else ""
            print(f"  UID={s['uid']}  ({s['count']} files){desc}")
        return

    img, surfaces, result = run_pipeline(
        args.input,
        args.export,
        series_uid=args.series,
        liver_mask_path=args.liver_mask,
        save_mask_path=args.save_mask,
        task008_mask_path=args.task008_mask,
    )
    print(f"[pipeline] method={result.used}, phase_hint={result.phase_hint}")

    if not args.no_gui:
        from .viewer import HpbViewer

        HpbViewer(image=img).show_with_surfaces(surfaces)


if __name__ == "__main__":
    main()
