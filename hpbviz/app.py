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
):
    """Run segmentation + meshing pipeline or load a precomputed mask."""
    img = load_dicom_series(input_path, series_uid=series_uid)  # DICOM dir, single DICOM file, or NIfTI

    if liver_mask_path:
        mask_img = sitk.Cast(sitk.ReadImage(liver_mask_path), sitk.sitkUInt8)
        mask_img = _resample_mask_to_image(mask_img, img)
        result = AutoLiverResult(mask=mask_img, phase_hint="unknown", used="precomputed")
    else:
        result = AutoLiver().run(img)
        mask_img = result.mask

    mask_np = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)  # z,y,x
    sx, sy, sz = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()
    surface = MeshBuilder().mask_to_surface(mask_np, spacing=(sx, sy, sz), origin=origin)

    if export_path:
        save_mesh(surface, export_path)

    if save_mask_path:
        sitk.WriteImage(mask_img, save_mask_path)
        print(f"[pipeline] saved liver mask to {save_mask_path}")

    return img, surface, result


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

    img, surface, result = run_pipeline(
        args.input,
        args.export,
        series_uid=args.series,
        liver_mask_path=args.liver_mask,
        save_mask_path=args.save_mask,
    )
    print(f"[pipeline] method={result.used}, phase_hint={result.phase_hint}")

    if not args.no_gui:
        from .viewer import HpbViewer

        HpbViewer(image=img).show_with_surface(surface)


if __name__ == "__main__":
    main()
