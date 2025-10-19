import argparse
from functools import lru_cache
from pathlib import Path
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


def _canonicalize_image(image: sitk.Image) -> sitk.Image:
    """
    Orient the image to LPS with identity direction and zero origin for consistent visualization.
    """
    oriented = sitk.DICOMOrient(image, "LPS")
    oriented.SetOrigin((0.0, 0.0, 0.0))
    return oriented


def _case_name_from_path(path: str) -> str:
    name = Path(path.rstrip("/"))
    if not name.name:
        return "case"
    base = name.name
    lowered = base.lower()
    for suffix in ("_dicom", "_dicoms"):
        if lowered.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base or "case"


def _discover_cases(raw_root: str, output_root: str) -> dict[str, dict[str, str]]:
    cases: dict[str, dict[str, str]] = {}
    raw_root_path = Path(raw_root).expanduser()
    output_root_path = Path(output_root).expanduser()

    if raw_root_path.is_dir():
        for raw_entry in raw_root_path.iterdir():
            if raw_entry.is_dir():
                case = _case_name_from_path(raw_entry.name)
                cases.setdefault(case, {})["dicom_path"] = str(raw_entry.resolve())
            elif raw_entry.suffix in {".nii", ".gz"}:
                case = _case_name_from_path(raw_entry.name)
                cases.setdefault(case, {})["dicom_path"] = str(raw_entry.resolve())

    if output_root_path.is_dir():
        for out_dir in output_root_path.iterdir():
            if not out_dir.is_dir():
                continue
            case = out_dir.name
            liver_mask = None
            task008_mask = None
            manual_mask = None
            for candidate in out_dir.glob("*.nii*"):
                name_lower = candidate.name.lower()
                if "liver" in name_lower and liver_mask is None:
                    liver_mask = str(candidate.resolve())
                elif ("task008" in name_lower or "hepatic" in name_lower or "tumor" in name_lower) and task008_mask is None:
                    task008_mask = str(candidate.resolve())
                elif ("input_mask" in name_lower or "manual" in name_lower) and manual_mask is None:
                    manual_mask = str(candidate.resolve())
            entry = cases.setdefault(case, {})
            if liver_mask:
                entry["liver_mask"] = liver_mask
            if task008_mask:
                entry["task008_mask"] = task008_mask
            if manual_mask:
                entry["manual_mask"] = manual_mask

    return {case: info for case, info in cases.items() if info.get("dicom_path") and info.get("liver_mask")}


def run_pipeline(
    input_path: str,
    export_path: str | None = None,
    series_uid: str | None = None,
    liver_mask_path: str | None = None,
    save_mask_path: str | None = None,
    task008_mask_path: str | None = None,
    manual_mask_path: str | None = None,
):
    """Run segmentation + meshing pipeline or load precomputed masks."""
    img = load_dicom_series(input_path, series_uid=series_uid)  # DICOM dir, single DICOM file, or NIfTI
    img = _canonicalize_image(img)

    if liver_mask_path:
        mask_img = sitk.Cast(sitk.ReadImage(liver_mask_path), sitk.sitkUInt8)
        mask_img = _resample_mask_to_image(mask_img, img)
        result = AutoLiverResult(mask=mask_img, phase_hint="unknown", used="precomputed")
    else:
        result = AutoLiver().run(img)
        mask_img = result.mask

    mask_img = _canonicalize_image(mask_img)

    if save_mask_path:
        sitk.WriteImage(mask_img, save_mask_path)
        print(f"[pipeline] saved liver mask to {save_mask_path}")

    mesh_builder = MeshBuilder()
    surfaces: dict[str, dict[str, np.ndarray]] = {}

    mask_img = _canonicalize_image(mask_img)
    mask_np = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)  # z,y,x
    sx, sy, sz = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()
    liver_surface = mesh_builder.mask_to_surface(mask_np, spacing=(sx, sy, sz), origin=origin)
    liver_surface["color"] = (1, 0.0, 0.0, 1.0)
    surfaces["liver"] = liver_surface

    if task008_mask_path:
        t8_img = sitk.Cast(sitk.ReadImage(task008_mask_path), sitk.sitkUInt8)
        t8_img = _resample_mask_to_image(t8_img, img)
        t8_img = _canonicalize_image(t8_img)
        t8_np = sitk.GetArrayFromImage(t8_img).astype(np.uint8)
        t8_spacing = t8_img.GetSpacing()
        t8_origin = t8_img.GetOrigin()
        label_map = {
            1: ("hepatic_vessels", (0.0, 0.0, 1.0, 1.0)),
            2: ("liver_tumors", (0.0, 1.0, 0.0, 1.0)),
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

    if manual_mask_path:
        manual_img = sitk.Cast(sitk.ReadImage(manual_mask_path), sitk.sitkUInt8)
        manual_img = _resample_mask_to_image(manual_img, img)
        manual_img = _canonicalize_image(manual_img)
        manual_np = sitk.GetArrayFromImage(manual_img).astype(np.int16)

        label_palette = {
            1: ("Provided Mask 1", (0.25, 0.40, 0.95, 0.85)),
            2: ("Provided Mask 2", (0.10, 0.75, 0.85, 0.80)),
            3: ("Provided Mask 3", (0.85, 0.45, 0.20, 0.80)),
        }
        fallback_colors = [
            (0.70, 0.30, 0.90, 0.80),
            (0.35, 0.85, 0.60, 0.80),
            (0.90, 0.60, 0.25, 0.80),
            (0.55, 0.25, 0.95, 0.80),
        ]

        labels = sorted(label for label in np.unique(manual_np) if label > 0)
        for idx, label in enumerate(labels):
            surface = mesh_builder.mask_to_surface(
                manual_np,
                spacing=tuple(manual_img.GetSpacing()),
                origin=tuple(manual_img.GetOrigin()),
                label=label,
            )
            if not surface["vertices"].size:
                continue

            name, color = label_palette.get(
                label,
                (f"Provided Mask {label}", fallback_colors[idx % len(fallback_colors)]),
            )
            surface["display_name"] = name
            surface["color"] = color
            surfaces[f"manual_mask_{label}"] = surface

    if export_path and "liver" in surfaces:
        save_mesh(surfaces["liver"], export_path)

    return img, surfaces, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to DICOM folder, single DICOM file, or NIfTI (.nii/.nii.gz)")
    p.add_argument("--export", help="Mesh output path (e.g., liver.obj)", default=None)
    p.add_argument("--no-gui", action="store_true", help="Do not open viewer")
    p.add_argument("--no-controls", action="store_true", help="Hide layer controls in the viewer")
    p.add_argument("--no-browser", action="store_true", help="Disable patient browser")
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
    p.add_argument(
        "--manual-mask",
        help="Path to provided/manual mask (NIfTI).",
        default=None,
    )
    p.add_argument(
        "--raw-root",
        default="data/raw",
        help="Root directory containing raw DICOM folders for case browsing",
    )
    p.add_argument(
        "--output-root",
        default="data/awsOutput",
        help="Root directory containing segmentation outputs for case browsing",
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
        manual_mask_path=args.manual_mask,
    )
    print(f"[pipeline] method={result.used}, phase_hint={result.phase_hint}")

    if not args.no_gui:
        from .viewer import HpbViewer
        initial_case = _case_name_from_path(args.input)
        case_cache: dict[str, tuple[sitk.Image, dict[str, dict[str, np.ndarray]], AutoLiverResult]] = {
            initial_case: (img, surfaces, result)
        }

        case_catalog = {}
        case_loader = None

        if not args.no_browser:
            case_catalog = _discover_cases(args.raw_root, args.output_root)
            entry = case_catalog.setdefault(initial_case, {})
            entry["dicom_path"] = str(Path(args.input).resolve())
            if args.liver_mask:
                entry["liver_mask"] = str(Path(args.liver_mask).resolve())
            if args.task008_mask:
                entry["task008_mask"] = str(Path(args.task008_mask).resolve())
            if args.manual_mask:
                entry["manual_mask"] = str(Path(args.manual_mask).resolve())

            case_catalog = {k: v for k, v in case_catalog.items() if v.get("liver_mask")}

            @lru_cache(maxsize=None)
            def _load_case(name: str):
                if name == initial_case:
                    return case_cache[initial_case]
                info = case_catalog.get(name)
                if not info:
                    raise RuntimeError(f"No case data available for {name}")
                return run_pipeline(
                    info["dicom_path"],
                    export_path=None,
                    series_uid=None,
                    liver_mask_path=info.get("liver_mask"),
                    save_mask_path=None,
                    task008_mask_path=info.get("task008_mask"),
                    manual_mask_path=info.get("manual_mask"),
                )

        case_loader = _load_case

        viewer = HpbViewer(
            image=img,
            show_controls=not args.no_controls,
            case_catalog=case_catalog if not args.no_browser else None,
            case_loader=case_loader,
            current_case=initial_case,
        )
        viewer.show_with_surfaces(surfaces)


if __name__ == "__main__":
    main()
