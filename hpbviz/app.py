import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import SimpleITK as sitk
import numpy as np

from .mesh import MeshBuilder
from .io import load_dicom_series, save_mesh, list_dicom_series

log = logging.getLogger("hpbviz.pipeline")
if not log.handlers:
    # Minimal default handler; feel free to configure elsewhere.
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.setLevel(logging.INFO)

RGBA = Tuple[float, float, float, float]

# ---- Label/color config ------------------------------------------------------

TASK08_LABELS: Dict[int, Tuple[str, RGBA]] = {
    1: ("hepatic_vessels_OLD", (0.0, 0.0, 1.0, 1.0)),
    2: ("liver_tumors",    (0.0, 1.0, 0.0, 1.0)),
}

VSNET_LABELS: Dict[int, Tuple[str, RGBA]] = {
    1: ("hepatic_vein", (0.25, 0.40, 0.95, 0.85)),
    2: ("portal_vein",   (0.10, 0.75, 0.85, 0.80)),
}

# ---- Core utilities ----------------------------------------------------------

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


def _read_mask_like(ref_img: sitk.Image, path: str) -> sitk.Image:
    """
    Read a mask file, cast to uint8, resample onto ref_img grid, canonicalize once.
    """
    m = sitk.Cast(sitk.ReadImage(path), sitk.sitkUInt8)
    m = _resample_mask_to_image(m, ref_img)
    return _canonicalize_image(m)


def _build_surface(
    mesh_builder: MeshBuilder,
    mask_img: sitk.Image,
    name: str,
    color: RGBA,
    label: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Centralized meshing with correct axis/spacing handling and guardrails.
    Returns a dict compatible with your viewer: {'vertices','faces','color', ...}
    """
    arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)  # (z, y, x)
    if label is None:
        mask_arr = (arr > 0).astype(np.uint8)
    else:
        mask_arr = (arr == label).astype(np.uint8)
    if not np.any(mask_arr):
        log.info("[mesh] '%s' is empty (label=%s)", name, label)
        return None

    spacing = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()
    try:
        mesh = mesh_builder.mask_to_surface(
            mask_arr,
            spacing=spacing,
            origin=origin,
            label=None,
        )
    except ValueError as e:
        log.warning("[mesh] '%s' marching cubes failed: %s", name, e)
        return None

    v = mesh.get("vertices", np.array([]))
    f = mesh.get("faces", np.array([]))
    if v.size == 0 or f.size == 0:
        log.info("[mesh] '%s' produced empty geometry", name)
        return None

    mesh["color"] = color
    mesh.setdefault("color", color)
    mesh.setdefault("display_name", name)
    return mesh

def _add_labeled_surfaces(
    *,
    mesh_builder: MeshBuilder,
    ref_img: sitk.Image,
    mask_path: Optional[str],
    label_map: Dict[int, Tuple[str, RGBA]],
    surfaces: Dict[str, Dict[str, np.ndarray]],
    log_tag: str,
    set_display_name: bool = False,
) -> None:
    """Common routine to read a multi-label mask, log present labels, and add surfaces."""
    if not mask_path:
        return
    m_img = _read_mask_like(ref_img, mask_path)
    arr = sitk.GetArrayFromImage(m_img)
    present = sorted(int(v) for v in np.unique(arr) if int(v) in label_map)
    log.info("[pipeline] %s labels present: %s", log_tag, present)

    for label, (name, color) in label_map.items():
        s = _build_surface(mesh_builder, m_img, name=name, color=color, label=label)
        if s:
            if set_display_name:
                s["display_name"] = name.replace("_", " ").title()
            surfaces[name] = s
# ---- Case discovery helpers --------------------------------------------------

def _case_name_from_path(path: str) -> str:
    name = Path(path.rstrip("/"))
    if not name.name:
        return "case"
    base = name.name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
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

    def _register_case(
        case: str,
        *,
        volume: Optional[Path] = None,
        liver: Optional[Path] = None,
        task: Optional[Path] = None,
        manual: Optional[Path] = None,
    ) -> None:
        entry = cases.setdefault(case, {})
        if volume and "dicom_path" not in entry:
            entry["dicom_path"] = str(volume.resolve())
        if liver:
            entry["liver_mask"] = str(liver.resolve())
        if task:
            entry["task008_mask"] = str(task.resolve())
        if manual:
            entry["manual_mask"] = str(manual.resolve())

    def _pick_volume_candidate(files: List[Path]) -> Optional[Path]:
        mask_tokens = ("mask", "liver", "task008", "hepatic", "tumor", "vsnet", "seg", "label", "manual")
        for candidate in files:
            name = candidate.name.lower()
            if not any(token in name for token in mask_tokens):
                return candidate
        return files[0] if files else None

    if output_root_path.is_dir():
        for out_entry in output_root_path.iterdir():
            if out_entry.is_dir():
                nii_files = sorted(out_entry.glob("*.nii*"))
                liver_mask = None
                task008_mask = None
                manual_mask = None
                for candidate in nii_files:
                    name_lower = candidate.name.lower()
                    if "liver" in name_lower and liver_mask is None:
                        liver_mask = candidate
                    elif (
                        "task008" in name_lower or "tumor" in name_lower
                    ) and task008_mask is None:
                        task008_mask = candidate
                    elif (
                        "vsnet" in name_lower or "inputvsnet" in name_lower or "manual" in name_lower
                    ) and manual_mask is None:
                        manual_mask = candidate
                base = _pick_volume_candidate(nii_files)
                case = _case_name_from_path(out_entry.name)
                _register_case(case, volume=base, liver=liver_mask, task=task008_mask, manual=manual_mask)
            elif out_entry.suffix in {".nii", ".gz"}:
                case = _case_name_from_path(out_entry.name)
                _register_case(case, volume=out_entry)

    if raw_root_path.is_dir():
        for raw_entry in raw_root_path.iterdir():
            if raw_entry.is_dir():
                nii_files = sorted(raw_entry.glob("*.nii*"))
                chosen = _pick_volume_candidate(nii_files)
                case = _case_name_from_path(raw_entry.name)
                target = chosen or raw_entry
                _register_case(case, volume=target)
            elif raw_entry.suffix in {".nii", ".gz"}:
                case = _case_name_from_path(raw_entry.name)
                _register_case(case, volume=raw_entry)

    return {case: info for case, info in cases.items() if info.get("dicom_path")}

# ---- Pipeline ---------------------------------------------------------------

def run_pipeline(
    input_path: str,
    export_path: str | None = None,
    series_uid: str | None = None,
    liver_mask_path: str | None = None,
    save_mask_path: str | None = None,
    task008_mask_path: str | None = None,
    manual_mask_path: str | None = None,
):
    """
    Run meshing pipeline with provided masks (no local segmentation).
    Returns: (img, surfaces, mask_img) for viewer compatibility.
    """
    # 1) Load & canonicalize reference image once
    img = _canonicalize_image(load_dicom_series(input_path, series_uid=series_uid))

    # 2) Liver mask is required (no AutoLiver fallback)
    if not liver_mask_path:
        raise RuntimeError("Liver mask is required. No local segmentation fallback is available.")
    mask_img = _read_mask_like(img, liver_mask_path)

    if save_mask_path:
        sitk.WriteImage(mask_img, save_mask_path)
        log.info("[pipeline] saved liver mask to %s", save_mask_path)

    # 3) Build surfaces via shared helper
    mesh_builder = MeshBuilder()
    surfaces: dict[str, dict[str, np.ndarray]] = {}

    # Liver
    liver_surface = _build_surface(mesh_builder, mask_img, name="liver", color=(1.0, 0.0, 0.0, 1.0))
    if liver_surface:
        surfaces["liver"] = liver_surface
    else:
        log.info("[pipeline] liver surface empty")

    # Task08 (optional)
    _add_labeled_surfaces(
        mesh_builder=mesh_builder,
        ref_img=img,
        mask_path=task008_mask_path,
        label_map=TASK08_LABELS,
        surfaces=surfaces,
        log_tag="Task08",
        set_display_name=True,  # keep previous behavior
    )

    # VSNet/manual (optional)
    _add_labeled_surfaces(
        mesh_builder=mesh_builder,
        ref_img=img,
        mask_path=manual_mask_path,
        label_map=VSNET_LABELS,
        surfaces=surfaces,
        log_tag="VSNet",
        set_display_name=True,   # match your previous manual block
    )

    # 4) Optional export (only if liver is present)
    if export_path and "liver" in surfaces:
        save_mesh(
            {"vertices": surfaces["liver"]["vertices"], "faces": surfaces["liver"]["faces"], "color": surfaces["liver"]["color"]},
            export_path,
        )
        log.info("[pipeline] exported liver mesh to %s", export_path)

    # Maintain API compatibility: third return value is the mask image
    return img, surfaces, mask_img

# ---- CLI / Viewer -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default=None, help="Path to DICOM folder, single DICOM file, or NIfTI (.nii/.nii.gz)")
    p.add_argument("--export", help="Mesh output path (e.g., liver.obj)", default=None)
    p.add_argument("--no-gui", action="store_true", help="Do not open viewer")
    p.add_argument("--no-controls", action="store_true", help="Hide layer controls in the viewer")
    p.add_argument("--no-browser", action="store_true", help="Disable patient browser")
    p.add_argument("--series", help="SeriesInstanceUID within a DICOM folder", default=None)
    p.add_argument(
        "--liver-mask",
        help="Path to precomputed liver mask (NIfTI). Required (no local segmentation).",
        default=None,
    )
    p.add_argument(
        "--save-mask",
        help="Optional path to save the liver mask (NIfTI). Useful for dev mode.",
        default=None,
    )
    p.add_argument(
        "--task008-mask",
        help="Path to nnU-Net Task008 output (NIfTI).",
        default=None,
    )
    p.add_argument(
        "--manual-mask",
        help="Path to provided/manual mask (NIfTI).",
        default=None,
    )
    p.add_argument(
        "--raw-root",
        default="data/awsInput",
        help="Fallback directory containing input volumes (folders or NIfTI files)",
    )
    p.add_argument(
        "--output-root",
        default="data/awsOutput",
        help="Primary directory containing processed cases and masks for browsing",
    )
    p.add_argument("--list-series", action="store_true", help="List DICOM series in the given folder and exit")
    args = p.parse_args()

    case_catalog = _discover_cases(args.raw_root, args.output_root)

    def _resolve(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return str(Path(path).expanduser().resolve())

    if args.input:
        input_path = Path(args.input).expanduser()
        initial_source = str(input_path.resolve())
        initial_case = _case_name_from_path(input_path.name)
    else:
        if not case_catalog:
            print("No cases discovered in", args.raw_root)
            return
        initial_case = sorted(case_catalog.keys())[0]
        initial_source = case_catalog[initial_case]["dicom_path"]
        args.input = initial_source

    entry = case_catalog.setdefault(initial_case, {})
    entry["dicom_path"] = _resolve(initial_source)

    initial_liver = _resolve(args.liver_mask) if args.liver_mask else _resolve(entry.get("liver_mask"))
    if initial_liver:
        entry["liver_mask"] = initial_liver

    initial_task = _resolve(args.task008_mask) if args.task008_mask else _resolve(entry.get("task008_mask"))
    if initial_task:
        entry["task008_mask"] = initial_task

    initial_manual = _resolve(args.manual_mask) if args.manual_mask else _resolve(entry.get("manual_mask"))
    if initial_manual:
        entry["manual_mask"] = initial_manual

    if args.list_series:
        if args.input:
            try:
                series = list_dicom_series(args.input)
            except Exception:
                series = []
        else:
            series = []
        if series:
            print("Found DICOM series:")
            for s in series:
                desc = f" — {s['description']}" if s['description'] else ""
                print(f"  UID={s['uid']}  ({s['count']} files){desc}")
        else:
            print("Available cases:")
            for name in sorted(case_catalog.keys()):
                print("  -", name)
        return

    img, surfaces, result = run_pipeline(
        entry["dicom_path"],
        args.export,
        series_uid=args.series,
        liver_mask_path=initial_liver,
        save_mask_path=args.save_mask,
        task008_mask_path=initial_task,
        manual_mask_path=initial_manual,
    )
    print("[pipeline] using provided masks")

    if not args.no_gui:
        from .viewer import HpbViewer
        initial_case = initial_case
        case_cache: dict[str, tuple[sitk.Image, dict[str, dict[str, np.ndarray]], sitk.Image]] = {
            initial_case: (img, surfaces, result)
        }
        catalog_for_viewer = case_catalog if not args.no_browser else None
        case_loader = None

        if catalog_for_viewer:

            @lru_cache(maxsize=None)
            def _load_case(name: str):
                if name == initial_case:
                    return case_cache[initial_case]
                info = catalog_for_viewer.get(name)
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
            case_catalog=catalog_for_viewer,
            case_loader=case_loader,
            current_case=initial_case,
        )
        viewer.show_with_surfaces(surfaces)


if __name__ == "__main__":
    main()
