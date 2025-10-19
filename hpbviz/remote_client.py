from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

import SimpleITK as sitk

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'requests' package is required. Install with 'pip install requests'.") from exc

from .remote import prepare_case


def _strip_case_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _is_mask_filename(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("mask", "label", "seg", "manual"))


def _post_file(url: str, file_path: Path, *, params: Optional[dict[str, str]] = None, timeout: int = 900) -> bytes:
    with file_path.open("rb") as fh:
        files = {"ct": (file_path.name, fh, "application/gzip")}
        response = requests.post(url, files=files, params=params or {}, timeout=timeout)
    if response.status_code != 200:
        detail = response.text.strip()
        raise RuntimeError(f"Request failed ({response.status_code}): {detail or 'no details'}")
    return response.content


def _write_meta(image: sitk.Image, case_dir: Path, case_id: str, source_path: Path) -> Path:
    meta = {
        "case_id": case_id,
        "source": str(source_path.resolve()),
        "spacing": list(map(float, image.GetSpacing())),
        "origin": list(map(float, image.GetOrigin())),
        "direction": list(map(float, image.GetDirection())),
        "size": list(map(int, image.GetSize())),
    }
    meta_path = case_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def _stage_nifti_case(
    ct_src: Path,
    case_dir: Path,
    case_id: str,
    manual_src: Optional[Path] = None,
) -> dict[str, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".nii.gz" if ct_src.name.endswith(".nii.gz") else ".nii"
    ct_dest = case_dir / f"{case_id}{suffix}"
    if not ct_dest.exists() or ct_dest.resolve() != ct_src.resolve():
        shutil.copy2(ct_src, ct_dest)

    nnunet_dest = case_dir / f"{case_id}_0000{suffix}"
    if not nnunet_dest.exists():
        shutil.copy2(ct_dest, nnunet_dest)

    image = sitk.ReadImage(str(ct_dest))
    meta_path = _write_meta(image, case_dir, case_id, ct_src)

    manual_dest = None
    if manual_src and manual_src.exists():
        manual_dest = case_dir / manual_src.name
        if not manual_dest.exists() or manual_dest.resolve() != manual_src.resolve():
            shutil.copy2(manual_src, manual_dest)

    return {
        "case_dir": case_dir,
        "totalseg_path": ct_dest,
        "nnunet_path": nnunet_dest,
        "meta_path": meta_path,
        "manual_path": manual_dest,
    }


def _maybe_find_manual_mask(case_id: str, raw_entry: Path, mask_root: Optional[Path]) -> Optional[Path]:
    candidates = []
    case_lower = case_id.lower()

    def consider(path: Optional[Path]) -> None:
        if path and path.exists() and _is_mask_filename(path.name):
            candidates.append(path)

    search_dirs = []
    if raw_entry.is_dir():
        search_dirs.append(raw_entry)
    else:
        search_dirs.append(raw_entry.parent)

    if mask_root:
        mask_root = mask_root.expanduser().resolve()
        case_dir = mask_root / case_id
        if case_dir.is_dir():
            search_dirs.append(case_dir)
        else:
            for ext in (".nii.gz", ".nii"):
                consider(mask_root / f"{case_id}{ext}")

    for directory in search_dirs:
        if not directory or not directory.exists():
            continue
        if directory.is_file():
            consider(directory)
            continue
        for path in directory.glob("*.nii*"):
            name_lower = path.name.lower()
            if not _is_mask_filename(name_lower):
                continue
            if case_lower in name_lower or directory == raw_entry:
                consider(path)

    if not candidates:
        return None

    candidates.sort(key=lambda p: (case_lower not in p.name.lower(), len(p.name)))
    return candidates[0]


def process_case(
    case_name: str,
    *,
    raw_root: Path,
    input_root: Path,
    output_root: Path,
    server: str,
    liver_endpoint: str = "/segment/liver",
    task008_endpoint: str = "/segment/task008",
    include_task008: bool = True,
    fast: bool = False,
    mask_root: Optional[Path] = None,
) -> dict[str, str]:
    raw_entry = (raw_root / case_name).resolve()
    mask_root = mask_root.resolve() if mask_root else None

    if not raw_entry.exists():
        raise FileNotFoundError(f"Case source not found: {raw_entry}")

    stage_info = {}
    manual_case_path = None

    if raw_entry.is_dir():
        nii_files = sorted(p for p in raw_entry.glob("*.nii*") if p.is_file())
        if nii_files:
            ct_src = next((p for p in nii_files if not _is_mask_filename(p.name)), nii_files[0])
            case_id = _strip_case_name(ct_src.name)
            manual_src = next((p for p in nii_files if p != ct_src and _is_mask_filename(p.name)), None)
            manual_src = manual_src or _maybe_find_manual_mask(case_id, raw_entry, mask_root)
            stage_info = _stage_nifti_case(ct_src, input_root / case_id, case_id, manual_src)
            manual_case_path = stage_info.get("manual_path")
        else:
            case_id = _strip_case_name(raw_entry.name)
            prep = prepare_case(
                dicom_path=str(raw_entry),
                output_dir=str(input_root),
                case_id=case_id,
                zip_result=False,
            )
            stage_info = {
                "case_dir": Path(prep["case_dir"]),
                "totalseg_path": Path(prep["totalseg_path"]),
                "nnunet_path": Path(prep["nnunet_path"]),
                "meta_path": Path(prep["meta_path"]),
            }
            manual_src = _maybe_find_manual_mask(prep["case_id"], raw_entry, mask_root)
            if manual_src:
                manual_case_path = stage_info["case_dir"] / manual_src.name
                if not manual_case_path.exists() or manual_case_path.resolve() != manual_src.resolve():
                    shutil.copy2(manual_src, manual_case_path)
            case_id = prep["case_id"]
            stage_info["manual_path"] = manual_case_path
    else:
        case_id = _strip_case_name(raw_entry.name)
        manual_src = _maybe_find_manual_mask(case_id, raw_entry, mask_root)
        stage_info = _stage_nifti_case(raw_entry, input_root / case_id, case_id, manual_src)
        manual_case_path = stage_info.get("manual_path")

    case_id = _strip_case_name(stage_info.get("case_dir", Path(case_name)).name)
    totalseg_path = stage_info["totalseg_path"]
    nnunet_path = stage_info["nnunet_path"]
    meta_path = stage_info["meta_path"]

    output_case_dir = output_root / case_id
    output_case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_path, output_case_dir / "meta.json")

    server = server.rstrip("/")
    params = {"fast": "true"} if fast else None

    liver_bytes = _post_file(f"{server}{liver_endpoint}", totalseg_path, params=params)
    liver_out = output_case_dir / f"{case_id}_liver.nii.gz"
    liver_out.write_bytes(liver_bytes)

    task_path = None
    if include_task008:
        task_bytes = _post_file(f"{server}{task008_endpoint}", nnunet_path, params=None)
        task_path = output_case_dir / f"{case_id}_task008.nii.gz"
        task_path.write_bytes(task_bytes)

    provided_path = None
    if manual_case_path and manual_case_path.exists():
        provided_path = output_case_dir / f"{case_id}_input_mask{manual_case_path.suffix}"
        if not provided_path.exists() or provided_path.resolve() != manual_case_path.resolve():
            shutil.copy2(manual_case_path, provided_path)

    return {
        "case": case_id,
        "liver_mask": str(liver_out),
        "task008_mask": str(task_path) if task_path else "",
        "provided_mask": str(provided_path) if provided_path else "",
        "meta": str(output_case_dir / "meta.json"),
    }


def iterate_cases(raw_root: Path, include: Optional[Iterable[str]] = None) -> list[str]:
    if include:
        return list(include)
    cases = []
    for path in raw_root.iterdir():
        if path.is_dir() or path.suffix in (".nii", ".gz"):
            cases.append(path.name)
    return sorted(cases)


def run_cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and upload cases to the remote AWS segmenter.")
    parser.add_argument("cases", nargs="*", help="Specific case directories/files under raw root. If omitted, process all.")
    parser.add_argument("--raw-root", default="data/raw", help="Root directory containing source data (DICOM folders or NIfTI files).")
    parser.add_argument("--input-root", default="data/awsInput", help="Where prepared cases should be staged.")
    parser.add_argument("--output-root", default="data/awsOutput", help="Where to write downloaded masks.")
    parser.add_argument("--mask-root", help="Optional directory containing provided/manual masks.")
    parser.add_argument(
        "--server",
        default="http://23.20.247.26:8080",
        help="Base URL of the segmentation server (no trailing slash).",
    )
    parser.add_argument("--no-task008", action="store_true", help="Skip nnU-Net Task008 upload.")
    parser.add_argument("--fast", action="store_true", help="Pass ?fast=true to the server for quicker inference.")
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root).expanduser().resolve()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve() if args.mask_root else None

    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    case_names = iterate_cases(raw_root, args.cases)
    if not case_names:
        print("No cases found.")
        return

    summaries = []
    for case_name in case_names:
        try:
            info = process_case(
                case_name,
                raw_root=raw_root,
                input_root=input_root,
                output_root=output_root,
                server=args.server,
                include_task008=not args.no_task008,
                fast=args.fast,
                mask_root=mask_root,
            )
            summaries.append(info)
            print(json.dumps(info, indent=2), flush=True)
        except Exception as exc:  # pragma: no cover - top-level reporting
            print(f"[error] {case_name}: {exc}", file=sys.stderr, flush=True)

    if summaries:
        print("\nProcessed cases:")
        for info in summaries:
            print(
                f"  {info['case']} -> liver={info['liver_mask']}"
                f" task008={info['task008_mask']} provided={info['provided_mask']}"
            )


if __name__ == "__main__":
    run_cli()

