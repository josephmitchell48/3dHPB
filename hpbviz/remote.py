from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import SimpleITK as sitk

from .io import load_dicom_series


def prepare_case(
    dicom_path: str,
    output_dir: str,
    *,
    case_id: Optional[str] = None,
    series_uid: Optional[str] = None,
    zip_result: bool = False,
) -> dict[str, str]:
    """
    Convert a DICOM directory to the nnU-Net/TotalSegmentator friendly layout.

    Writes:
      - <case_id>/<case_id>.nii.gz      (for TotalSegmentator)
      - <case_id>/<case_id>_0000.nii.gz (for nnU-Net v1 Task008)
      - <case_id>/meta.json             (basic spacing/origin metadata)
    Optionally produces <case_id>.zip alongside the directory.
    """
    dicom_path = str(dicom_path)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image = load_dicom_series(dicom_path, series_uid=series_uid)

    if case_id is None:
        candidate = Path(dicom_path.rstrip("/")).name or "case"
        case_id = candidate
        if not case_id.strip():
            case_id = "case"
        if (output_root / case_id).exists():
            case_id = f"{case_id}_{uuid.uuid4().hex[:8]}"

    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    raw_path = case_dir / f"{case_id}.nii.gz"
    channel0_path = case_dir / f"{case_id}_0000.nii.gz"
    meta_path = case_dir / "meta.json"

    # Write images (preserve original type)
    sitk.WriteImage(image, str(raw_path))
    sitk.WriteImage(image, str(channel0_path))

    meta = {
        "case_id": case_id,
        "source": os.path.abspath(dicom_path),
        "spacing": list(map(float, image.GetSpacing())),
        "origin": list(map(float, image.GetOrigin())),
        "direction": list(map(float, image.GetDirection())),
        "size": list(map(int, image.GetSize())),
        "series_uid": series_uid,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    zip_path = None
    if zip_result:
        zip_path = shutil.make_archive(str(case_dir), "zip", root_dir=case_dir)

    result = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "totalseg_path": str(raw_path),
        "nnunet_path": str(channel0_path),
        "meta_path": str(meta_path),
    }
    if zip_path:
        result["zip_path"] = zip_path
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare a DICOM series for remote segmentation services.")
    p.add_argument("dicom", help="Path to DICOM directory.")
    p.add_argument("--output", "-o", default="prepared_cases", help="Output directory for prepared case.")
    p.add_argument("--series", help="Optional SeriesInstanceUID to select within the DICOM folder.")
    p.add_argument("--case-id", help="Explicit case identifier to use instead of deriving from the folder.")
    p.add_argument("--zip", action="store_true", help="Create a <case_id>.zip alongside the output directory.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    info = prepare_case(
        args.dicom,
        args.output,
        case_id=args.case_id,
        series_uid=args.series,
        zip_result=args.zip,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

