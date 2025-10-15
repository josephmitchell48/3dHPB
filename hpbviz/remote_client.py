from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'requests' package is required. Install with 'pip install requests'.") from exc

from .remote import prepare_case


def _post_file(url: str, file_path: Path, *, params: Optional[dict[str, str]] = None, timeout: int = 900) -> bytes:
    with file_path.open("rb") as fh:
        files = {"ct": (file_path.name, fh, "application/gzip")}
        response = requests.post(url, files=files, params=params or {}, timeout=timeout)
    if response.status_code != 200:
        detail = response.text.strip()
        raise RuntimeError(f"Request failed ({response.status_code}): {detail or 'no details'}")
    return response.content


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
) -> dict[str, str]:
    raw_dir = raw_root / case_name
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw case directory not found: {raw_dir}")

    # Derive case id (strip common suffixes like _DICOM)
    base = case_name
    for suffix in ("_dicom", "_DICOM"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    prep = prepare_case(
        dicom_path=str(raw_dir),
        output_dir=str(input_root),
        case_id=base,
        zip_result=False,
    )

    output_case_dir = output_root / base
    output_case_dir.mkdir(parents=True, exist_ok=True)

    # Copy meta for reference
    shutil.copy2(prep["meta_path"], output_case_dir / "meta.json")

    server = server.rstrip("/")
    params = {"fast": "true"} if fast else None

    # Liver request
    liver_bytes = _post_file(f"{server}{liver_endpoint}", Path(prep["totalseg_path"]), params=params)
    liver_out = output_case_dir / f"{base}_liver.nii.gz"
    liver_out.write_bytes(liver_bytes)

    task_path = None
    if include_task008:
        task_bytes = _post_file(f"{server}{task008_endpoint}", Path(prep["nnunet_path"]), params=None)
        task_path = output_case_dir / f"{base}_task008.nii.gz"
        task_path.write_bytes(task_bytes)

    return {
        "case": base,
        "liver_mask": str(liver_out),
        "task008_mask": str(task_path) if task_path else "",
        "meta": str(output_case_dir / "meta.json"),
    }


def iterate_cases(raw_root: Path, include: Optional[Iterable[str]] = None) -> list[str]:
    if include:
        return list(include)
    return sorted([p.name for p in raw_root.iterdir() if p.is_dir()])


def run_cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and upload cases to the remote AWS segmenter.")
    parser.add_argument("cases", nargs="*", help="Specific case directories under raw root. If omitted, process all.")
    parser.add_argument("--raw-root", default="data/raw", help="Root directory containing raw DICOM folders.")
    parser.add_argument("--input-root", default="data/awsInput", help="Where prepared cases should be stored.")
    parser.add_argument("--output-root", default="data/awsOutput", help="Where to write downloaded masks.")
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
            )
            summaries.append(info)
            print(json.dumps(info, indent=2), flush=True)
        except Exception as exc:  # pragma: no cover - top-level reporting
            print(f"[error] {case_name}: {exc}", file=sys.stderr, flush=True)

    if summaries:
        print("\nProcessed cases:")
        for info in summaries:
            print(f"  {info['case']} -> liver={info['liver_mask']} task008={info['task008_mask']}")


if __name__ == "__main__":
    run_cli()

