# segment.py
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable

import SimpleITK as sitk


@dataclass
class AutoLiverResult:
    mask: sitk.Image          # uint8 (0/1)
    phase_hint: str           # "noncontrast" | "arterial/venous/unknown"
    used: str                 # "totalsegmentator"


class AutoLiver:
    """
    One-call liver segmentation wrapper around TotalSegmentator.
    Requires the TotalSegmentator CLI to be available on PATH.
    """
    def __init__(self) -> None:
        pass

    def run(self, image: sitk.Image) -> AutoLiverResult:
        if not self._totalseg_available():
            raise RuntimeError(
                "TotalSegmentator executable not found on PATH. "
                "Install it in the active environment or adjust PATH."
            )
        try:
            mask = self._run_totalseg_liver(image)
        except Exception as e:
            raise RuntimeError(f"TotalSegmentator inference failed: {e}") from e
        return AutoLiverResult(
            mask=sitk.Cast(mask > 0, sitk.sitkUInt8),
            phase_hint="unknown",
            used="totalsegmentator",
        )

    def _totalseg_available(self) -> bool:
        return shutil.which("TotalSegmentator") is not None

    def _run_totalseg(self, cmd: Iterable[str]) -> subprocess.CompletedProcess:
        """
        Run TotalSegmentator with helpful error reporting.
        """
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                list(cmd),
                output=proc.stdout,
                stderr=proc.stderr,
            )
        return proc

    def _is_task_argument_error(self, err: subprocess.CalledProcessError) -> bool:
        msg = f"{err.output}\n{err.stderr}".lower()
        return any(
            token in msg
            for token in [
                "invalid choice",
                "unrecognized arguments",
                "--task",
                "roi_subset",
            ]
        )

    def _run_totalseg_liver(self, image: sitk.Image) -> sitk.Image:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "ct.nii.gz")
            outd = os.path.join(td, "out")
            os.makedirs(outd, exist_ok=True)
            sitk.WriteImage(image, src)
            preferred_cmd = [
                "TotalSegmentator",
                "-i",
                src,
                "-o",
                outd,
                "--task",
                "total",
                "--roi_subset",
                "liver",
            ]
            try:
                self._run_totalseg(preferred_cmd)
            except subprocess.CalledProcessError as err:
                if self._is_task_argument_error(err):
                    fallback_cmd = [
                        "TotalSegmentator",
                        "-i",
                        src,
                        "-o",
                        outd,
                        "--task",
                        "organ",
                    ]
                    try:
                        self._run_totalseg(fallback_cmd)
                    except subprocess.CalledProcessError as fallback_err:
                        raise RuntimeError(
                            self._format_totalseg_error(fallback_err)
                        ) from fallback_err
                else:
                    raise RuntimeError(self._format_totalseg_error(err)) from err
            liver_path = os.path.join(outd, "liver.nii.gz")
            if not os.path.exists(liver_path):
                raise FileNotFoundError("TotalSegmentator output missing liver.nii.gz")
            return sitk.Cast(sitk.ReadImage(liver_path), sitk.sitkUInt8)

    def _format_totalseg_error(self, err: subprocess.CalledProcessError) -> str:
        parts = [
            "TotalSegmentator failed",
            f"command: {' '.join(err.cmd)}",
            f"exit code: {err.returncode}",
        ]
        if err.output:
            parts.append(f"stdout:\n{err.output.strip()}")
        if err.stderr:
            parts.append(f"stderr:\n{err.stderr.strip()}")
        return "\n".join(parts)
