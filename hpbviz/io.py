from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List, TextIO

import sys

import os
import SimpleITK as sitk
import numpy as np

_DICOM_INFO_TAGS = {
    "0008|103e": "SeriesDescription",
    "0008|1030": "StudyDescription",
    "0008|0060": "Modality",
    "0008|0070": "Manufacturer",
    "0008|1090": "ManufacturerModelName",
    "0018|0050": "SliceThickness",
    "0018|0088": "SpacingBetweenSlices",
    "0020|000e": "SeriesInstanceUID",
    "0020|0011": "SeriesNumber",
    "0028|0030": "PixelSpacing",
}


def _read_dicom_metadata(file_name: str) -> Dict[str, Any]:
    """
    Read a small set of commonly useful DICOM tags from a file.
    """
    info: Dict[str, Any] = {}
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_name)
        reader.ReadImageInformation()
        for tag, name in _DICOM_INFO_TAGS.items():
            if reader.HasMetaDataKey(tag):
                info[name] = reader.GetMetaData(tag)
    except Exception:
        pass
    return info


# ----------------------------- Data Class ----------------------------- #

@dataclass
class DicomSeries:
    """
    Convenience wrapper if you want metadata + numpy in one place.
    Not required by the pipeline, but handy for debugging/inspection.
    """
    image: sitk.Image
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]
    metadata: Dict[str, Any]

    @classmethod
    def from_image(cls, image: sitk.Image, metadata: Optional[Dict[str, Any]] = None) -> "DicomSeries":
        meta = dict(metadata or {})
        spacing = tuple(float(v) for v in image.GetSpacing())
        origin = tuple(float(v) for v in image.GetOrigin())
        direction = tuple(float(v) for v in image.GetDirection())
        return cls(image=image, spacing=spacing, origin=origin, direction=direction, metadata=meta)

    def size(self) -> Tuple[int, ...]:
        return tuple(int(v) for v in self.image.GetSize())

    def dimension(self) -> int:
        return int(self.image.GetDimension())

    def _format_tuple(self, values, precision: Optional[int] = None) -> str:
        parts: List[str] = []
        for v in values:
            try:
                fv = float(v)
                parts.append(f"{fv:.{precision}f}" if precision is not None else f"{fv}")
            except (TypeError, ValueError):
                parts.append(str(v))
        return "(" + ", ".join(parts) + ")"

    def _format_pixel_spacing(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value)
        parts = [p for p in raw.replace("\\", " ").split() if p]
        floats: List[str] = []
        for p in parts:
            try:
                floats.append(f"{float(p):.3f}")
            except ValueError:
                floats.append(p)
        if not floats:
            return None
        return " x ".join(floats)

    def _direction_rows(self) -> List[str]:
        dim = self.dimension()
        values = list(self.direction)
        if dim <= 0 or len(values) != dim * dim:
            return []
        rows: List[str] = []
        for i in range(dim):
            row = values[i * dim : (i + 1) * dim]
            formatted = ", ".join(f"{float(v):.4f}" for v in row)
            rows.append(f"[{formatted}]")
        return rows

    def summary_items(self) -> List[Tuple[str, Any]]:
        info: List[Tuple[str, Any]] = []
        meta = self.metadata

        def add(key: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str) and not value.strip():
                return
            info.append((key, value))

        source = meta.get("SourceFolder") or meta.get("SourcePath")
        add("Source", source)
        add("Source type", meta.get("SourceType"))
        add("Series ID", meta.get("SeriesID") or meta.get("SeriesInstanceUID"))
        add("Series Number", meta.get("SeriesNumber"))
        add("Series Description", meta.get("SeriesDescription"))
        add("Study Description", meta.get("StudyDescription"))
        add("Modality", meta.get("Modality"))
        add("Manufacturer", meta.get("Manufacturer"))
        add("Model", meta.get("ManufacturerModelName"))
        num_files = meta.get("NumFiles")
        if isinstance(num_files, int):
            add("File Count", num_files)
        elif isinstance(num_files, str) and num_files.isdigit():
            add("File Count", int(num_files))

        pixel_spacing = self._format_pixel_spacing(meta.get("PixelSpacing"))
        add("Pixel Spacing (row, col) mm", pixel_spacing)

        def _format_mm(value: Any) -> Optional[str]:
            try:
                return f"{float(value):.3f}"
            except (TypeError, ValueError):
                return None

        add("Slice Thickness (mm)", _format_mm(meta.get("SliceThickness")))
        add("Spacing Between Slices (mm)", _format_mm(meta.get("SpacingBetweenSlices")))

        dim = self.dimension()
        add("Dimension", dim)

        size = self.size()
        if size:
            add("Size (x, y, z)", " x ".join(str(v) for v in size))

        add("Spacing (x, y, z) mm", self._format_tuple(self.spacing, precision=3))
        add("Origin (x, y, z)", self._format_tuple(self.origin, precision=3))

        direction_rows = self._direction_rows()
        if direction_rows:
            add("Direction cosines", direction_rows)

        try:
            pixel_type = self.image.GetPixelIDTypeAsString()
        except RuntimeError:
            pixel_type = None
        add("Pixel type", pixel_type)

        if size:
            voxel_count = 1
            for v in size:
                voxel_count *= int(v)
            add("Voxel count", f"{voxel_count:,}")
            try:
                spacing_product = float(np.prod(self.spacing))
                physical_volume = voxel_count * spacing_product
                add("Volume (mm^3)", f"{physical_volume:,.2f}")
            except Exception:
                pass

        return info

    def print_summary(self, stream: TextIO = sys.stdout, prefix: str = "[dicom]") -> None:
        items = self.summary_items()
        header_hint = (
            self.metadata.get("SeriesDescription")
            or self.metadata.get("SeriesID")
            or self.metadata.get("SeriesInstanceUID")
            or "Series"
        )
        print(f"{prefix} Series summary: {header_hint}", file=stream)
        if not items:
            print(f"{prefix}   (no additional metadata)", file=stream)
            return

        width = max(len(key) for key, _ in items)
        for key, value in items:
            if isinstance(value, list):
                if not value:
                    continue
                first, *rest = value
                print(f"{prefix}   {key.ljust(width)} : {first}", file=stream)
                for extra in rest:
                    print(f"{prefix}   {'':<{width}}   {extra}", file=stream)
            else:
                print(f"{prefix}   {key.ljust(width)} : {value}", file=stream)

    @classmethod
    def load_from_folder(cls, folder: str, series_uid: Optional[str] = None) -> "DicomSeries":
        """
        Load a DICOM series from a directory. If series_uid is not provided,
        the series with the most files is chosen.
        """
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Not a folder: {folder}")

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(folder) or []
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in: {folder}")

        if series_uid and series_uid not in series_ids:
            raise RuntimeError(f"Series UID not found. Available: {series_ids}")

        if series_uid:
            chosen = series_uid
        else:
            # Heuristic: pick the series with the most files
            counts = {sid: len(reader.GetGDCMSeriesFileNames(folder, sid)) for sid in series_ids}
            chosen = max(counts, key=counts.get)

        file_names = reader.GetGDCMSeriesFileNames(folder, chosen)
        reader.SetFileNames(file_names)
        image = reader.Execute()

        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()

        # Basic metadata (from first file)
        meta: Dict[str, Any] = {
            "SeriesID": chosen,
            "NumFiles": len(file_names),
            "SourceFolder": os.path.abspath(folder),
            "SourceType": "DICOM Series",
        }
        meta.update(_read_dicom_metadata(file_names[0]))
        meta.setdefault("SeriesDescription", os.path.basename(os.path.normpath(folder)))

        return cls(image=image, spacing=spacing, origin=origin, direction=direction, metadata=meta)

    def to_numpy(self) -> np.ndarray:
        """
        Returns (z, y, x) numpy array. Applies RescaleSlope/Intercept if available.
        """
        arr = sitk.GetArrayFromImage(self.image).astype(np.float32)  # (slices, rows, cols)
        slope = 1.0
        intercept = 0.0
        try:
            if self.image.HasMetaDataKey("0028|1053"):
                slope = float(self.image.GetMetaData("0028|1053"))
            if self.image.HasMetaDataKey("0028|1052"):
                intercept = float(self.image.GetMetaData("0028|1052"))
        except Exception:
            pass
        return (arr * slope + intercept).astype(np.float32)


# ----------------------------- Helpers ----------------------------- #

def _is_nifti(path: str) -> bool:
    p = path.lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")


# ----------------------------- Public API ----------------------------- #

def list_dicom_series(path: str) -> List[Dict[str, Any]]:
    """
    List all DICOM series in a directory.

    Returns: [{"uid": <SeriesInstanceUID>, "count": <num files>, "description": <SeriesDescription or "">}, ...]
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not a directory: {path}")

    r = sitk.ImageSeriesReader()
    series_ids = r.GetGDCMSeriesIDs(path) or []
    result: List[Dict[str, Any]] = []

    for uid in series_ids:
        files = r.GetGDCMSeriesFileNames(path, uid)
        desc = ""
        try:
            fr = sitk.ImageFileReader()
            fr.SetFileName(files[0])
            fr.ReadImageInformation()
            if fr.HasMetaDataKey("0008|103e"):
                desc = fr.GetMetaData("0008|103e")
        except Exception:
            pass
        result.append({"uid": uid, "count": len(files), "description": desc})

    return result


def load_dicom_series(path: str, series_uid: Optional[str] = None) -> sitk.Image:
    """
    Load a volume from:
      - a DICOM directory (optionally selecting a SeriesInstanceUID),
      - a single DICOM file,
      - or a NIfTI file (.nii/.nii.gz).

    Returns a 3D SimpleITK image (int16).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    if os.path.isdir(path):
        series = DicomSeries.load_from_folder(path, series_uid=series_uid)
        series.print_summary()
        return sitk.Cast(series.image, sitk.sitkInt16)

    # File path
    if _is_nifti(path):
        img = sitk.ReadImage(path)
        metadata = {
            "SourcePath": os.path.abspath(path),
            "SourceType": "NIfTI",
            "NumFiles": 1,
        }
        metadata.setdefault("SeriesDescription", os.path.basename(path))
        series = DicomSeries.from_image(img, metadata=metadata)
        series.print_summary()
        return sitk.Cast(img, sitk.sitkInt16)

    # Single-file DICOM (or other readable format)
    img = sitk.ReadImage(path)  # will raise if unreadable
    metadata = {
        "SourcePath": os.path.abspath(path),
        "SourceType": "Image file",
        "NumFiles": 1,
    }
    dicom_meta = _read_dicom_metadata(path)
    if dicom_meta:
        metadata["SourceType"] = "Single DICOM"
    metadata.update(dicom_meta)
    metadata.setdefault("SeriesDescription", os.path.basename(path))
    series = DicomSeries.from_image(img, metadata=metadata)
    series.print_summary()
    return sitk.Cast(img, sitk.sitkInt16)


def save_mesh(surface: Dict[str, Any], path: str) -> None:
    """
    Write a triangle mesh dict to Wavefront OBJ.
    surface = {"vertices": (N,3) array-like, "faces": (M,3) int array-like}
    """
    V = surface["vertices"]
    F = surface["faces"]
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
        for tri in F:
            # OBJ uses 1-based indexing
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


__all__ = [
    "DicomSeries",
    "list_dicom_series",
    "load_dicom_series",
    "save_mesh",
]
