#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import SimpleITK as sitk

def read_dicom_series(dicom_dir: str) -> sitk.Image:
    """Read a DICOM series with proper slice ordering and HU scaling."""
    reader = sitk.ImageSeriesReader()
    series_uids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_uids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")

    # Heuristic: pick the series with most files (typical axial series)
    best_uid, best_files = None, []
    for uid in series_uids:
        files = reader.GetGDCMSeriesFileNames(dicom_dir, uid)
        if len(files) > len(best_files):
            best_uid, best_files = uid, files

    reader.SetFileNames(best_files)
    # Enable rescale to HU using DICOM tags (RescaleSlope/Intercept)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()
    return img

def save_nifti(img: sitk.Image, out_path: str, hu_clip: tuple|None):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    if hu_clip is not None:
        lo, hi = hu_clip
        arr = np.clip(arr, lo, hi, out=arr)

    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(img)  # keep spacing/origin/direction
    sitk.WriteImage(out_img, out_path, useCompression=True)

def main():
    p = argparse.ArgumentParser(description="Convert DICOM series to NIfTI (.nii.gz) in HU.")
    p.add_argument("dicom_dir", help="Folder containing DICOM slices")
    p.add_argument("out_nii", help="Output path, e.g. /path/study.nii.gz")
    p.add_argument("--hu_clip", type=int, nargs=2, metavar=("LO","HI"),
                   default=(-1000, 1000),
                   help="Optional HU clipping range, default -1000 1000")
    args = p.parse_args()

    if not os.path.isdir(args.dicom_dir):
        sys.exit(f"Not a directory: {args.dicom_dir}")

    img = read_dicom_series(args.dicom_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_nii)), exist_ok=True)
    save_nifti(img, args.out_nii, tuple(args.hu_clip) if args.hu_clip else None)
    print(f"✔ Wrote {args.out_nii}")

if __name__ == "__main__":
    main()