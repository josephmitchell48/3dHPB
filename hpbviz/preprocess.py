
from __future__ import annotations
import SimpleITK as sitk

class VolumeProcessor:
    def __init__(self, target_spacing: float = 1.0):
        self.target_spacing = float(target_spacing)

    def resample_isotropic(self, image: sitk.Image) -> sitk.Image:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        spacing = [self.target_spacing] * 3

        new_size = [
            int(round(osz * (ospc / tspc)))
            for osz, ospc, tspc in zip(original_size, original_spacing, spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetDefaultPixelValue(-1024)  # background for CT
        return resampler.Execute(image)
