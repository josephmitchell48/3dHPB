
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import vtk
from vtk.util import numpy_support

class MeshBuilder:
    def __init__(self):
        pass

    def mask_to_surface(
        self,
        mask_zyx: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float] | None = None,
        label: int = 1,
    ) -> Dict[str, Any]:
        """
        mask_zyx: binary or label mask (z, y, x)
        spacing: image spacing (x_spacing, y_spacing, z_spacing).
        origin: physical origin (x, y, z). Defaults to (0, 0, 0).
        label: label value to extract (used if mask is multi-label).
        Returns dict: { 'vertices': (N, 3), 'faces': (M, 3) }
        """
        if mask_zyx.ndim != 3:
            raise ValueError("Mask must be a 3D array (z, y, x).")

        mask_bool = (mask_zyx == label) if label is not None else mask_zyx > 0
        if not np.any(mask_bool):
            raise ValueError("Mask is empty; nothing to mesh.")

        nz, ny, nx = mask_bool.shape
        vtk_arr = numpy_support.numpy_to_vtk(
            mask_bool.astype(np.uint8).ravel(order="C"),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )

        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)
        image.SetSpacing(spacing)
        image.SetOrigin(origin if origin is not None else (0.0, 0.0, 0.0))
        image.SetExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
        image.GetPointData().SetScalars(vtk_arr)
        image.Modified()

        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(image)
        dmc.SetValue(0, 1)
        dmc.Update()

        poly = dmc.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("Marching cubes returned an empty mesh.")

        vertices = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())

        polys = poly.GetPolys()
        if polys is None or polys.GetNumberOfCells() == 0:
            raise ValueError("Marching cubes returned a mesh without faces.")
        faces = numpy_support.vtk_to_numpy(polys.GetData()).reshape(-1, 4)[:, 1:]

        return {
            "vertices": vertices.astype(np.float32, copy=False),
            "faces": faces.astype(np.int32, copy=False),
        }
