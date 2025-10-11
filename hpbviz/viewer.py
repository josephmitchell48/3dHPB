# viewer.py
from typing import Optional, Dict, Any
import numpy as np
import SimpleITK as sitk
import napari
from magicgui import magicgui

from .io import save_mesh
from .mesh import MeshBuilder


class HpbViewer:
    def __init__(self, image: sitk.Image, volume_name: str = "CT"):
        self.image_sitk = image
        self.volume_name = volume_name
        self.viewer: Optional[napari.Viewer] = None
        self._volume_data: Optional[np.ndarray] = None
        self._liver_mesh_world: Optional[Dict[str, Any]] = None
        self._abdomen_mesh_world: Optional[Dict[str, Any]] = None

        sx, sy, sz = image.GetSpacing()
        self.spacing_xyz = (float(sx), float(sy), float(sz))
        self.spacing_zyx = (float(sz), float(sy), float(sx))

        self.origin_xyz = np.array(image.GetOrigin(), dtype=np.float64)
        self.direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

        self.vol_layer = None
        self.liver_layer = None
        self.abdo_layer = None
        self._mesher = MeshBuilder()

    # ---------- public entry points ----------

    def show(self):
        """Open the viewer with just the CT volume."""
        self._ensure_viewer()
        napari.run()

    def show_with_surface(self, surface: Dict[str, Any], build_abdomen: bool = True, hide_volume: bool = False):
        """
        Open the viewer, add the liver surface, and optionally build a full-abdomen surface.
        """
        self._ensure_viewer()

        # Liver mesh
        verts = np.asarray(surface["vertices"], dtype=np.float32)
        faces = np.asarray(surface["faces"], dtype=np.int32)
        if verts.size and faces.size:
            self._liver_mesh_world = {"vertices": verts, "faces": faces}
            verts_napari = self._world_to_data_coords(verts)
            self.liver_layer = self.viewer.add_surface(
                (verts_napari, faces),
                name="Liver",
                opacity=0.95,
                shading="smooth",
            )
            self.liver_layer.scale = self.spacing_zyx
            # Make it pop
            try:
                self.liver_layer.face_color = [0.1, 0.8, 0.2, 1.0]  # RGBA
            except Exception:
                pass

        # Optional: full-abdomen surface
        if build_abdomen:
            try:
                abdo = self._build_abdomen_surface()
                if abdo is not None:
                    self._abdomen_mesh_world = abdo
                    abdo_napari = {
                        "vertices": self._world_to_data_coords(abdo["vertices"]),
                        "faces": abdo["faces"],
                    }
                    self.abdo_layer = self.viewer.add_surface(
                        (abdo_napari["vertices"], abdo_napari["faces"]),
                        name="Abdomen",
                        opacity=0.25,
                        shading="flat",
                    )
                    self.abdo_layer.scale = self.spacing_zyx
                    try:
                        self.abdo_layer.face_color = [0.6, 0.6, 0.6, 1.0]
                    except Exception:
                        pass
            except Exception as e:
                print(f"[viewer] Abdomen surface build failed: {e}")

        # Volume visibility preference
        if hide_volume and self.vol_layer is not None:
            self.vol_layer.visible = False

        # Switch to 3D and center
        self.viewer.dims.ndisplay = 3
        self.viewer.reset_view()
        napari.run()

    # ---------- internal helpers ----------

    def _ensure_viewer(self) -> None:
        if self.viewer is not None:
            return

        if self._volume_data is None:
            self._volume_data = sitk.GetArrayFromImage(self.image_sitk).astype(np.float32)  # z,y,x
        vol = self._volume_data
        vmin, vmax = np.percentile(vol, (5, 95))

        self.viewer = napari.Viewer()
        self.vol_layer = self.viewer.add_image(
            vol,
            name=self.volume_name,
            contrast_limits=(float(vmin), float(vmax)),
            scale=self.spacing_zyx,     # physical scaling
            rendering="mip",
            blending="translucent",
            opacity=0.35,               # faint so meshes pop
            visible=True,
        )

        self.viewer.window.add_dock_widget(self._controls_widget(), area="right")

    def _build_abdomen_surface(self) -> Optional[Dict[str, Any]]:
        """
        Build a coarse 'body/abdomen' surface by thresholding the CT,
        suitable for context around the liver.
        """
        img = sitk.Cast(self.image_sitk, sitk.sitkInt16)

        # Stabilize intensities
        clamp = sitk.IntensityWindowing(img, -200, 250, -200, 250)

        # Body mask (exclude air); clean a bit
        body = sitk.BinaryThreshold(clamp, lowerThreshold=-300, upperThreshold=3071)
        body = sitk.BinaryMorphologicalClosing(body, [2, 2, 2])
        body = sitk.VotingBinaryHoleFilling(body, radius=[1, 1, 1], majorityThreshold=1)

        arr = (sitk.GetArrayFromImage(body) > 0).astype(np.uint8)  # z,y,x
        if arr.max() == 0:
            return None

        sx, sy, sz = self.spacing_xyz
        origin = tuple(float(v) for v in self.image_sitk.GetOrigin())
        surface = self._mesher.mask_to_surface(arr, spacing=(sx, sy, sz), origin=origin)
        return surface

    # ---------- UI panel ----------

    def _controls_widget(self):
        @magicgui(call_button="Toggle CT Volume")
        def toggle_volume():
            if self.vol_layer is None:
                print("No CT volume layer.")
                return
            self.vol_layer.visible = not self.vol_layer.visible

        @magicgui(call_button="Toggle Liver Mesh")
        def toggle_liver():
            if self.liver_layer is None:
                print("No liver surface layer.")
                return
            self.liver_layer.visible = not self.liver_layer.visible

        @magicgui(call_button="Toggle Abdomen Mesh")
        def toggle_abdomen():
            if self.abdo_layer is None:
                print("No abdomen surface layer.")
                return
            self.abdo_layer.visible = not self.abdo_layer.visible

        @magicgui(call_button="Center View")
        def center_view():
            self.viewer.dims.ndisplay = 3
            self.viewer.reset_view()

        @magicgui(call_button="Export Liver Mesh")
        def export_liver(path: str = "liver.obj"):
            if self._liver_mesh_world is None:
                print("No liver mesh to export.")
                return
            save_mesh(self._liver_mesh_world, path)
            print(f"Saved {path}")

        @magicgui(call_button="Export Abdomen Mesh")
        def export_abdomen(path: str = "abdomen.obj"):
            if self._abdomen_mesh_world is None:
                print("No abdomen mesh to export.")
                return
            save_mesh(self._abdomen_mesh_world, path)
            print(f"Saved {path}")

        @magicgui(call_button="Save Screenshot")
        def save_screenshot(path: str = "viewer.png"):
            self.viewer.screenshot(path=path, canvas_only=True, flash=False)
            print(f"Saved {path}")

        # Build a stacked controls panel
        from qtpy.QtWidgets import QWidget, QVBoxLayout
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(toggle_volume.native)
        layout.addWidget(toggle_liver.native)
        layout.addWidget(toggle_abdomen.native)
        layout.addWidget(center_view.native)
        layout.addWidget(export_liver.native)
        layout.addWidget(export_abdomen.native)
        layout.addWidget(save_screenshot.native)
        return w

    def _world_to_data_coords(self, verts_xyz: np.ndarray) -> np.ndarray:
        """
        Convert physical (x, y, z) coordinates into napari data coordinates (z, y, x)
        consistent with the volume array indexing.
        """
        if verts_xyz.size == 0:
            return verts_xyz

        rel = verts_xyz.astype(np.float64) - self.origin_xyz
        aligned = self.direction.T @ rel.T  # -> (3, N)
        spacing = np.array(self.spacing_xyz, dtype=np.float64)
        idx = (aligned.T / spacing).astype(np.float32)  # (N, 3) index coords (x, y, z)
        napari_coords = idx[:, [2, 1, 0]]  # -> (z, y, x)
        return napari_coords
    
