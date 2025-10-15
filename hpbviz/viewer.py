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
        self.surface_layers: Dict[str, napari.layers.Surface] = {}
        self._surface_meshes_world: Dict[str, Dict[str, Any]] = {}
        self._controls_dock = None

        sx, sy, sz = image.GetSpacing()
        self.spacing_xyz = (float(sx), float(sy), float(sz))
        self.spacing_zyx = (float(sz), float(sy), float(sx))

        self.origin_xyz = np.array(image.GetOrigin(), dtype=np.float64)
        self.direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

        self.vol_layer = None
        self._mesher = MeshBuilder()

    # ---------- public entry points ----------

    def show(self):
        """Open the viewer with just the CT volume."""
        self._ensure_viewer()
        self._setup_controls()
        napari.run()

    def show_with_surfaces(
        self,
        surfaces: Dict[str, Dict[str, Any]],
        build_abdomen: bool = True,
        hide_volume: bool = False,
    ):
        """Open the viewer, add multiple surfaces, and optionally build a context abdomen surface."""
        self._ensure_viewer()

        self._add_surfaces(surfaces, clear_existing=True)

        # Optional: full-abdomen surface
        if build_abdomen:
            try:
                abdo = self._build_abdomen_surface()
                if abdo is not None:
                    abdo.setdefault("color", (0.6, 0.6, 0.6, 1.0))
                    abdo.setdefault("opacity", 0.25)
                    abdo.setdefault("display_name", "Abdomen")
                    self._add_surfaces({"abdomen": abdo})
            except Exception as e:
                print(f"[viewer] Abdomen surface build failed: {e}")

        # Volume visibility preference
        if hide_volume and self.vol_layer is not None:
            self.vol_layer.visible = False

        # Switch to 3D and center
        self.viewer.dims.ndisplay = 3
        self.viewer.reset_view()
        self._setup_controls()
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

    def _setup_controls(self) -> None:
        if self.viewer is None:
            return
        if self._controls_dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._controls_dock)
            except Exception:
                pass
            self._controls_dock = None
        widget = self._controls_widget()
        self._controls_dock = self.viewer.window.add_dock_widget(widget, area="right")

    def _add_surfaces(self, surfaces: Dict[str, Dict[str, Any]], clear_existing: bool = False) -> None:
        if self.viewer is None:
            return

        if clear_existing:
            for layer in self.surface_layers.values():
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
            self.surface_layers.clear()
            self._surface_meshes_world.clear()

        for key, surface in surfaces.items():
            if not surface:
                continue
            verts = np.asarray(surface["vertices"], dtype=np.float32)
            faces = np.asarray(surface["faces"], dtype=np.int32)
            if verts.size == 0 or faces.size == 0:
                continue
            verts_napari = self._world_to_data_coords(verts)
            display_name = surface.get("display_name") or key.replace("_", " ").title()
            opacity = surface.get("opacity", 0.9)
            shading = surface.get("shading", "smooth")

            layer = self.viewer.add_surface(
                (verts_napari, faces),
                name=display_name,
                opacity=opacity,
                shading=shading,
            )
            layer.scale = self.spacing_zyx
            color = surface.get("color")
            if color is not None:
                try:
                    layer.face_color = color
                except Exception:
                    pass

            self.surface_layers[display_name] = layer
            self._surface_meshes_world[display_name] = {
                "vertices": verts,
                "faces": faces,
            }

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

        @magicgui(call_button="Center View")
        def center_view():
            self.viewer.dims.ndisplay = 3
            self.viewer.reset_view()

        @magicgui(call_button="Save Screenshot")
        def save_screenshot(path: str = "viewer.png"):
            self.viewer.screenshot(path=path, canvas_only=True, flash=False)
            print(f"Saved {path}")

        # Build a stacked controls panel
        from qtpy.QtWidgets import QWidget, QVBoxLayout
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(toggle_volume.native)
        layout.addWidget(center_view.native)

        for name in sorted(self.surface_layers.keys()):
            layer = self.surface_layers[name]
            mesh_world = self._surface_meshes_world.get(name)
            label = name

            def make_toggle(target_layer=layer, display_name=label):
                @magicgui(call_button=f"Toggle {display_name}")
                def toggle():
                    if target_layer is None:
                        print(f"No {display_name} surface layer.")
                        return
                    target_layer.visible = not target_layer.visible
                return toggle

            toggle_widget = make_toggle()
            layout.addWidget(toggle_widget.native)

            if mesh_world is not None:
                def make_export(mesh=mesh_world, display_name=label):
                    default_name = display_name.lower().replace(" ", "_") + ".obj"

                    @magicgui(call_button=f"Export {display_name} Mesh")
                    def export(path: str = default_name):
                        save_mesh(mesh, path)
                        print(f"Saved {path}")

                    return export

                export_widget = make_export()
                layout.addWidget(export_widget.native)

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
    
