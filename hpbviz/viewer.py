# viewer.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List

import numpy as np
import SimpleITK as sitk
import napari
from magicgui import magicgui
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, QSignalBlocker
from qtpy.QtWidgets import QListWidget, QWidget, QVBoxLayout, QLabel, QFrame

_NAPARI_CACHE = Path.cwd() / ".napari_cache"
_NAPARI_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["NAPARI_CACHE_DIR"] = str(_NAPARI_CACHE)
os.environ["NAPARI_CONFIG_DIR"] = str(_NAPARI_CACHE)
os.environ["NAPARI_CONFIG"] = str(_NAPARI_CACHE)

from .io import save_mesh
from .mesh import MeshBuilder


CaseLoader = Callable[[str], Tuple[sitk.Image, Dict[str, Dict[str, np.ndarray]], Any]]


class HpbViewer:
    def __init__(
        self,
        image: sitk.Image,
        volume_name: str = "CT",
        show_controls: bool = True,
        case_catalog: Optional[Dict[str, Dict[str, str]]] = None,
        case_loader: Optional[CaseLoader] = None,
        current_case: Optional[str] = None,
    ) -> None:
        self.image_sitk = image
        self.volume_name = volume_name
        self.viewer: Optional[napari.Viewer] = None
        self._volume_data: Optional[np.ndarray] = None

        self.surface_layers: Dict[str, napari.layers.Surface] = {}
        self._surface_meshes_world: Dict[str, Dict[str, Any]] = {}

        self._side_dock = None
        self._case_list_widget: Optional[QListWidget] = None
        self._suppress_case_signal = False

        self.show_controls = bool(show_controls)
        self.case_catalog = case_catalog or {}
        self.case_loader = case_loader
        self.current_case = current_case

        sx, sy, sz = image.GetSpacing()
        self.spacing_xyz = (float(sx), float(sy), float(sz))
        self.spacing_zyx = (float(sz), float(sy), float(sx))

        self.origin_xyz = np.array(image.GetOrigin(), dtype=np.float64)
        self.direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

        self.vol_layer = None
        self._mesher = MeshBuilder()

    # ---------- public entry points ----------

    def show(self) -> None:
        self._ensure_viewer()
        self._setup_side_panel()
        napari.run()

    def show_with_surfaces(
        self,
        surfaces: Dict[str, Dict[str, Any]],
        build_abdomen: bool = True,
        hide_volume: bool = False,
    ) -> None:
        self._ensure_viewer()

        self._add_surfaces(surfaces, clear_existing=True)

        if build_abdomen:
            try:
                abdo = self._build_abdomen_surface()
                if abdo is not None:
                    abdo.setdefault("color", (0.6, 0.6, 0.6, 1.0))
                    abdo.setdefault("opacity", 0.25)
                    abdo.setdefault("display_name", "Abdomen")
                    self._add_surfaces({"abdomen": abdo})
            except Exception as exc:  # pragma: no cover - GUI log only
                print(f"[viewer] Abdomen surface build failed: {exc}")

        if hide_volume and self.vol_layer is not None:
            self.vol_layer.visible = False

        self.viewer.dims.ndisplay = 3
        self.viewer.reset_view()

        self._setup_side_panel()
        napari.run()

    # ---------- internal helpers ----------

    def _ensure_viewer(self) -> None:
        if self.viewer is not None:
            return

        if self._volume_data is None:
            self._volume_data = sitk.GetArrayFromImage(self.image_sitk).astype(np.float32)
        vol = self._volume_data
        vmin, vmax = np.percentile(vol, (5, 95))

        self.viewer = napari.Viewer()
        self.vol_layer = self.viewer.add_image(
            vol,
            name=self._volume_layer_name(),
            contrast_limits=(float(vmin), float(vmax)),
            scale=self.spacing_zyx,
            rendering="mip",
            blending="translucent",
            opacity=0.35,
            visible=True,
        )

    def _setup_side_panel(self) -> None:
        if self.viewer is None:
            return

        if self._side_dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._side_dock)
            except Exception:
                pass
            self._side_dock = None
            self._case_list_widget = None

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        if self.show_controls:
            layout.addWidget(self._controls_widget())

        if self.case_catalog and self.case_loader:
            if layout.count() > 0:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                layout.addWidget(separator)

            layout.addWidget(QLabel("Patients"))
            list_widget = QListWidget()
            list_widget.setSelectionMode(QListWidget.SingleSelection)
            list_widget.addItems(sorted(self.case_catalog.keys()))
            list_widget.currentTextChanged.connect(self._on_case_selected)
            self._case_list_widget = list_widget
            layout.addWidget(list_widget)

        self._side_dock = self.viewer.window.add_dock_widget(widget, area="right")

        if self._case_list_widget is not None:
            blocker = QSignalBlocker(self._case_list_widget)
            try:
                if self.current_case:
                    matches = self._case_list_widget.findItems(self.current_case, Qt.MatchExactly)
                    if matches:
                        self._case_list_widget.setCurrentItem(matches[0])
                    elif self._case_list_widget.count() > 0:
                        self._case_list_widget.setCurrentRow(0)
                elif self._case_list_widget.count() > 0:
                    self._case_list_widget.setCurrentRow(0)
            finally:
                del blocker

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

            values = surface.get("values")
            if values is None:
                values = np.ones(len(verts_napari), dtype=np.float32)
            else:
                values = np.asarray(values, dtype=np.float32)
                if values.ndim != 1 or values.size not in (len(verts_napari), len(faces)):
                    raise ValueError("Surface values must be 1D and match vertices or faces length.")

            layer = self.viewer.add_surface(
                (verts_napari, faces, values),
                name=display_name,
                opacity=opacity,
                shading=shading,
                blending="translucent_no_depth",
            )
            layer.scale = self.spacing_zyx

            color = surface.get("color")
            if color is not None:
                try:
                    rgba = np.array(color, dtype=np.float32)
                    if rgba.ndim == 1:
                        if rgba.size == 3:
                            rgba = np.concatenate([rgba, np.array([1.0], dtype=rgba.dtype)])
                        if rgba.size != 4:
                            raise ValueError("Expected RGB or RGBA tuple")
                        rgba = np.vstack([rgba, rgba])
                    elif rgba.ndim == 2 and rgba.shape[1] == 3:
                        alpha = np.ones((rgba.shape[0], 1), dtype=rgba.dtype)
                        rgba = np.concatenate([rgba, alpha], axis=1)
                    elif rgba.ndim != 2 or rgba.shape[1] != 4:
                        raise ValueError("Color array must be RGB or RGBA")

                    cmap = Colormap(colors=rgba, name=f"{display_name}_const")
                    layer.colormap = cmap
                    layer.contrast_limits = (0.0, 1.0)
                except Exception as exc:
                    print(f"[viewer] failed to apply color for {display_name}: {exc}")

            self.surface_layers[display_name] = layer
            self._surface_meshes_world[display_name] = {
                "vertices": verts,
                "faces": faces,
            }

    def _on_case_selected(self, case_name: str) -> None:
        if self._suppress_case_signal:
            return
        if not case_name or case_name == self.current_case:
            return
        if not self.case_loader:
            return
        try:
            img, surfaces, _ = self.case_loader(case_name)
        except Exception as exc:
            print(f"[viewer] failed to load case {case_name}: {exc}")
            return

        self.current_case = case_name
        self.image_sitk = img
        self._volume_data = sitk.GetArrayFromImage(img).astype(np.float32)
        sx, sy, sz = img.GetSpacing()
        self.spacing_xyz = (float(sx), float(sy), float(sz))
        self.spacing_zyx = (float(sz), float(sy), float(sx))
        self.origin_xyz = np.array(img.GetOrigin(), dtype=np.float64)
        self.direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

        if self.vol_layer is not None:
            self.vol_layer.data = self._volume_data
            self.vol_layer.scale = self.spacing_zyx
            self.vol_layer.name = self._volume_layer_name()

        self._add_surfaces(surfaces, clear_existing=True)

        self._suppress_case_signal = True
        try:
            self._setup_side_panel()
        finally:
            self._suppress_case_signal = False

    def _build_abdomen_surface(self) -> Optional[Dict[str, Any]]:
        img = sitk.Cast(self.image_sitk, sitk.sitkInt16)

        clamp = sitk.IntensityWindowing(img, -200, 250, -200, 250)
        body = sitk.BinaryThreshold(clamp, lowerThreshold=-300, upperThreshold=3071)
        body = sitk.BinaryMorphologicalClosing(body, [2, 2, 2])
        body = sitk.VotingBinaryHoleFilling(body, radius=[1, 1, 1], majorityThreshold=1)

        arr = (sitk.GetArrayFromImage(body) > 0).astype(np.uint8)
        if arr.max() == 0:
            return None

        sx, sy, sz = self.spacing_xyz
        origin = tuple(float(v) for v in self.image_sitk.GetOrigin())
        return self._mesher.mask_to_surface(arr, spacing=(sx, sy, sz), origin=origin)

    def _controls_widget(self) -> QWidget:
        @magicgui(call_button="Center View")
        def center_view():
            self.viewer.dims.ndisplay = 3
            self.viewer.reset_view()

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(center_view.native)

        for name in self._sorted_layer_names():
            mesh_world = self._surface_meshes_world.get(name)
            if not mesh_world:
                continue

            def make_export(mesh=mesh_world, display_name=name):
                default_name = display_name.lower().replace(" ", "_") + ".obj"

                @magicgui(call_button=f"Export {display_name} Mesh")
                def export(path: str = default_name):
                    save_mesh(mesh, path)
                    print(f"Saved {path}")

                return export

            layout.addWidget(make_export().native)

        return widget

    def _world_to_data_coords(self, verts_xyz: np.ndarray) -> np.ndarray:
        if verts_xyz.size == 0:
            return verts_xyz

        rel = verts_xyz.astype(np.float64) - self.origin_xyz
        aligned = self.direction.T @ rel.T
        spacing = np.array(self.spacing_xyz, dtype=np.float64)
        idx = (aligned.T / spacing).astype(np.float32)
        return idx[:, [2, 1, 0]]

    def _volume_layer_name(self) -> str:
        if self.current_case:
            return f"{self.volume_name} ({self.current_case})"
        return self.volume_name

    def _sorted_layer_names(self) -> List[str]:
        def sort_key(name: str):
            parts = re.split(r"(\d+)", name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        return sorted(self.surface_layers.keys(), key=sort_key)
