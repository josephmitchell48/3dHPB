# viewer.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple

import numpy as np
import SimpleITK as sitk
import napari
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, QSignalBlocker
from qtpy.QtWidgets import QListWidget, QCheckBox, QPushButton

_NAPARI_CACHE = Path.cwd() / ".napari_cache"
_NAPARI_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["NAPARI_CACHE_DIR"] = str(_NAPARI_CACHE)
os.environ["NAPARI_CONFIG_DIR"] = str(_NAPARI_CACHE)
os.environ["NAPARI_CONFIG"] = str(_NAPARI_CACHE)

from .mesh import MeshBuilder
from .ui import SidebarMixin, ViewerActionsMixin, ThemeMixin


CaseLoader = Callable[[str], Tuple[sitk.Image, Dict[str, Dict[str, np.ndarray]], Any]]


class HpbViewer(SidebarMixin, ViewerActionsMixin, ThemeMixin):
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

        self._side_docks: list[Any] = []
        self._case_list_widgets: list[QListWidget] = []
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
        self._surface_toggle_widgets: Dict[str, list[QCheckBox]] = {}
        self._volume_toggle_widgets: list[QCheckBox] = []
        self._theme_buttons: list[QPushButton] = []
        self._display_mode_buttons: list[QPushButton] = []

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
        finite = vol[np.isfinite(vol)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(finite, (5, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = float(finite.min()), float(finite.max())
            else:
                vmin, vmax = float(vmin), float(vmax)
            if vmin == vmax:
                vmax = vmin + 1.0

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
        self._apply_window_customizations()

    def _setup_side_panel(self) -> None:
        if self.viewer is None:
            return

        window = getattr(self.viewer, "window", None)
        if window is None:
            return

        if self._side_docks:
            for dock in list(self._side_docks):
                try:
                    window.remove_dock_widget(dock)
                except Exception:
                    pass
            self._side_docks.clear()

        self._case_list_widgets = []
        self._volume_toggle_widgets = []
        self._theme_buttons = []
        self._display_mode_buttons = []
        self._surface_toggle_widgets = {}

        left_widget = self._build_sidebar_widget(
            include_controls=True,
            include_surfaces=True,
            include_cases=False,
            header_title="HPB Navigator",
        )
        left_dock = window.add_dock_widget(left_widget, area="left")
        self._configure_dock(left_dock, "HPB Navigator")
        self._side_docks.append(left_dock)

        right_widget = self._build_sidebar_widget(
            include_controls=False,
            include_surfaces=False,
            include_cases=True,
            header_title="Patient Library",
        )
        right_dock = window.add_dock_widget(right_widget, area="right")
        self._configure_dock(right_dock, "Patient Library")
        self._side_docks.append(right_dock)

        self._sync_case_list_selection()

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

    def _apply_window_customizations(self) -> None:
        if not self.viewer:
            return
        window = getattr(self.viewer, "window", None)
        if window is None:
            return

        qt_viewer = getattr(window, "_qt_viewer", None)
        if qt_viewer is None:
            return

        for dock_name in ("dockLayerList", "dockLayerControls"):
            dock_widget = getattr(qt_viewer, dock_name, None)
            if dock_widget is None:
                continue
            try:
                dock_widget.setVisible(False)
                window.remove_dock_widget(dock_widget)
            except Exception:
                pass

    def _configure_dock(self, dock_widget: Any, title: str) -> None:
        if dock_widget is None:
            return
        try:
            dock_widget.setWindowTitle(title)
            dock_widget.setMinimumWidth(320)
        except Exception:
            pass

    def _sync_case_list_selection(self) -> None:
        if not self._case_list_widgets:
            return
        for widget in self._case_list_widgets:
            blocker = QSignalBlocker(widget)
            try:
                if self.current_case:
                    matches = widget.findItems(self.current_case, Qt.MatchExactly)
                    if matches:
                        widget.setCurrentItem(matches[0])
                    elif widget.count() > 0:
                        widget.setCurrentRow(0)
                elif widget.count() > 0:
                    widget.setCurrentRow(0)
            finally:
                del blocker
