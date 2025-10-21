from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING, Any, List

import numpy as np
import napari
import SimpleITK as sitk
from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import QFileDialog

from ..io import save_mesh
from ..mesh import MeshBuilder

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from qtpy.QtWidgets import QCheckBox, QListWidget, QPushButton


class ViewerActionsMixin:
    """Shared actions and utilities for the HPB napari viewer."""

    viewer: Optional[napari.Viewer]
    vol_layer: Optional[napari.layers.Image]
    _volume_toggle_widgets: List["QCheckBox"]
    _theme_buttons: List["QPushButton"]
    _case_list_widgets: List["QListWidget"]
    _display_mode_buttons: List["QPushButton"]
    _surface_meshes_world: Dict[str, Dict[str, Any]]
    _mesher: MeshBuilder
    _volume_data: Optional[np.ndarray]
    surface_layers: Dict[str, napari.layers.Surface]
    current_case: Optional[str]
    volume_name: str
    spacing_xyz: tuple[float, float, float]
    spacing_zyx: tuple[float, float, float]
    origin_xyz: np.ndarray
    direction: np.ndarray
    image_sitk: sitk.Image

    def _center_view(self) -> None:
        if not self.viewer:
            return
        self.viewer.dims.ndisplay = 3
        self.viewer.reset_view()
        self._update_display_button_text()

    def _toggle_display_mode(self) -> None:
        if not self.viewer:
            return
        current = getattr(self.viewer.dims, "ndisplay", 3)
        new_value = 2 if current == 3 else 3
        self.viewer.dims.ndisplay = new_value
        if new_value == 3:
            self.viewer.reset_view()
        self._update_display_button_text()

    def _set_volume_visible(self, visible: bool) -> None:
        if self.vol_layer is None:
            return
        self.vol_layer.visible = bool(visible)
        for toggle in getattr(self, "_volume_toggle_widgets", []):
            blocker = QSignalBlocker(toggle)
            try:
                toggle.setChecked(self.vol_layer.visible)
            finally:
                del blocker

    def _set_layer_visible(self, layer: napari.layers.Layer, visible: bool) -> None:
        layer.visible = bool(visible)

    def _open_export_dialog(self, display_name: str) -> None:
        mesh = self._surface_meshes_world.get(display_name)
        if not mesh:
            print(f"[viewer] No mesh data available for {display_name}")
            return

        default_name = display_name.lower().replace(" ", "_") + ".obj"
        default_path = str((Path.cwd() / default_name).resolve())

        path, _ = QFileDialog.getSaveFileName(
            None,
            f"Export {display_name} Mesh",
            default_path,
            "Wavefront OBJ (*.obj);;All Files (*)",
        )
        if path:
            save_mesh(mesh, path)
            print(f"Saved {path}")

    def _header_subtitle_text(self) -> str:
        if self.current_case:
            return f"Currently viewing: {self.current_case}"
        return "Ready whenever you are—load a case to begin."

    def _header_info_text(self) -> str:
        if self._volume_data is None:
            return ""
        shape = self._volume_data.shape
        spacing = tuple(round(v, 2) for v in self.spacing_xyz)
        lines = [
            f"Volume voxels: {shape[2]} × {shape[1]} × {shape[0]}",
            f"Spacing (mm): {spacing[0]}, {spacing[1]}, {spacing[2]}",
        ]
        return "\n".join(lines)

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

    def _update_display_button_text(self) -> None:
        if not getattr(self, "viewer", None):
            return
        current = getattr(self.viewer.dims, "ndisplay", 3)
        label = "Switch to 2D View" if current == 3 else "Switch to 3D View"
        for button in getattr(self, "_display_mode_buttons", []):
            button.setText(label)
