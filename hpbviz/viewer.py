# viewer.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List

import numpy as np
import SimpleITK as sitk
import napari
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, QSignalBlocker
from qtpy.QtWidgets import (
    QListWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
    QHBoxLayout,
    QCheckBox,
    QScrollArea,
    QSpacerItem,
    QSizePolicy,
    QFileDialog,
)

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
        self._surface_toggle_widgets: Dict[str, QCheckBox] = {}
        self._volume_toggle_widget: Optional[QCheckBox] = None
        self._theme_button: Optional[QPushButton] = None

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

        widget = self._build_sidebar_widget()
        self._side_dock = self.viewer.window.add_dock_widget(widget, area="right")
        try:
            self._side_dock.setWindowTitle("HPB Navigator")
            self._side_dock.setMinimumWidth(320)
        except Exception:
            pass

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

    def _build_sidebar_widget(self) -> QWidget:
        self._case_list_widget = None
        self._surface_toggle_widgets = {}
        self._volume_toggle_widget = None
        self._theme_button = None

        root = QWidget()
        root.setObjectName("SidebarRoot")
        root.setStyleSheet(self._sidebar_stylesheet())

        outer_layout = QVBoxLayout(root)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setObjectName("SidebarScroll")
        outer_layout.addWidget(scroll)

        container = QWidget()
        container.setObjectName("SidebarContainer")
        scroll.setWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        layout.addWidget(self._build_header_section())

        if self.show_controls:
            layout.addWidget(self._build_controls_section())

        if self.surface_layers:
            layout.addWidget(self._build_surface_section())

        if self.case_catalog and self.case_loader:
            layout.addWidget(self._build_case_section())

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return root

    def _build_header_section(self) -> QWidget:
        header = QFrame()
        header.setObjectName("SidebarHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(6)

        title = QLabel("HPB Visualizer")
        title.setObjectName("SidebarTitle")
        layout.addWidget(title)

        subtitle = QLabel(self._header_subtitle_text())
        subtitle.setObjectName("SidebarSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        info_text = self._header_info_text()
        if info_text:
            info = QLabel(info_text)
            info.setObjectName("SidebarMeta")
            info.setWordWrap(True)
            layout.addWidget(info)

        return header

    def _build_controls_section(self) -> QWidget:
        section = QFrame()
        section.setObjectName("SidebarSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Viewer Controls")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        center_btn = QPushButton("Center View")
        center_btn.setObjectName("PrimaryButton")
        center_btn.clicked.connect(self._center_view)
        layout.addWidget(center_btn)

        theme_btn = QPushButton()
        theme_btn.setObjectName("SecondaryButton")
        theme_btn.clicked.connect(self._toggle_theme)
        self._theme_button = theme_btn
        self._update_theme_button_text()
        layout.addWidget(theme_btn)

        if self.vol_layer is not None:
            volume_toggle = QCheckBox("Show Volume")
            volume_toggle.setObjectName("AccentCheckBox")
            volume_toggle.setChecked(self.vol_layer.visible)
            volume_toggle.stateChanged.connect(
                lambda state: self._set_volume_visible(state == Qt.Checked)
            )
            layout.addWidget(volume_toggle)
            self._volume_toggle_widget = volume_toggle

        return section

    def _build_surface_section(self) -> QWidget:
        section = QFrame()
        section.setObjectName("SidebarSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Structures")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        description = QLabel("Toggle visibility or export any structure mesh.")
        description.setObjectName("SectionHint")
        description.setWordWrap(True)
        layout.addWidget(description)

        for name in self._sorted_layer_names():
            layer = self.surface_layers.get(name)
            if layer is None:
                continue

            row = QFrame()
            row.setObjectName("SurfaceRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 10, 12, 10)
            row_layout.setSpacing(8)

            toggle = QCheckBox(name)
            toggle.setObjectName("SurfaceToggle")
            toggle.setChecked(layer.visible)
            toggle.stateChanged.connect(
                lambda state, layer=layer: self._set_layer_visible(layer, state == Qt.Checked)
            )
            row_layout.addWidget(toggle)

            row_layout.addStretch(1)

            export_btn = QPushButton("Export")
            export_btn.setObjectName("TertiaryButton")
            export_btn.clicked.connect(lambda _, display=name: self._open_export_dialog(display))
            row_layout.addWidget(export_btn)

            layout.addWidget(row)
            self._surface_toggle_widgets[name] = toggle

        return section

    def _build_case_section(self) -> QWidget:
        section = QFrame()
        section.setObjectName("SidebarSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Patient Library")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        hint = QLabel("Load a prepared case to explore its structures.")
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        list_widget = QListWidget()
        list_widget.setObjectName("CaseList")
        list_widget.setSelectionMode(QListWidget.SingleSelection)
        list_widget.addItems(sorted(self.case_catalog.keys()))
        list_widget.currentTextChanged.connect(self._on_case_selected)
        self._case_list_widget = list_widget
        layout.addWidget(list_widget)

        return section

    def _center_view(self) -> None:
        if not self.viewer:
            return
        self.viewer.dims.ndisplay = 3
        self.viewer.reset_view()

    def _toggle_theme(self) -> None:
        if not self.viewer:
            return
        current = getattr(self.viewer, "theme", "dark")
        next_theme = "light" if current == "dark" else "dark"
        self.viewer.theme = next_theme
        self._update_theme_button_text()

    def _update_theme_button_text(self) -> None:
        if not self.viewer or not self._theme_button:
            return
        current = getattr(self.viewer, "theme", "dark")
        if current == "dark":
            self._theme_button.setText("Use Light Theme")
        else:
            self._theme_button.setText("Use Dark Theme")

    def _set_volume_visible(self, visible: bool) -> None:
        if self.vol_layer is None:
            return
        self.vol_layer.visible = bool(visible)
        if self._volume_toggle_widget:
            blocker = QSignalBlocker(self._volume_toggle_widget)
            try:
                self._volume_toggle_widget.setChecked(self.vol_layer.visible)
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

    def _sidebar_stylesheet(self) -> str:
        return """
QWidget#SidebarRoot {
    background: transparent;
}
QScrollArea#SidebarScroll {
    border: none;
}
QWidget#SidebarContainer {
    background-color: #1a1f2d;
    border-radius: 18px;
    color: #e6e9f0;
}
QFrame#SidebarHeader {
    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e62, stop:1 #1b1f32);
    border-radius: 16px;
}
QLabel#SidebarTitle {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 0.4px;
}
QLabel#SidebarSubtitle {
    color: rgba(255, 255, 255, 0.78);
    font-size: 13px;
}
QLabel#SidebarMeta {
    color: rgba(255, 255, 255, 0.55);
    font-size: 12px;
}
QFrame#SidebarSection {
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
}
QLabel#SectionTitle {
    font-size: 14px;
    font-weight: 600;
    color: #f1f4ff;
}
QLabel#SectionHint {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
}
QPushButton#PrimaryButton {
    background-color: #3a7afe;
    border: none;
    color: #ffffff;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 14px;
}
QPushButton#PrimaryButton:hover {
    background-color: #5a92ff;
}
QPushButton#SecondaryButton {
    background-color: rgba(58, 122, 254, 0.18);
    border: 1px solid rgba(58, 122, 254, 0.5);
    color: #c9d8ff;
    font-weight: 500;
    border-radius: 10px;
    padding: 9px 14px;
}
QPushButton#SecondaryButton:hover {
    background-color: rgba(58, 122, 254, 0.25);
}
QPushButton#TertiaryButton {
    background-color: transparent;
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 8px 12px;
}
QPushButton#TertiaryButton:hover {
    border-color: rgba(58, 122, 254, 0.7);
    color: #ffffff;
}
QCheckBox#AccentCheckBox, QCheckBox#SurfaceToggle {
    spacing: 8px;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.45);
    background-color: rgba(0, 0, 0, 0.15);
}
QCheckBox::indicator:checked {
    border: 1px solid rgba(58, 122, 254, 0.8);
    background-color: rgba(58, 122, 254, 0.8);
}
QCheckBox::indicator:hover {
    border: 1px solid rgba(58, 122, 254, 0.6);
}
QFrame#SurfaceRow {
    background-color: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
}
QFrame#SurfaceRow:hover {
    background-color: rgba(255, 255, 255, 0.04);
}
QListWidget#CaseList {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 6px;
}
QListWidget#CaseList::item {
    padding: 10px;
    border-radius: 10px;
}
QListWidget#CaseList::item:selected {
    background-color: rgba(58, 122, 254, 0.35);
    color: #ffffff;
}
"""

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
