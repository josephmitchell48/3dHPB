from __future__ import annotations

import re
from typing import Dict, Optional, TYPE_CHECKING, Callable, Any, List, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QListWidget,
    QListWidgetItem,
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
)


if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from napari.layers import Layer


class SidebarMixin:
    """Composable sidebar builder for the HPB napari viewer."""

    surface_layers: Dict[str, "Layer"]
    show_controls: bool
    case_catalog: Dict[str, Dict[str, str]]
    case_loader: Optional[Callable[[str], Any]]
    _surface_toggle_widgets: Dict[str, List[QCheckBox]]
    _volume_toggle_widgets: List[QCheckBox]
    _theme_buttons: List[QPushButton]
    _display_mode_buttons: List[QPushButton]
    _case_list_widgets: List[QListWidget]
    _side_docks: List[Any]
    vol_layer: Optional["Layer"]
    current_metrics: Optional[Dict[str, Any]]

    def _build_sidebar_widget(
        self,
        *,
        include_controls: bool = True,
        include_surfaces: bool = True,
        include_cases: bool = True,
        include_metrics: bool = False,
        header_title: str = "HPB Visualizer",
    ) -> QWidget:
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

        handles = self._empty_sidebar_handles()

        layout.addWidget(self._build_header_section(header_title))

        if include_controls and self.show_controls:
            controls, control_handles = self._build_controls_section()
            layout.addWidget(controls)
            self._merge_sidebar_handles(handles, control_handles)

        if include_surfaces and self.surface_layers:
            surfaces, surface_handles = self._build_surface_section()
            layout.addWidget(surfaces)
            self._merge_sidebar_handles(handles, surface_handles)

        if include_metrics:
            metrics_widget = self._build_metrics_section()
            if metrics_widget is not None:
                layout.addWidget(metrics_widget)

        if include_cases and self.case_catalog and self.case_loader:
            cases, case_handles = self._build_case_section()
            layout.addWidget(cases)
            self._merge_sidebar_handles(handles, case_handles)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._register_sidebar_handles(handles)
        return root

    def _build_header_section(self, title_text: str) -> QWidget:
        header = QFrame()
        header.setObjectName("SidebarHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(6)

        title = QLabel(title_text)
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

    def _build_controls_section(self) -> Tuple[QWidget, Dict[str, List[Any]]]:
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

        handles = self._empty_sidebar_handles()

        display_btn = QPushButton()
        display_btn.setObjectName("SecondaryButton")
        display_btn.clicked.connect(self._toggle_display_mode)
        layout.addWidget(display_btn)
        handles["display_buttons"].append(display_btn)

        theme_btn = QPushButton()
        theme_btn.setObjectName("SecondaryButton")
        theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(theme_btn)
        handles["theme_buttons"].append(theme_btn)

        if self.vol_layer is not None:
            volume_toggle = QCheckBox("Show Volume")
            volume_toggle.setObjectName("AccentCheckBox")
            volume_toggle.setChecked(self.vol_layer.visible)
            volume_toggle.stateChanged.connect(
                lambda state: self._set_volume_visible(state == Qt.Checked)
            )
            layout.addWidget(volume_toggle)
            handles["volume_toggles"].append(volume_toggle)

        return section, handles

    def _build_metrics_section(self) -> Optional[QWidget]:
        metrics = getattr(self, "current_metrics", None)
        section = QFrame()
        section.setObjectName("SidebarSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(18, 20, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Tumour Metrics")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        if not metrics:
            empty = QLabel("No tumour metrics available.")
            empty.setObjectName("SectionHint")
            empty.setWordWrap(True)
            layout.addWidget(empty)
            return section

        total_mm3 = float(metrics.get("total_mm3", 0.0))
        total_ml = float(metrics.get("total_ml", total_mm3 / 1000.0 if total_mm3 else 0.0))
        summary = QLabel(f"Total Volume: {total_ml:.2f} mL ({total_mm3:.0f} mm³)")
        summary.setObjectName("SectionHint")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        components = metrics.get("components") or []
        if not components:
            none_label = QLabel("No individual tumours detected.")
            none_label.setObjectName("SectionHint")
            none_label.setWordWrap(True)
            layout.addWidget(none_label)
            return section

        for idx, comp in enumerate(components, start=1):
            if isinstance(comp, dict):
                volume_ml = float(comp.get("volume_ml", 0.0))
                volume_mm3 = float(comp.get("volume_mm3", volume_ml * 1000.0))
                voxels = int(comp.get("voxel_count", 0))
                label_id = comp.get("label_id", idx)
            else:
                volume_ml = float(getattr(comp, "volume_ml", 0.0))
                volume_mm3 = float(getattr(comp, "volume_mm3", volume_ml * 1000.0))
                voxels = int(getattr(comp, "voxel_count", 0))
                label_id = getattr(comp, "label_id", idx)

            desc = QLabel(
                f"Tumour {idx} (label {label_id}): {volume_ml:.2f} mL "
                f"({volume_mm3:.0f} mm³, {voxels} voxels)"
            )
            desc.setObjectName("SectionHint")
            desc.setWordWrap(True)
            layout.addWidget(desc)

        return section

    def _build_surface_section(self) -> Tuple[QWidget, Dict[str, List[Any]]]:
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

        handles = self._empty_sidebar_handles()

        for name in self._sorted_layer_names():
            layer = self.surface_layers.get(name)
            if layer is None:
                continue

            if name == "Hepatic Vessels Old":
                layer.visible = False

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
            handles["surface_toggles"].append((name, toggle))

        return section, handles

    def _build_case_section(self) -> Tuple[QWidget, Dict[str, List[Any]]]:
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
        case_names = sorted(self.case_catalog.keys(), key=self._natural_sort_key)
        print("[sidebar] building case list:", case_names)
        list_widget.blockSignals(True)
        for case_name in case_names:
            sort_key = self._natural_sort_key(case_name)
            print(f"[sidebar] add item '{case_name}' sort_key={sort_key}")
            item = QListWidgetItem(case_name)
            item.setData(Qt.UserRole, case_name)
            list_widget.addItem(item)
        list_widget.blockSignals(False)
        list_widget.currentItemChanged.connect(self._on_case_item_changed)
        layout.addWidget(list_widget)

        handles = self._empty_sidebar_handles()
        handles["case_lists"].append(list_widget)

        return section, handles

    def _empty_sidebar_handles(self) -> Dict[str, List[Any]]:
        return {
            "case_lists": [],
            "volume_toggles": [],
            "theme_buttons": [],
            "surface_toggles": [],
            "display_buttons": [],
        }

    def _merge_sidebar_handles(
        self,
        dest: Dict[str, List[Any]],
        src: Dict[str, List[Any]],
    ) -> None:
        for key, values in src.items():
            dest.setdefault(key, [])
            dest[key].extend(values)

    def _register_sidebar_handles(self, handles: Dict[str, List[Any]]) -> None:
        for widget in handles.get("case_lists", []):
            self._case_list_widgets.append(widget)
        for toggle in handles.get("volume_toggles", []):
            self._volume_toggle_widgets.append(toggle)
        for button in handles.get("theme_buttons", []):
            self._theme_buttons.append(button)
        for button in handles.get("display_buttons", []):
            self._display_mode_buttons.append(button)
        for name, toggle in handles.get("surface_toggles", []):
            self._surface_toggle_widgets.setdefault(name, []).append(toggle)
        self._update_theme_button_text()
        self._update_display_button_text()

    def _on_case_item_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        current_name = current.data(Qt.UserRole) if current else None
        previous_name = previous.data(Qt.UserRole) if previous else None
        widget = current.listWidget() if current else (previous.listWidget() if previous else None)
        has_focus = widget.hasFocus() if widget else False
        print(
            f"[sidebar] item changed: previous={previous_name} current={current_name} "
            f"focus={has_focus}"
        )
        if current is None:
            print("[sidebar] current item is None, ignoring")
            return
        case_name = current.data(Qt.UserRole)
        if case_name is None:
            case_name = current.text()
        if not case_name:
            print("[sidebar] case_name empty, ignoring")
            return
        if widget and not widget.hasFocus():
            print(f"[sidebar] ignoring non-focused change to {case_name}")
            return
        print(f"[sidebar] dispatch _on_case_selected('{case_name}')")
        self._on_case_selected(str(case_name))

    def _find_case_item(self, widget: QListWidget, case_name: str) -> Optional[QListWidgetItem]:
        for idx in range(widget.count()):
            item = widget.item(idx)
            value = item.data(Qt.UserRole)
            if value is None:
                value = item.text()
            if str(value) == case_name:
                return item
        return None

    def _natural_sort_key(self, value: str) -> Tuple[Any, ...]:
        parts = re.split(r"(\d+)", value)
        return tuple(int(part) if part.isdigit() else part.lower() for part in parts)
