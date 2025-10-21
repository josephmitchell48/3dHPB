from __future__ import annotations

from typing import Optional, TYPE_CHECKING, List


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from qtpy.QtWidgets import QPushButton


class ThemeMixin:
    """Shared theme helpers for the HPB napari viewer."""

    viewer: Optional["napari.Viewer"]
    _theme_buttons: List["QPushButton"]

    def _toggle_theme(self) -> None:
        """Toggle between napari light and dark themes."""
        if not getattr(self, "viewer", None):
            return
        current = getattr(self.viewer, "theme", "dark")
        next_theme = "light" if current == "dark" else "dark"
        self.viewer.theme = next_theme
        self._update_theme_button_text()

    def _update_theme_button_text(self) -> None:
        """Refresh the theme toggle button text to match current mode."""
        if not getattr(self, "viewer", None):
            return
        current = getattr(self.viewer, "theme", "dark")
        for button in getattr(self, "_theme_buttons", []):
            button.setText("Use Light Theme" if current == "dark" else "Use Dark Theme")

    def _sidebar_stylesheet(self) -> str:
        """Return the sidebar QSS snippet applied to the dock widget."""
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
