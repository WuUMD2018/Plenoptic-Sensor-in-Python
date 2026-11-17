"""Plenoptic Studio: an interactive, image-driven wavefront laboratory.

The GUI deliberately keeps input construction, sensor inference and display
separate.  The optical model lives in :mod:`plenoptic_studio.optics` and can be
used without a Qt installation.
"""
from __future__ import annotations

import csv
import json
import sys
import threading
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

from .optics import (MODES, Parameters, make_wavefront, simulate,
                     validate_all, validate_vortex_charges, validate_parameters, phase_residual)

BG = "#090f1c"
PANEL = "#101c2e"
TEXT = "#e5edf8"
MUTED = "#8fa4bf"
CYAN = "#50dfd0"
AMBER = "#f3b969"

STYLE = """
QWidget { color: #e5edf8; font-family: 'DejaVu Sans'; font-size: 12px; }
QMainWindow, QDialog { background: #090f1c; }
QFrame#card { background: #101c2e; border: 1px solid #24344b; border-radius: 12px; }
QFrame#banner { background: #101c2e; border: 1px solid #24344b; border-radius: 14px; }
QLabel#eyebrow { color: #50dfd0; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
QLabel#title { color: #f2f6ff; font-size: 29px; font-weight: 700; }
QLabel#subtitle { color: #9eb2ce; font-size: 12px; }
QLabel#section { color: #f2f6ff; font-size: 14px; font-weight: 600; }
QLabel#muted { color: #8fa4bf; font-size: 11px; }
QLabel#metric { font-size: 23px; font-weight: 650; color: #50dfd0; }
QLabel#warning { color: #f3b969; background: #29251f; border-radius: 7px; padding: 9px; font-size: 11px; }
QLabel#okay { color: #77ddbd; background: #142d2f; border-radius: 7px; padding: 9px; font-size: 11px; }
QDoubleSpinBox, QSpinBox, QComboBox { background: #081321; border: 1px solid #32435b; border-radius: 5px; padding: 5px 6px; selection-background-color: #277e83; }
QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #50dfd0; }
QComboBox QAbstractItemView { background-color: #101c2e; color: #e5edf8; selection-background-color: #24535b; selection-color: #e5edf8; border: 1px solid #374a65; outline: 0; }
QComboBox QAbstractItemView::item { min-height: 24px; padding: 3px 6px; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button, QSpinBox::up-button, QSpinBox::down-button { width: 16px; border: none; }
QSlider::groove:horizontal { height: 4px; background: #2a3950; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3baba8; border-radius: 2px; }
QSlider::handle:horizontal { background: #6be6da; border: 3px solid #163c47; width: 10px; height: 10px; margin: -6px 0; border-radius: 8px; }
QPushButton { background: #1c2c43; border: 1px solid #374a65; border-radius: 7px; padding: 9px 12px; font-weight: 600; }
QPushButton:hover { background: #283d58; border-color: #6b93b9; }
QPushButton:pressed { background: #0d1e31; }
QPushButton:disabled { color: #61718a; background: #142035; border-color: #273448; }
QPushButton#primary { color: #062422; background: #50dfd0; border: none; }
QPushButton#primary:hover { background: #83eee4; }
QPushButton#primary:disabled { color: #3d696c; background: #1c4a50; }
QPushButton#sign { padding: 3px; font-size: 15px; }
QPushButton#sign:checked { color: #f3b969; background: #3a2922; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #58708e; border-radius: 4px; background: #0a1628; }
QCheckBox::indicator:checked { background: #50dfd0; border: 3px solid #163d47; }
QTabWidget::pane { border: none; background: #101c2e; }
QTabBar::tab { color: #91a6c3; background: #101c2e; padding: 10px 16px; border-bottom: 2px solid #24344b; }
QTabBar::tab:selected { color: #50dfd0; border-bottom: 2px solid #50dfd0; }
QTabBar::tab:hover { color: #e5edf8; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { width: 7px; background: #101b2c; }
QScrollBar::handle:vertical { background: #3a4d68; border-radius: 3px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar { border: none; background: #21324a; border-radius: 3px; min-height: 5px; max-height: 5px; }
QProgressBar::chunk { background: #50dfd0; border-radius: 3px; }
QTableWidget { background: #101c2e; alternate-background-color: #132238; gridline-color: #28394f; border: 1px solid #293d55; selection-background-color: #24535b; }
QHeaderView::section { background: #18283f; color: #a9bdd5; border: none; border-bottom: 1px solid #334762; padding: 6px; }
QToolBar { background: #101c2e; border: none; spacing: 4px; }
QToolButton { background: transparent; border: none; padding: 3px; }
QToolButton:hover { background: #2b405b; }
QStatusBar { color: #91a6c3; background: #090f1c; }
QToolTip { color: #e5edf8; background: #1b3047; border: 1px solid #4d6d8b; padding: 6px; }
"""

PARAMETERS = (
    ("objective_diameter_mm", "Objective aperture", "mm", 3),
    ("objective_focal_mm", "Objective focal length", "mm", 3),
    ("mla_pitch_um", "Square MLA pitch", "µm", 3),
    ("mla_focal_mm", "MLA focal length", "mm", 4),
    ("beam_diameter_mm", "Input beam diameter", "mm", 3),
    ("phase_ptp_rad", "Zernike peak-to-peak", "rad", 3),
    ("pixel_um", "Camera pixel size", "µm", 4),
    ("wavelength_nm", "Central wavelength", "nm", 3),
    ("bandwidth_nm", "Spectral FWHM", "nm", 4),
)


def label(text, name=None, wrap=False):
    item = QtWidgets.QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(wrap)
    return item


def card():
    item = QtWidgets.QFrame()
    item.setObjectName("card")
    return item


def configure_app(app):
    """Use Matplotlib's bundled font, including on headless Windows hosts."""
    import matplotlib
    font_directory = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for filename in ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"]:
        QtGui.QFontDatabase.addApplicationFont(str(font_directory / filename))
    for family in ["Segoe UI", "Arial", "sans-serif"]:
        QtGui.QFont.insertSubstitution(family, "DejaVu Sans")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QtGui.QFont("DejaVu Sans", 10))


class AnchoredParameter(QtWidgets.QWidget):
    """An editable value and a slider relative to its last explicit entry."""
    value_changed = QtCore.Signal()

    def __init__(self, title, unit, value, decimals=3, parent=None):
        super().__init__(parent)
        self.anchor = float(value)
        self._typed = False
        self.unit = unit
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(2)
        line = QtWidgets.QHBoxLayout()
        line.setSpacing(8)
        line.addWidget(label(title))
        line.addStretch()
        self.box = QtWidgets.QDoubleSpinBox()
        self.box.setDecimals(decimals)
        self.box.setRange(0, 1e9)
        self.box.setValue(value)
        self.box.setSuffix(" " + unit)
        self.box.setKeyboardTracking(False)
        self.box.setFixedWidth(110)
        self.box.setFixedHeight(27)
        self.box.setSingleStep(max(10 ** -decimals, value * .01))
        self.box.setToolTip("Edit this value to set a new 100% slider reference.")
        line.addWidget(self.box)
        layout.addLayout(line)
        lower = QtWidgets.QHBoxLayout()
        lower.setSpacing(6)
        lower.addWidget(label("0", "muted"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 2000)
        self.slider.setValue(1000)
        self.slider.setToolTip("0–200% of your most recently entered value.")
        lower.addWidget(self.slider, 1)
        self.percent = label("100%", "muted")
        self.percent.setFixedWidth(38)
        self.percent.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        lower.addWidget(self.percent)
        layout.addLayout(lower)
        self.slider.valueChanged.connect(self._slider_changed)
        self.box.valueChanged.connect(self._entry_changed)
        self.box.lineEdit().textEdited.connect(self._text_edited)
        self.box.editingFinished.connect(self._finish_entry)

    def _text_edited(self, _text):
        self._typed = True

    def _finish_entry(self):
        # Retyping the displayed value is still an explicit new reference,
        # even though QDoubleSpinBox does not emit valueChanged in that case.
        if self._typed:
            self._typed = False
            self._entry_changed(self.box.value())

    def _slider_changed(self, position):
        blocker = QtCore.QSignalBlocker(self.box)
        self.box.setValue(self.anchor * position / 1000)
        del blocker
        self.percent.setText(f"{position / 10:g}%")
        self.value_changed.emit()

    def _entry_changed(self, value):
        self.anchor = float(value)
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(1000)
        del blocker
        self.percent.setText("100%")
        self.value_changed.emit()

    def value(self):
        return self.box.value()

    def reset(self, value):
        self.box.setValue(value)
        self._entry_changed(value)


class VortexChargeControl(QtWidgets.QWidget):
    """An absolute integer charge, independent of the Zernike amplitude slider."""
    value_changed = QtCore.Signal()

    def __init__(self, value=0, parent=None):
        super().__init__(parent)
        self.setObjectName("vortex_charge")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)
        line = QtWidgets.QHBoxLayout()
        line.addWidget(label("Vortex charge ℓ", "section"), 1)
        self.box = QtWidgets.QSpinBox()
        self.box.setRange(0, 6)
        self.box.setSingleStep(1)
        self.box.setValue(int(value))
        self.box.setFixedWidth(72)
        self.box.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        line.addWidget(self.box)
        layout.addLayout(line)
        line = QtWidgets.QHBoxLayout()
        line.addWidget(label("0 · off", "muted"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 6)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setValue(int(value))
        line.addWidget(self.slider, 1)
        line.addWidget(label("6", "muted"))
        layout.addLayout(line)
        layout.addWidget(label("Adds ℓθ phase to the Gaussian beam.\nZernike amplitude does not change the charge.", "muted", True))
        self.setToolTip("Integer topological charge 0–6. Zero disables the vortex. The Gaussian intensity is unchanged; this is a phase-only vortex, not a Laguerre–Gaussian amplitude profile.")
        self.slider.valueChanged.connect(self._slide)
        self.box.valueChanged.connect(self._edit)

    def _slide(self, value):
        blocker = QtCore.QSignalBlocker(self.box)
        self.box.setValue(value)
        del blocker
        self.value_changed.emit()

    def _edit(self, value):
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(value)
        del blocker
        self.value_changed.emit()

    def value(self):
        return self.box.value()

    def reset(self, value=0):
        self.box.setValue(int(value))


class MetricCard(QtWidgets.QFrame):
    def __init__(self, heading, value="—", foot="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.heading_label = label(heading.upper(), "eyebrow")
        layout.addWidget(self.heading_label)
        self.value_label = label(value, "metric")
        layout.addWidget(self.value_label)
        self.foot_label = label(foot, "muted", True)
        layout.addWidget(self.foot_label)


class ScaledImageLabel(QtWidgets.QLabel):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.source = QtGui.QPixmap(str(path))
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.source.isNull():
            self.setPixmap(self.source.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                              QtCore.Qt.TransformationMode.SmoothTransformation))


def style_axes(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor("#0a1525")
    ax.set_title(title, color=TEXT, loc="left", fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_color("#30425c")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


class ResultCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 6), dpi=100, facecolor=PANEL)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(550, 450)
        self.current_result = None
        self.empty()

    def empty(self):
        self.fig.clear()
        titles = ["01  PLENOPTIC CAMERA", "02  INPUT PHASE", "03  RECONSTRUCTED PHASE", "04  PHASE RESIDUAL"]
        for index, title in enumerate(titles):
            ax = self.fig.add_subplot(2, 2, index + 1)
            style_axes(ax, title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(.5, .55, "Ready to explore", color="#60738e", ha="center", transform=ax.transAxes, fontsize=12)
            ax.text(.5, .43, "Set the input, then run the sensor", color="#455b79", ha="center", transform=ax.transAxes, fontsize=9)
        self.fig.subplots_adjust(left=.07, right=.96, bottom=.075, top=.93, hspace=.38, wspace=.28)
        self.draw_idle()

    def show_result(self, result, title_suffix=""):
        self.current_result = result
        self.fig.clear()
        wf = result.wavefront
        mask = np.asarray(wf.mask, dtype=bool)
        wrapped = bool(result.metadata.get("wrapped_phase_comparison", False))
        residual = np.where(mask, phase_residual(wf.phase, result.reconstructed_phase,
                                                mask, wrapped=wrapped), np.nan)
        if wrapped:
            # Match the core's circular global-phase gauge. Wrapping each image
            # independently and then subtracting would create false 2π errors.
            truth = np.where(mask, np.angle(np.exp(1j * wf.phase)), np.nan)
            rec = np.angle(np.exp(1j * (truth + residual)))
        else:
            truth = np.where(mask, wf.phase, np.nan)
            rec = np.where(mask, result.reconstructed_phase, np.nan)
            if np.any(mask):
                truth = truth - np.nanmean(truth)
                rec = rec - np.nanmean(rec)
        extent = [float(wf.x_mm[0]), float(wf.x_mm[-1]), float(wf.y_mm[0]), float(wf.y_mm[-1])]
        phase_limit = np.pi if wrapped else max(float(np.nanmax(np.abs(truth))) if np.any(mask) else 0,
                                                float(np.nanmax(np.abs(rec))) if np.any(mask) else 0, 1e-6)
        suffix = " · WRAPPED" if wrapped else ""
        fields = [(truth, "02  INPUT PHASE" + suffix, phase_limit),
                  (rec, "03  RECONSTRUCTED PHASE" + suffix, phase_limit),
                  (residual, "04  CIRCULAR RESIDUAL" if wrapped else "04  PHASE RESIDUAL",
                   max(float(np.nanmax(np.abs(residual))), 1e-6))]
        camera = self.fig.add_subplot(2, 2, 1)
        image_data = np.maximum(np.asarray(result.image, dtype=float), 0)
        normed = image_data / max(float(image_data.max()), 1e-30)
        im = camera.imshow(np.log10(1 + 999 * normed), origin="lower", cmap="magma", vmin=0, vmax=3,
                           interpolation="nearest", aspect="equal")
        style_axes(camera, "01  PLENOPTIC IMAGE · LOG", "Camera x (pixel)", "Camera y (pixel)")
        cbar = self.fig.colorbar(im, ax=camera, fraction=.044, pad=.035, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["0", ".009", ".099", "1"])
        cbar.set_label("Relative intensity", color=MUTED, fontsize=8)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        cbar.outline.set_edgecolor("#30425c")
        for position, (data, name, limit) in enumerate(fields, 2):
            ax = self.fig.add_subplot(2, 2, position)
            artist = ax.imshow(data, extent=extent, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            style_axes(ax, name, "Pupil x (mm)", "Pupil y (mm)")
            cb = self.fig.colorbar(artist, ax=ax, fraction=.044, pad=.035)
            cb.set_label("Phase (rad)", color=MUTED, fontsize=8)
            cb.ax.tick_params(colors=MUTED, labelsize=7)
            cb.outline.set_edgecolor("#30425c")
        self.fig.subplots_adjust(left=.065, right=.95, bottom=.075, top=.88, hspace=.43, wspace=.35)
        if title_suffix:
            self.fig.suptitle(title_suffix, color=MUTED, fontsize=9, y=.995)
        self.draw_idle()


class CoefficientRow(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, mode, value, parent=None):
        super().__init__(parent)
        self.mode = mode
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        display_name = {12: "2nd vertical astig.", 13: "2nd oblique astig."}.get(mode.j, mode.name)
        title = label(f"{mode.j:02d}  {display_name}")
        title.setMinimumWidth(155)
        title.setToolTip(f"{mode.name}\nNoll j={mode.j}; radial n={mode.n}, azimuthal m={mode.m}")
        row.addWidget(title, 1)
        self.sign = QtWidgets.QPushButton("−" if value < 0 else "+")
        self.sign.setObjectName("sign")
        self.sign.setCheckable(True)
        self.sign.setChecked(bool(value < 0))
        self.sign.setFixedSize(25, 27)
        self.sign.setToolTip("Coefficient sign. Slider magnitude remains 0–100%.")
        row.addWidget(self.sign)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(round(abs(value) * 1000))
        self.slider.setMinimumWidth(70)
        row.addWidget(self.slider, 1)
        self.box = QtWidgets.QDoubleSpinBox()
        self.box.setRange(0, 100)
        self.box.setDecimals(1)
        self.box.setSuffix("%")
        self.box.setValue(abs(value) * 100)
        self.box.setKeyboardTracking(False)
        self.box.setFixedWidth(72)
        row.addWidget(self.box)
        self.slider.valueChanged.connect(self._slide)
        self.box.valueChanged.connect(self._edit)
        self.sign.toggled.connect(self._signed)

    def _slide(self, value):
        blocker = QtCore.QSignalBlocker(self.box)
        self.box.setValue(value / 10)
        del blocker
        self.changed.emit()

    def _edit(self, value):
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(round(value * 10))
        del blocker
        self.changed.emit()

    def _signed(self, checked):
        self.sign.setText("−" if checked else "+")
        self.changed.emit()

    def value(self):
        return self.box.value() / 100 * (-1 if self.sign.isChecked() else 1)

    def reset(self, value=0):
        self.sign.setChecked(bool(value < 0))
        self.box.setValue(abs(value) * 100)


class WavefrontDialog(QtWidgets.QDialog):
    def __init__(self, params, weights, parent=None):
        super().__init__(parent)
        self.params = params
        self.live = False
        self.ax = None
        self.setWindowTitle("Input composer · Gaussian vortex + Zernike modes")
        self.resize(1120, 820)
        self.setMinimumSize(920, 620)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.addWidget(label("WAVEFRONT COMPOSER", "eyebrow"))
        layout.addWidget(label("Build the distortion", "title"))
        layout.addWidget(label(
            f"Zernike mixture: {params.phase_ptp_rad:g} rad peak-to-peak. "
            f"Vortex charge ℓ = {params.vortex_charge} adds an independent ℓθ phase.",
            "subtitle", True))
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = card()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.addWidget(label("NOLL 01–21   /   MAGNITUDE 0–100%", "eyebrow"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        rows = QtWidgets.QVBoxLayout(body)
        rows.setContentsMargins(0, 2, 4, 2)
        rows.setSpacing(4)
        self.rows = []
        for mode, value in zip(MODES, weights):
            item = CoefficientRow(mode, value)
            item.changed.connect(self._changed)
            rows.addWidget(item)
            self.rows.append(item)
        rows.addStretch()
        scroll.setWidget(body)
        ll.addWidget(scroll, 1)
        self.reset_button = QtWidgets.QPushButton("Clear all coefficients")
        self.reset_button.clicked.connect(self._clear)
        ll.addWidget(self.reset_button)
        split.addWidget(left)
        right = card()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(10, 12, 10, 10)
        rl.addWidget(label("ROTATABLE PHASE SURFACE", "eyebrow"))
        self.figure = Figure(figsize=(5.6, 5.6), dpi=100, facecolor=PANEL)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(340, 330)
        rl.addWidget(self.canvas, 1)
        self.note = label("Click View to reveal the phase. Drag the surface to rotate; coefficient changes then update live.", "muted", True)
        rl.addWidget(self.note)
        self.view_button = QtWidgets.QPushButton("View · enable live preview")
        self.view_button.setObjectName("primary")
        self.view_button.clicked.connect(self._enable_view)
        rl.addWidget(self.view_button)
        split.addWidget(right)
        split.setSizes([480, 580])
        layout.addWidget(split, 1)
        layout.addWidget(label("Piston (Noll 01) is unobservable and removed. Clear coefficients removes the Zernike phase; the vortex charge is set in the main window.", "muted", True))
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(label("Gaussian intensity σ = beam diameter / 4", "muted"), 1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        bottom.addWidget(cancel)
        finish = QtWidgets.QPushButton("Finish setup")
        finish.setObjectName("primary")
        finish.clicked.connect(self.accept)
        bottom.addWidget(finish)
        layout.addLayout(bottom)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._draw)
        self._placeholder()

    @property
    def weights(self):
        return np.array([row.value() for row in self.rows])

    def _placeholder(self):
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        ax.text(.5, .55, "Your wavefront starts here", ha="center", color=MUTED, transform=ax.transAxes, fontsize=15)
        ax.text(.5, .45, "Mix modes. Set the shape. Inspect in 3D.", ha="center", color="#506985", transform=ax.transAxes, fontsize=10)
        self.canvas.draw_idle()

    def _clear(self):
        for row in self.rows:
            row.reset()
        self._changed()

    def _changed(self):
        if self.live:
            self.timer.start()

    def _enable_view(self):
        self.live = True
        self.view_button.setText("View · live preview active")
        self._draw()

    def _draw(self):
        try:
            wf = make_wavefront(self.params, self.weights, resolution=96)
        except Exception as exc:
            self.note.setText(str(exc))
            return
        elev, azim = (self.ax.elev, self.ax.azim) if self.ax else (28, -55)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        ax = self.ax
        x, y = np.meshgrid(wf.x_mm, wf.y_mm)
        wrapped = self.params.vortex_charge != 0
        phase = np.angle(np.exp(1j * wf.phase)) if wrapped else wf.phase
        z = np.where(wf.mask, phase, np.nan)
        if wrapped:
            # Leave a gap at each plotting branch cut rather than draw a false
            # continuous surface between opposite sides of the principal phase.
            cuts = np.zeros_like(wf.mask)
            cuts[:, 1:] |= np.abs(np.diff(phase, axis=1)) > np.pi
            cuts[1:, :] |= np.abs(np.diff(phase, axis=0)) > np.pi
            z[cuts] = np.nan
        peak = float(np.nanmax(z) - np.nanmin(z))
        surface_sampling = {"rstride": 1, "cstride": 1} if wrapped else {"rcount": 80, "ccount": 80}
        ax.plot_surface(x, y, z, cmap="turbo", linewidth=0, antialiased=True, **surface_sampling)
        ax.view_init(elev=elev, azim=azim)
        ax.set_facecolor(PANEL)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((.06, .11, .18, 1))
            axis._axinfo["grid"]["color"] = (.2, .3, .4, .45)
            axis.line.set_color("#415a79")
        ax.tick_params(colors=MUTED, labelsize=8, pad=0)
        ax.set_xlabel("x (mm)", color=MUTED, fontsize=9, labelpad=2)
        ax.set_ylabel("y (mm)", color=MUTED, fontsize=9, labelpad=2)
        ax.set_zlabel("Wrapped phase (rad)" if wrapped else "Phase (rad)", color=MUTED, fontsize=9, labelpad=4)
        ax.set_box_aspect((1, 1, .65))
        self.figure.subplots_adjust(left=0, right=1, bottom=.03, top=.98)
        if wrapped:
            self.note.setText(f"ℓ = {self.params.vortex_charge} · phase wrapped to ±π for display; gaps are plotting branch cuts. "
                              "Drag to rotate. Zernike amplitude scales only the smooth phase.")
        else:
            self.note.setText(f"Actual peak-to-peak: {peak:.3f} rad · drag to rotate. Relative weights set the shape; Zernike amplitude sets its scale.")
        self.canvas.draw_idle()


class SimulationWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    result = QtCore.Signal(int, object)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, params, weights, validation=False, validation_kind="zernike"):
        super().__init__()
        self.params = params
        self.weights = np.array(weights, copy=True)
        self.validation = validation
        self.validation_kind = validation_kind
        self.cancel_event = threading.Event()

    @QtCore.Slot()
    def run(self):
        try:
            if self.validation:
                validator = validate_vortex_charges if self.validation_kind == "vortex" else validate_all
                import inspect
                if "result_callback" in inspect.signature(validator).parameters:
                    validator(self.params, progress=self.progress.emit,
                              cancel=self.cancel_event.is_set,
                              result_callback=lambda index, result: self.result.emit(index + 1, result))
                else:
                    results = validator(self.params, progress=self.progress.emit,
                                        cancel=self.cancel_event.is_set)
                    for index, result in enumerate(results, 1):
                        self.result.emit(index, result)
            else:
                result = simulate(self.params, self.weights, progress=self.progress.emit,
                                  cancel=self.cancel_event.is_set)
                self.result.emit(0, result)
        except Exception as exc:
            if self.cancel_event.is_set():
                self.progress.emit("Cancelled. Completed results remain available.")
            else:
                self.error.emit(str(exc))
        finally:
            self.finished.emit()


class BeamProfileDialog(QtWidgets.QDialog):
    """Show the exact Gaussian intensity and pupil support used by the core."""

    def __init__(self, params, weights, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaussian input beam")
        self.resize(880, 540)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(label("INPUT AMPLITUDE", "eyebrow"))
        layout.addWidget(label(f"Gaussian intensity · σ = {params.beam_diameter_mm / 4:g} mm", "section"))
        layout.addWidget(label("I(x,y) = exp[−(x²+y²)/(2σ²)] inside the circular transmitted pupil. "
                               f"Vortex charge ℓ = {params.vortex_charge} changes phase, not this Gaussian intensity profile; "
                               "the singular center has undefined phase.", "muted", True))
        wf = make_wavefront(params, weights, resolution=128)
        fig = Figure(figsize=(8.2, 4), dpi=100, facecolor=PANEL)
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas, 1)
        ax = fig.add_subplot(121)
        extent = [wf.x_mm[0], wf.x_mm[-1], wf.y_mm[0], wf.y_mm[-1]]
        ax.imshow(wf.intensity, origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1)
        style_axes(ax, "GAUSSIAN INTENSITY", "x (mm)", "y (mm)")
        profile = fig.add_subplot(122)
        profile.plot(wf.x_mm, wf.intensity[len(wf.y_mm)//2], color=CYAN, linewidth=2)
        profile.fill_between(wf.x_mm, wf.intensity[len(wf.y_mm)//2], color=CYAN, alpha=.12)
        profile.set_ylim(0, 1.07)
        profile.axvline(-params.beam_diameter_mm/4, color=AMBER, linewidth=.8, linestyle="--")
        profile.axvline(params.beam_diameter_mm/4, color=AMBER, linewidth=.8, linestyle="--")
        style_axes(profile, "CENTRAL CROSS SECTION · ±σ", "x (mm)", "Normalized intensity")
        fig.subplots_adjust(left=.08, right=.97, bottom=.16, top=.88, wspace=.38)
        done = QtWidgets.QPushButton("Close beam preview")
        done.clicked.connect(self.accept)
        layout.addWidget(done)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plenoptic Studio · laser wavefront laboratory")
        self.resize(1480, 1040)
        self.setMinimumSize(1120, 780)
        self.weights = np.zeros(21)
        self.weights[3], self.weights[5], self.weights[7] = .6, .25, .15
        self.current_result = None
        self.single_result = None
        self._validation_kind = "zernike"
        self.validation_results_by_kind = {"zernike": {}, "vortex": {}}
        self.thread = None
        self.worker = None
        self.busy = False
        self.last_run_params = None
        self.validation_params = None
        self.last_error = None
        self.last_run_weights = self.weights.copy()
        self._build()
        self._parameters_changed()

    def _build(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 8)
        layout.setSpacing(12)
        banner = QtWidgets.QFrame()
        self.banner = banner
        banner.setObjectName("banner")
        banner.setFixedHeight(138)
        bl = QtWidgets.QHBoxLayout(banner)
        bl.setContentsMargins(22, 10, 12, 10)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(3)
        text.addWidget(label("COMPUTATIONAL OPTICS  /  PLENOPTIC SENSING", "eyebrow"))
        text.addWidget(label("Plenoptic Studio", "title"))
        text.addWidget(label("A laser wavefront laboratory", "subtitle"))
        text.addWidget(label("Shape the beam. Observe the sensor. Recover the phase.", "muted"))
        bl.addLayout(text, 4)
        asset = Path(__file__).parent / "assets" / "optics_bench.png"
        if asset.exists():
            self.bench = ScaledImageLabel(asset)
        else:
            self.bench = label("")
            self.bench.setText("OBJECTIVE      →      SQUARE MLA      →      CAMERA\n\nShared-focus Keplerian sensor · ideal thin lenses")
            self.bench.setStyleSheet("color:#50dfd0; font-size:13px;")
        self.bench.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(self.bench, 6)
        layout.addWidget(banner)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setHandleWidth(10)
        split.setChildrenCollapsible(False)
        split.addWidget(self._controls())
        split.addWidget(self._workspace())
        split.setSizes([345, 1090])
        layout.addWidget(split, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.status_text = label("Ready · set an input or run the initial mixture.", "muted", True)
        bottom.addWidget(self.status_text, 1)
        self.cancel_button = QtWidgets.QPushButton("Cancel computation")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel)
        bottom.addWidget(self.cancel_button)
        layout.addLayout(bottom)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    def _controls(self):
        panel = card()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(440)
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(8)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label("Optical controls", "section"))
        heading.addStretch()
        self.defaults_button = QtWidgets.QPushButton("Reset")
        self.defaults_button.setStyleSheet("padding:4px 9px; font-size:11px;")
        self.defaults_button.clicked.connect(self._reset)
        heading.addWidget(self.defaults_button)
        outer.addLayout(heading)
        self.vortex_control = VortexChargeControl()
        self.vortex_control.value_changed.connect(self._parameters_changed)
        outer.addWidget(self.vortex_control)
        outer.addWidget(label("Physical-parameter sliders span 0–200% of the last typed value.\nVortex charge uses fixed integer steps.", "muted", True))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(body)
        form.setContentsMargins(0, 1, 4, 0)
        form.setSpacing(4)
        self.controls = {"vortex_charge": self.vortex_control}
        defaults = Parameters()
        for key, title, unit, decimals in PARAMETERS:
            widget = AnchoredParameter(title, unit, getattr(defaults, key), decimals)
            widget.setObjectName(key)
            widget.value_changed.connect(self._parameters_changed)
            self.controls[key] = widget
            form.addWidget(widget)
        form.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.f_status = label("", "okay", True)
        outer.addWidget(self.f_status)
        self.setup_button = QtWidgets.QPushButton("Setup phase · vortex + Zernike modes")
        self.setup_button.clicked.connect(self._setup)
        outer.addWidget(self.setup_button)
        self.beam_button = QtWidgets.QPushButton("Preview Gaussian input beam")
        self.beam_button.setStyleSheet("padding:6px 10px; font-size:11px;")
        self.beam_button.clicked.connect(self._beam)
        outer.addWidget(self.beam_button)
        self.run_button = QtWidgets.QPushButton("Run plenoptic sensor  →")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run)
        outer.addWidget(self.run_button)
        self.validate_mode = QtWidgets.QCheckBox("Enable validation sweep")
        self.validate_mode.toggled.connect(self._validation_toggled)
        outer.addWidget(self.validate_mode)
        self.validation_kind = QtWidgets.QComboBox()
        self.validation_kind.addItem("Zernike modes (21)", "zernike")
        self.validation_kind.addItem("Vortex charges (1–6)", "vortex")
        self.validation_kind.setToolTip("Zernike: validate each Noll mode with the selected vortex charge.\n"
                                       "Vortex: sweep pure Gaussian vortices 1–6 with zero Zernike phase; "
                                       "the main charge slider and Zernike mixture are ignored. Other optical settings are preserved.")
        self.validation_kind.currentIndexChanged.connect(self._validation_kind_changed)
        outer.addWidget(self.validation_kind)
        self.validate_button = QtWidgets.QPushButton("Validate all 21 modes")
        self.validate_button.setEnabled(False)
        self.validate_button.clicked.connect(self._validate)
        outer.addWidget(self.validate_button)
        outer.addWidget(label("Gaussian intensity, σ = diameter / 4.\nInteger vortex phase is added to the Zernike mixture.", "muted", True))
        return panel

    def _workspace(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        metrics = QtWidgets.QHBoxLayout()
        metrics.setSpacing(10)
        self.similarity_card = MetricCard("Phase similarity", "—", "Piston removed · amplitude-sensitive")
        self.similarity_card.setToolTip("Synthetic same-model comparison, not experimental accuracy. Vortices use circular RMSE divided by π; non-vortex results use RMSE divided by truth RMS. Recovered charge is reported separately. Ground truth is used only to score the image-derived reconstruction.")
        self.error_card = MetricCard("Phase RMSE", "—", "Radians across the illuminated pupil")
        self.geometry_card = MetricCard("Objective / MLA", "—", "f-number ratio · requested target > 1")
        for item in [self.similarity_card, self.error_card, self.geometry_card]:
            metrics.addWidget(item, 1)
        layout.addLayout(metrics)
        results = card()
        rl = QtWidgets.QVBoxLayout(results)
        rl.setContentsMargins(10, 6, 10, 10)
        rl.setSpacing(5)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._tab_changed)
        self.single_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.single_tab, "Sensor & reconstruction")
        self.validation_tab = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(self.validation_tab)
        vl.setContentsMargins(4, 8, 4, 4)
        self.validation_table = QtWidgets.QTableWidget(21, 4)
        self.validation_table.setHorizontalHeaderLabels(["Noll mode", "Similarity", "RMSE (rad)", "State"])
        self.validation_table.verticalHeader().setVisible(False)
        self.validation_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.validation_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.validation_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.validation_table.setAlternatingRowColors(True)
        self.validation_table.setMaximumHeight(160)
        self.validation_table.setMinimumHeight(72)
        self.validation_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in range(1, 4):
            self.validation_table.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for i, mode in enumerate(MODES):
            self.validation_table.setRowHeight(i, 25)
            for col, value in enumerate([f"{mode.j:02d}  {mode.name}", "—", "—", "Waiting"]):
                self.validation_table.setItem(i, col, QtWidgets.QTableWidgetItem(value))
        self.validation_table.itemSelectionChanged.connect(self._select_validation)
        vl.addWidget(self.validation_table)
        self.validation_hint = label("Select a completed mode to inspect its camera image, reconstruction and residual below.", "muted", True)
        vl.addWidget(self.validation_hint)
        self.tabs.addTab(self.validation_tab, "21-mode validation")
        self.tabs.setMaximumHeight(225)
        self.tabs.setMinimumHeight(42)
        self.tabs.setFixedHeight(42)
        rl.addWidget(self.tabs)
        self.canvas = ResultCanvas()
        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plot_scroll.setMinimumHeight(200)
        self.plot_scroll.setWidget(self.canvas)
        rl.addWidget(self.plot_scroll, 1)
        footer = QtWidgets.QHBoxLayout()
        self.plot_label = label("Model: scalar paraxial optics · known Gaussian amplitude", "muted", True)
        footer.addWidget(self.plot_label, 1)
        self.export_button = QtWidgets.QPushButton("Export results")
        self.export_button.setStyleSheet("padding:6px 10px; font-size:11px;")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        footer.addWidget(self.export_button)
        rl.addLayout(footer)
        layout.addWidget(results, 1)
        self.result_warning = label("Reconstruction uses the simulated camera image. Ground truth is used only to measure reconstruction error.", "muted", True)
        layout.addWidget(self.result_warning)
        return widget

    def parameters(self):
        return Parameters(**{key: control.value() for key, control in self.controls.items()})

    def _parameters_changed(self):
        if not hasattr(self, "f_status"):
            return
        p = self.parameters()
        try:
            validate_parameters(p)
            valid = True
        except ValueError as exc:
            valid = False
            self.f_status.setText(str(exc))
            self.f_status.setObjectName("warning")
        if valid:
            fobj = p.objective_focal_mm / p.objective_diameter_mm
            fmla = p.mla_focal_mm / (p.mla_pitch_um * .001)
            ratio = fobj / fmla
            self.geometry_card.value_label.setText(f"{ratio:.2f}×")
            self.geometry_card.foot_label.setText(f"Objective f/{fobj:.2f}  ·  MLA f/{fmla:.2f}")
            if fobj <= fmla:
                self.f_status.setObjectName("warning")
                match = "Matched boundary; no margin." if np.isclose(fobj, fmla) else "Pupil microimages can overlap."
                self.f_status.setText(f"⚠  f/{fobj:.2f} ≤ f/{fmla:.2f}\n{match} Potential reconstruction inaccuracy; requested condition is Fobjective > FMLA.")
            else:
                self.f_status.setObjectName("okay")
                self.f_status.setText(f"✓  f/{fobj:.2f} > f/{fmla:.2f}\nObjective f-number exceeds the MLA cell f-number.")
        else:
            self.geometry_card.value_label.setText("—")
            self.geometry_card.foot_label.setText("Enter positive physical dimensions")
        self.f_status.style().unpolish(self.f_status)
        self.f_status.style().polish(self.f_status)
        self.run_button.setEnabled(valid and not self.busy)
        self.setup_button.setEnabled(valid and not self.busy)
        self.beam_button.setEnabled(valid and not self.busy)
        self.validate_button.setEnabled(valid and not self.busy and self.validate_mode.isChecked())
        if self.current_result is not None and not self.busy:
            self.status_text.setText("Parameters edited · displayed results are from the previous run. Run again to update.")

    def _reset(self):
        defaults = Parameters()
        for key, control in self.controls.items():
            control.reset(getattr(defaults, key))
        self._parameters_changed()

    def _setup(self):
        dialog = WavefrontDialog(self.parameters(), self.weights, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.weights = dialog.weights
            active = int(np.count_nonzero(self.weights[1:]))
            p = self.parameters()
            self.status_text.setText(f"Input saved · ℓ = {p.vortex_charge} · {active} non-piston modes · "
                                     f"{p.phase_ptp_rad:g} rad Zernike peak-to-peak. Run to update the sensor.")

    def _beam(self):
        BeamProfileDialog(self.parameters(), self.weights, self).exec()

    @property
    def validation_results(self):
        """Keep both sweep snapshots while retaining the original GUI API."""
        return self.validation_results_by_kind[self._validation_kind]

    @validation_results.setter
    def validation_results(self, results):
        self.validation_results_by_kind[self._validation_kind] = results

    def _validation_name(self, index, kind=None):
        kind = self._validation_kind if kind is None else kind
        return f"Pure Gaussian vortex · ℓ = {index}" if kind == "vortex" else f"Noll {index:02d} · {MODES[index - 1].name}"

    def _update_validation_row(self, index, result):
        row = index - 1
        if not 0 <= row < self.validation_table.rowCount():
            return
        sim = result.similarity_pct
        if self._validation_kind == "vortex":
            recovered = result.metadata.get("recovered_vortex_charge")
            match = recovered == index
            state = "Charge match" if match else "Charge mismatch"
            if result.warnings:
                state += " · review warnings"
            values = [str(recovered) if recovered is not None else "—",
                      "N/A" if sim is None else f"{sim:.1f}%", f"{result.rms_error_rad:.4g}", state]
        else:
            vortex = int(result.metadata.get("input_vortex_charge", 0))
            state = ("Piston removed; vortex retained" if index == 1 and vortex else
                     "Piston unobservable" if index == 1 else
                     "Review warnings" if result.warnings else "Complete")
            values = [("N/A · piston" if index == 1 else "N/A · flat") if sim is None else f"{sim:.1f}%",
                      f"{result.rms_error_rad:.4g}", state]
        for column, value in enumerate(values, 1):
            self.validation_table.item(row, column).setText(value)

    def _populate_validation_table(self):
        vortex = self._validation_kind == "vortex"
        blocker = QtCore.QSignalBlocker(self.validation_table)
        self.validation_table.clearContents()
        self.validation_table.setColumnCount(5 if vortex else 4)
        self.validation_table.setRowCount(6 if vortex else 21)
        headers = ["Input ℓ", "Recovered ℓ", "Circular score", "RMSE (rad)", "State"] if vortex else ["Noll mode", "Similarity", "RMSE (rad)", "State"]
        self.validation_table.setHorizontalHeaderLabels(headers)
        stretch_column = 4 if vortex else 0
        for col in range(self.validation_table.columnCount()):
            resize_mode = (QtWidgets.QHeaderView.ResizeMode.Stretch if col == stretch_column else
                           QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            self.validation_table.horizontalHeader().setSectionResizeMode(col, resize_mode)
        for row in range(self.validation_table.rowCount()):
            name = str(row + 1) if vortex else f"{row + 1:02d}  {MODES[row].name}"
            values = [name] + ["—"] * (self.validation_table.columnCount() - 2) + ["Waiting"]
            self.validation_table.setRowHeight(row, 20 if vortex else 25)
            for col, value in enumerate(values):
                self.validation_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        for index, result in self.validation_results.items():
            self._update_validation_row(index, result)
        if self.validation_results:
            self.validation_table.selectRow(max(self.validation_results) - 1)
        else:
            self.validation_table.clearSelection()
            self.validation_table.setCurrentCell(-1, -1)
        del blocker

    def _validation_kind_changed(self, _index):
        if self.busy:
            blocker = QtCore.QSignalBlocker(self.validation_kind)
            self.validation_kind.setCurrentIndex(self.validation_kind.findData(self._validation_kind))
            del blocker
            return
        self._validation_kind = self.validation_kind.currentData()
        vortex = self._validation_kind == "vortex"
        self.validate_button.setText("Validate vortex charges 1–6" if vortex else "Validate all 21 modes")
        self.tabs.setTabText(1, "Vortex validation · 1–6" if vortex else "21-mode validation")
        self.validation_hint.setText("Pure Gaussian vortices 1–6; Zernike phase is zero. Charge match and circular score are separate checks." if vortex else
                                     "Select a completed mode to inspect its camera image, reconstruction and residual below.")
        self._populate_validation_table()
        self._update_validation_layout()
        if self.tabs.currentIndex() == 1:
            self._select_validation()
        self._parameters_changed()

    def _validation_toggled(self, checked):
        self.tabs.setCurrentIndex(1 if checked else 0)
        self._parameters_changed()

    def _update_validation_layout(self):
        if not hasattr(self, "validation_hint"):
            return
        compact = self.height() < 900
        vortex = self._validation_kind == "vortex"
        # Six vortex rows and their scope note stay visible at the minimum
        # window size; the larger Zernike table retains its compact scroll view.
        self.validation_hint.setVisible(vortex or not compact)
        expanded_height = 225 if vortex or not compact else 170
        self.tabs.setFixedHeight(expanded_height if self.tabs.currentIndex() == 1 else 42)

    def _tab_changed(self, index):
        self._update_validation_layout()
        if index == 1:
            self._select_validation()
        elif self.single_result is not None:
            self._display(self.single_result, "Custom mixture")
        elif hasattr(self, "canvas"):
            self._clear_display()

    def _run(self):
        self.tabs.setCurrentIndex(0)
        self.last_run_params = self.parameters()
        self.last_run_weights = self.weights.copy()
        self._start(False)

    def _validate(self):
        if self.busy:
            return
        self.validation_results = {}
        self._clear_display()
        self.validation_params = self.parameters()
        self._populate_validation_table()
        self.tabs.setCurrentIndex(1)
        self._start(True)

    def _start(self, validation):
        if self.busy:
            return
        self.busy = True
        self.last_error = None
        self._parameters_changed()
        for control in self.controls.values():
            control.setEnabled(False)
        self.defaults_button.setEnabled(False)
        self.validate_mode.setEnabled(False)
        self.validation_kind.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_text.setText("Preparing the optical model…")
        self.thread = QtCore.QThread(self)
        self.worker = SimulationWorker(self.parameters(), self.weights, validation, self._validation_kind)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.status_text.setText)
        self.worker.result.connect(self._receive_result)
        self.worker.error.connect(self._error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @QtCore.Slot(int, object)
    def _receive_result(self, index, result):
        if index == 0:
            result.metadata["relative_weights"] = self.last_run_weights.tolist()
            self.single_result = result
            if self.tabs.currentIndex() == 0:
                self._display(result, "Custom mixture")
        else:
            kind = result.metadata.get("validation_kind", self._validation_kind)
            index = int(result.metadata.get("validation_charge" if kind == "vortex" else "mode_j", index))
            self.validation_results_by_kind[kind][index] = result
            if kind != self._validation_kind:
                return
            self._update_validation_row(index, result)
            self.validation_table.selectRow(index - 1)
            if self.tabs.currentIndex() == 1:
                self._display(result, self._validation_name(index))

    def _display(self, result, name):
        self.current_result = result
        self.canvas.show_result(result, name)
        similarity = result.similarity_pct
        wrapped = bool(result.metadata.get("wrapped_phase_comparison", False))
        self.similarity_card.heading_label.setText("CIRCULAR PHASE SCORE" if wrapped else "PHASE SIMILARITY")
        self.similarity_card.value_label.setText("N/A" if similarity is None else f"{similarity:.1f}%")
        self.similarity_card.foot_label.setText("Piston / flat phase is unobservable" if similarity is None else
                                               "100 × max(0, 1 − circular RMSE / π)" if wrapped else
                                               "100 × max(0, 1 − RMSE / truth RMS)")
        self.error_card.value_label.setText(f"{result.rms_error_rad:.4f} rad")
        self.error_card.foot_label.setText("Circular error after global phase alignment" if wrapped else
                                          "Piston removed across the illuminated pupil")
        self.export_button.setEnabled(True)
        if result.warnings:
            self.result_warning.setText("⚠  " + "  •  ".join(str(w) for w in result.warnings))
            self.result_warning.setStyleSheet("color:#f3b969; font-size:11px;")
        else:
            self.result_warning.setText("Intensity-only reconstruction with known Gaussian amplitude, charge candidates 0–6, and low-order Zernike phase. Ground truth is used only for comparison.")
            self.result_warning.setStyleSheet("color:#8fa4bf; font-size:11px;")
        shape = result.image.shape
        fit_error = result.metadata.get("image_fit_error")
        fit_text = f"  ·  image fit {fit_error:.2e}" if isinstance(fit_error, (int, float)) else ""
        charge = result.metadata.get("input_vortex_charge")
        recovered = result.metadata.get("recovered_vortex_charge")
        charge_text = f"  ·  ℓ input {charge}, recovered {recovered}" if charge is not None and recovered is not None else ""
        self.plot_label.setText(f"{name}{charge_text}  ·  {shape[1]} × {shape[0]} camera pixels{fit_text}")
        self.plot_label.setToolTip("Image fit is the L2 distance between square roots of the two unit-sum intensity images. A small residual indicates consistency with the assumed optical model; it does not establish real-instrument accuracy.")

    def _select_validation(self):
        if not hasattr(self, "validation_table"):
            return
        if self.tabs.currentIndex() != 1:
            return
        row = self.validation_table.currentRow()
        index = row + 1
        if index in self.validation_results:
            self._display(self.validation_results[index], self._validation_name(index))
        elif hasattr(self, "canvas"):
            self._clear_display()

    def _clear_display(self):
        self.current_result = None
        self.canvas.empty()
        self.similarity_card.heading_label.setText("PHASE SIMILARITY")
        self.similarity_card.value_label.setText("—")
        self.similarity_card.foot_label.setText("Metric follows the displayed result")
        self.error_card.value_label.setText("—")
        self.error_card.foot_label.setText("Radians across the illuminated pupil")
        self.export_button.setEnabled(False)
        self.result_warning.setText("Run the sensor or a validation sweep to inspect results here.")
        self.plot_label.setText("Model: scalar paraxial optics · known Gaussian amplitude")

    def _cancel(self):
        if self.worker:
            self.worker.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status_text.setText("Cancelling at the next solver checkpoint…")

    def _error(self, message):
        self.last_error = message
        self.status_text.setText("The run could not finish. Check the parameter values and sampling limits.")
        QtWidgets.QMessageBox.warning(self, "Simulation could not finish", message)

    def _finished(self):
        cancelled = self.worker.cancel_event.is_set() if self.worker else False
        self.busy = False
        self.thread = None
        self.worker = None
        for control in self.controls.values():
            control.setEnabled(True)
        self.defaults_button.setEnabled(True)
        self.validate_mode.setEnabled(True)
        self.validation_kind.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.progress.setVisible(False)
        self._parameters_changed()
        if cancelled:
            self.status_text.setText("Cancelled · completed results remain available.")
        elif self.last_error:
            self.status_text.setText("Run failed · previous completed results remain available.")
        elif self.single_result is not None or self.validation_results:
            self.status_text.setText("Complete · inspect the plots or export the numerical data and figures.")

    def _export(self):
        if self.current_result is None:
            return
        target = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose an export folder")
        if not target:
            return
        directory = Path(target)
        try:
            info = self._result_settings(self.current_result)
            settings_name = "run_settings.json"
            if self.tabs.currentIndex() == 1 and self.validation_results:
                results = sorted(self.validation_results.items())
                vortex = self._validation_kind == "vortex"
                csv_name = "vortex_summary.csv" if vortex else "validation_summary.csv"
                pdf_name = "vortex_charges_1_to_6.pdf" if vortex else "validation_21_modes.pdf"
                settings_name = "vortex_run_settings.json" if vortex else "run_settings.json"
                with (directory / csv_name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    if vortex:
                        writer.writerow(["input_charge", "recovered_charge", "charge_match", "circular_score_percent",
                                         "circular_rmse_rad", "image_fit_error", "warnings"])
                    else:
                        writer.writerow(["noll_j", "mode", "similarity_percent", "phase_rmse_rad", "warnings",
                                         "input_vortex_charge", "recovered_vortex_charge", "phase_comparison"])
                    for index, result in results:
                        if vortex:
                            recovered = result.metadata.get("recovered_vortex_charge")
                            writer.writerow([index, recovered, recovered == index, result.similarity_pct,
                                             result.rms_error_rad, result.metadata.get("image_fit_error"),
                                             " | ".join(result.warnings)])
                        else:
                            writer.writerow([index, MODES[index - 1].name, result.similarity_pct,
                                             result.rms_error_rad, " | ".join(result.warnings),
                                             result.metadata.get("input_vortex_charge", 0),
                                             result.metadata.get("recovered_vortex_charge", 0),
                                             result.metadata.get("phase_comparison", "unwrapped")])
                previous = self.current_result
                expected = 6 if vortex else 21
                completion = "Complete sweep" if len(results) == expected else f"Partial sweep · {len(results)}/{expected} completed"
                try:
                    with PdfPages(directory / pdf_name) as pdf:
                        for index, result in results:
                            self.canvas.show_result(result, f"{self._validation_name(index)} · {completion}")
                            pdf.savefig(self.canvas.fig, facecolor=PANEL)
                            filename = f"vortex_charge_{index:02d}.npz" if vortex else f"noll_{index:02d}.npz"
                            self._save_result(directory / filename, result)
                finally:
                    self.canvas.show_result(previous)
                info.update({"validation_kind": self._validation_kind,
                             "expected_count": expected, "completed_count": len(results),
                             "complete": len(results) == expected,
                             "completed_indices": [index for index, _ in results],
                             "selected_result": "Top-level parameters describe only the selected result; validation_results contains every completed snapshot.",
                             "validation_results": [{"index": index, **self._result_settings(result)} for index, result in results]})
            else:
                self.canvas.fig.savefig(directory / "plenoptic_reconstruction.png", dpi=200, facecolor=PANEL)
                self._save_result(directory / "plenoptic_result.npz", self.current_result)
            (directory / settings_name).write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            count_text = f" · {info['completed_count']}/{info['expected_count']} completed" if "completed_count" in info else ""
            self.status_text.setText(f"Exported figures and numerical arrays to {directory.name}{count_text}.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    @staticmethod
    def _result_settings(result):
        return {"parameters": result.metadata.get("parameters", {}),
                "relative_weights": result.metadata.get("relative_weights", "Individual unit Noll modes 1–21"),
                "similarity_definition": result.metadata.get("similarity_definition",
                    "100 * max(0, 1 - RMSE / ground-truth RMS), after piston removal; flat phase N/A"),
                "metadata": result.metadata, "warnings": result.warnings}

    @staticmethod
    def _save_result(path, result):
        wf = result.wavefront
        np.savez_compressed(path, camera_image=result.image, input_phase_rad=wf.phase,
                            reconstruction_phase_rad=result.reconstructed_phase,
                            pupil_mask=wf.mask, input_intensity=wf.intensity,
                            x_mm=wf.x_mm, y_mm=wf.y_mm,
                            input_coefficients_rad=wf.coefficients,
                            reconstructed_coefficients=result.coefficients,
                            input_vortex_charge=result.metadata.get("input_vortex_charge", getattr(wf, "vortex_charge", 0)),
                            reconstructed_vortex_charge=result.metadata.get("recovered_vortex_charge", 0),
                            parameters_json=json.dumps(result.metadata.get("parameters", {}), default=str),
                            metadata_json=json.dumps(result.metadata, default=str),
                            phase_residual_rad=phase_residual(wf.phase, result.reconstructed_phase, wf.mask,
                                wrapped=bool(result.metadata.get("wrapped_phase_comparison", False))))

    def closeEvent(self, event):
        if self.busy:
            self._cancel()
            self.status_text.setText("Stopping computation before closing. Close again after cancellation completes.")
            event.ignore()
        else:
            event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "banner"):
            compact = self.height() < 900
            self.banner.setFixedHeight(112 if compact else 138)
            self._update_validation_layout()


def launch(argv=None):
    app = QtWidgets.QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Plenoptic Studio")
    app.setOrganizationName("Plenoptic Studio")
    configure_app(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch())
