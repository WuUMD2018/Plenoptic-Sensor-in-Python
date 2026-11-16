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
                     validate_all, validate_parameters)

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
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 16px; border: none; }
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
    ("phase_ptp_rad", "Phase peak-to-peak", "rad", 3),
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


class MetricCard(QtWidgets.QFrame):
    def __init__(self, heading, value="—", foot="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        layout.addWidget(label(heading.upper(), "eyebrow"))
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
        truth = np.where(mask, wf.phase, np.nan)
        rec = np.where(mask, result.reconstructed_phase, np.nan)
        # Phase is compared after piston removal by the optical core.
        if np.any(mask):
            truth = truth - np.nanmean(truth)
            rec = rec - np.nanmean(rec)
        residual = rec - truth
        extent = [float(wf.x_mm[0]), float(wf.x_mm[-1]), float(wf.y_mm[0]), float(wf.y_mm[-1])]
        phase_limit = max(float(np.nanmax(np.abs(truth))) if np.any(mask) else 0,
                          float(np.nanmax(np.abs(rec))) if np.any(mask) else 0, 1e-6)
        fields = [(truth, "02  INPUT PHASE", phase_limit),
                  (rec, "03  RECONSTRUCTED PHASE", phase_limit),
                  (residual, "04  PHASE RESIDUAL", max(float(np.nanmax(np.abs(residual))), 1e-6))]
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
        self.setWindowTitle("Input composer · 21 Zernike modes")
        self.resize(1120, 820)
        self.setMinimumSize(920, 620)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.addWidget(label("WAVEFRONT COMPOSER", "eyebrow"))
        layout.addWidget(label("Build the distortion", "title"))
        layout.addWidget(label(
            f"Relative Noll coefficients · normalized to {params.phase_ptp_rad:g} rad peak-to-peak on the illuminated pupil.",
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
        layout.addWidget(label("Piston (Noll 01) is a constant phase offset: intensity cannot sense it and its peak-to-peak is zero. A piston-only input remains flat.", "muted", True))
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
        z = np.where(wf.mask, wf.phase, np.nan)
        peak = float(np.nanmax(z) - np.nanmin(z))
        ax.plot_surface(x, y, z, cmap="turbo", linewidth=0, antialiased=True, rcount=80, ccount=80)
        ax.view_init(elev=elev, azim=azim)
        ax.set_facecolor(PANEL)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((.06, .11, .18, 1))
            axis._axinfo["grid"]["color"] = (.2, .3, .4, .45)
            axis.line.set_color("#415a79")
        ax.tick_params(colors=MUTED, labelsize=8, pad=0)
        ax.set_xlabel("x (mm)", color=MUTED, fontsize=9, labelpad=2)
        ax.set_ylabel("y (mm)", color=MUTED, fontsize=9, labelpad=2)
        ax.set_zlabel("Phase (rad)", color=MUTED, fontsize=9, labelpad=4)
        ax.set_box_aspect((1, 1, .65))
        self.figure.subplots_adjust(left=0, right=1, bottom=.03, top=.98)
        self.note.setText(f"Actual peak-to-peak: {peak:.3f} rad · drag to rotate. Relative weights set the shape; the main amplitude sets its scale.")
        self.canvas.draw_idle()


class SimulationWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    result = QtCore.Signal(int, object)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, params, weights, validation=False):
        super().__init__()
        self.params = params
        self.weights = np.array(weights, copy=True)
        self.validation = validation
        self.cancel_event = threading.Event()

    @QtCore.Slot()
    def run(self):
        try:
            if self.validation:
                # validate_all owns reuse of the calibrated propagation model.
                import inspect
                if "result_callback" in inspect.signature(validate_all).parameters:
                    validate_all(self.params, progress=self.progress.emit,
                                 cancel=self.cancel_event.is_set,
                                 result_callback=lambda index, result: self.result.emit(index + 1, result))
                else:
                    results = validate_all(self.params, progress=self.progress.emit,
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
        layout.addWidget(label("I(x,y) = exp[−(x²+y²)/(2σ²)] inside the circular transmitted pupil. The optical field amplitude is √I.", "muted", True))
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
        self.validation_results = {}
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
        outer.addWidget(label("Sliders span 0–200% of the last typed value.\nTyping a value resets its slider to 100%.", "muted", True))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(body)
        form.setContentsMargins(0, 1, 4, 0)
        form.setSpacing(4)
        self.controls = {}
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
        self.setup_button = QtWidgets.QPushButton("Setup input · 21 Zernike modes")
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
        self.validate_mode = QtWidgets.QCheckBox("Validate individual Zernike modes")
        self.validate_mode.toggled.connect(self._validation_toggled)
        outer.addWidget(self.validate_mode)
        self.validate_button = QtWidgets.QPushButton("Validate all 21 modes")
        self.validate_button.setEnabled(False)
        self.validate_button.clicked.connect(self._validate)
        outer.addWidget(self.validate_button)
        outer.addWidget(label("Input intensity: Gaussian, σ = diameter / 4.\nPiston is unobservable; all phases are piston-free.", "muted", True))
        return panel

    def _workspace(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        metrics = QtWidgets.QHBoxLayout()
        metrics.setSpacing(10)
        self.similarity_card = MetricCard("Phase similarity", "—", "Piston removed · amplitude-sensitive")
        self.similarity_card.setToolTip("Synthetic same-model comparison, not experimental accuracy. Ground truth is used only for this metric; the reconstruction uses camera intensity. Piston is removed, but phase amplitude and sign are not fitted away.")
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
        self.validation_hint = label("Select a completed mode to inspect its camera image, reconstruction and residual below.", "muted")
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
            self.status_text.setText(f"Input saved · {active} non-piston modes · {self.parameters().phase_ptp_rad:g} rad requested peak-to-peak. Run to update the sensor.")

    def _beam(self):
        BeamProfileDialog(self.parameters(), self.weights, self).exec()

    def _validation_toggled(self, checked):
        self.tabs.setCurrentIndex(1 if checked else 0)
        self._parameters_changed()

    def _tab_changed(self, index):
        self.tabs.setFixedHeight((170 if self.height() < 900 else 225) if index == 1 else 42)
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
        self.validation_results = {}
        self._clear_display()
        self.validation_params = self.parameters()
        for row in range(21):
            for col, value in enumerate(["—", "—", "Waiting"], 1):
                self.validation_table.item(row, col).setText(value)
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
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_text.setText("Preparing the optical model…")
        self.thread = QtCore.QThread(self)
        self.worker = SimulationWorker(self.parameters(), self.weights, validation)
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
            # Core callbacks use the displayed Noll number (1–21).
            index = int(result.metadata.get("mode_j", index))
            self.validation_results[index] = result
            row = index - 1
            if not 0 <= row < 21:
                return
            sim = result.similarity_pct
            self.validation_table.item(row, 1).setText(("N/A · piston" if index == 1 else "N/A · flat") if sim is None else f"{sim:.1f}%")
            self.validation_table.item(row, 2).setText(f"{result.rms_error_rad:.4g}")
            status = "Piston unobservable" if index == 1 else ("Review warnings" if result.warnings else "Complete")
            self.validation_table.item(row, 3).setText(status)
            self.validation_table.selectRow(row)
            if self.tabs.currentIndex() == 1:
                self._display(result, f"Noll {index:02d} · {MODES[row].name}")

    def _display(self, result, name):
        self.current_result = result
        self.canvas.show_result(result, name)
        similarity = result.similarity_pct
        self.similarity_card.value_label.setText("N/A" if similarity is None else f"{similarity:.1f}%")
        self.similarity_card.foot_label.setText("Piston / flat phase is unobservable" if similarity is None else "100 × max(0, 1 − normalized RMSE)")
        self.error_card.value_label.setText(f"{result.rms_error_rad:.4f} rad")
        self.export_button.setEnabled(True)
        if result.warnings:
            self.result_warning.setText("⚠  " + "  •  ".join(str(w) for w in result.warnings))
            self.result_warning.setStyleSheet("color:#f3b969; font-size:11px;")
        else:
            self.result_warning.setText("Intensity-only reconstruction; ground truth is used only for this comparison. Known Gaussian amplitude and a 21-mode phase basis are assumed.")
            self.result_warning.setStyleSheet("color:#8fa4bf; font-size:11px;")
        shape = result.image.shape
        fit_error = result.metadata.get("image_fit_error")
        fit_text = f"  ·  image fit {fit_error:.2e}" if isinstance(fit_error, (int, float)) else ""
        self.plot_label.setText(f"{name}  ·  {shape[1]} × {shape[0]} camera pixels{fit_text}")
        self.plot_label.setToolTip("Image fit is the L2 distance between square roots of the two unit-sum intensity images. A small residual indicates consistency with the assumed optical model; it does not establish real-instrument accuracy.")

    def _select_validation(self):
        if not hasattr(self, "validation_table"):
            return
        if self.tabs.currentIndex() != 1:
            return
        row = self.validation_table.currentRow()
        index = row + 1
        if index in self.validation_results:
            self._display(self.validation_results[index], f"Noll {index:02d} · {MODES[row].name}")
        elif hasattr(self, "canvas"):
            self._clear_display()

    def _clear_display(self):
        self.current_result = None
        self.canvas.empty()
        self.similarity_card.value_label.setText("—")
        self.error_card.value_label.setText("—")
        self.export_button.setEnabled(False)
        self.result_warning.setText("Run the sensor or validate the 21 modes to inspect results here.")
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
            if self.tabs.currentIndex() == 1 and self.validation_results:
                results = sorted(self.validation_results.items())
                with (directory / "validation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["noll_j", "mode", "similarity_percent", "phase_rmse_rad", "warnings"])
                    for index, result in results:
                        writer.writerow([index, MODES[index - 1].name, result.similarity_pct,
                                         result.rms_error_rad, " | ".join(result.warnings)])
                previous = self.current_result
                with PdfPages(directory / "validation_21_modes.pdf") as pdf:
                    for index, result in results:
                        self.canvas.show_result(result, f"Noll {index:02d} · {MODES[index - 1].name}")
                        pdf.savefig(self.canvas.fig, facecolor=PANEL)
                        self._save_result(directory / f"noll_{index:02d}.npz", result)
                self.canvas.show_result(previous)
            else:
                self.canvas.fig.savefig(directory / "plenoptic_reconstruction.png", dpi=200, facecolor=PANEL)
                self._save_result(directory / "plenoptic_result.npz", self.current_result)
            info = {"parameters": self.current_result.metadata.get("parameters", {}),
                    "relative_weights": self.current_result.metadata.get("relative_weights", "Individual unit Noll modes 1–21"),
                    "similarity_definition": "100 * max(0, 1 - RMSE / ground-truth RMS), after piston removal; flat phase N/A",
                    "metadata": self.current_result.metadata,
                    "warnings": self.current_result.warnings}
            (directory / "run_settings.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            self.status_text.setText(f"Exported figures and numerical arrays to {directory.name}.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    @staticmethod
    def _save_result(path, result):
        wf = result.wavefront
        np.savez_compressed(path, camera_image=result.image, input_phase_rad=wf.phase,
                            reconstruction_phase_rad=result.reconstructed_phase,
                            pupil_mask=wf.mask, input_intensity=wf.intensity,
                            x_mm=wf.x_mm, y_mm=wf.y_mm,
                            input_coefficients_rad=wf.coefficients,
                            reconstructed_coefficients=result.coefficients)

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
            self.validation_hint.setVisible(not compact)
            self.tabs.setFixedHeight((170 if compact else 225) if self.tabs.currentIndex() == 1 else 42)


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
