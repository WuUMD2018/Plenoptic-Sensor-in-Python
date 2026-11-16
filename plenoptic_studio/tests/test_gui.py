"""Interaction regressions; run with QT_QPA_PLATFORM=offscreen when headless."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from PySide6 import QtCore, QtTest, QtWidgets
    from plenoptic_studio import gui
    from plenoptic_studio.optics import Parameters, SimulationResult, make_wavefront
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@unittest.skipUnless(GUI_AVAILABLE, "PySide6 and matplotlib are optional for headless core tests")
class GuiInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        gui.configure_app(cls.app)

    def test_slider_anchor_and_explicit_same_value_entry(self):
        widget = gui.AnchoredParameter("Aperture", "mm", 10.0)
        widget.slider.setValue(500)
        self.assertAlmostEqual(widget.value(), 5)
        self.assertAlmostEqual(widget.anchor, 10)
        # Retype the already displayed value: this must still create a new anchor.
        widget.box.lineEdit().setFocus()
        widget.box.lineEdit().selectAll()
        QtTest.QTest.keyClicks(widget.box.lineEdit(), "5")
        QtTest.QTest.keyClick(widget.box.lineEdit(), QtCore.Qt.Key.Key_Return)
        self.assertEqual(widget.slider.value(), 1000)
        self.assertAlmostEqual(widget.anchor, 5)
        widget.slider.setValue(2000)
        self.assertAlmostEqual(widget.value(), 10)
        widget.deleteLater()

    def test_invalid_zero_and_requested_f_number_warning(self):
        window = gui.MainWindow()
        window.controls["objective_diameter_mm"].box.setValue(0)
        self.assertFalse(window.run_button.isEnabled())
        window.controls["objective_diameter_mm"].box.setValue(10)
        window.controls["objective_focal_mm"].box.setValue(200)
        self.assertEqual(window.f_status.objectName(), "warning")
        self.assertTrue(window.run_button.isEnabled())
        window.close()

    def test_mode_preview_and_piston_behavior(self):
        weights = np.zeros(21)
        weights[3] = 1
        dialog = gui.WavefrontDialog(Parameters(), weights)
        dialog._enable_view()
        self.assertIsNotNone(dialog.ax)
        self.assertTrue(dialog.live)
        dialog._clear()
        dialog.rows[0].box.setValue(100)
        dialog._draw()
        self.assertIn("0.000 rad", dialog.note.text())
        dialog.close()

    def test_validation_does_not_replace_single_tab(self):
        window = gui.MainWindow()
        wf = make_wavefront(Parameters(), np.zeros(21), resolution=16)
        result = SimulationResult(wf, np.ones((16, 16)), wf.phase.copy(), None,
                                  0.0, [], {"mode_j": 1}, np.zeros(21))
        window._receive_result(1, result)
        self.assertIsNone(window.current_result)
        self.assertFalse(window.export_button.isEnabled())
        window.tabs.setCurrentIndex(1)
        self.assertIs(window.current_result, result)
        window.validation_table.selectRow(3)
        self.assertIsNone(window.current_result)
        self.assertFalse(window.export_button.isEnabled())
        window.validation_table.selectRow(0)
        self.assertIs(window.current_result, result)
        window.tabs.setCurrentIndex(0)
        self.assertIsNone(window.current_result)
        self.assertFalse(window.export_button.isEnabled())
        window.close()

    def test_worker_delivers_result_without_blocking_the_gui(self):
        window = gui.MainWindow()
        wf = make_wavefront(Parameters(), np.zeros(21), resolution=16)
        result = SimulationResult(wf, np.ones((16, 16)), wf.phase.copy(), None,
                                  0.0, [], {"parameters": {}}, np.zeros(21))
        with patch.object(gui, "simulate", return_value=result):
            window._run()
            self.assertTrue(window.busy)
            timeout = time.monotonic() + 3
            while window.busy and time.monotonic() < timeout:
                self.app.processEvents()
                QtTest.QTest.qWait(5)
            self.assertFalse(window.busy)
            self.assertIs(window.single_result, result)
            self.assertTrue(window.run_button.isEnabled())
        window.close()

    def test_export_uses_successful_result_settings_after_input_edits(self):
        window = gui.MainWindow()
        wf = make_wavefront(Parameters(), np.zeros(21), resolution=16)
        result = SimulationResult(wf, np.ones((16, 16)), wf.phase.copy(), None,
                                  0.0, [], {"parameters": {"wavelength_nm": 633}}, np.zeros(21))
        window._receive_result(0, result)
        window.controls["wavelength_nm"].box.setValue(532)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=directory):
                window._export()
            settings = json.loads((Path(directory) / "run_settings.json").read_text())
            self.assertEqual(settings["parameters"]["wavelength_nm"], 633)
            self.assertTrue((Path(directory) / "plenoptic_reconstruction.png").is_file())
            self.assertTrue((Path(directory) / "plenoptic_result.npz").is_file())
        window.close()


if __name__ == "__main__":
    unittest.main()
