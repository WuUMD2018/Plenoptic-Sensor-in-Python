"""Vortex controls, circular displays, and saved-run metadata regressions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
import time
import unittest
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
class VortexGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        gui.configure_app(cls.app)

    def test_charge_slider_is_integer_absolute_and_resets(self):
        window = gui.MainWindow()
        charge = window.vortex_control
        self.assertEqual((charge.slider.minimum(), charge.slider.maximum()), (0, 6))
        for value in range(7):
            charge.slider.setValue(value)
            self.assertEqual(charge.box.value(), value)
            self.assertEqual(window.parameters().vortex_charge, value)
            self.assertIsInstance(window.parameters().vortex_charge, int)
        charge.box.setValue(3)
        self.assertEqual(charge.slider.value(), 3)
        window.controls["phase_ptp_rad"].slider.setValue(0)
        self.assertEqual(window.parameters().vortex_charge, 3)
        window._reset()
        self.assertEqual(window.parameters().vortex_charge, 0)
        self.assertEqual(charge.slider.value(), 0)
        window.close()

    def test_composer_clear_retains_vortex(self):
        dialog = gui.WavefrontDialog(Parameters(vortex_charge=4, phase_ptp_rad=0), np.ones(21))
        dialog._clear()
        dialog._enable_view()
        self.assertTrue(np.all(dialog.weights == 0))
        self.assertEqual(dialog.params.vortex_charge, 4)
        self.assertIsNotNone(dialog.ax)
        self.assertIn("ℓ = 4", dialog.note.text())
        self.assertIn("wrapped", dialog.note.text())
        dialog.close()

    def test_worker_snapshots_and_locks_charge(self):
        window = gui.MainWindow()
        window.vortex_control.slider.setValue(4)
        params = window.parameters()
        wf = make_wavefront(params, np.zeros(21), resolution=16)
        result = SimulationResult(wf, np.ones((16, 16)), wf.phase.copy(), 100.0,
                                  0.0, [], {"parameters": {}, "input_vortex_charge": 4,
                                  "recovered_vortex_charge": 4, "wrapped_phase_comparison": True}, np.zeros(21))
        with patch.object(gui, "simulate", return_value=result) as simulate:
            window._run()
            self.assertTrue(window.busy)
            self.assertFalse(window.vortex_control.isEnabled())
            deadline = time.monotonic() + 4
            while window.busy and time.monotonic() < deadline:
                self.app.processEvents()
                QtTest.QTest.qWait(5)
            self.assertFalse(window.busy)
            self.assertTrue(window.vortex_control.isEnabled())
            self.assertEqual(simulate.call_args.args[0].vortex_charge, 4)
            self.assertEqual(window.last_run_params.vortex_charge, 4)
        window.close()

    def test_wrapped_display_and_exports_preserve_completed_charge(self):
        window = gui.MainWindow()
        params = Parameters(vortex_charge=4, phase_ptp_rad=0)
        wf = make_wavefront(params, np.zeros(21), resolution=16)
        # Equivalent phase with a different global gauge and a branch-cut shift.
        reconstruction = wf.phase + 0.37
        reconstruction[:, 8:] += 2 * np.pi
        metadata = {"parameters": {"vortex_charge": 4, "phase_ptp_rad": 0},
                    "input_vortex_charge": 4, "recovered_vortex_charge": 4,
                    "wrapped_phase_comparison": True, "phase_comparison": "wrapped",
                    "similarity_definition": "100 * max(0, 1 - circular_RMSE / pi)"}
        result = SimulationResult(wf, np.ones((16, 16)), reconstruction, 100.0,
                                  0.0, [], metadata, np.zeros(21))
        window._receive_result(0, result)
        self.assertIn("CIRCULAR", window.similarity_card.heading_label.text())
        self.assertIn("RMSE / π", window.similarity_card.foot_label.text())
        self.assertIn("ℓ input 4, recovered 4", window.plot_label.text())
        residual_axis = next(ax for ax in window.canvas.fig.axes if "CIRCULAR RESIDUAL" in ax.get_title(loc="left"))
        residual = residual_axis.images[0].get_array()
        self.assertLess(float(np.max(np.abs(residual.compressed()))), 1e-12)
        window.vortex_control.slider.setValue(2)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=directory):
                window._export()
            settings = json.loads((Path(directory) / "run_settings.json").read_text())
            self.assertEqual(settings["parameters"]["vortex_charge"], 4)
            self.assertEqual(settings["similarity_definition"], metadata["similarity_definition"])
            with np.load(Path(directory) / "plenoptic_result.npz") as data:
                self.assertEqual(int(data["input_vortex_charge"]), 4)
                self.assertEqual(int(data["reconstructed_vortex_charge"]), 4)
                self.assertLess(float(np.max(np.abs(data["phase_residual_rad"][wf.mask]))), 1e-12)
        window.close()


if __name__ == "__main__":
    unittest.main()
