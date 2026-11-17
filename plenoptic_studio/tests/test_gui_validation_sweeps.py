"""Validation routing, snapshot isolation, partial cancellation and export checks."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import json
import tempfile
import time
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from PySide6 import QtTest, QtWidgets
    from plenoptic_studio import gui
    from plenoptic_studio.optics import CancelledError, Parameters, SimulationResult, make_wavefront
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@unittest.skipUnless(GUI_AVAILABLE, "GUI dependencies are optional for headless core tests")
class ValidationSweepGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        gui.configure_app(cls.app)

    def result(self, charge=1, recovered=None, kind="vortex", noll=1, wavelength=633):
        p = Parameters(vortex_charge=charge, phase_ptp_rad=0, wavelength_nm=wavelength)
        wf = make_wavefront(p, np.zeros(21), resolution=16)
        metadata = {"parameters": asdict(p), "relative_weights": [0.0] * 21,
                    "validation_kind": kind, "input_vortex_charge": charge,
                    "recovered_vortex_charge": charge if recovered is None else recovered,
                    "wrapped_phase_comparison": bool(charge),
                    "phase_comparison": "wrapped" if charge else "unwrapped",
                    "similarity_definition": "circular test score" if charge else "linear test score"}
        metadata["validation_charge" if kind == "vortex" else "mode_j"] = charge if kind == "vortex" else noll
        return SimulationResult(wf, np.ones((16, 16)) / 256, wf.phase.copy(), 100.0,
                                0.0, [], metadata, np.zeros(21))

    def select_vortex(self, window):
        window.validation_kind.setCurrentIndex(window.validation_kind.findData("vortex"))
        window.validate_mode.setChecked(True)

    def drain_until(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            QtTest.QTest.qWait(5)
        self.assertTrue(condition(), "GUI worker did not reach its expected state")

    def test_kind_switch_keeps_snapshots_and_pending_row_clears_display(self):
        window = gui.MainWindow()
        custom = self.result(4)
        window._receive_result(0, custom)
        noll = self.result(0, kind="zernike", noll=3)
        window._receive_result(3, noll)
        self.select_vortex(window)
        self.assertEqual(window.validation_table.rowCount(), 6)
        self.assertEqual(window.validation_table.columnCount(), 5)
        self.assertIn("1–6", window.validate_button.text())
        self.assertIsNone(window.current_result)
        vortex = self.result(1, recovered=2)
        window._receive_result(1, vortex)
        self.assertIs(window.current_result, vortex)
        self.assertIn("mismatch", window.validation_table.item(0, 4).text())
        self.assertEqual(window.validation_table.item(0, 2).text(), "100.0%")
        window.validation_table.selectRow(5)
        self.assertIsNone(window.current_result)
        self.assertFalse(window.export_button.isEnabled())
        window.validation_kind.setCurrentIndex(window.validation_kind.findData("zernike"))
        self.assertEqual(window.validation_table.rowCount(), 21)
        self.assertIs(window.current_result, noll)
        window.validation_kind.setCurrentIndex(window.validation_kind.findData("vortex"))
        self.assertIs(window.current_result, vortex)
        window.tabs.setCurrentIndex(0)
        self.assertIs(window.current_result, custom)
        window.close()

    def test_vortex_worker_routes_six_results_and_locks_selector(self):
        window = gui.MainWindow()
        window.controls["wavelength_nm"].box.setValue(532)
        window.vortex_control.slider.setValue(6)
        self.select_vortex(window)
        calls = []

        def sweep(params, progress=None, cancel=None, result_callback=None):
            calls.append(params)
            results = []
            for charge in range(1, 7):
                result = self.result(charge, wavelength=params.wavelength_nm)
                results.append(result)
                result_callback(charge - 1, result)
            return results

        with patch.object(gui, "validate_vortex_charges", new=sweep), patch.object(gui, "validate_all") as noll:
            window._validate()
            self.assertFalse(window.validation_kind.isEnabled())
            self.drain_until(lambda: not window.busy)
            self.assertTrue(window.validation_kind.isEnabled())
            self.assertEqual(set(window.validation_results), set(range(1, 7)))
            self.assertEqual(calls[0].wavelength_nm, 532)
            noll.assert_not_called()
            self.assertEqual(window.current_result.metadata["validation_charge"], 6)
            self.assertEqual(window.vortex_control.value(), 6)
        window.close()

    def test_cancel_keeps_completed_rows_and_restart_clears_only_active_kind(self):
        window = gui.MainWindow()
        noll_result = self.result(0, kind="zernike", noll=2)
        window._receive_result(2, noll_result)
        self.select_vortex(window)

        def cancelled_sweep(params, progress=None, cancel=None, result_callback=None):
            result_callback(0, self.result(1))
            while not cancel():
                time.sleep(.005)
            raise CancelledError("cancelled")

        with patch.object(gui, "validate_vortex_charges", new=cancelled_sweep):
            window._validate()
            self.drain_until(lambda: 1 in window.validation_results)
            window._cancel()
            self.drain_until(lambda: not window.busy)
        self.assertEqual(list(window.validation_results), [1])
        self.assertTrue(window.export_button.isEnabled())
        self.assertIn("Cancelled", window.status_text.text())
        with patch.object(window, "_start") as start:
            window._validate()
            start.assert_called_once_with(True)
        self.assertEqual(window.validation_results, {})
        self.assertFalse(window.export_button.isEnabled())
        self.assertIs(window.validation_results_by_kind["zernike"][2], noll_result)
        window.close()

    def test_partial_vortex_export_has_all_actual_snapshots_and_distinct_names(self):
        window = gui.MainWindow()
        self.select_vortex(window)
        window._receive_result(1, self.result(1, wavelength=532))
        window._receive_result(2, self.result(2, recovered=3, wavelength=532))
        window.controls["wavelength_nm"].box.setValue(780)
        window.vortex_control.slider.setValue(6)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=folder), patch.object(QtWidgets.QMessageBox, "warning") as warning:
                window._export()
            warning.assert_not_called()
            self.assertTrue((directory / "vortex_charges_1_to_6.pdf").is_file())
            self.assertFalse((directory / "validation_summary.csv").exists())
            settings = json.loads((directory / "vortex_run_settings.json").read_text())
            self.assertFalse(settings["complete"])
            self.assertEqual(settings["completed_indices"], [1, 2])
            self.assertEqual((settings["completed_count"], settings["expected_count"]), (2, 6))
            self.assertEqual([r["parameters"]["vortex_charge"] for r in settings["validation_results"]], [1, 2])
            for snapshot in settings["validation_results"]:
                self.assertEqual(snapshot["parameters"]["wavelength_nm"], 532)
                self.assertEqual(snapshot["parameters"]["phase_ptp_rad"], 0)
                self.assertEqual(snapshot["relative_weights"], [0.0] * 21)
            with (directory / "vortex_summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["charge_match"] for r in rows], ["True", "False"])
            for charge in (1, 2):
                with np.load(directory / f"vortex_charge_{charge:02d}.npz", allow_pickle=False) as data:
                    parameters = json.loads(str(data["parameters_json"]))
                    self.assertEqual(parameters["vortex_charge"], charge)
                    self.assertEqual(parameters["wavelength_nm"], 532)
            self.assertIn("2/6", window.status_text.text())
        window.close()


if __name__ == "__main__":
    unittest.main()
