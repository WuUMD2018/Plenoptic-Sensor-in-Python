"""Independent analytic checks of branch-cut integration and camera mapping."""
import json
from types import SimpleNamespace
import unittest

import numpy as np

from plenoptic_studio.vortex_diagnostics import (
    analyze_vortex_camera, reconstruct_gradient_field,
)


def analytic_vortex(charge, size=80, cut=0.0):
    axis = (np.arange(size)-(size-1)/2)*4/size
    x, y = np.meshgrid(axis, axis)
    r2 = x*x+y*y
    mask = r2 < 1.9**2
    gx, gy = -charge*y/r2, charge*x/r2
    phase = charge*np.mod(np.arctan2(y, x)-cut, 2*np.pi)
    return axis, gx, gy, mask, phase


def piston_rmse(phase, truth, mask):
    residual = phase[mask]-truth[mask]
    residual -= residual.mean()
    return float(np.sqrt(np.mean(residual**2)))


class _CameraGeometry:
    """Geometry/calibration stub with no generating phase or charge member."""
    def __init__(self):
        self.n = 64
        self.dx = 1e-5
        self.pitch = 1e-4
        self.a, self.b = 0.4, 0.005
        self.camera_axis = (np.arange(self.n)-(self.n-1)/2)*self.dx
        cx, cy = np.meshgrid(self.camera_axis, self.camera_axis)
        self.cell_x = self.pitch*np.floor(cx/self.pitch+0.5)
        self.cell_y = self.pitch*np.floor(cy/self.pitch+0.5)
        self.axis = (np.arange(32)-15.5)*2/32
        x, y = np.meshgrid(self.axis, self.axis)
        self.mask = x*x+y*y < 0.95**2
        self.p = SimpleNamespace(wavelength_nm=633.0, bandwidth_nm=0.0)
        self.flat = np.exp(-(cx*cx+cy*cy)/(2*(2e-5)**2))
        self.flat /= self.flat.sum()
        self.reference_calls = []

    def forward(self, coefficients, *, vortex_charge):
        self.reference_calls.append((np.array(coefficients), vortex_charge))
        return self.flat.copy()


class VortexDiagnosticsTests(unittest.TestCase):
    def test_branch_cut_recovers_hidden_vortex_phase_that_global_lmse_misses(self):
        for charge in (1, 4, 6):
            with self.subTest(charge=charge):
                axis, gx, gy, mask, truth = analytic_vortex(charge)
                result = reconstruct_gradient_field(axis, gx, gy, mask, center_mm=(0, 0))
                region = result["valid_mask"]
                cut_error = piston_rmse(result["phase_branchcut"], truth, region)
                global_error = piston_rmse(result["phase_lmse"], truth, region)
                self.assertLess(cut_error, 0.025*charge)
                self.assertGreater(global_error, 1.5*charge)
                self.assertAlmostEqual(result["diagnostics"]["circulation_charge_estimate"], charge, delta=0.01)
                self.assertLess(result["diagnostics"]["branchcut_lmse"]["gradient_residual_rms_rad_per_mm"],
                                result["diagnostics"]["global_lmse"]["gradient_residual_rms_rad_per_mm"]*0.05)
                self.assertEqual(result["diagnostics"]["branchcut_lmse"]["connected_components"], 1)

    def test_curl_center_is_inferred_from_gradients(self):
        axis, gx, gy, mask, _ = analytic_vortex(4)
        result = reconstruct_gradient_field(axis, gx, gy, mask)
        center = result["diagnostics"]["center_mm"]
        self.assertLess(np.hypot(*center), 1e-8)
        self.assertEqual(result["diagnostics"]["center_source"], "inner absolute-curl peak")

    def test_arbitrary_cut_changes_gauge_but_recovers_same_vortex(self):
        cut = 0.61
        axis, gx, gy, mask, truth = analytic_vortex(4, cut=cut)
        result = reconstruct_gradient_field(axis, gx, gy, mask, center_mm=(0, 0), cut_angle_rad=cut)
        self.assertLess(piston_rmse(result["phase_branchcut"], truth, result["valid_mask"]), 0.10)

    def test_smooth_phase_is_recovered_by_both_integrations(self):
        axis, _, _, mask, _ = analytic_vortex(0, size=48)
        x, y = np.meshgrid(axis, axis)
        truth = 0.4*x*x+0.3*y*y+0.2*x-0.1*y
        result = reconstruct_gradient_field(axis, 0.8*x+0.2, 0.6*y-0.1, mask, center_mm=(0, 0))
        for key in ("phase_lmse", "phase_branchcut"):
            self.assertLess(piston_rmse(result[key], truth, result["valid_mask"]), 1e-5)
        self.assertAlmostEqual(result["diagnostics"]["circulation_charge_estimate"], 0.0, delta=1e-10)

    def test_camera_diagnostics_and_explicit_flat_calibration(self):
        model = _CameraGeometry()
        image = model.flat.copy()
        raw = analyze_vortex_camera(model, image, center_mm=(0, 0))
        corrected = analyze_vortex_camera(model, image, center_mm=(0, 0), subtract_flat_reference=True)
        np.testing.assert_array_equal(image, model.flat)
        self.assertEqual(len(model.reference_calls), 1)
        coefficients, charge = model.reference_calls[0]
        np.testing.assert_array_equal(coefficients, np.zeros(21))
        self.assertEqual(charge, 0)
        self.assertFalse(raw["diagnostics"]["flat_reference_subtracted"])
        self.assertTrue(corrected["diagnostics"]["flat_reference_subtracted"])
        for result in (raw, corrected):
            for key in ("gx", "gy", "curl", "phase_lmse", "phase_branchcut", "valid_mask"):
                self.assertEqual(result[key].shape, model.mask.shape)
            self.assertTrue(np.all(np.isfinite(result["phase_branchcut"][result["valid_mask"]])))
            json.dumps(result["diagnostics"], allow_nan=False)
        self.assertLess(np.max(np.abs(corrected["phase_branchcut"][corrected["valid_mask"]])), 1e-12)

    def test_invalid_inputs_fail_informatively(self):
        axis, gx, gy, mask, _ = analytic_vortex(1, size=20)
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            reconstruct_gradient_field(axis[::-1], gx, gy, mask)
        bad = gx.copy()
        bad[mask] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            reconstruct_gradient_field(axis, bad, gy, mask)
        with self.assertRaisesRegex(ValueError, "match"):
            reconstruct_gradient_field(axis, gx[:-1], gy, mask)
        with self.assertRaisesRegex(ValueError, "too few"):
            reconstruct_gradient_field(axis, gx, gy, mask, core_radius_mm=100)
        model = _CameraGeometry()
        for image in (np.ones((2, 2)), -model.flat, np.full_like(model.flat, np.nan), np.zeros_like(model.flat)):
            with self.assertRaises(ValueError):
                analyze_vortex_camera(model, image)
        with self.assertRaisesRegex(ValueError, "lie in"):
            analyze_vortex_camera(model, model.flat, min_relative_intensity=1)


if __name__ == "__main__":
    unittest.main()
