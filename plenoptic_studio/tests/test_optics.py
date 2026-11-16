"""Run from project root: python -m unittest discover -s plenoptic_studio/tests -v."""
import unittest
from dataclasses import replace
import numpy as np
from plenoptic_studio.optics import (Parameters, MODES, OpticalModel, make_wavefront,
                    validate_parameters, compare_wavefronts, CancelledError, simulate)


class WavefrontTests(unittest.TestCase):
    def test_noll_order_and_phase_range(self):
        self.assertEqual([(m.n, m.m) for m in MODES[:6]],
                         [(0,0), (1,1), (1,-1), (2,0), (2,-2), (2,2)])
        weights = np.arange(21)/20
        wave = make_wavefront(Parameters(), weights)
        self.assertAlmostEqual(np.ptp(wave.phase[wave.mask]), Parameters().phase_ptp_rad, places=10)
        self.assertAlmostEqual(wave.phase[wave.mask].mean(), 0, places=12)
        # Intensity, not electric-field amplitude, has the specified Gaussian sigma.
        x, y = np.meshgrid(wave.x_mm, wave.y_mm)
        expected = np.exp(-(x*x+y*y)/(2*2.0**2))
        np.testing.assert_allclose(wave.intensity[wave.mask], expected[wave.mask])

    def test_piston_is_explicitly_unobservable(self):
        weights = np.zeros(21)
        weights[0] = 1
        wave = make_wavefront(Parameters(), weights)
        self.assertEqual(np.ptp(wave.phase), 0)
        similarity, rmse = compare_wavefronts(wave.phase, wave.phase, wave.mask)
        self.assertIsNone(similarity)
        self.assertEqual(rmse, 0)

    def test_metric_does_not_hide_amplitude_error(self):
        mask = np.ones((2,2), bool)
        phase = np.array([[-1.,0.],[0.,1.]])
        score, _ = compare_wavefronts(phase, 0.5*phase+100, mask)
        self.assertAlmostEqual(score, 50)

    def test_invalid_zero_and_equal_fnumber_warning(self):
        with self.assertRaises(ValueError):
            validate_parameters(replace(Parameters(), pixel_um=0))
        warnings = validate_parameters(replace(Parameters(), objective_focal_mm=1000/3))
        self.assertTrue(any("F-number warning" in item for item in warnings))


class PropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = replace(Parameters(), bandwidth_nm=0, phase_ptp_rad=1.5)
        cls.model = OpticalModel(cls.p, pupil_resolution=48, camera_min=128)

    def test_normalized_intensity_and_piston_invariance(self):
        coefficients = np.zeros(21)
        a = self.model.forward(coefficients)
        coefficients[0] = 12.3
        b = self.model.forward(coefficients)
        self.assertAlmostEqual(a.sum(), 1, places=12)
        self.assertTrue(np.all(a >= 0))
        np.testing.assert_allclose(a, b, atol=2e-17)

    def test_analytic_intensity_jacobian(self):
        coefficients = np.linspace(-0.03, 0.03, 20)
        _, jac = self.model.forward(coefficients, derivatives=True)
        h = 1e-5
        for k in (0, 2, 9, 19):
            plus, minus = coefficients.copy(), coefficients.copy()
            plus[k] += h
            minus[k] -= h
            numerical = (self.model.forward(plus)-self.model.forward(minus))/(2*h)
            relative = np.linalg.norm(numerical-jac[k])/np.linalg.norm(numerical)
            self.assertLess(relative, 2e-6)

    def test_image_only_low_aberration_reconstruction(self):
        # Reconstruction is supplied image only; perturbing generation metadata is impossible here.
        coeff = np.zeros(20)
        coeff[2] = 0.12
        coeff[6] = -0.08
        image = self.model.forward(coeff)
        _, recovered, info = self.model.reconstruct(image, max_nfev=15)
        np.testing.assert_allclose(recovered[1:], coeff, atol=2e-3)
        self.assertLess(info["image_fit_error"], 1e-3)

    def test_cancel_before_work(self):
        with self.assertRaises(CancelledError):
            simulate(self.p, np.ones(21), cancel=lambda: True)

    def test_objective_fourier_nyquist_warning(self):
        undersampled = OpticalModel(replace(self.p, objective_focal_mm=50))
        self.assertTrue(any("Objective Fourier sampling warning" in item for item in undersampled.warnings))
        self.assertFalse(any("Objective Fourier sampling warning" in item for item in self.model.warnings))

    def test_independent_pupil_sampling_reconstruction(self):
        # Generate on a different pupil grid, avoiding a strictly identical
        # forward/inverse discretization in this additional convergence check.
        low = OpticalModel(self.p, pupil_resolution=80)
        high = OpticalModel(self.p, pupil_resolution=160)
        coeff = np.zeros(20)
        coeff[2], coeff[6], coeff[9] = 0.8, 0.3, 0.2
        image = high.forward(coeff)
        _, recovered, info = low.reconstruct(image, max_nfev=20)
        relative = np.linalg.norm(recovered[1:]-coeff)/np.linalg.norm(coeff)
        self.assertLess(relative, 0.02)
        self.assertLess(info["image_fit_error"], 0.02)

    def test_independent_camera_window_reconstruction(self):
        # A larger propagation window changes Fresnel wraparound and provides
        # an independent numerical check beyond same-window self-consistency.
        parameters = Parameters()
        weights = np.zeros(21)
        weights[3], weights[4], weights[7] = 1.0, 0.25, 0.2
        wavefront = make_wavefront(parameters, weights)
        low = OpticalModel(parameters)
        high = OpticalModel(parameters, camera_min=384)
        large_image = high.forward(wavefront.coefficients)
        offset = (high.n-low.n)//2
        image = large_image[offset:offset+low.n, offset:offset+low.n].copy()
        image /= image.sum()
        phase, _, info = low.reconstruct(image)
        similarity, _ = compare_wavefronts(wavefront.phase, phase, wavefront.mask)
        self.assertGreater(similarity, 99)
        self.assertLess(info["image_fit_error"], 0.03)


if __name__ == "__main__":
    unittest.main()
