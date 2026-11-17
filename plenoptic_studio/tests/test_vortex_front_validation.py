"""Core coverage for the front-panel pure-vortex validation sweep."""
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np

from plenoptic_studio import optics
from plenoptic_studio.optics import Parameters, CancelledError, validate_vortex_charges


class VortexValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Non-default settings expose accidental resets while retaining a small
        # numerical grid. A nonzero selected smooth amplitude must be overridden.
        cls.parameters = Parameters(objective_diameter_mm=16, objective_focal_mm=640,
                                    beam_diameter_mm=8, pixel_um=10, wavelength_nm=532,
                                    bandwidth_nm=0, phase_ptp_rad=9, vortex_charge=5)
        cls.streamed = []
        cls.progress = []
        cls.results = validate_vortex_charges(
            cls.parameters, progress=cls.progress.append,
            result_callback=lambda index, result: cls.streamed.append((index, result)),
            max_nfev=10)

    def test_all_six_charges_are_independently_inferred(self):
        self.assertEqual(len(self.results), 6)
        for charge, result in enumerate(self.results, 1):
            with self.subTest(charge=charge):
                self.assertEqual(result.wavefront.vortex_charge, charge)
                self.assertEqual(result.metadata["recovered_vortex_charge"], charge)
                self.assertEqual(result.metadata["charge_candidates"], list(range(7)))
                self.assertEqual(result.metadata["vortex_charge_error"], 0)
                self.assertTrue(result.metadata["wrapped_phase_comparison"])
                self.assertAlmostEqual(result.metadata["input_winding_number"], charge)
                self.assertAlmostEqual(result.metadata["reconstructed_winding_number"], charge)
                self.assertGreater(result.similarity_pct, 99.999)
                self.assertLess(result.metadata["image_fit_error"], 1e-8)
                self.assertTrue(np.isfinite(result.image).all())
                self.assertAlmostEqual(result.image.sum(), 1)

    def test_optical_settings_preserved_and_zernike_phase_cleared(self):
        self.assertEqual(self.parameters.phase_ptp_rad, 9)
        self.assertEqual(self.parameters.vortex_charge, 5)
        for charge, result in enumerate(self.results, 1):
            with self.subTest(charge=charge):
                expected = replace(self.parameters, vortex_charge=charge, phase_ptp_rad=0.0)
                self.assertEqual(result.metadata["parameters"], asdict(expected))
                np.testing.assert_array_equal(result.wavefront.coefficients, np.zeros(21))
                self.assertEqual(result.metadata["actual_zernike_phase_ptp_rad"], 0)
                self.assertEqual(result.metadata["relative_weights"], [0.0]*21)
                self.assertFalse(any("No nonpiston Zernike" in warning for warning in result.warnings))

    def test_results_stream_in_order_with_distinct_vortex_metadata(self):
        self.assertEqual([index for index, _ in self.streamed], list(range(6)))
        for index, result in self.streamed:
            self.assertIs(result, self.results[index])
            self.assertEqual(result.metadata["validation_kind"], "vortex")
            self.assertEqual(result.metadata["validation_charge"], index+1)
            self.assertIn(f"q={index+1}", result.metadata["validation_name"])
            self.assertNotIn("mode_j", result.metadata)
        for charge in range(1, 7):
            self.assertTrue(any(message.startswith(f"Vortex {charge}/6 · ") for message in self.progress))

    def test_cancellation_before_start_does_not_compute(self):
        with patch.object(optics, "simulate") as simulation:
            with self.assertRaises(CancelledError):
                validate_vortex_charges(self.parameters, cancel=lambda: True)
            simulation.assert_not_called()

    def test_cancellation_preserves_completed_streamed_cases(self):
        completed = []
        def receive(index, result):
            completed.append((index, result))
        with self.assertRaises(CancelledError):
            validate_vortex_charges(self.parameters, cancel=lambda: len(completed) == 2,
                                   result_callback=receive, max_nfev=10)
        self.assertEqual([index for index, _ in completed], [0, 1])
        self.assertEqual([result.metadata["recovered_vortex_charge"] for _, result in completed], [1, 2])
        self.assertEqual([result.metadata["validation_charge"] for _, result in completed], [1, 2])


if __name__ == "__main__":
    unittest.main()
