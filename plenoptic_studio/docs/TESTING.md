# Verification record

## Front-panel validation, version 1.2 — 2026-11-16

**46 tests passed in 70.816 seconds**, including the six-charge numerical sweep, streaming and cancellation, preservation of optical settings, independent charge inference, GUI sweep selection, separate result snapshots, mismatch display and partial exports. After the final dropdown and compact-layout changes, all **14 GUI tests passed again in 3.886 seconds**.

The real Qt worker completed both **6/6 vortex cases** and **21/21 Zernike cases**, with row selection and CSV/PDF/NPZ/JSON export checks. These runs used the default optics with monochromatic 633 nm light. The vortex sweep took 24.50 seconds and the Zernike sweep 185.31 seconds including result drawing and row inspection, under concurrent local testing; these are not performance benchmarks. The six vortex charges were correctly inferred, and every exported result retained its original parameters after input edits. The PDF exports contained six and 21 pages respectively. [Detailed GUI run evidence](../../docs/vortex_validation/gui_sweep_results.json) and [test log](../../docs/vortex_validation/front_validation_tests.txt) are included.

A separate native-window run at the default 0.1 nm spectral FWHM recovered charges 1–6. The dropdown, all six table rows and phase maps were visually checked, and the compact 1120×780 layout was rendered and inspected. The vortex table and its zero-Zernike scope note remain visible at that size; plots can be scrolled. [GUI screenshot](../../docs/vortex_validation/gui_sweep.png).

These are noiseless, same-model software consistency checks. They do not establish experimental accuracy or validate the separate paper-inspired geometric reconstruction.

## Vortex extension, 2026-11-16

**37 tests passed** in 61.162 seconds on Windows/Python 3.13.2 (NumPy 2.5.5, SciPy 1.18.1, Matplotlib 3.11.2, PySide6 6.12.0). This includes the original regressions, integer charge validation, additive phase and winding, zero-charge compatibility, nonzero-charge analytic derivatives, image-only candidate selection, wrapped metrics, GUI snapshots/exports, and gradient integration with branch cuts.

The native GUI was checked separately for the charge slider and a charge-4 run. A visual defect in the integer spinbox was fixed and rechecked at 0/4/6. The 0–6 pure/mixed sweep recovered all 14 input charges; an independent 120→80 pupil-grid check recovered charge 4. [VORTEX.md](VORTEX.md) contains numerical results and the geometric diagnostic's limitations, including unsuccessful charge recovery using the paper's lens dimensions. The demo and paper-geometry command-line entry points were also executed successfully.

## Publication check — 2026-11-16

Re-ran `python -m unittest discover -s plenoptic_studio/tests -v` on Windows using the tested source folder's virtual environment: **17 tests passed in 4.139 seconds** (6.18 seconds including process startup). Environment: Python 3.14.5, NumPy 2.5.5, SciPy 1.18.1, Matplotlib 3.11.2, and PySide6 6.12.0. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `QT_QPA_PLATFORM=offscreen`.

This check covers the existing GUI and numerical regression suite. The earlier verification, generated examples, visual inspection, and saved `test_results.txt` below are retained as historical evidence; they were not all regenerated during this publication check.

## Earlier verification

Executed on Windows with Python 3.13.2, NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.7, and PySide6 6.10.0. GUI checks used Qt's offscreen platform. Dependencies passed `pip check`.

## Automated checks

**17 tests passed**. The complete output is in `test_results.txt`.

- Gaussian intensity convention and requested phase peak-to-peak.
- Noll ordering, piston invariance/unobservability, and amplitude-sensitive similarity.
- Invalid zero inputs, f-number warning, and objective Fourier sampling warning.
- Analytic camera-image Jacobian compared with finite differences.
- Image-only inversion, with no generating coefficients supplied to the inverse.
- Independent input-pupil sampling and finite-camera-window checks.
- Cancellation before numerical work.
- Main slider anchoring, including explicitly retyping the current value.
- 3D composer and piston-only behavior.
- Separation of custom-run and validation-tab state.
- Background-thread delivery and restoration of controls.
- Export files and preservation of the successful run's settings after input edits.

## Numerical examples

`examples/validation.csv` contains a fresh SciPy run of all Noll modes 1–21 at the default 6 rad peak-to-peak. All 20 non-piston modes achieved greater than 99.999999% same-model similarity. Piston reports N/A. The mixed-mode example includes its camera image, phase arrays, coefficients and metadata.

`validation_reference.json` contains earlier NumPy-fallback results, explicitly labeled by backend. This file also records independent pupil-grid and propagation-window checks. A representative larger-window test gave approximately 99.926% phase similarity, 0.00102 rad RMSE, and 0.01194 square-root intensity discrepancy when reconstructed using a smaller window. The pupil-refinement check gave approximately 0.141% relative coefficient error. These representative tests do not prove convergence for arbitrary slider settings.

**Interpretation:** excellent same-model recovery is a software consistency check. It is not a prediction of physical accuracy, a unique-solution guarantee, or evidence of robustness to unknown beam amplitude, detector noise, calibration errors, vortices, or unmodeled higher-order modes.

## Visual inspection

The actual desktop widgets were rendered and inspected at 1480×1040 and at a compact 1120×780 size. Fonts, nine parameter controls, optical-bench illustration, phase maps, color bars, labels, and the rotatable composer were checked. At smaller heights, the plot and control panels scroll. Screenshots in this folder show actual widget output, not design mockups.

See `gui_smoke_results.json` for the completed full GUI validation/export workflow and its measured timing; `validation_preview.png` shows the completed validation browser.
