"""Compare charge-aware camera fitting with smooth and gradient-based inverses.

Run ``python -m plenoptic_studio.vortex_validation --output vortex_results``.
The paper-inspired calculation is a geometric diagnostic applied to coherent
synthetic images, not a replication of the paper's physical phase plate.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .optics import OpticalModel, Parameters, make_wavefront, phase_residual
from .vortex_diagnostics import analyze_vortex_camera


def circular_rmse(truth, phase, mask):
    valid = mask & np.isfinite(truth) & np.isfinite(phase)
    if not np.any(valid):
        return None
    residual = phase_residual(truth, phase, valid, wrapped=True)
    return float(np.sqrt(np.mean(residual[valid]**2)))


def _plot_case(destination, model, truth, image, fitted, paper, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    extent = [model.axis[0], model.axis[-1], model.axis[0], model.axis[-1]]
    fields = [
        (np.log10(image/max(image.max(), 1e-30)+1e-5), "Sensor intensity (log)", "magma", None),
        (np.where(truth.mask, np.angle(np.exp(1j*truth.phase)), np.nan), "Input phase (wrapped rad)", "twilight", np.pi),
        (np.where(truth.mask, np.angle(np.exp(1j*fitted)), np.nan), "Charge-aware fit (wrapped rad)", "twilight", np.pi),
        (np.where(paper["valid_mask"], paper["curl"], np.nan), "Camera-derived curl (rad/mm²)", "RdBu_r", None),
        (np.where(paper["valid_mask"], np.angle(np.exp(1j*paper["phase_lmse"])), np.nan), "Geometric global LMSE (wrapped rad)", "twilight", np.pi),
        (np.where(paper["valid_mask"], np.angle(np.exp(1j*paper["phase_branchcut"])), np.nan), "Geometric cut LMSE (wrapped rad)", "twilight", np.pi),
    ]
    for index, (ax, (data, name, cmap, limit)) in enumerate(zip(axes.flat, fields)):
        options = dict(origin="lower", cmap=cmap)
        if index:
            options["extent"] = extent
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
        else:
            ax.set(xlabel="camera x (pixel)", ylabel="camera y (pixel)")
        if limit is not None:
            options.update(vmin=-limit, vmax=limit)
        im = ax.imshow(data, **options)
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(title)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def run(output, charges=range(7), cases=("pure", "mixed"), max_nfev=35,
        refinement=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Monochromatic 633 nm matches the paper's wavelength; the default app
    # geometry is retained and is not advertised as the paper's hardware.
    base = Parameters(bandwidth_nm=0.0, vortex_charge=0)
    inference = OpticalModel(base)
    rows, details = [], []
    start = time.perf_counter()
    for case in cases:
        weights = np.zeros(21)
        if case == "mixed":
            weights[3], weights[5], weights[7] = .6, .25, .15
        for charge in charges:
            print(f"Validating {case}, charge {charge}", flush=True)
            params = replace(base, vortex_charge=charge)
            truth = make_wavefront(params, weights, inference.pupil_resolution)
            image = inference.forward(truth.coefficients, vortex_charge=charge)
            # Inference object remains charge=0 even when generator charge is not.
            phase, coefficients, fit = inference.reconstruct(image, max_nfev=max_nfev)
            smooth, _, smooth_fit = inference.reconstruct(image, max_nfev=max_nfev,
                                                          charge_candidates=(0,))
            paper = analyze_vortex_camera(inference, image)
            calibrated = analyze_vortex_camera(inference, image, subtract_flat_reference=True)
            row = {
                "case": case, "input_charge": charge,
                "recovered_charge": fit["recovered_vortex_charge"],
                "charge_correct": int(fit["recovered_vortex_charge"] == charge),
                "circular_rmse_rad": circular_rmse(truth.phase, phase, truth.mask),
                "image_fit_error": fit["image_fit_error"],
                "smooth_only_circular_rmse_rad": circular_rmse(truth.phase, smooth, truth.mask),
                "smooth_only_image_fit_error": smooth_fit["image_fit_error"],
                "paper_global_circular_rmse_rad": circular_rmse(truth.phase, paper["phase_lmse"], paper["valid_mask"] & truth.mask),
                "paper_cut_circular_rmse_rad": circular_rmse(truth.phase, paper["phase_branchcut"], paper["valid_mask"] & truth.mask),
                "calibrated_cut_circular_rmse_rad": circular_rmse(truth.phase, calibrated["phase_branchcut"], calibrated["valid_mask"] & truth.mask),
                "paper_circulation_charge": paper["diagnostics"]["circulation_charge_estimate"],
                "paper_nearest_integer_charge": paper["diagnostics"]["nearest_integer_charge"],
                "optimizer_success": bool(fit["optimizer_success"]),
            }
            rows.append(row)
            details.append({**row, "camera_fit": fit, "smooth_only_fit": smooth_fit,
                            "paper": paper["diagnostics"],
                            "calibrated_paper": calibrated["diagnostics"]})
            stem = f"{case}_charge_{charge}"
            np.savez_compressed(output/f"{stem}.npz", image=image, truth=truth.phase,
                reconstruction=phase, coefficients=coefficients, mask=truth.mask,
                smooth_reconstruction=smooth, x_mm=truth.x_mm, y_mm=truth.y_mm,
                gradient_x=paper["gx"], gradient_y=paper["gy"], curl=paper["curl"],
                gradient_mask=paper["valid_mask"], global_lmse=paper["phase_lmse"],
                branchcut_lmse=paper["phase_branchcut"])
            if charge in (1, 4, 6):
                _plot_case(output/f"{stem}.png", inference, truth, image, phase, paper,
                           f"{case.title()} Gaussian vortex · charge {charge} · 633 nm")
            print(json.dumps(row), flush=True)
            # Persist partial evidence so an interrupted long run is inspectable.
            (output/"results.json").write_text(json.dumps({"parameters": asdict(base),
                "rows": rows, "details": details}, indent=2), encoding="utf-8")
    extra = None
    if refinement:
        print("Independent pupil-grid check: charge 4, 120 -> 80 samples", flush=True)
        weights = np.zeros(21)
        weights[3], weights[5], weights[7] = .6, .25, .15
        p4 = replace(base, vortex_charge=4)
        coarse_truth = make_wavefront(p4, weights, inference.pupil_resolution)
        fine = OpticalModel(base, pupil_resolution=120)
        image = fine.forward(coarse_truth.coefficients, vortex_charge=4)
        phase, coeff, fit = inference.reconstruct(image, max_nfev=max_nfev)
        extra = {"generator_pupil_resolution": 120, "inverse_pupil_resolution": 80,
            "input_charge": 4, "recovered_charge": fit["recovered_vortex_charge"],
            "circular_rmse_rad": circular_rmse(coarse_truth.phase, phase, coarse_truth.mask),
            "diagnostics": fit, "same_physical_coefficients": True}
        print(json.dumps(extra), flush=True)
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"parameters": asdict(base), "rows": rows, "details": details,
        "independent_grid": extra, "elapsed_seconds": time.perf_counter()-start,
        "scope": "Ideal centered phase-only Gaussian vortex; charge0–6 prior; same-model noiseless fits plus optional independent pupil grid. Geometric diagnostic is inspired by Wu/Ko/Davis 2016, not experimental replication.",
        "paper_doi": "10.1364/OL.41.003169"}
    (output/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("vortex_results"))
    parser.add_argument("--charges", type=int, nargs="+", choices=range(7), default=list(range(7)))
    parser.add_argument("--cases", nargs="+", choices=("pure", "mixed"), default=["pure", "mixed"])
    parser.add_argument("--max-nfev", type=int, default=35)
    parser.add_argument("--refinement", action="store_true")
    args = parser.parse_args()
    run(args.output, args.charges, args.cases, args.max_nfev, args.refinement)


if __name__ == "__main__":
    main()
