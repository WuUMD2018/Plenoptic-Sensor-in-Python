"""One charge-4 camera case using the paper's listed optical dimensions."""
from pathlib import Path
import os
import argparse
import json
from dataclasses import asdict
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from plenoptic_studio.optics import Parameters, OpticalModel, make_wavefront, phase_residual
from plenoptic_studio.vortex_diagnostics import analyze_vortex_camera


def phase_error(truth, phase, mask):
    mask = mask & np.isfinite(truth) & np.isfinite(phase)
    error = phase_residual(truth, phase, mask, wrapped=True)
    return float(np.sqrt(np.mean(error[mask]**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("vortex_results"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    p = Parameters(objective_diameter_mm=30, objective_focal_mm=3000,
                   mla_pitch_um=500, mla_focal_mm=46.7, beam_diameter_mm=30,
                   pixel_um=7, wavelength_nm=633, bandwidth_nm=0,
                   vortex_charge=4)
    model = OpticalModel(p, pupil_resolution=160)
    truth = make_wavefront(p, np.zeros(21), resolution=160)
    image = model.forward(truth.coefficients)
    edge = np.ones_like(image, dtype=bool)
    edge[5:-5, 5:-5] = False
    edge_energy = float(image[edge].sum())
    warnings = list(model.warnings)
    if edge_energy > 0.01:
        warnings.append(f"{100*edge_energy:.3f}% camera energy in outer five pixels: finite-window effects may be material.")
    arrays = {"camera_image": image, "input_phase": truth.phase, "input_mask": truth.mask,
              "input_intensity": truth.intensity, "x_mm": model.axis,
              "camera_axis_mm": model.camera_axis*1e3}
    cases = {}
    data = {}
    for calibrated in (False, True):
        for center in (None, (0, 0)):
            name = ("calibrated" if calibrated else "uncalibrated")+"_"+("detected_center" if center is None else "known_center")
            result = analyze_vortex_camera(model, image, subtract_flat_reference=calibrated, center_mm=center)
            mask = result["valid_mask"] & truth.mask
            cases[name] = {
                "global_wrapped_rmse_rad": phase_error(truth.phase, result["phase_lmse"], mask),
                "branchcut_wrapped_rmse_rad": phase_error(truth.phase, result["phase_branchcut"], mask),
                "evaluation_samples": int(mask.sum()),
                "diagnostics": result["diagnostics"],
            }
            data[name] = result
            for key in ("gx", "gy", "curl", "valid_mask", "phase_lmse", "phase_branchcut"):
                arrays[name+"_"+key] = result[key]
            print(json.dumps({"case": name, "charge": result["diagnostics"]["circulation_charge_estimate"],
                              "center_mm": result["diagnostics"]["center_mm"],
                              "global_rmse": cases[name]["global_wrapped_rmse_rad"],
                              "cut_rmse": cases[name]["branchcut_wrapped_rmse_rad"]}), flush=True)
    report = {
        "paper_doi": "10.1364/OL.41.003169", "parameters": asdict(p),
        "pupil_resolution": 160, "camera_shape": list(image.shape),
        "camera_extent_mm": model.n*model.dx*1e3,
        "camera_width_in_mla_pitches": model.n*model.dx/model.pitch,
        "objective_nyquist_halfwidth_mm": model.objective_nyquist_halfwidth*1e3,
        "edge_energy_fraction_outer_five_pixels": edge_energy,
        "warnings": warnings, "cases": cases,
        "comparison": "Uniform circular phase RMSE on each case's supported mask, excluding its resolution-dependent core and removing circular piston. No scale/sign/rotation fit.",
        "scope": "One ideal, centered, pure charge-4 Gaussian vortex; coherent camera and paper-inspired geometric reconstruction. No generating charge or phase is supplied to diagnostics.",
        "limitations": [
            "The paper's optical dimensions are used, but the 384-pixel camera cap covers only about 5.4 lenslet pitches, not the physical 20x20-cell MLA.",
            "Phase is imprinted at the modeled entrance pupil; the paper expanded a physical phase-plate-modulated beam and propagated it over 3 m before sensing.",
            "An ideal phase imprint does not reproduce scattering/diffraction at the manufactured phase plate's thickness steps.",
            "No detector noise, aberrations, or calibration error. Single geometry and grid are not a convergence study.",
            "Explicit known center and optional flat-gradient subtraction are declared calibration assumptions, not inferred charge or ground truth."
        ],
        "elapsed_seconds": time.perf_counter()-start,
    }
    (output/"paper_geometry_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output/"paper_geometry_results.npz", **arrays)
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.0), layout="constrained")
    extent = [model.axis[0], model.axis[-1], model.axis[0], model.axis[-1]]
    cextent = [model.camera_axis[0]*1e3, model.camera_axis[-1]*1e3]*2
    inputs = [
        (np.log10(image/image.max()+1e-6), "Coherent sensor image (log10 relative intensity)", "magma", cextent, None),
        (np.where(truth.mask, np.angle(np.exp(1j*truth.phase)), np.nan), "Input charge 4 (wrapped phase, rad)", "twilight", extent, np.pi),
        (np.where(truth.mask, truth.intensity, np.nan), "Input Gaussian intensity", "magma", extent, None),
    ]
    for ax, (field, title, cmap, bounds, limit) in zip(axes[0], inputs):
        options = {"vmin": -limit, "vmax": limit} if limit is not None else {}
        im = ax.imshow(field, origin="lower", extent=bounds, cmap=cmap, **options)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax, shrink=.8)
    for row, name in enumerate(("uncalibrated_known_center", "calibrated_known_center"), start=1):
        result, summary = data[name], cases[name]
        label = "Raw geometric" if row == 1 else "Flat-calibrated"
        charge = result["diagnostics"]["circulation_charge_estimate"]
        fields = [(result["curl"], f"{label} curl (rad/mm²)\nCirculation charge: {charge:.3f}", "RdBu_r", None),
                  (np.where(result["valid_mask"], np.angle(np.exp(1j*result["phase_lmse"])), np.nan),
                   f"{label} global LMSE\nCircular RMSE: {summary['global_wrapped_rmse_rad']:.3f} rad", "twilight", np.pi),
                  (np.where(result["valid_mask"], np.angle(np.exp(1j*result["phase_branchcut"])), np.nan),
                   f"{label} branch-cut LMSE\nCircular RMSE: {summary['branchcut_wrapped_rmse_rad']:.3f} rad", "twilight", np.pi)]
        for ax, (field, title, cmap, limit) in zip(axes[row], fields):
            options = {"vmin": -limit, "vmax": limit} if limit is not None else {}
            im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, **options)
            ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
            fig.colorbar(im, ax=ax, shrink=.8)
    for ax in axes.flat:
        ax.title.set_fontsize(10)
    fig.suptitle("Charge-4 diagnostic at the paper's optical dimensions\n160×160 pupil; 384×384 camera; geometric center supplied for plotted reconstructions", fontsize=13)
    fig.savefig(output/"paper_geometry_results.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"edge_energy_fraction": edge_energy, "warnings": warnings,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
