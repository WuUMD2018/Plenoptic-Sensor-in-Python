"""Reproducible, headless examples: python -m plenoptic_studio.demo [--validate]."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from .optics import MODES, Parameters, simulate, validate_all


def plot_result(result, destination, title="Plenoptic Studio · image-only wavefront reconstruction"):
    """Save physical values in NPZ separately; display camera with a log stretch."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), layout="constrained")
    fig.patch.set_facecolor("#101c2e")
    fig.suptitle(title, color="#e5edf8", fontsize=15)
    wf = result.wavefront
    radius = result.metadata["parameters"]["beam_diameter_mm"]/2
    extent = [-radius, radius, -radius, radius]
    magnitude = max(float(np.max(np.abs(wf.phase[wf.mask]))), .01)
    data = [np.log10(result.image/max(float(result.image.max()), 1e-20)+1e-5),
            np.where(wf.mask, wf.phase, np.nan),
            np.where(wf.mask, result.reconstructed_phase, np.nan),
            np.where(wf.mask, result.reconstructed_phase-wf.phase, np.nan)]
    labels = ["Camera · log₁₀ relative intensity", "Ground truth · rad", "Reconstruction · rad", "Residual · rad"]
    for i, (ax, values, text) in enumerate(zip(axes, data, labels)):
        ax.set_facecolor("#101c2e")
        ax.set_title(text, color="#e5edf8", fontsize=10)
        ax.tick_params(colors="#8fa4bf", labelsize=8)
        if i == 0:
            half = result.metadata["camera_extent_mm"]/2
            image = ax.imshow(values, origin="lower", extent=[-half, half, -half, half], cmap="magma", vmin=-5, vmax=0)
        elif i == 3:
            bound = max(float(np.nanmax(np.abs(values))), 1e-7)
            image = ax.imshow(values, origin="lower", extent=extent, cmap="RdBu_r", vmin=-bound, vmax=bound)
        else:
            image = ax.imshow(values, origin="lower", extent=extent, cmap="RdBu_r", vmin=-magnitude, vmax=magnitude)
        ax.set_xlabel("x / mm", color="#8fa4bf", fontsize=9)
        ax.set_ylabel("y / mm", color="#8fa4bf", fontsize=9)
        colorbar = fig.colorbar(image, ax=ax, shrink=.65, pad=.03)
        colorbar.ax.tick_params(colors="#8fa4bf", labelsize=7)
    score = "N/A" if result.similarity_pct is None else f"{result.similarity_pct:.5f}%"
    fig.text(.5, .005, f"Similarity {score}  |  RMSE {result.rms_error_rad:.3g} rad  |  noiseless simulation consistency; not experimental accuracy",
             ha="center", color="#a8bfd6", fontsize=10)
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example_output"))
    parser.add_argument("--validate", action="store_true", help="Also run and export all 21 Noll modes")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = np.zeros(21)
    weights[3], weights[4], weights[7] = 1, .25, .2
    result = simulate(Parameters(), weights, progress=print)
    plot_result(result, args.output/"simulation.png")
    np.savez_compressed(args.output/"simulation.npz", camera=result.image,
        truth=result.wavefront.phase, reconstruction=result.reconstructed_phase,
        mask=result.wavefront.mask, intensity=result.wavefront.intensity,
        x_mm=result.wavefront.x_mm, y_mm=result.wavefront.y_mm,
        coefficients=result.coefficients, truth_coefficients=result.wavefront.coefficients)
    (args.output/"simulation.json").write_text(json.dumps({**result.metadata,
        "similarity_pct":result.similarity_pct, "rms_error_rad":result.rms_error_rad,
        "warnings":result.warnings}, indent=2), encoding="utf-8")
    if args.validate:
        results = validate_all(Parameters(), progress=print)
        with (args.output/"validation.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Noll index", "mode", "similarity percent", "RMSE rad", "image fit residual", "warnings"])
            for mode, res in zip(MODES, results):
                writer.writerow([mode.j, mode.name, res.similarity_pct, res.rms_error_rad,
                                 res.metadata["image_fit_error"], " | ".join(res.warnings)])
                plot_result(res, args.output/f"zernike_{mode.j:02}.png", f"Noll {mode.j} · {mode.name}")
        fig, axes = plt.subplots(21, 3, figsize=(9, 42), layout="constrained")
        for row, (mode, res) in enumerate(zip(MODES, results)):
            wf = res.wavefront
            vmax = max(np.max(np.abs(wf.phase[wf.mask])), .01)
            maps = [np.where(wf.mask, wf.phase, np.nan), np.log10(res.image/res.image.max()+1e-5),
                    np.where(wf.mask, res.reconstructed_phase, np.nan)]
            for col, data in enumerate(maps):
                axes[row,col].imshow(data, origin="lower", cmap="magma" if col==1 else "RdBu_r",
                    vmin=-5 if col==1 else -vmax, vmax=0 if col==1 else vmax)
                axes[row,col].set_xticks([])
                axes[row,col].set_yticks([])
            score = "N/A · piston" if res.similarity_pct is None else f"{res.similarity_pct:.4f}%"
            axes[row,0].set_ylabel(f"j{mode.j} {mode.name}", fontsize=8)
            axes[row,2].set_title(score, fontsize=8)
        for ax, title in zip(axes[0], ["Ground truth / rad", "Camera / log intensity", "Reconstructed / rad"]):
            ax.set_title(title, fontsize=11)
        fig.suptitle("21 Zernike modes · noiseless model consistency", fontsize=14)
        fig.savefig(args.output/"validation_gallery.png", dpi=100)
        plt.close(fig)
    print(f"Saved examples in {args.output.resolve()}")


if __name__ == "__main__":
    main()
