"""Paper-inspired geometric diagnostics for an intensity-only plenoptic image.

Wu, Ko & Davis, Optics Letters (2016), doi:10.1364/OL.41.003169,
Eqs. (2)--(4), motivate the pixel-to-pupil gradient mapping below. The
comparison integrates the *same measured gradients* with global least squares
and with one explicitly chosen radial branch cut. No generating phase, charge,
or Zernike coefficients enter either reconstruction.

This is a geometric diagnostic, not an exact reproduction of the experiment:
coherent diffraction, overlapping light paths, finite windows and sampling can
bias the mapped gradients. An ideal phase vortex does not model scattering at
the physical thickness steps of the paper's phase plate. A chosen cut is a
phase gauge, not a claim to have measured those physical plate edges.
"""
from __future__ import annotations

import numpy as np


def _scipy():
    try:
        from scipy import ndimage, sparse
        from scipy.sparse import csgraph, linalg
    except ImportError as exc:
        raise RuntimeError(
            "Paper-inspired vortex diagnostics require SciPy (sparse LSQR and image interpolation). "
            "The main numerical model remains usable without these optional diagnostics."
        ) from exc
    return ndimage, sparse, csgraph, linalg


def _grid(axis_mm, gx, gy, valid_mask, weights):
    axis = np.asarray(axis_mm, dtype=float)
    if axis.ndim != 1 or len(axis) < 5 or not np.all(np.isfinite(axis)):
        raise ValueError("axis_mm must be a finite one-dimensional grid with at least five points.")
    step = float(axis[1] - axis[0])
    if step <= 0 or not np.allclose(np.diff(axis), step, rtol=1e-8, atol=step*1e-9):
        raise ValueError("axis_mm must be uniformly spaced and strictly increasing.")
    shape = (len(axis), len(axis))
    gx, gy = np.asarray(gx, dtype=float), np.asarray(gy, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if gx.shape != shape or gy.shape != shape or valid.shape != shape:
        raise ValueError("Gradient arrays and valid_mask must match the square pupil grid.")
    if not np.all(np.isfinite(gx[valid])) or not np.all(np.isfinite(gy[valid])):
        raise ValueError("Gradients must be finite on valid_mask.")
    weight = np.ones(shape) if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != shape or np.any(~np.isfinite(weight[valid])) or np.any(weight[valid] < 0):
        raise ValueError("weights must match the grid and be finite and nonnegative on valid_mask.")
    valid = valid & (weight > 0)
    if np.count_nonzero(valid) < 16:
        raise ValueError("At least 16 supported pupil samples are needed.")
    weight = np.where(valid, weight / weight[valid].max(), 0.0)
    return axis, gx, gy, valid, weight, step


def _curl(gx, gy, valid, step):
    ndi, _, _, _ = _scipy()
    # An eroded mask keeps zero-filled exterior samples out of the derivatives.
    inside = ndi.binary_erosion(valid, structure=np.ones((3, 3)))
    value = (np.gradient(np.where(valid, gy, 0.0), step, axis=1)
             - np.gradient(np.where(valid, gx, 0.0), step, axis=0))
    return np.where(inside, value, np.nan), inside


def _center_from_curl(axis, curl, curl_valid, step):
    xx, yy = np.meshgrid(axis, axis)
    radius = float(np.max(np.hypot(xx[curl_valid], yy[curl_valid]))) if np.any(curl_valid) else 0.0
    search = curl_valid & (np.hypot(xx, yy) <= 0.5*radius)
    if not np.any(search):
        return (0.0, 0.0), False, 0.0
    magnitude = np.where(search, np.abs(curl), 0.0)
    peak = float(np.max(magnitude))
    if peak < 1e-10:
        return (0.0, 0.0), False, peak
    row, col = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    # Combine equal peaks around an even-grid singularity without combining
    # distant artifacts. This remains a local peak estimate, not a detector proof.
    near = search & (np.hypot(xx-axis[col], yy-axis[row]) <= 2*step) & (magnitude >= 0.5*peak)
    total = float(magnitude[near].sum())
    return (float((xx[near]*magnitude[near]).sum()/total),
            float((yy[near]*magnitude[near]).sum()/total)), True, peak


def _circulation(axis, gx, gy, valid, center, core_radius):
    ndi, _, _, _ = _scipy()
    xx, yy = np.meshgrid(axis, axis)
    rr = np.hypot(xx-center[0], yy-center[1])
    outer = float(np.max(rr[valid]))
    step = float(axis[1]-axis[0])
    samples = max(360, 8*len(axis))
    angles = np.arange(samples)*2*np.pi/samples
    radii = np.unique(np.maximum(np.array([0.25, 0.4, 0.55, 0.7, 0.8])*outer, 1.6*core_radius))
    results = []
    for radius in radii:
        if radius >= 0.9*outer:
            continue
        x = center[0] + radius*np.cos(angles)
        y = center[1] + radius*np.sin(angles)
        coords = [(y-axis[0])/step, (x-axis[0])/step]
        support = ndi.map_coordinates(valid.astype(float), coords, order=1, mode="constant", cval=0.0)
        coverage = float(np.mean(support >= 0.999))
        # Do not infer a whole contour by averaging over missing angular arcs.
        if coverage < 1.0:
            results.append({"radius_mm": float(radius), "coverage": coverage, "charge": None})
            continue
        sx = ndi.map_coordinates(np.where(valid, gx, 0.0), coords, order=1, mode="constant", cval=0.0)
        sy = ndi.map_coordinates(np.where(valid, gy, 0.0), coords, order=1, mode="constant", cval=0.0)
        charge = float(radius*np.mean(-sx*np.sin(angles)+sy*np.cos(angles)))
        results.append({"radius_mm": float(radius), "coverage": coverage, "charge": charge})
    charges = [row["charge"] for row in results if row["charge"] is not None]
    estimate = float(np.median(charges)) if charges else None
    return results, estimate


def _integrate(gx, gy, valid, weight, step, theta=None):
    _, sparse, csgraph, linalg = _scipy()
    count = int(np.count_nonzero(valid))
    indices = np.full(valid.shape, -1, dtype=int)
    indices[valid] = np.arange(count)
    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    removed = 0
    if theta is not None:
        cross_h = np.abs(theta[:, 1:]-theta[:, :-1]) > np.pi
        cross_v = np.abs(theta[1:, :]-theta[:-1, :]) > np.pi
        removed = int(np.count_nonzero(horizontal & cross_h)+np.count_nonzero(vertical & cross_v))
        horizontal &= ~cross_h
        vertical &= ~cross_v
    first = np.r_[indices[:, :-1][horizontal], indices[:-1, :][vertical]]
    second = np.r_[indices[:, 1:][horizontal], indices[1:, :][vertical]]
    targets = step*np.r_[0.5*(gx[:, :-1]+gx[:, 1:])[horizontal],
                        0.5*(gy[:-1, :]+gy[1:, :])[vertical]]
    edge_weight = np.r_[np.minimum(weight[:, :-1], weight[:, 1:])[horizontal],
                        np.minimum(weight[:-1, :], weight[1:, :])[vertical]]
    if len(first) == 0:
        raise ValueError("Supported gradient samples do not have any neighboring edges.")
    root_weight = np.sqrt(edge_weight)
    edge_count = len(first)
    graph = sparse.coo_matrix((np.ones(2*edge_count),
                              (np.r_[first, second], np.r_[second, first])), shape=(count, count)).tocsr()
    components, labels = csgraph.connected_components(graph, directed=False)
    anchors = np.array([np.flatnonzero(labels == k)[0] for k in range(components)])
    rows = np.r_[np.arange(edge_count), np.arange(edge_count), edge_count+np.arange(components)]
    cols = np.r_[first, second, anchors]
    values = np.r_[-root_weight, root_weight, np.ones(components)]
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(edge_count+components, count)).tocsr()
    rhs = np.r_[targets*root_weight, np.zeros(components)]
    fit = linalg.lsqr(matrix, rhs, atol=1e-9, btol=1e-9, iter_lim=min(10000, max(1000, 4*count)))
    phase_values = fit[0]
    for k in range(components):
        region = labels == k
        phase_values[region] -= phase_values[region].mean()
    residual = phase_values[second]-phase_values[first]-targets
    phase = np.full(valid.shape, np.nan)
    phase[valid] = phase_values
    return phase, {
        "edge_count": int(edge_count), "cut_edges_removed": removed,
        "connected_components": int(components), "lsqr_stop_code": int(fit[1]),
        "lsqr_iterations": int(fit[2]),
        "gradient_residual_rms_rad_per_mm": float(np.sqrt(np.average(residual**2, weights=edge_weight))/step),
        "piston_gauge": "zero mean independently in each connected component",
    }


def reconstruct_gradient_field(axis_mm, gx, gy, valid_mask, *, weights=None,
                               center_mm=None, cut_angle_rad=0.0, core_radius_mm=None):
    """Compare global and radial-cut LSQR integration of measured gradients.

    Gradients are in rad/mm on a uniform square pupil grid. ``center_mm=None``
    estimates the center from the strongest inner curl feature; supplying a
    center declares prior geometric calibration. Neither phase nor charge is an
    input. The radial cut is arbitrary, measured counterclockwise from +x.
    Phase outputs are NaN outside ``valid_mask`` and inside the excluded core.
    """
    axis, gx, gy, valid, weight, step = _grid(axis_mm, gx, gy, valid_mask, weights)
    if not np.isfinite(cut_angle_rad):
        raise ValueError("cut_angle_rad must be finite.")
    curl, curl_valid = _curl(gx, gy, valid, step)
    detected_center, detected, peak = _center_from_curl(axis, curl, curl_valid, step)
    if center_mm is None:
        center = detected_center
        center_source = "inner absolute-curl peak" if detected else "origin fallback; no significant curl peak"
    else:
        pair = np.asarray(center_mm, dtype=float)
        if pair.shape != (2,) or not np.all(np.isfinite(pair)):
            raise ValueError("center_mm must be a finite (x, y) pair or None.")
        center = (float(pair[0]), float(pair[1]))
        center_source = "explicitly supplied geometric center"
    core = 2.5*step if core_radius_mm is None else float(core_radius_mm)
    if not np.isfinite(core) or core < 0:
        raise ValueError("core_radius_mm must be finite and nonnegative.")
    xx, yy = np.meshgrid(axis, axis)
    solve_mask = valid & (np.hypot(xx-center[0], yy-center[1]) >= core)
    if np.count_nonzero(solve_mask) < 16:
        raise ValueError("Core exclusion leaves too few supported pupil samples.")
    theta = np.mod(np.arctan2(yy-center[1], xx-center[0])-cut_angle_rad, 2*np.pi)
    ordinary, ordinary_info = _integrate(gx, gy, solve_mask, weight, step)
    branchcut, cut_info = _integrate(gx, gy, solve_mask, weight, step, theta)
    circles, charge = _circulation(axis, gx, gy, solve_mask, center, core)
    charge_values = [r["charge"] for r in circles if r["charge"] is not None]
    warnings = []
    if center_mm is None:
        warnings.append("Curl-peak localization is heuristic; diffraction or sampling artifacts can move the inferred center.")
    if not charge_values:
        warnings.append("No complete supported contour is available for a circulation/charge diagnostic.")
    if ordinary_info["connected_components"] > 1 or cut_info["connected_components"] > 1:
        warnings.append("Disconnected support has independently undetermined phase offsets.")
    if ordinary_info["lsqr_stop_code"] == 7 or cut_info["lsqr_stop_code"] == 7:
        warnings.append("At least one least-squares integration reached its iteration limit.")
    diagnostics = {
        "method": "measured-gradient global LSQR versus one radial branch cut",
        "gradient_units": "rad/mm", "curl_units": "rad/mm^2",
        "center_mm": list(center), "center_source": center_source,
        "inner_curl_peak_center_mm": list(detected_center), "inner_curl_peak_abs": peak,
        "core_radius_mm": core, "cut_angle_rad": float(cut_angle_rad),
        "supported_samples": int(np.count_nonzero(solve_mask)),
        "circulation_contours": circles,
        "circulation_charge_estimate": charge,
        "nearest_integer_charge": int(np.rint(charge)) if charge is not None else None,
        "circulation_charge_spread": float(np.ptp(charge_values)) if charge_values else None,
        "global_lmse": ordinary_info, "branchcut_lmse": cut_info, "warnings": warnings,
        "interpretation": "Circulation is a diagnostic of the measured gradient field, not a guaranteed optical charge measurement.",
    }
    return {"x_mm": axis.copy(), "y_mm": axis.copy(), "gx": gx.copy(), "gy": gy.copy(),
            "weight": weight, "curl": curl, "curl_valid_mask": curl_valid,
            "gradient_valid_mask": valid, "valid_mask": solve_mask,
            "phase_lmse": ordinary, "phase_branchcut": branchcut, "diagnostics": diagnostics}


def _map_camera(model, image, relative_threshold):
    ndi, _, _, _ = _scipy()
    image = np.asarray(image, dtype=float)
    if image.shape != (model.n, model.n) or not np.all(np.isfinite(image)) or np.any(image < 0):
        raise ValueError("Camera image must be finite, nonnegative, and match the calibrated camera grid.")
    if image.sum() <= 0:
        raise ValueError("Camera image must contain positive intensity.")
    image = image / image.sum()
    xx, yy = np.meshgrid(np.asarray(model.axis)*1e-3, np.asarray(model.axis)*1e-3)
    local_x, local_y = -model.b/model.a*xx, -model.b/model.a*yy
    support = np.asarray(model.mask, dtype=bool) & (np.abs(local_x) <= model.pitch/2) & (np.abs(local_y) <= model.pitch/2)
    den, sx, sy = np.zeros_like(xx), np.zeros_like(xx), np.zeros_like(xx)
    centers_x = np.unique(model.cell_x)
    centers_y = np.unique(model.cell_y)
    factor = 2*np.pi/(model.p.wavelength_nm*1e-9)/model.a*1e-3
    for cx in centers_x:
        for cy in centers_y:
            coords = [(cy+local_y-model.camera_axis[0])/model.dx,
                      (cx+local_x-model.camera_axis[0])/model.dx]
            values = ndi.map_coordinates(image, coords, order=1, mode="constant", cval=0.0)*support
            den += values
            sx += values*(cx-local_x)
            sy += values*(cy-local_y)
    if not np.any(den > 0):
        raise ValueError("No camera intensity maps into the supported entrance pupil.")
    valid = support & (den > float(den.max())*relative_threshold)
    gx = factor*sx/np.maximum(den, np.finfo(float).tiny)
    gy = factor*sy/np.maximum(den, np.finfo(float).tiny)
    return gx, gy, den, valid


def analyze_vortex_camera(model, image, *, subtract_flat_reference=False, center_mm=None,
                          cut_angle_rad=0.0, core_radius_mm=None, min_relative_intensity=1e-4):
    """Map a calibrated camera image to paper-inspired vortex diagnostics.

    ``model`` supplies only sensor geometry, wavelength, pupil support, and (if
    requested) the forward calculation of a flat zero-charge calibration beam.
    ``subtract_flat_reference=True`` explicitly subtracts that known-flat
    gradient bias; it is an extra calibration, not a step claimed by the paper.
    Use the central wavelength for geometric mapping of polychromatic images.
    """
    if not np.isfinite(min_relative_intensity) or not 0 <= min_relative_intensity < 1:
        raise ValueError("min_relative_intensity must lie in [0, 1).")
    ndi, _, _, _ = _scipy()
    gx, gy, weight, valid = _map_camera(model, image, min_relative_intensity)
    if subtract_flat_reference:
        reference = model.forward(np.zeros(21), vortex_charge=0)
        rx, ry, _, reference_valid = _map_camera(model, reference, min_relative_intensity)
        valid &= reference_valid
        gx, gy = gx-rx, gy-ry
    # The paper uses a 3x3 Gaussian with sigma=1 pixel. Normalize at support
    # boundaries instead of leaking zero exterior values into the gradients.
    coordinate = np.arange(-1, 2, dtype=float)
    kernel = np.exp(-0.5*(coordinate[:, None]**2+coordinate[None, :]**2))
    kernel /= kernel.sum()
    norm = ndi.convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
    gx = ndi.convolve(np.where(valid, gx, 0.0), kernel, mode="constant", cval=0.0)/np.maximum(norm, 1e-30)
    gy = ndi.convolve(np.where(valid, gy, 0.0), kernel, mode="constant", cval=0.0)/np.maximum(norm, 1e-30)
    result = reconstruct_gradient_field(model.axis, gx, gy, valid, weights=weight,
                                       center_mm=center_mm, cut_angle_rad=cut_angle_rad,
                                       core_radius_mm=core_radius_mm)
    info = result["diagnostics"]
    info.update({"source": "camera intensity and calibrated geometry only",
                 "reference": "Wu, Ko & Davis (2016), doi:10.1364/OL.41.003169, Eqs. 2-4",
                 "flat_reference_subtracted": bool(subtract_flat_reference),
                 "smoothing": "normalized masked 3x3 Gaussian, sigma=1 pupil-grid pixel",
                 "min_relative_mapped_intensity": float(min_relative_intensity),
                 "wavelength_for_mapping_nm": float(model.p.wavelength_nm)})
    info["warnings"].append("Geometric camera mapping neglects coherent diffraction and path overlap; compare against independent data before interpreting phase accuracy.")
    if model.p.bandwidth_nm > 0:
        info["warnings"].append("Gradient mapping uses the central wavelength for the spectrally integrated image.")
    return result
