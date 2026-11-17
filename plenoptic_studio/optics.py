"""Scalar paraxial plenoptic sensor simulator and image-only vortex/Zernike inversion.

Geometry follows Wu, Ko & Davis (2015), doi:10.1364/JOSAA.32.000964:
pupil -- f_o -- objective -- (f_o + f_l) -- MLA -- f_l -- camera.
The objective Fourier plane is propagated f_l to the physical microlens array.
Spectral components share Zernike optical path difference and add in intensity.
An optional ideal achromatic phase imprint exp(i*q*theta) supplies integer winding.

The objective diameter is represented by an entrance-pupil stop; separate
aberration-dependent vignetting at the physical objective is not modeled.
This is a finite-grid, ideal-lens model, not a calibrated instrument model.
No ground-truth coefficients or phase map are supplied to reconstruction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import factorial
from typing import Callable
import time

import numpy as np
try:
    from scipy import fft, ndimage, optimize
    NUMERICAL_BACKEND = "SciPy FFT / trust-region least squares"
except ImportError:
    # The NumPy fallback keeps the numerical engine usable in restricted/offline
    # installations. SciPy's trust-region solver remains the preferred path.
    from types import SimpleNamespace
    NUMERICAL_BACKEND = "NumPy FFT / damped Gauss-Newton fallback"

    class _FFT:
        @staticmethod
        def next_fast_len(n):
            while True:
                value = n
                for factor in (2, 3, 5, 7):
                    while value % factor == 0:
                        value //= factor
                if value == 1:
                    return n
                n += 1

        fftfreq = staticmethod(np.fft.fftfreq)

        @staticmethod
        def fft2(a, axes=(-2,-1), workers=1):
            return np.fft.fft2(a, axes=axes)

        @staticmethod
        def ifft2(a, axes=(-2,-1), workers=1):
            return np.fft.ifft2(a, axes=axes)

    class _NDImage:
        @staticmethod
        def binary_erosion(mask, iterations=1):
            result = mask.copy()
            for _ in range(iterations):
                inner = np.zeros_like(result)
                inner[1:-1,1:-1] = (result[1:-1,1:-1] & result[:-2,1:-1] &
                    result[2:,1:-1] & result[1:-1,:-2] & result[1:-1,2:])
                result = inner
            return result

        @staticmethod
        def map_coordinates(data, coords, order=1, mode="constant", cval=0):
            row, col = coords
            valid = (row >= 0) & (row <= data.shape[0]-1) & (col >= 0) & (col <= data.shape[1]-1)
            rr, cc = np.clip(row,0,data.shape[0]-1), np.clip(col,0,data.shape[1]-1)
            r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
            r1, c1 = np.minimum(r0+1,data.shape[0]-1), np.minimum(c0+1,data.shape[1]-1)
            fr, fc = rr-r0, cc-c0
            values = ((1-fr)*(1-fc)*data[r0,c0] + fr*(1-fc)*data[r1,c0] +
                      (1-fr)*fc*data[r0,c1] + fr*fc*data[r1,c1])
            return np.where(valid, values, cval)

    def _numpy_least_squares(fun, x0, jac, method="trf", max_nfev=35,
                             ftol=2e-6, xtol=2e-6, gtol=2e-6):
        """Damped Gauss-Newton with monotone backtracking, for 20 variables."""
        x = np.array(x0, dtype=float, copy=True)
        residual = fun(x)
        matrix = jac(x)
        nfev, damping, success = 1, 1e-3, False
        message = "Maximum image evaluations reached (NumPy damped Gauss-Newton)."
        while nfev < max_nfev:
            normal = matrix.T @ matrix
            gradient = matrix.T @ residual
            if np.max(np.abs(gradient)) < gtol:
                success, message = True, "Gradient tolerance satisfied (NumPy solver)."
                break
            diagonal = np.maximum(np.diag(normal), 1e-8)
            step = np.linalg.solve(normal+damping*np.diag(diagonal), -gradient)
            length = np.linalg.norm(step)
            if length > 2:
                step *= 2/length
            old_cost = np.dot(residual, residual)
            candidate = x+step
            trial = fun(candidate)
            nfev += 1
            new_cost = np.dot(trial, trial)
            if new_cost < old_cost:
                x, residual = candidate, trial
                matrix = jac(x)
                damping = max(damping/3, 1e-9)
                if np.linalg.norm(step) < xtol*(xtol+np.linalg.norm(x)):
                    success, message = True, "Step tolerance satisfied (NumPy solver)."
                    break
                if old_cost-new_cost < ftol*max(old_cost, 1e-15):
                    success, message = True, "Cost tolerance satisfied (NumPy solver)."
                    break
            else:
                damping = min(damping*10, 1e10)
        return SimpleNamespace(x=x, jac=matrix, success=success, message=message, nfev=nfev)

    fft, ndimage = _FFT(), _NDImage()
    optimize = SimpleNamespace(least_squares=_numpy_least_squares)


@dataclass(frozen=True)
class Parameters:
    objective_diameter_mm: float = 10.0
    objective_focal_mm: float = 400.0
    mla_pitch_um: float = 150.0
    mla_focal_mm: float = 5.0
    beam_diameter_mm: float = 8.0
    phase_ptp_rad: float = 6.0
    pixel_um: float = 5.0
    wavelength_nm: float = 633.0
    bandwidth_nm: float = 0.1
    vortex_charge: int = 0


@dataclass(frozen=True)
class Mode:
    j: int
    n: int
    m: int
    name: str


MODES = (
    Mode(1, 0, 0, "Piston"), Mode(2, 1, 1, "X tilt"),
    Mode(3, 1, -1, "Y tilt"), Mode(4, 2, 0, "Defocus"),
    Mode(5, 2, -2, "Oblique astigmatism"),
    Mode(6, 2, 2, "Vertical astigmatism"),
    Mode(7, 3, -1, "Y coma"), Mode(8, 3, 1, "X coma"),
    Mode(9, 3, -3, "Y trefoil"), Mode(10, 3, 3, "X trefoil"),
    Mode(11, 4, 0, "Primary spherical"),
    Mode(12, 4, 2, "Secondary vertical astigmatism"),
    Mode(13, 4, -2, "Secondary oblique astigmatism"),
    Mode(14, 4, 4, "X quadrafoil"), Mode(15, 4, -4, "Y quadrafoil"),
    Mode(16, 5, 1, "Secondary X coma"),
    Mode(17, 5, -1, "Secondary Y coma"),
    Mode(18, 5, 3, "Secondary X trefoil"),
    Mode(19, 5, -3, "Secondary Y trefoil"),
    Mode(20, 5, 5, "X pentafoil"), Mode(21, 5, -5, "Y pentafoil"),
)


@dataclass
class Wavefront:
    x_mm: np.ndarray
    y_mm: np.ndarray
    phase: np.ndarray
    mask: np.ndarray
    intensity: np.ndarray
    coefficients: np.ndarray
    vortex_charge: int = 0


@dataclass
class SimulationResult:
    wavefront: Wavefront
    image: np.ndarray
    reconstructed_phase: np.ndarray
    similarity_pct: float | None
    rms_error_rad: float
    warnings: list[str]
    metadata: dict
    coefficients: np.ndarray


class CancelledError(RuntimeError):
    """Raised between numerical steps when cancellation is requested."""


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise CancelledError("Simulation cancelled.")


def validate_parameters(p: Parameters) -> list[str]:
    values = asdict(p)
    for name, value in values.items():
        if name == "vortex_charge":
            _validate_charge(value)
            continue
        if not np.isfinite(value):
            raise ValueError(f"{name.replace('_', ' ')} must be finite.")
        if name in ("phase_ptp_rad", "bandwidth_nm"):
            if value < 0:
                raise ValueError(f"{name.replace('_', ' ')} cannot be negative.")
        elif value <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be greater than zero.")
    if p.bandwidth_nm > 0.5 * p.wavelength_nm:
        raise ValueError("Spectral FWHM must be at most half the central wavelength in this Gaussian-spectrum model.")
    warnings = []
    f_obj = p.objective_focal_mm / p.objective_diameter_mm
    f_mla = p.mla_focal_mm / (p.mla_pitch_um * 1e-3)
    if f_obj <= f_mla:
        warnings.append(
            f"F-number warning: objective f/{f_obj:.2f} is not larger than MLA f/{f_mla:.2f}. "
            "Potential overlap / reconstruction inaccuracy; equality is the theoretical matched limit, with no margin."
        )
    if p.beam_diameter_mm > p.objective_diameter_mm:
        warnings.append("The objective clips the specified beam; metrics use only the transmitted circular pupil.")
    diffraction_um = p.wavelength_nm * 1e-3 * f_mla
    if p.pixel_um > diffraction_um / 2:
        warnings.append("Camera pixels undersample the characteristic microlens diffraction scale (lambda f/p); numerical accuracy may be poor.")
    if p.mla_pitch_um / p.pixel_um < 8:
        warnings.append("Fewer than eight camera pixels span a lenslet pitch; spatial pupil sampling is coarse.")
    if p.bandwidth_nm / p.wavelength_nm > 0.05:
        warnings.append("Broad spectrum: five-point spectral quadrature and achromatic ideal lenses may be insufficient.")
    if p.phase_ptp_rad > 40:
        warnings.append("Large phase range: nonlinear inversion can converge to a local minimum; inspect image residual and validation.")
    return warnings


def _validate_charge(charge):
    try:
        valid = (np.isscalar(charge) and not isinstance(charge, (bool, np.bool_)) and
                 np.isfinite(charge) and charge == int(charge) and 0 <= charge <= 6)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("Vortex topological charge must be an integer from 0 to 6.")
    return int(charge)


def _azimuth(axis):
    xx, yy = np.meshgrid(axis, axis)
    return np.arctan2(yy, xx), (xx == 0) & (yy == 0)


def zernike(n: int, m: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Real, unit-RMS Noll convention on the unit disk (cos m>0, sin m<0)."""
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    am = abs(m)
    radial = np.zeros_like(rho, dtype=float)
    for k in range((n - am) // 2 + 1):
        radial += ((-1) ** k * factorial(n-k) /
                   (factorial(k) * factorial((n+am)//2-k) * factorial((n-am)//2-k))) * rho ** (n-2*k)
    if m == 0:
        return np.sqrt(n+1) * radial
    angular = np.cos(am*theta) if m > 0 else np.sin(am*theta)
    return np.sqrt(2*(n+1)) * radial * angular


@lru_cache(maxsize=16)
def _pupil_grid(beam_mm: float, objective_mm: float, resolution: int):
    # Pixel centers are symmetric; the explicit circular support has diameter beam_mm.
    axis = (np.arange(resolution) - (resolution-1)/2) * beam_mm / resolution
    xx, yy = np.meshgrid(axis, axis)
    r2 = xx*xx + yy*yy
    radius = min(beam_mm, objective_mm)/2
    mask = r2 <= radius*radius
    if np.count_nonzero(mask) < 16:
        raise ValueError("The transmitted aperture is too small for the numerical pupil grid; reduce beam diameter or increase objective diameter.")
    sigma = beam_mm/4
    intensity = np.exp(-r2/(2*sigma*sigma)) * mask
    basis = np.stack([zernike(m.n, m.m, xx/(beam_mm/2), yy/(beam_mm/2)) for m in MODES])
    basis[:, ~mask] = 0
    return axis, mask, intensity, basis


def make_wavefront(p: Parameters, weights, resolution: int = 80) -> Wavefront:
    """Build a Gaussian-intensity beam with phase normalized on transmitted pupil.

    The intensity sigma equals one quarter of the nominal beam diameter.
    Piston is removed: a constant phase has zero peak-to-peak and is unobservable.
    Zero/only-piston weights produce no smooth distortion. The additive vortex
    q*atan2(y,x) is never scaled by the requested Zernike peak-to-peak.
    """
    validate_parameters(p)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (21,) or not np.all(np.isfinite(weights)):
        raise ValueError("Exactly 21 finite relative Zernike coefficients are required.")
    axis, mask, intensity, basis = _pupil_grid(p.beam_diameter_mm, p.objective_diameter_mm, resolution)
    coeff = weights.copy()
    coeff[0] = 0
    phase = np.einsum("k,kij->ij", coeff, basis)
    ptp = np.ptp(phase[mask])
    if ptp > 1e-12 and p.phase_ptp_rad > 0:
        coeff *= p.phase_ptp_rad / ptp
        phase *= p.phase_ptp_rad / ptp
    else:
        coeff[:] = 0
        phase[:] = 0
    mask, intensity = mask.copy(), intensity.copy()
    if p.vortex_charge:
        theta, center = _azimuth(axis)
        phase += p.vortex_charge * theta
        # An exact origin has undefined vortex phase. Even grids avoid it;
        # odd grids explicitly assign zero field and exclude it from metrics.
        mask[center] = False
        intensity[center] = 0
    # Fix the piston gauge over the transmitted pupil, including clipped beams.
    phase[mask] -= phase[mask].mean()
    phase[~mask] = 0
    return Wavefront(axis.copy(), axis.copy(), phase, mask, intensity, coeff, int(p.vortex_charge))


def _spectral_nodes(p):
    if p.bandwidth_nm == 0:
        return np.array([p.wavelength_nm*1e-9]), np.array([1.0])
    count = 5 if p.bandwidth_nm/p.wavelength_nm > 0.01 else 3
    nodes, weights = np.polynomial.hermite.hermgauss(count)
    sigma = p.bandwidth_nm/(2*np.sqrt(2*np.log(2)))
    wavelengths = (p.wavelength_nm + np.sqrt(2)*sigma*nodes)*1e-9
    return wavelengths, weights/np.sqrt(np.pi)


class OpticalModel:
    """Fixed geometry forward model. Reconstruction has access only to its image.

    The camera grid consists of actual pixel centers. Pixel flux uses midpoint
    area quadrature; it is not an exact subpixel integral. FFT propagation uses
    a periodic finite window. The reported edge-energy diagnostic warns about
    clipping/wraparound, and increasing sampling is needed for convergence work.
    """

    def __init__(self, p: Parameters, pupil_resolution: int = 80, camera_min: int = 192):
        self.p = p
        self.warnings = validate_parameters(p)
        self.axis, self.mask, self.intensity, self.basis = _pupil_grid(
            p.beam_diameter_mm, p.objective_diameter_mm, pupil_resolution)
        self.pupil_resolution = pupil_resolution
        self.theta, self.vortex_center = _azimuth(self.axis)
        self.dx = p.pixel_um*1e-6
        self.pitch = p.mla_pitch_um*1e-6
        # Seven lenslets across is a fixed simulation field, not inferred from truth.
        desired = max(camera_min, int(np.ceil(7*self.pitch/self.dx)))
        self.n = int(fft.next_fast_len(desired))
        if self.n % 2:
            self.n += 1
        if self.n > 384:
            self.n = 384
            self.warnings.append("Camera calculation capped at 384 pixels per axis; small pixels reduce field of view. Inspect edge energy.")
        self.camera_axis = (np.arange(self.n)-(self.n-1)/2)*self.dx
        xx, yy = np.meshgrid(self.camera_axis, self.camera_axis)
        self.cell_x = self.pitch*np.floor(xx/self.pitch+0.5)
        self.cell_y = self.pitch*np.floor(yy/self.pitch+0.5)
        local_r2 = (xx-self.cell_x)**2 + (yy-self.cell_y)**2
        f = fft.fftfreq(self.n, self.dx)
        f2 = f[:, None]**2+f[None, :]**2
        pupil_m = self.axis*1e-3
        self.a = p.objective_focal_mm*1e-3
        self.b = p.mla_focal_mm*1e-3
        self.wavelengths, self.spectral_weights = _spectral_nodes(p)
        self.transforms = []
        dxp = p.beam_diameter_mm*1e-3/pupil_resolution
        self.objective_nyquist_halfwidth = float(np.min(self.wavelengths)*self.a/(2*dxp))
        if self.n*self.dx/2 > self.objective_nyquist_halfwidth:
            self.warnings.append(
                "Objective Fourier sampling warning: camera half-width exceeds the pupil-grid "
                "Nyquist field (lambda_min f_obj / (2 pupil_step)); aliased focal-plane light can "
                "invalidate the result. Increase pupil resolution or reduce camera field of view."
            )
        for wavelength in self.wavelengths:
            mat = np.exp(-2j*np.pi*np.outer(self.camera_axis, pupil_m)/(wavelength*self.a))
            # The common Fourier-prefactor phase is irrelevant; amplitude is physical up to input normalization.
            mat *= dxp/np.sqrt(wavelength*self.a)
            transfer = np.exp(-1j*np.pi*wavelength*self.b*f2)
            lens = np.exp(-1j*np.pi*local_r2/(wavelength*self.b))
            self.transforms.append((mat, transfer, lens, p.wavelength_nm*1e-9/wavelength))
        self._reference_images = {}

    @staticmethod
    def _propagate(field, transfer):
        return fft.ifft2(fft.fft2(field, axes=(-2, -1), workers=1)*transfer,
                         axes=(-2, -1), workers=1)

    def forward(self, coefficients, derivatives=False, cancel=None, *, vortex_charge=None):
        """Normalized detector intensity; optionally derivatives w.r.t j2..21.

        Generation defaults to the requested charge. Reconstruction always
        overrides it explicitly, so the input charge is never an inverse prior.
        The vortex factor is achromatic; only Zernike OPD scales with wavelength.
        """
        charge = _validate_charge(self.p.vortex_charge if vortex_charge is None else vortex_charge)
        coeff = np.asarray(coefficients, dtype=float)
        if coeff.shape == (20,):
            coeff = np.r_[0.0, coeff]
        if coeff.shape != (21,):
            raise ValueError("Expected 20 nonpiston or 21 Noll coefficients.")
        phase = np.einsum("k,kij->ij", coeff, self.basis)
        amplitude = np.sqrt(self.intensity)
        if charge:
            amplitude = amplitude.copy()
            amplitude[self.vortex_center] = 0
        raw = np.zeros((self.n, self.n))
        derivative = np.zeros((20, self.n, self.n)) if derivatives else None
        for weight, (mat, transfer, lens, spectral_scale) in zip(self.spectral_weights, self.transforms):
            _check_cancel(cancel)
            pupil = amplitude*np.exp(1j*(spectral_scale*phase + charge*self.theta))
            if derivatives:
                fields = np.concatenate((pupil[None], 1j*spectral_scale*self.basis[1:]*pupil), axis=0)
            else:
                fields = pupil[None]
            focus = np.matmul(np.matmul(mat, fields), mat.T)
            mla = self._propagate(focus, transfer)
            camera = self._propagate(mla*lens, transfer)
            raw += weight*np.abs(camera[0])**2
            if derivatives:
                derivative += weight*2*np.real(camera[1:]*np.conjugate(camera[0]))
        energy = raw.sum()
        if not np.isfinite(energy) or energy <= 1e-30:
            raise ValueError("No sampled energy reaches the camera; adjust aperture, wavelength, focal lengths or pixel size.")
        image = raw/energy
        if derivatives:
            dtotal = derivative.sum(axis=(1, 2))
            jac = (derivative-dtotal[:, None, None]*image)/energy
            return image, jac
        return image

    def geometric_start(self, image, vortex_charge=0):
        """Image-derived centroid backprojection as an approximate initial guess.

        A pixel t in lenslet c maps to x=-(a/b)(t-c); lenslet centroid
        differences from a flat reference encode angle approximately as dc/a.
        This is an initializer only, not the final coherent reconstruction.
        """
        if vortex_charge not in self._reference_images:
            self._reference_images[vortex_charge] = self.forward(np.zeros(20), vortex_charge=vortex_charge)
        # Keep inversion inside camera pixel centers; interpolate each pupil image.
        axis = self.axis*1e-3
        px, py = np.meshgrid(axis, axis)
        centers = np.arange(-int(self.n*self.dx/(2*self.pitch)),
                            int(self.n*self.dx/(2*self.pitch))+1)*self.pitch
        def centroid(data):
            den = np.zeros_like(px)
            sx = np.zeros_like(px)
            sy = np.zeros_like(px)
            for cx in centers:
                for cy in centers:
                    tx = cx-self.b/self.a*px
                    ty = cy-self.b/self.a*py
                    # All samples belong to this lenslet image if the f-number is safe.
                    support = (np.abs(tx-cx) <= self.pitch/2) & (np.abs(ty-cy) <= self.pitch/2)
                    coords = np.array([(ty-self.camera_axis[0])/self.dx,
                                       (tx-self.camera_axis[0])/self.dx])
                    values = ndimage.map_coordinates(data, coords, order=1, mode="constant", cval=0)*support
                    den += values
                    sx += cx*values
                    sy += cy*values
            return sx/np.maximum(den, 1e-20), sy/np.maximum(den, 1e-20), den
        cx, cy, weight = centroid(image)
        rx, ry, _ = centroid(self._reference_images[vortex_charge])
        factor = 2*np.pi/(self.p.wavelength_nm*1e-9)/self.a
        gx = factor*(cx-rx)
        gy = factor*(cy-ry)
        step = axis[1]-axis[0]
        by, bx = np.gradient(self.basis[1:], step, step, axis=(1, 2))
        interior = ndimage.binary_erosion(self.mask, iterations=2)
        w = np.sqrt(weight[interior]/max(weight.max(), 1e-20))
        design = np.concatenate((bx[:, interior].T*w[:, None],
                                 by[:, interior].T*w[:, None]), axis=0)
        target = np.r_[gx[interior]*w, gy[interior]*w]
        if len(target) < 20:
            return np.zeros(20)
        initial = np.linalg.lstsq(design, target, rcond=1e-7)[0]
        return np.clip(initial, -20, 20)

    def reconstruct(self, image, progress=None, cancel=None, max_nfev=35,
                    *, charge_candidates=tuple(range(7))):
        """Infer integer charge and j2..21 from the camera image alone.

        All allowed charges are ranked by their zero-Zernike camera templates.
        Each tested charge fits the same 20 unknown smooth coefficients. Search
        stops early only at a numerical exact fit (image residual < 1e-8);
        otherwise all candidates are fitted. This is not a uniqueness proof.
        Neither self.p.vortex_charge, generating weights nor target PTP guides
        inference. charge_candidates=(0,) explicitly requests the old basis.
        """
        image = np.asarray(image, dtype=float)
        if image.shape != (self.n, self.n) or not np.all(np.isfinite(image)) or np.any(image < 0) or image.sum() <= 0:
            raise ValueError("Camera image must be finite, nonnegative, and match the model grid.")
        charges = tuple(dict.fromkeys(_validate_charge(q) for q in charge_candidates))
        if not charges:
            raise ValueError("At least one candidate vortex charge is required.")
        target = np.sqrt(image/image.sum()+1e-12)
        _check_cancel(cancel)
        template_costs = {}
        for charge in charges:
            _check_cancel(cancel)
            if charge not in self._reference_images:
                self._reference_images[charge] = self.forward(np.zeros(20), cancel=cancel, vortex_charge=charge)
            template_costs[charge] = float(np.linalg.norm(np.sqrt(self._reference_images[charge]+1e-12)-target))
        ordered = sorted(charges, key=lambda q: (template_costs[q], q))
        tested, best, exact_stop = [], None, False
        for charge in ordered:
            _check_cancel(cancel)
            if progress:
                progress(f"Inferring vortex charge · testing q={charge}…")
            phase, coeff, info = self._fit_charge(image, target, charge, progress, cancel, max_nfev)
            tested.append({"charge": charge, "image_fit_error": info["image_fit_error"],
                           "optimizer_success": info["optimizer_success"],
                           "optimizer_evaluations": info["optimizer_evaluations"]})
            if best is None or info["image_fit_error"] < best[2]["image_fit_error"]:
                best = (phase, coeff, info)
            if info["image_fit_error"] < 1e-8:
                exact_stop = len(tested) < len(ordered)
                break
        phase, coeff, info = best
        info.update({
            "charge_candidates": list(charges),
            "charge_template_residuals": {str(q): template_costs[q] for q in charges},
            "charge_fits": tested,
            "charge_search_exact_fit_early_stop": exact_stop,
            "charge_search_complete": len(tested) == len(ordered),
            "charge_search_note": "Discrete image-only model selection over the listed charges; an exact numerical fit does not establish uniqueness.",
            "total_optimizer_evaluations": sum(item["optimizer_evaluations"] for item in tested),
        })
        return phase, coeff, info

    def _fit_charge(self, image, target, charge, progress, cancel, max_nfev):
        """Fit smooth phase at one explicit candidate charge (no truth inputs)."""
        geometric = self.geometric_start(image, vortex_charge=charge)
        candidates = [np.zeros(20)] + [scale*geometric for scale in (1, 2, 4, 8)]
        labels = ["zero Zernike phase"] + [f"camera centroid × {scale}" for scale in (1, 2, 4, 8)]
        for defocus in (-4, -2, -1, 1, 2, 4):
            seed = np.zeros(20)
            seed[2] = defocus
            candidates.append(seed)
            labels.append(f"image-selected defocus seed {defocus} rad")
        costs = [np.sum((np.sqrt(self.forward(c, cancel=cancel, vortex_charge=charge)+1e-12)-target)**2)
                 for c in candidates]
        initial_index = int(np.argmin(costs))
        cache, evaluations = {}, 0
        def evaluate(c):
            nonlocal evaluations
            _check_cancel(cancel)
            if "x" not in cache or not np.array_equal(c, cache["x"]):
                prediction, jac = self.forward(c, derivatives=True, cancel=cancel, vortex_charge=charge)
                square = np.sqrt(prediction+1e-12)
                cache["x"] = c.copy()
                cache["residual"] = 100*(square-target).ravel()
                cache["jacobian"] = (50*jac/square).reshape(20, -1).T.copy()
                evaluations += 1
                if progress:
                    progress(f"Fitting q={charge} · Zernike iteration {evaluations}")
            return cache
        fit = optimize.least_squares(lambda c: evaluate(c)["residual"], candidates[initial_index],
                                     jac=lambda c: evaluate(c)["jacobian"],
                                     method="trf", max_nfev=max_nfev,
                                     ftol=2e-6, xtol=2e-6, gtol=2e-6)
        coeff = np.r_[0.0, fit.x]
        phase = np.einsum("k,kij->ij", coeff, self.basis) + charge*self.theta
        mask = self.mask & ~self.vortex_center if charge else self.mask
        phase[mask] -= phase[mask].mean()
        phase[~mask] = 0
        prediction = self.forward(coeff, cancel=cancel, vortex_charge=charge)
        image_error = np.linalg.norm(np.sqrt(prediction)-np.sqrt(image/image.sum()))
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        condition = float(singular[0]/max(singular[-1], 1e-15))
        return phase, coeff, {
            "optimizer_success": bool(fit.success),
            "optimizer_message": str(fit.message),
            "optimizer_evaluations": int(fit.nfev),
            "image_fit_error": float(image_error),
            "jacobian_condition": condition,
            "initialization": labels[initial_index],
            "recovered_vortex_charge": int(charge),
        }


SIMILARITY_DEFINITION = "100 × max(0, 1 − RMSE / RMS_truth), after independent piston removal on the transmitted pupil; no amplitude or sign fitting. Flat truth: N/A."
WRAPPED_SIMILARITY_DEFINITION = "100 × max(0, 1 − circular_phase_RMSE / pi), after one global phase-offset alignment; residuals wrapped to [-pi, pi]. Report topological charge separately. This score is not experimental accuracy."


def phase_residual(truth, reconstruction, mask, wrapped=False):
    """Piston-aligned reconstruction-minus-truth, zero outside the metric mask."""
    mask = np.asarray(mask, dtype=bool)
    difference = np.asarray(reconstruction, dtype=float) - np.asarray(truth, dtype=float)
    residual = np.zeros_like(difference, dtype=float)
    values = difference[mask]
    if not len(values):
        raise ValueError("Phase comparison needs at least one valid pupil sample.")
    if wrapped:
        mean_phasor = np.mean(np.exp(1j*values))
        offset = float(np.angle(mean_phasor)) if abs(mean_phasor) > 1e-12 else 0.0
        residual[mask] = np.angle(np.exp(1j*(values-offset)))
    else:
        residual[mask] = values-values.mean()
    return residual


def compare_wavefronts(truth, reconstruction, mask, wrapped=False):
    a = np.asarray(truth)[mask].astype(float)
    a -= a.mean()
    residual = phase_residual(truth, reconstruction, mask, wrapped=wrapped)
    rmse = float(np.sqrt(np.mean(residual[mask]**2)))
    if wrapped:
        return float(100*max(0, 1-rmse/np.pi)), rmse
    rms = float(np.sqrt(np.mean(a*a)))
    similarity = None if rms < 1e-10 else float(100*max(0, 1-rmse/rms))
    return similarity, rmse


def phase_winding_number(phase, mask):
    """Sampled circulation on a centered interior square, in turns.

    Adjacent phase increments are wrapped before summing. The loop avoids the
    vortex core; unresolved increments can alias, so this is a grid diagnostic.
    """
    phase, mask = np.asarray(phase), np.asarray(mask, dtype=bool)
    cy, cx = (np.array(mask.shape)-1)//2
    max_radius = min(cy, cx, mask.shape[0]-1-cy, mask.shape[1]-1-cx)
    for radius in range(max_radius, 0, -1):
        top, bottom, left, right = cy-radius, cy+radius, cx-radius, cx+radius
        ys = np.r_[np.full(2*radius, top), np.arange(top, bottom),
                   np.full(2*radius, bottom), np.arange(bottom, top, -1)]
        xs = np.r_[np.arange(left, right), np.full(2*radius, right),
                   np.arange(right, left, -1), np.full(2*radius, left)]
        if np.all(mask[ys, xs]):
            values = phase[ys, xs]
            return float(np.angle(np.exp(1j*(np.roll(values, -1)-values))).sum()/(2*np.pi))
    return None


def simulate(p: Parameters, weights, progress: Callable[[str], None] | None = None,
             cancel: Callable[[], bool] | None = None, *, model: OpticalModel | None = None,
             max_nfev: int = 35) -> SimulationResult:
    start = time.perf_counter()
    _check_cancel(cancel)
    model = model or OpticalModel(p)
    if model.p != p:
        raise ValueError("Optical model parameters differ from simulation parameters.")
    wf = make_wavefront(p, weights, model.pupil_resolution)
    if progress:
        progress("Propagating the Gaussian beam through the objective and microlens array…")
    image = model.forward(wf.coefficients, cancel=cancel)
    phase, coefficients, metadata = model.reconstruct(image, progress, cancel, max_nfev=max_nfev)
    recovered_charge = metadata["recovered_vortex_charge"]
    wrapped = bool(p.vortex_charge or recovered_charge)
    similarity, rmse = compare_wavefronts(wf.phase, phase, wf.mask, wrapped=wrapped)
    warnings = list(model.warnings)
    weights = np.asarray(weights)
    if abs(weights[0]) > 0:
        warnings.append("Noll j=1 piston is unobservable in intensity and has zero peak-to-peak. It is removed from generation and reconstruction.")
    if not np.any(np.abs(weights[1:]) > 1e-12) and p.phase_ptp_rad > 0:
        warnings.append("No nonpiston Zernike mode is selected: no smooth distortion is added; the requested Zernike peak-to-peak cannot be applied.")
    if recovered_charge != p.vortex_charge:
        warnings.append(f"Topological-charge mismatch: generated q={p.vortex_charge}, inferred q={recovered_charge}. Inspect charge diagnostics and image residual.")
    if not metadata["optimizer_success"]:
        warnings.append("Reconstruction reached its iteration limit; the current estimate is shown. Convergence is not guaranteed.")
    if metadata["image_fit_error"] > 0.05:
        warnings.append("Camera image fit remains imperfect; this reconstruction may be trapped in a local minimum.")
    edge = np.ones_like(image, dtype=bool)
    edge[5:-5, 5:-5] = False
    edge_energy = float(image[edge].sum())
    if edge_energy > 0.01:
        warnings.append(f"{100*edge_energy:.1f}% of camera energy lies in its outer five pixels; finite-window propagation may wrap or clip light.")
    if metadata["jacobian_condition"] > 1e8:
        warnings.append("The local image Jacobian is poorly conditioned; some fitted modes are weakly identifiable.")
    # High phase frequencies can be unresolved on the fixed pupil sampling.
    if p.vortex_charge:
        # Raw differences across the atan2 branch cut are artificial. Wrapped
        # increments instead flag unresolved field variation near the core.
        steps = []
        for axis in (0, 1):
            delta = np.angle(np.exp(1j*np.diff(wf.phase, axis=axis)))
            pair_mask = ((wf.mask[1:, :] & wf.mask[:-1, :]) if axis == 0
                         else (wf.mask[:, 1:] & wf.mask[:, :-1]))
            steps.append(float(np.max(np.abs(delta[pair_mask]))) if np.any(pair_mask) else 0.0)
        max_step = max(steps)
    else:
        gradient = np.gradient(wf.phase)
        interior = ndimage.binary_erosion(wf.mask)
        max_step = max(float(np.max(np.abs(g[interior]))) for g in gradient) if np.any(interior) else 0
    if max_step > np.pi/2:
        warnings.append("Pupil phase changes by more than pi/2 per sample; increase numerical pupil resolution for convergence studies.")
    metadata.update({
        "camera_shape": list(image.shape), "sampling_um": p.pixel_um,
        "camera_extent_mm": model.n*p.pixel_um*1e-3,
        "pupil_resolution": model.pupil_resolution,
        "objective_nyquist_halfwidth_mm": model.objective_nyquist_halfwidth*1e3,
        "numerical_backend": NUMERICAL_BACKEND,
        "spectral_samples": len(model.wavelengths),
        "spectral_wavelengths_nm": (model.wavelengths*1e9).tolist(),
        "f_number_objective": p.objective_focal_mm/p.objective_diameter_mm,
        "f_number_mla": p.mla_focal_mm/(p.mla_pitch_um*1e-3),
        "edge_energy_fraction": edge_energy,
        "elapsed_seconds": time.perf_counter()-start,
        "similarity_definition": WRAPPED_SIMILARITY_DEFINITION if wrapped else SIMILARITY_DEFINITION,
        "phase_comparison": "wrapped" if wrapped else "unwrapped",
        "wrapped_phase_comparison": wrapped,
        "input_vortex_charge": int(p.vortex_charge),
        "vortex_charge_error": int(recovered_charge-p.vortex_charge),
        "input_winding_number": phase_winding_number(wf.phase, wf.mask),
        "reconstructed_winding_number": phase_winding_number(phase, wf.mask),
        "vortex_spectral_convention": "Ideal achromatic phase imprint exp(i*q*theta); smooth Zernike OPD scales as lambda_center/lambda. Gaussian amplitude is unchanged except zero field at the singular origin; not a Laguerre-Gaussian radial mode.",
        "model": "Scalar paraxial Fourier + Fresnel propagation; shared-focus 4f geometry; entrance-pupil stop approximation; ideal achromatic lenses and vortex phase imprint; midpoint camera-pixel sampling; no detector noise.",
        "requested_phase_ptp_rad": p.phase_ptp_rad,
        "actual_phase_ptp_rad": float(np.ptp(wf.phase[wf.mask])),
        "actual_zernike_phase_ptp_rad": float(np.ptp(np.einsum("k,kij->ij", wf.coefficients, model.basis)[wf.mask])),
        "validation_scope": "Same-model noiseless self-consistency, not experimental accuracy; use grid refinement and independent calibration for quantitative design.",
        "parameters": asdict(p),
    })
    return SimulationResult(wf, image, phase, similarity, rmse, warnings, metadata, coefficients)


def validate_all(p: Parameters, progress=None, cancel=None, result_callback=None,
                 *, max_nfev=35) -> list[SimulationResult]:
    """Simulate Noll j1..21 with the selected vortex; infer charge and 20 modes."""
    model = OpticalModel(p)
    results = []
    for index, mode in enumerate(MODES):
        _check_cancel(cancel)
        weights = np.zeros(21)
        weights[index] = 1
        def report(message, j=mode.j):
            if progress:
                progress(f"Zernike {j}/21 · {message}")
        result = simulate(p, weights, report, cancel, model=model, max_nfev=max_nfev)
        result.metadata["mode_j"] = mode.j
        result.metadata["mode_name"] = mode.name
        results.append(result)
        if result_callback:
            result_callback(index, result)
    return results
