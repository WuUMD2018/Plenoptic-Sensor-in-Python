# Implemented numerical model

All physical calculations use meters and radians internally. GUI unit conversions occur inside the numerical core. Phase plots are unwrapped, piston-free phases at the central wavelength.

## Forward operator

For the circular input support, `I=exp(-(x²+y²)/(2σ²))`, with `σ=B/4`. The field is `U=sqrt(I) exp(iφ)`. The support radius is `min(B,D)/2`; Zernikes are defined on radius `B/2`, so objective clipping does not silently rescale their coordinate system.

At each wavelength, the objective's focal-plane field is evaluated by a separable matrix Fourier integral:

\[
U_F(u,v)=\frac{1}{i\lambda f_o}\iint U(x,y)
\exp\left[-\frac{2\pi i}{\lambda f_o}(ux+vy)\right]dx\,dy.
\]

A common prefactor phase is omitted because it does not change detector intensity. Quadrature factors preserve relative spectral normalization before the final total-intensity normalization.

The Fresnel propagation operator at distance `b=f_MLA` is

\[
\mathcal P_b[U]=\mathcal F^{-1}\{\mathcal F[U]\,
\exp[-i\pi\lambda b(\nu_x^2+\nu_y^2)]\}.
\]

For each sample, the physical lenslet center is `c_x=p floor(u/p+1/2)`, and similarly in y. Its transmission is

\[
T(u,v)=\exp\left[-\frac{i\pi}{\lambda b}
((u-c_x)^2+(v-c_y)^2)\right].
\]

The camera field is `P_b[T · P_b[U_F]]`. All cells propagate together, retaining coherent cross terms; no independent intensity tiles are pasted together. Ideal cells have no gaps. The finite computational window is not a separately specified physical MLA aperture.

For spectral nodes λk and Gaussian quadrature weights wk, the camera records `Σ wk |Uk|²`, then the sampled image is normalized to sum one. The detector spacing is exactly the entered pixel size. Midpoint quadrature samples intensities at pixel centers; absolute power, exposure and detector counts are unspecified.

## Phase parameterization and inference

Noll 1–21 use RMS-normalized real polynomials, with positive signed m denoting cosine and negative m denoting sine. Piston is fixed at zero. User weights are normalized to the requested sampled phase range, while inference fits 20 **unconstrained signed coefficients in radians**. It does not assume the entered magnitude or selected generating mode.

The residual per pixel is `100 [sqrt(I_pred+epsilon) − sqrt(I_observed+epsilon)]`, with `epsilon=1e-12` on normalized intensity. Analytic derivatives propagate `i Z_j U` through the same optical system, including normalization and spectral scaling. SciPy uses trust-region least squares with at most 35 evaluations. The NumPy fallback uses damped Gauss–Newton with backtracking. The fixed search seeds and image-derived centroid estimate are selected using camera residual, not truth error.

Piston removal is the only registration used for scoring. Phase RMSE is unweighted over the transmitted circular pupil. The similarity score is clipped at zero and is N/A for flat truth. The input Gaussian is known to the solver; reconstructing an unknown amplitude distribution is outside this implementation.

## Numerical diagnostics

The output metadata records camera dimensions, pitch in physical units, pupil resolution, spectral nodes, f-numbers, camera edge-energy fraction, optimizer termination, image residual, local Jacobian condition number, initialization, backend and elapsed time.

- **F-number:** warn for `fo/D <= b/p` as requested, with equality identified as the matched boundary.
- **Detector sampling:** warn if the pixel exceeds half the characteristic `λ b/p` diffraction scale or fewer than eight samples span a lenslet.
- **Objective transform:** warn when the focal-plane half-width exceeds the Nyquist interval `λ_min fo/(2 Δx_input)` of the pupil quadrature.
- **Phase sampling:** warn if the true test phase changes by more than π/2 per pupil sample. This is a simulation diagnostic, not information used by the inverse fit.
- **Window:** report energy in the camera's outer five pixels; warn above 1%. This cannot detect every intermediate-plane wraparound artifact.
- **Inverse fit:** warn at the iteration limit, square-root camera discrepancy above 0.05, or image-Jacobian condition number above 1e8. These are engineering diagnostics, not rigorous confidence intervals.

Independent-grid tests generate a camera image using twice the pupil resolution, then reconstruct on the coarser grid. The window test generates at 384×384, crops to the default 210×210 field, and reconstructs using the smaller propagation domain. These expose discretization mismatch that same-model tests cannot reveal. They are representative checks, not a general convergence proof for every slider setting.

## Scope relative to the literature

The geometry and sensing interpretation are informed by the sources in RESEARCH.md. The coherent forward implementation and the nonlinear Zernike inverse fit are original code for this application. They do not claim to reproduce the published maximum-spanning-tree inversion, full plenoptic mapping algorithm, experimental amplitude retrieval, vortex recovery, or an adaptive-optics control loop.
