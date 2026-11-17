# Plenoptic sensing of laser wavefront distortion

Research notes for Plenoptic Studio · 15 November 2026

## What the sensor measures

The relevant laser-beam configuration is the **shared-focus Keplerian plenoptic sensor**, rather than an ordinary photographic light-field camera. In Wu, Ko and Davis [1], the objective back focal plane coincides with the microlens array (MLA) front focal plane. Thus the objective-to-MLA separation is `f_obj + f_mla`, and the detector is `f_mla` behind the MLA. A reference input plane lies `f_obj` in front of the objective.

The objective separates angular content; the microlenses form spatially resolved subimages. The index of a subimage mainly encodes direction, while position within that subimage encodes input-plane position. The paper treats coherent propagation with Fourier optics and also develops geometrical interpretations. It requires `NA_obj ≤ NA_cell`, corresponding in the paraxial approximation to

\[
F_{\rm obj}=\frac{f_{\rm obj}}{D_{\rm obj}}\geq
F_{\rm MLA}=\frac{f_{\rm MLA}}{p}.
\]

Equality is the matched boundary, not a universally invalid design. The requested GUI applies a stricter `>` criterion and warns when it is not met; equality should be read as “matched, without margin.” [1]

## Why plenoptic information helps

The later plenoptic-mapping paper [2] emphasizes using the illuminated pixels across the MLA image, rather than collapsing each lenslet image to one centroid. It develops mappings between a distorted coherent field and its sensor representation and demonstrates their use for adaptive-optics correction. This provides the mechanism behind spatially varying wavefront estimation: different positions can contribute to different angular cells, and the cells retain spatial information about those contributions. It is a 2016 paper, although some bibliographic services indexed it in early 2017.

The graph-based approach [3] treats reconstruction points as vertices. Candidate connections receive weights from relevant image intensities; a maximum spanning tree selects informative connections and avoids relying on dark regions. Phase differences along the selected edges determine relative phases. That approach is useful when scintillation makes uniform integration unreliable, but a tree reconstruction does not by itself provide an absolute piston reference. The paper's reported large-distortion correction is experimental evidence for its particular arrangement, not an accuracy guarantee for another simulator.

Vortex reconstruction is a separate, more difficult case. Wu et al. [4] demonstrate detecting phase structures around intensity nulls. A fit restricted to smooth low-order Zernike modes cannot reproduce such singularities. The present low-order validation therefore does not establish vortex or arbitrary strong-turbulence performance.

## Independent derivation of the geometrical mapping

The following is a direct paraxial ray-matrix calculation for this application's coordinate convention. All lengths must use the same units; angles are in radians. It makes the signs and plane locations explicit.

Let an input-plane ray have coordinate `x` and slope `θx`. For a thin objective of focal length `f1`, followed by an MLA of focal length `f2` and pitch `p`:

\[
X=x+f_1\theta_x,\qquad
u=f_1\theta_x-\frac{f_2}{f_1}x,\qquad
M=\left\lfloor\frac{u}{p}+\frac12\right\rfloor,
\]

\[
s=Mp-\frac{f_2}{f_1}x.
\]

`X` is the ray height at the objective, `u` the physical MLA-plane position, `M` the lenslet index, and `s` the detector coordinate. Repeat independently in `y`. The objective stop is circular in `(X,Y)`. Square lenslet boundaries apply in `(u,v)`.

For a known lenslet index, detector position gives

\[
x=-\frac{f_1}{f_2}(s-Mp).
\]

Approximating the unknown coordinate within the MLA cell by its center gives

\[
\widehat{\theta}_x=\frac{2Mp-s}{f_1},\qquad
\left|\theta_x-\widehat{\theta}_x\right|\lesssim\frac{p}{2f_1}.
\]

The last bound describes ideal geometrical cell quantization, not a statistical confidence interval. Finite pixels, overlap, diffraction, and noise add errors. Dropping the `s` term is an extra approximation, not the exact mapping above. Lenslet indices cannot be assigned reliably from detector tiles when subimages substantially overlap.

For an optical-path difference `W(x,y)` in air,

\[
\phi_{\lambda}(x,y)=\frac{2\pi W(x,y)}{\lambda},\qquad
\boldsymbol\theta\simeq\nabla W
=\frac{\lambda_0}{2\pi}\nabla\phi_{\lambda_0}.
\]

Consequently, an achromatic fixed-OPD screen has nearly wavelength-independent geometrical ray slopes. Changing bandwidth must not arbitrarily rescale those slopes. Wavelength still controls diffraction and the phase expressed in radians. Spectral components are added as intensities for a time-averaging detector when they are mutually incoherent.

## Gaussian beam and relative phase controls

Define the requested beam size `B` as the diameter of the circular phase-support region. The validation intensity profile is

\[
I(x,y)=\exp\!\left[-\frac{x^2+y^2}{2\sigma^2}\right],\qquad
\sigma=B/4.
\]

This convention concerns **intensity**, so the complex field amplitude is `sqrt(I)`. The circular support at radius `B/2` truncates the otherwise unbounded Gaussian. An objective stop may clip it further.

For relative coefficients `c_j`, construct `q=Σ c_j Z_j`. Remove its mean on the selected support and multiply by `P/[max(q)-min(q)]` to obtain requested peak-to-peak phase `P`, provided the denominator is nonzero. Coefficients are relative weights; their numerical values are not independent phase amplitudes after this shared normalization. Multiplying every coefficient by the same positive number leaves the normalized phase unchanged.

Piston is constant. Its peak-to-peak value is zero at any amplitude, and intensity measurements are invariant under a common phase factor. A piston-only validation must be labeled **unobservable / zero P–V**, with no invented recovery percentage. Zero phase is likewise a special case for normalized error metrics.

## Lowest 21 Zernike modes

The application uses **Noll indices 1–21**, including piston, and RMS-normalized real polynomials on the unit disk. This is not “21 non-piston modes.” Signed `m` below is an implementation convention: positive denotes cosine, negative sine. Noll ordering and normalization follow [5]; names and orientations can differ between software packages.

| Noll j | n | signed m | Mode |
|---:|---:|---:|---|
| 1 | 0 | 0 | Piston |
| 2 | 1 | +1 | Horizontal tilt |
| 3 | 1 | −1 | Vertical tilt |
| 4 | 2 | 0 | Defocus |
| 5 | 2 | −2 | Oblique astigmatism |
| 6 | 2 | +2 | Horizontal astigmatism |
| 7 | 3 | −1 | Vertical coma |
| 8 | 3 | +1 | Horizontal coma |
| 9 | 3 | −3 | Sine trefoil |
| 10 | 3 | +3 | Cosine trefoil |
| 11 | 4 | 0 | Primary spherical |
| 12 | 4 | +2 | Cosine secondary astigmatism |
| 13 | 4 | −2 | Sine secondary astigmatism |
| 14 | 4 | +4 | Cosine quadrafoil |
| 15 | 4 | −4 | Sine quadrafoil |
| 16 | 5 | +1 | Horizontal secondary coma |
| 17 | 5 | −1 | Vertical secondary coma |
| 18 | 5 | +3 | Cosine secondary trefoil |
| 19 | 5 | −3 | Sine secondary trefoil |
| 20 | 5 | +5 | Cosine pentafoil |
| 21 | 5 | −5 | Sine pentafoil |

## What a simulation can establish

A geometrical forward model is useful for exploring image organization, sampling, f-number compatibility, and low-order reconstruction. A finite spectral mixture and a diffraction-width blur are useful approximations but do not become a coherent propagation solver simply because wavelength is an input.

A full scalar-wave model instead propagates `sqrt(I) exp(iφ)` through the objective aperture, intervening distances, square MLA apertures and lens phases, then integrates detector-pixel intensities. It must resolve quadratic phases, lenslet edges, diffraction, and the detector. Coherent cross terms matter when light from neighboring lenslets overlaps. Sampling convergence should be demonstrated before claiming physical accuracy.

Likewise, an inverse fit must use the sensor image and known calibration/geometry. Passing the generating Zernike coefficients or ground-truth slopes directly to reconstruction would be a demonstration of the forward model, not sensor reconstruction. Recovering a phase from the same low-order basis used to generate it is an in-model test; it does not measure robustness against unknown higher-order modes or incorrect beam profiles.

Useful diagnostics include piston-removed phase RMS error, reconstruction P–V, an explicitly defined similarity percentage, usable pupil coverage, image clipping, matrix rank/conditioning, and data residual. Similarity should not silently remove tilt or fit an arbitrary gain, because that would hide physically significant errors. No single percentage proves experimental performance.

For an actual instrument, independently measure lenslet centers, plate scale, camera orientation, dark/flat response, focal distances, and wavelength dependence. Validate known positive and negative modes, mixtures, different phase amplitudes, flat input, undersampling, and misalignment. Simulated tests should include cases that fail gracefully, rather than only favorable images.

## References and access notes

1. C. Wu, J. Ko, and C. C. Davis, **Determining the phase and amplitude distortion of a wavefront using a plenoptic sensor**, *JOSA A* **32**, 964–978 (2015). [DOI](https://doi.org/10.1364/JOSAA.32.000964) · [Author preprint, full text](https://arxiv.org/pdf/1504.02503). Geometry and numerical-aperture condition checked in Section 2; reconstruction in Section 3.
2. C. Wu, J. Ko, and C. C. Davis, **Plenoptic mapping for imaging and retrieval of the complex field amplitude of a laser beam**, *Optics Express* **24**, 29852–29871 (2016). [DOI](https://doi.org/10.1364/OE.24.029852) · [Publisher](https://opg.optica.org/oe/abstract.cfm?uri=oe-24-26-29852). Bibliographic details and author abstract checked; publisher full text was not consistently accessible during this research.
3. C. Wu, J. Ko, and C. C. Davis, **Using Graph Theory and a Plenoptic Sensor to Recognize Phase Distortions of a Laser Beam**, arXiv:1506.00310 (2015). [Preprint](https://arxiv.org/abs/1506.00310) · [Full text](https://arxiv.org/pdf/1506.00310). Maximum-spanning-tree method checked in Section 2.
4. C. Wu, J. Ko, and C. C. Davis, **Using a plenoptic sensor to reconstruct vortex phase structures**, *Optics Letters* **41**, 3169–3172 (2016). [DOI](https://doi.org/10.1364/OL.41.003169) · [Publisher](https://opg.optica.org/ol/abstract.cfm?uri=ol-41-14-3169). Full user-supplied accepted manuscript reviewed for the vortex extension. Eqs. (2)–(4), gradient smoothing, curl localization and cut-domain integration inform the separate diagnostic; [VORTEX.md](VORTEX.md) records the implementation differences and failed geometric-recovery cases. The physical experiment is not reproduced.
5. R. J. Noll, **Zernike polynomials and atmospheric turbulence**, *JOSA* **66**, 207–211 (1976). [DOI](https://doi.org/10.1364/JOSA.66.000207) · [Article scan hosted by University of Auckland](https://ftp.bioeng.auckland.ac.nz/pub/pub/jtur044/references/fitting/NOLL1976.pdf). Index mapping checked against Table I. A convenient independent institutional table is [Greg A. Smith, University of Arizona](https://webs.optics.arizona.edu/gsmith/Zernike.html).

All equations in the independent derivation and the simulation-design discussion are provided as the implementation's explicit assumptions and reasoning; they are not claims that the published experimental algorithms have been reproduced in full.
