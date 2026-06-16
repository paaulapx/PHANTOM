"""
perturbations.py
bias models, transforms and variants for perturbing the tracer maps and host pdfs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulation import host_pdf_from_tracer

_FLOOR = 1e-6        # icarogw takes log -> tracer strictly positive
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# bias models  b(z)
# --------------------------------------------------------------------------- #
class BiasModel:
    label = "bias"

    def __call__(self, z):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(label={self.label!r})"


class ConstantBias(BiasModel):
    def __init__(self, b, label="constant"):
        self.b = float(b); self.label = label

    def __call__(self, z):
        return np.full(np.shape(z), self.b, dtype=float)


class PowerLawBias(BiasModel):
    """b(z) = b0 (1+z)^alpha -- gw spatial-bias form."""
    def __init__(self, b0, alpha, label="powerlaw"):
        self.b0 = float(b0); self.alpha = float(alpha); self.label = label

    def __call__(self, z):
        return self.b0 * (1.0 + np.asarray(z, dtype=float)) ** self.alpha


class PolynomialBias(BiasModel):
    """b(z) = sum c_i z^i (e.g. hi/21cm fit)."""
    def __init__(self, coeffs, label="polynomial"):
        self.coeffs = [float(c) for c in coeffs]; self.label = label

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        return sum(c * z ** i for i, c in enumerate(self.coeffs))


def tracer_library(config=None):
    """fiducial bias tracers. matter = unbiased baseline. gw added from config.bias if given."""
    lib = {
        "matter":   ConstantBias(1.0, "matter"),
        "galaxies": PowerLawBias(1.1, 0.5, "galaxies"),           # flux-limited optical
        "clusters": PowerLawBias(2.5, 1.0, "clusters"),           # high bias
        "hi":       PolynomialBias([0.667, 0.178, 0.050], "hi"),  # tng-like fit
    }
    if config is not None:
        lib["gw"] = PowerLawBias(config.bias.b_gw, config.bias.alpha_gw, "gw")
    return lib


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def tracer_from_delta(delta_matrix, redshift_grid, bias, *, floor=_FLOOR):
    """T = 1 + b(z) delta, floored positive."""
    b = np.asarray(bias(np.asarray(redshift_grid, dtype=float)), dtype=float)[:, None]
    return np.maximum(1.0 + b * np.asarray(delta_matrix, dtype=float), floor)


def angular_blur(maps, fwhm_deg, *, renormalize=None, floor=0.0):
    """per-shell gaussian beam, B_l = exp(-l(l+1) sigma^2/2).
    negative ringing clipped at floor."""
    import healpy as hp
    maps = np.asarray(maps, dtype=float)
    if not fwhm_deg or fwhm_deg <= 0:
        return maps.copy()
    fwhm = np.deg2rad(float(fwhm_deg))
    out = np.empty_like(maps)
    for i in range(maps.shape[0]):
        out[i] = np.maximum(hp.smoothing(maps[i], fwhm=fwhm), floor)
    return _renorm(out, renormalize)


def add_gaussian_noise(maps, sigma, *, mode="relative", rng, renormalize=None, floor=None):
    """additive noise. mode: 'absolute' = sigma, 'relative' = sigma * shell std,
    'relative_global' = sigma * global std."""
    maps = np.asarray(maps, dtype=float).copy()
    if not sigma or sigma <= 0:
        return maps
    if mode == "absolute":
        s = np.full(maps.shape[0], float(sigma))
    elif mode == "relative":
        s = float(sigma) * maps.std(axis=1)
    elif mode == "relative_global":
        s = np.full(maps.shape[0], float(sigma) * maps.std())
    else:
        raise ValueError("mode must be 'absolute', 'relative' or 'relative_global'")
    out = maps + rng.standard_normal(maps.shape) * s[:, None]
    if floor is not None:
        out = np.maximum(out, floor)
    return _renorm(out, renormalize)


def _renorm(maps, how):
    if how in (None, False, "none"):
        return maps
    if how == "row":
        ssum = maps.sum(axis=1, keepdims=True)
        return np.where(ssum > 0, maps / np.where(ssum > 0, ssum, 1.0), maps)
    if how == "global":
        total = maps.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("cannot global-normalise a non-positive map")
        return maps / total
    raise ValueError("renormalize must be None, 'row' or 'global'")


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
@dataclass
class Variant:
    """one experiment: bias + optional blur + noise.
    *_target: 'tracer' -> both maps (consistent), 'host' -> moca only
    (mismatch test, icarogw map stays clean)."""
    name: str
    bias: BiasModel = field(default_factory=lambda: ConstantBias(1.0, "matter"))
    blur_fwhm_deg: float = 0.0
    blur_target: str = "host"
    noise_sigma: float = 0.0
    noise_mode: str = "relative"
    noise_target: str = "host"
    noise_seed: int = 0
    floor: float = _FLOOR

    def describe(self):
        bits = [f"bias={self.bias.label}"]
        if self.blur_fwhm_deg > 0:
            bits.append(f"blur={self.blur_fwhm_deg:g}deg->{self.blur_target}")
        if self.noise_sigma > 0:
            bits.append(f"noise={self.noise_sigma:g}({self.noise_mode})->{self.noise_target}")
        return ", ".join(bits)


def build_variant_products(sim, variant):
    """variant -> (icarogw_tracer, moca_host). sim must be built."""
    if sim.delta_matrix is None:
        raise RuntimeError("Simulation.build() must run before building variants.")
    rng = np.random.default_rng(sim.config.box.seed + 100 + variant.noise_seed)

    tracer = tracer_from_delta(sim.delta_matrix, sim.shells, variant.bias,
                               floor=variant.floor)

    # tracer-target perturbations -> both products
    if variant.blur_fwhm_deg > 0 and variant.blur_target in ("tracer", "both"):
        tracer = angular_blur(tracer, variant.blur_fwhm_deg, floor=variant.floor)
    if variant.noise_sigma > 0 and variant.noise_target in ("tracer", "both"):
        tracer = add_gaussian_noise(tracer, variant.noise_sigma,
                                    mode=variant.noise_mode, rng=rng,
                                    floor=variant.floor)
    icarogw_tracer = tracer

    # host pdf, then host-only perturbations (moca sees a degraded map)
    host = host_pdf_from_tracer(tracer, sim.redshift_weights())
    if variant.blur_fwhm_deg > 0 and variant.blur_target in ("host", "both"):
        host = angular_blur(host, variant.blur_fwhm_deg, renormalize="global", floor=0.0)
    if variant.noise_sigma > 0 and variant.noise_target in ("host", "both"):
        host = add_gaussian_noise(host, variant.noise_sigma, mode=variant.noise_mode,
                                  rng=rng, renormalize="global", floor=0.0)
    return icarogw_tracer, host


def default_variants(config):
    """standard suite: baseline + em tracers + blur + noise."""
    lib = tracer_library(config)
    return [
        Variant("baseline_matter", bias=lib["matter"]),
        Variant("biased_gw",       bias=lib["gw"]),
        Variant("biased_galaxies", bias=lib["galaxies"]),
        Variant("biased_clusters", bias=lib["clusters"]),
        Variant("biased_hi",       bias=lib["hi"]),
        Variant("blur_5deg",       bias=lib["matter"], blur_fwhm_deg=5.0, blur_target="both"),
        Variant("noise_10pct",     bias=lib["matter"], noise_sigma=0.10,
                noise_mode="relative", noise_target="host"),
    ]


def angular_cross_spectrum(map1, map2, *, use_contrast=True):
    """r_l = cl12 / sqrt(cl11 cl22) between two healpix maps."""
    import healpy as hp
    m1 = np.asarray(map1, dtype=float)
    m2 = np.asarray(map2, dtype=float)
    if use_contrast:
        m1 = m1 - m1.mean()
        m2 = m2 - m2.mean()
    cl11 = hp.anafast(m1)
    cl22 = hp.anafast(m2)
    cl12 = hp.anafast(m1, m2)
    denom = np.sqrt(np.abs(cl11 * cl22))
    with np.errstate(invalid="ignore", divide="ignore"):
        r_l = np.where(denom > 0, cl12 / denom, 0.0)
    return {"l": np.arange(len(r_l)), "r_l": r_l, "cl12": cl12, "cl11": cl11, "cl22": cl22}
