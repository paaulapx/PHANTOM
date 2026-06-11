"""
selftest.py
Unit tests for the dark-siren mock host pipeline components.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .cosmology import Cosmology
from .field import GaussianFieldGenerator


def _measure_pk(delta, L):
    """radially-averaged p(k) of a real cube."""
    n = delta.shape[0]
    Vcell = (L / n) ** 3
    V = L ** 3
    dk = np.fft.fftn(delta) * Vcell                  # physical delta_k
    Pk = (np.abs(dk) ** 2) / V
    kf = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY, KZ = np.meshgrid(kf, kf, kf, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    return kmag.ravel(), Pk.ravel()


def test_field_normalisation():
    n, L = 64, 1000.0
    Vcell = (L / n) ** 3
    gen = GaussianFieldGenerator(n, L, smooth_mpch=0.0, kmax=None, seed=0,
                                 coherent=True, dtype=np.float64)

    # white spectrum: var = p0 / vcell
    P0 = 5.0e3
    d = gen.realize(lambda k: np.full_like(k, P0))
    ratio = d.var() / (P0 / Vcell)
    assert 0.95 < ratio < 1.05, f"variance ratio {ratio:.3f} off unity"

    # power law recovered in the resolved band
    def Ppl(k):
        out = np.zeros_like(k); m = k > 0
        out[m] = 2e4 * (k[m] / 0.05) ** (-1.5)
        return out
    d2 = gen.realize(Ppl)
    kmag, Pk = _measure_pk(d2, L)
    bins = np.linspace(0.04, 0.30, 10)
    idx = np.digitize(kmag, bins)
    worst = 0.0
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() < 8:
            continue
        kc = kmag[sel].mean()
        meas = Pk[sel].mean()
        theo = Ppl(np.array([kc]))[0]
        worst = max(worst, abs(meas / theo - 1.0))
    assert worst < 0.15, f"P(k) recovery off by {worst:.2%}"
    return f"field: Var ratio={ratio:.3f}, max P(k) dev={worst:.2%}"


def test_cosmology_background():
    cfg = Config.default()
    cosmo = Cosmology(cfg)

    # psi(0) = 1
    psi0 = float(cosmo.madau_dickinson(0.0))
    assert abs(psi0 - 1.0) < 1e-9, f"psi(0)={psi0} != 1"

    # chi monotone, chi(0)=0
    z = np.linspace(0.0, cfg.redshift.z_max, 50)
    chi = cosmo.comoving_distance(z)
    assert chi[0] == 0.0 and np.all(np.diff(chi) > 0), "chi(z) not monotone"

    # kernel positive on sampling range
    zk = np.linspace(cfg.redshift.z_min, cfg.redshift.z_max, 200)
    ker = cosmo.redshift_kernel(zk)
    assert np.all(np.isfinite(ker)) and ker.min() > 0, "kernel non-positive"

    # b(z) = b0 (1+z)^alpha
    cfg.bias.b_gw, cfg.bias.alpha_gw = 1.5, 1.0
    cosmo2 = Cosmology(cfg)
    assert np.isclose(cosmo2.bias(1.0), 1.5 * 2.0), "bias model wrong"

    # chi(z_max) sphere inside half-box
    chi_max = float(cosmo.comoving_distance(cfg.redshift.z_max))
    fits = chi_max < 0.5 * cfg.box.size
    return (f"cosmo: psi(0)={psi0:.6f}, chi(z_max)={chi_max:.0f} Mpc/h "
            f"(half-box={0.5*cfg.box.size:.0f}, fits={fits})")

def test_power_spline_accuracy():
    cfg = Config.default()
    cfg.box.n_grid = 64   # speed
    cfg.redshift.n_shells = 5
    cosmo = Cosmology(cfg)
    cosmo.compute_power_spectra()
    kh, z_camb, pk_raw = cosmo.power_spectrum_grid()
    # mid shell, spline vs raw camb
    z_target = cfg.shells[len(cfg.shells)//2]
    power_fn = cosmo.power_at(z_target)
    ks = np.geomspace(cosmo.k_min*1.1, cosmo.k_max*0.9, 10)
    pk_spline = power_fn(ks)
    z_idx = np.argmin(np.abs(z_camb - z_target))
    pk_interp = np.interp(ks, kh, pk_raw[z_idx])
    rel_err = np.abs(pk_spline / pk_interp - 1.0)
    assert np.all(rel_err < 0.01), f"P(k) spline error up to {rel_err.max():.3%}"
    return f"spline error max {rel_err.max():.3%}"


def test_perturbations():
    from types import SimpleNamespace
    from .perturbations import (
        ConstantBias, PowerLawBias, PolynomialBias, tracer_library,
        tracer_from_delta, add_gaussian_noise, Variant, build_variant_products,
    )
    from .simulation import host_pdf_from_tracer

    # bias models
    assert np.allclose(ConstantBias(1.0)(np.array([0., 2.])), 1.0)
    assert np.isclose(PowerLawBias(2.0, 1.0)(1.0), 4.0)           # 2*(1+1)^1
    assert np.isclose(PolynomialBias([0.6, 0.1])(2.0), 0.8)       # 0.6+0.1*2

    # tracer floored > 0 even in deep voids
    z = np.array([0.1, 1.0, 3.0])
    delta = np.array([[0.0, 0.5, -2.0],
                      [0.0, -0.3, 0.2],
                      [0.1, 0.0, -0.9]])
    T = tracer_from_delta(delta, z, PowerLawBias(2.0, 0.0))       # b=2 everywhere
    assert T.min() > 0 and np.isclose(T[0, 0], 1.0)               # delta=0 -> 1

    # noise: scatter up, global renorm -> sum 1, stays positive
    rng = np.random.default_rng(0)
    host0 = host_pdf_from_tracer(np.abs(T) + 0.1, np.array([1.0, 2.0, 1.5]))
    hn = add_gaussian_noise(host0, 0.2, mode="relative", rng=rng,
                            renormalize="global", floor=0.0)
    assert np.isclose(hn.sum(), 1.0) and hn.min() >= 0.0
    assert hn.std() >= host0.std() * 0.5                         # noise added scatter

    # full variant path, no blur -> no healpy
    sim = SimpleNamespace(
        delta_matrix=delta, shells=z,
        redshift_weights=lambda: np.array([1.0, 2.0, 1.5]),
        config=SimpleNamespace(box=SimpleNamespace(seed=42)),
    )
    v = Variant("test", bias=tracer_library()["clusters"],
                noise_sigma=0.1, noise_target="host")
    tr, ho = build_variant_products(sim, v)
    assert tr.min() > 0 and np.isclose(ho.sum(), 1.0)
    return ("perturbations: bias/tracer/noise ok, variant tracer>0, "
            f"host sum={ho.sum():.4f}")


def main():
    results = []
    for name, fn in [("field normalisation", test_field_normalisation),
                     ("cosmology background", test_cosmology_background),
                     ("perturbations", test_perturbations),
                     ("power spline", test_power_spline_accuracy)]:
        try:
            msg = fn()
            results.append((name, True, msg))
        except AssertionError as e:
            results.append((name, False, str(e)))
    print("=" * 64)
    ok = True
    for name, passed, msg in results:
        tag = "PASS" if passed else "FAIL"
        ok &= passed
        print(f"[{tag}] {name}: {msg}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
