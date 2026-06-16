"""
smoketest.py
End-to-end smoke test of the dark-siren mock host pipeline, with the CAMB realisation and a minimal healpy stub for directions and plotting.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import numpy as np


def _install_healpy_stub():
    """minimal healpy: fibonacci-sphere directions, ring-like."""
    hp = types.ModuleType("healpy")

    def nside2npix(nside):
        return 12 * nside * nside

    def _dirs(nside):
        npix = nside2npix(nside)
        i = np.arange(npix)
        # uniform-ish, deterministic, distinct
        z = 1.0 - (2.0 * i + 1.0) / npix
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = (np.pi * (1.0 + 5.0 ** 0.5) * i) % (2.0 * np.pi)
        return theta, phi

    def pix2ang(nside, ipix, nest=False):
        theta, phi = _dirs(nside)
        return theta[ipix], phi[ipix]

    def pix2vec(nside, ipix, nest=False):
        theta, phi = _dirs(nside)
        st = np.sin(theta[ipix])
        return st * np.cos(phi[ipix]), st * np.sin(phi[ipix]), np.cos(theta[ipix])

    # plotting stand-ins, offline coverage only. fills the whole ellipse
    # like real mollview: grid to (lon,lat) mesh + pcolormesh
    def mollview(m, fig=None, sub=None, title="", cmap="inferno", min=None,
                 max=None, cbar=True, unit="", nest=False, xsize=800, **kw):
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata
        m = np.asarray(m, dtype=float)
        nside = int(round((len(m) / 12) ** 0.5))
        theta, phi = _dirs(nside)
        lon = np.mod(phi + np.pi, 2 * np.pi) - np.pi
        lat = np.pi / 2 - theta
        gl = np.linspace(-np.pi, np.pi, 360)
        gb = np.linspace(-np.pi / 2, np.pi / 2, 180)
        LON, LAT = np.meshgrid(gl, gb)
        grid = griddata((lon, lat), m, (LON, LAT), method="linear")
        if np.any(np.isnan(grid)):                         # edges
            near = griddata((lon, lat), m, (LON, LAT), method="nearest")
            grid = np.where(np.isnan(grid), near, grid)
        figure = plt.figure(fig) if fig is not None else plt.gcf()
        spec = sub if sub is not None else 111
        if isinstance(spec, (tuple, list)):
            ax = figure.add_subplot(*spec, projection="mollweide")
        else:
            ax = figure.add_subplot(spec, projection="mollweide")
        pm = ax.pcolormesh(LON, LAT, grid, cmap=cmap, vmin=min, vmax=max,
                           shading="auto", rasterized=True)
        ax.set_title(title); ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)
        if cbar:
            figure.colorbar(pm, ax=ax, shrink=0.6, label=unit)

    def graticule(*a, **k):
        return None

    def anafast(m, map2=None, lmax=None, **k):
        lmax = lmax or (3 * int(round((len(m) / 12) ** 0.5)) - 1)
        ell = np.arange(lmax + 1)
        return 1e-4 / (1.0 + ell) ** 2          # crude decaying c_ell

    def smoothing(m, fwhm=0.0, **k):
        # beam stand-in: grid -> gaussian -> back
        m = np.asarray(m, dtype=float)
        if not fwhm or fwhm <= 0:
            return m.copy()
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
        nside = int(round((len(m) / 12) ** 0.5))
        theta, phi = _dirs(nside)
        lon = np.mod(phi + np.pi, 2 * np.pi) - np.pi
        lat = np.pi / 2 - theta
        nlon, nlat = 720, 360
        gl = np.linspace(-np.pi, np.pi, nlon)
        gb = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        LON, LAT = np.meshgrid(gl, gb)
        grid = griddata((lon, lat), m, (LON, LAT), method="nearest")
        sig = fwhm / 2.3548                                     # fwhm -> sigma (rad)
        grid = gaussian_filter(grid, sigma=(sig / (np.pi / nlat),
                                            sig / (2 * np.pi / nlon)),
                               mode=("nearest", "wrap"))
        out = griddata((LON.ravel(), LAT.ravel()), grid.ravel(),
                       (lon, lat), method="linear")
        return np.where(np.isfinite(out), out, m)

    hp.nside2npix = nside2npix
    hp.pix2ang = pix2ang
    hp.pix2vec = pix2vec
    hp.mollview = mollview
    hp.graticule = graticule
    hp.anafast = anafast
    hp.smoothing = smoothing
    sys.modules["healpy"] = hp
    return hp


def _small_config():
    from .config import Config
    cfg = Config.default()
    cfg.redshift.z_max = 1.5
    cfg.redshift.n_shells = 8
    cfg.redshift.n_z_sampling = 400
    cfg.box.size = 8000.0          # chi(1.5) ~ 3 gpc/h < half-box
    cfg.box.n_grid = 64
    cfg.box.smooth_mpch = None      # auto 1 voxel
    cfg.healpix.nside = 8          # npix 768
    cfg.events.total_events = 300
    return cfg


def main():
    _install_healpy_stub()
    from .pipeline import MockHostPipeline
    from .config import Config  # noqa: F401  (ensures package import works)

    cfg = _small_config()
    with tempfile.TemporaryDirectory() as td:
        cfg.out_dir = Path(td) / "run"
        pipe = MockHostPipeline(cfg, verbose=True, make_plots=True)
        paths = pipe.run()

        sim = pipe.sim
        n_z, n_pix = len(sim.shells), cfg.healpix.npix

        # shape + content
        assert sim.tracer_matrix.shape == (n_z, n_pix)
        assert sim.host_matrix.shape == (n_z, n_pix)
        assert sim.tracer_matrix.min() > 0.0
        assert abs(sim.host_matrix.sum() - 1.0) < 1e-6
        assert len(pipe.catalog) == cfg.events.total_events
        assert np.all(np.isfinite(pipe.catalog[["z", "ra", "dec", "delta"]].values))

        # reload hdf5 exactly as icarogw density_map.__init__ does
        import h5py
        with h5py.File(paths["icarogw_density_map"], "r") as f:
            zg = f["redshift_grid"][:]
            pg = f["pixel_grid"][:]
            dm = f["density_matrix"][:]
        log_dm = np.log(dm)                      # icarogw ingest
        log_avg = np.log(dm.mean(axis=1))        # density_matrix_average
        assert dm.shape == (n_z, n_pix)
        assert zg.shape == (n_z,) and pg.shape == (n_pix,)
        assert np.all(np.isfinite(log_dm)), "log(density_matrix) has -inf/NaN!"
        assert np.all(np.isfinite(log_avg)), "log sky-average has -inf/NaN!"

        # moca csv: columns + norm
        import pandas as pd
        df = pd.read_csv(paths["moca_probability_map"])
        assert list(df.columns) == ["ra", "dec", "z", "host_probability"]
        assert abs(df["host_probability"].sum() - 1.0) < 1e-6
        assert len(df) == n_z * n_pix

        # figures: >=8 of 11 must render with the stub
        figs = sorted((cfg.out_dir / "figures").glob("*.pdf"))
        n_fig = len(figs)
        assert n_fig >= 8, f"only {n_fig} figures rendered: {[f.name for f in figs]}"

        # main pipeline must be the unbiased baseline 1 + delta
        floor = 1e-6
        expected = np.maximum(1.0 + sim.delta_matrix, floor)
        assert np.allclose(sim.tracer_matrix, expected), \
            "main tracer is not the unbiased matter field 1+delta!"

        # experiments = separate step: reload baseline, layer on top
        from darksiren_hosts import experiments_run
        experiments_run.main(["--out", str(cfg.out_dir)])
        vroot = cfg.out_dir / "variants"
        assert {"baseline_matter", "biased_galaxies", "biased_clusters",
                "biased_hi", "blur_5deg", "noise_10pct"} <= {
                    d.name for d in vroot.iterdir() if d.is_dir()}
        n_var = 0
        for vdir in sorted(d for d in vroot.iterdir()
                           if d.is_dir() and d.name != "figures"):
            h5 = vdir / cfg.products.icarogw_density_map
            csv = vdir / cfg.products.moca_probability_map
            assert h5.exists() and csv.exists(), f"variant {vdir.name} missing products"
            with h5py.File(h5, "r") as f:
                tr = f["density_matrix"][:]
            assert np.all(np.isfinite(np.log(tr))), f"{vdir.name}: log(tracer) not finite"
            assert abs(pd.read_csv(csv)["host_probability"].sum() - 1.0) < 1e-6
            n_var += 1
        # comparison sheets
        assert (vroot / "comparison_overdensity.png").exists()
        assert (vroot / "comparison_host_pdf.png").exists()

    print("=" * 64)
    print("[PASS] end-to-end smoke test (CAMB real, healpy directions stubbed)")
    print(f"       n_z={n_z}, n_pix={n_pix}, events={len(pipe.catalog)}")
    print(f"       tracer>0, host PDF sum=1, ICAROGW log(map) finite, MOCA CSV ok")
    print(f"       {n_fig} report figures rendered")
    print(f"       main pipeline is UNBIASED (tracer == 1+delta)")
    print(f"       {n_var} experiments (separate) written + 2 shared-colorbar sheets")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
