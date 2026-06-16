"""
pipeline.py
the main pipeline: build the simulation, sample the catalogue, write products. also runs variants on top of the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .simulation import Simulation


class MockHostPipeline:
    def __init__(self, config, *, verbose=True, make_plots=True, log_file=None):
        self.config = config
        self.verbose = verbose
        self.make_plots = make_plots
        self.out_dir = Path(config.out_dir)
        self.sim = Simulation(config, verbose=verbose)
        self.catalog = None
        self.log_file = log_file
        if log_file:
            import sys
            self._log_fh = open(log_file, 'w')
            self._original_stdout = sys.stdout
            sys.stdout = self._log_fh
        else:
            self._log_fh = None

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def __del__(self):
        if self._log_fh:
            sys.stdout = self._original_stdout
            self._log_fh.close()

    # ------------------------------------------------------------------ #
    def run(self):
        """build maps, sample catalogue, write products."""
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.sim.build()
        self._log("sampling cosmic-truth host catalogue ...")
        self.catalog = self.sim.sample_catalog()

        self._validate()

        paths = {}
        paths["host_catalogue"] = self.write_host_catalogue()
        paths["moca_probability_map"] = self.write_moca_probability_map()
        paths.update(self.write_icarogw_density_map())
        paths["truths"] = self.write_truths()
        paths["config"] = self.write_config()

        if self.make_plots:
            paths["figures"] = self.write_figures()

        self._log("done. products:")
        for k, v in paths.items():
            self._log(f"  {k:24s} -> {v}")
        return paths

    # ------------------------------------------------------------------ #
    # low-level writers (baseline + variants)
    # ------------------------------------------------------------------ #
    def _write_density_hdf5(self, path, tracer_matrix, attrs):
        """icarogw hdf5: bare tracer, ring, axis0=z axis1=pix.
        loads as density_map(redshift_grid, pixel_grid, density_matrix)."""
        with h5py.File(path, "w") as f:
            f.create_dataset("redshift_grid", data=self.sim.shells)
            f.create_dataset("pixel_grid", data=self.sim.pixel_grid)
            f.create_dataset("density_matrix", data=tracer_matrix,
                             compression="gzip")
            f.attrs["nside"] = self.config.healpix.nside
            f.attrs["ordering"] = "RING"
            f.attrs["content"] = "spatial_tracer_1plus_bdelta"
            f.attrs["note"] = ("bare tracer; icarogw applies psi(z) and dVc/dz "
                               "internally — do not pre-weight.")
            for k, v in attrs.items():
                f.attrs[k] = v
        return path

    def _write_moca_csv(self, path, host_matrix):
        """moca csv: joint host pdf, long format."""
        ra, dec = self.sim.projector.pixel_radec_deg()
        n_z = len(self.sim.shells)
        n_pix = self.config.healpix.npix
        pd.DataFrame({
            "ra": np.tile(ra, n_z),
            "dec": np.tile(dec, n_z),
            "z": np.repeat(self.sim.shells, n_pix),
            "host_probability": np.asarray(host_matrix).reshape(-1),
        }).to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------ #
    # baseline writers
    # ------------------------------------------------------------------ #
    def write_icarogw_density_map(self):
        # baseline = unbiased matter, 1 + delta (b=1)
        p = self.config.products
        attrs = {"b_gw": 1.0, "alpha_gw": 0.0,
                 "variant": "theoretical_baseline_matter"}
        written = {}
        written["icarogw_density_map"] = self._write_density_hdf5(
            self.out_dir / p.icarogw_density_map, self.sim.tracer_matrix, attrs)
        written["icarogw_density_map_alias"] = self._write_density_hdf5(
            self.out_dir / p.icarogw_density_map_alias, self.sim.tracer_matrix, attrs)
        return written

    def write_moca_probability_map(self):
        return self._write_moca_csv(
            self.out_dir / self.config.products.moca_probability_map,
            self.sim.host_matrix)

    def write_host_catalogue(self):
        path = self.out_dir / self.config.products.host_catalogue
        self.catalog.to_csv(path, index=False)
        return path

    def write_truths(self):
        """cosmology + rate + grid truths -> moca/icarogw match the universe."""
        c, r = self.config.cosmo, self.config.rate
        truths = {
            "cosmology": {"H0": c.H0, "Omega_m": self.sim.cosmo.Omega_m,
                          "ombh2": c.ombh2, "omch2": c.omch2, "ns": c.ns,
                          "As": c.As, "mnu": c.mnu},
            "rate_evolution": {"model": r.model, "gamma": r.gamma,
                               "kappa": r.kappa, "zp": r.zp},
            # baseline unbiased; config bias only feeds the experiments layer
            "baseline_bias": {"b_gw": 1.0, "alpha_gw": 0.0},
            "experiments_gw_bias": {"b_gw": self.config.bias.b_gw,
                                    "alpha_gw": self.config.bias.alpha_gw},
            "grid": {"nside": self.config.healpix.nside,
                     "ordering": "RING",
                     "z_min": self.config.redshift.z_min,
                     "z_max": self.config.redshift.z_max,
                     "n_shells": self.config.redshift.n_shells},
            "products": {
                "host_catalogue": self.config.products.host_catalogue,
                "moca_probability_map": self.config.products.moca_probability_map,
                "icarogw_density_map": self.config.products.icarogw_density_map,
            },
        }
        path = self.out_dir / self.config.products.truths
        path.write_text(json.dumps(truths, indent=2))
        return path

    def write_config(self):
        path = self.out_dir / self.config.products.config
        path.write_text(json.dumps(self.config.to_dict(), indent=2, default=str))
        return path

    def write_figures(self):
        """figures into out_dir/figures, best-effort."""
        fig_dir = self.out_dir / "figures"
        try:
            from .plots import ReportPlots
        except Exception as e:                       # noqa: BLE001
            self._log(f"  [skip] figures: plotting deps unavailable ({e}).")
            return fig_dir
        self._log("rendering report figures ...")
        ReportPlots(self.sim, self.catalog, fig_dir, verbose=self.verbose).make_all()
        return fig_dir

    # ------------------------------------------------------------------ #
    # variants layer
    # ------------------------------------------------------------------ #
    def run_variants(self, variants=None, *, make_figs=True):
        """experiments on top of the one baseline delta_matrix. baseline
        products untouched. each variant -> variants/<name>/ with its own
        bare-tracer hdf5 + host csv (+ figures, comparison sheets)."""
        from .perturbations import build_variant_products, default_variants

        if self.sim.delta_matrix is None:
            self.sim.build()
        variants = default_variants(self.config) if variants is None else variants
        root = self.out_dir / "variants"
        p = self.config.products
        written = {}
        i_mid = len(self.sim.shells) // 2
        gallery = []                                      # (name, contrast_map, host_map)

        self._log(f"running {len(variants)} experiments (separate from baseline) ...")
        for v in variants:
            vdir = root / v.name
            vdir.mkdir(parents=True, exist_ok=True)
            tracer, host = build_variant_products(self.sim, v)

            if not np.all(np.isfinite(tracer)) or tracer.min() <= 0:
                raise ValueError(f"variant {v.name}: icarogw tracer must be >0.")
            if not np.isclose(host.sum(), 1.0, rtol=1e-6):
                raise ValueError(f"variant {v.name}: host pdf not normalised.")

            attrs = {"variant": v.name, "bias": v.bias.label,
                     "description": v.describe()}
            self._write_density_hdf5(vdir / p.icarogw_density_map, tracer, attrs)
            self._write_moca_csv(vdir / p.moca_probability_map, host)
            if make_figs:
                self._variant_figures(vdir / "figures", v, tracer, host)
            gallery.append((v.name, tracer[i_mid] - 1.0, host[i_mid]))
            written[v.name] = vdir
            self._log(f"  {v.name:18s} [{v.describe()}] -> {vdir}")

        if make_figs:
            from .plots import ReportPlots
            rp = ReportPlots(self.sim, self.catalog, root / "figures", verbose=False)
            for vv in variants:
                if vv.name != "baseline_matter":
                    rp.plot_tracer_cross_correlation(vv.name)
            self._variant_comparison_sheets(root, gallery, float(self.sim.shells[i_mid]))
        return written

    def _variant_comparison_sheets(self, root, gallery, z):
        """shared-colorbar sheets, all variants, one shell."""
        try:
            from .plots import ReportPlots
        except Exception:                                
            return
        nest = self.config.healpix.nest
        names = [g[0] for g in gallery]
        ReportPlots.comparison_sheet(
            [g[1] for g in gallery], names, root / "comparison_overdensity.png",
            unit=r"$b\,\delta$", symmetric=True, nest=nest,
            suptitle=fr"Tracer overdensity by experiment, $z={z:.2f}$ (shared scale)")
        ReportPlots.comparison_sheet(
            [g[2] for g in gallery], names, root / "comparison_host_pdf.png",
            unit=r"$p(z,\hat n)$", symmetric=False, nest=nest,
            suptitle=fr"Host PDF by experiment, $z={z:.2f}$ (shared scale)")
        self._log(f"  comparison sheets -> {root}")

    def _variant_figures(self, fig_dir, variant, tracer, host):
        """per-variant mollweides, best-effort."""
        try:
            from .plots import ReportPlots
        except Exception:                              
            return
        fig_dir.mkdir(parents=True, exist_ok=True)
        i = len(self.sim.shells) // 2
        z = float(self.sim.shells[i])
        delta_t = tracer[i] - 1.0                        # b(z)*delta
        ReportPlots.mollweide(
            delta_t, fig_dir / "tracer_overdensity_mollweide.png",
            title=fr"{variant.name}: $b\,\delta(\hat n)$ at $z={z:.2f}$",
            unit=r"$b\,\delta$", symmetric=True)
        ReportPlots.mollweide(
            host[i], fig_dir / "host_pdf_mollweide.png",
            title=fr"{variant.name}: host PDF at $z={z:.2f}$",
            unit=r"$p(z,\hat n)$", symmetric=False)

    # ------------------------------------------------------------------ #
    def _validate(self):
        """cheap invariants, fail before corrupting downstream."""
        T = self.sim.tracer_matrix
        H = self.sim.host_matrix
        if not np.all(np.isfinite(T)) or T.min() <= 0:
            raise ValueError("icarogw tracer must be finite and strictly > 0 "
                             "(it is log-transformed on ingest).")
        if not np.all(np.isfinite(H)) or H.min() < 0:
            raise ValueError("moca host pdf must be finite and non-negative.")
        if not np.isclose(H.sum(), 1.0, rtol=1e-6):
            raise ValueError(f"moca host pdf must be normalised; sum={H.sum()}.")
        if not bool(self.catalog["delta"].notna().all()):
            raise ValueError("catalogue contains nan delta.")
        self._log(f"validation ok: tracer>0, host pdf sum={H.sum():.6f}, "
                  f"{len(self.catalog)} hosts.")
