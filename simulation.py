"""
simulation.py
Core simulation logic: field generation, projection, host pdf construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cosmology import Cosmology
from .field import GaussianFieldGenerator
from .projection import SkyProjector


def host_pdf_from_tracer(tracer_matrix, weight_z, *, eps=1e-12):
    """tracer -> joint host pdf. p(pix|z) = row-normalised tracer,
    z-marginal = weight_z = psi dvc/dz /(1+z) dz, total = 1.
    shared by baseline build and variants."""
    ang = np.asarray(tracer_matrix, dtype=float) + eps
    ang = ang / ang.sum(axis=1, keepdims=True)
    host = ang * np.asarray(weight_z, dtype=float)[:, None]
    total = host.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("host pdf has non-positive / non-finite total.")
    return host / total


class Simulation:
    def __init__(self, config, *, verbose=True):
        self.config = config
        self.verbose = verbose
        config.sanity_check()

        self.cosmo = Cosmology(config)
        self.projector = SkyProjector(config, self.cosmo)
        self.field = GaussianFieldGenerator(
            n_grid=config.box.n_grid, box_size=config.box.size,
            smooth_mpch=config.box.resolved_smooth(), kmax=config.box.kmax,
            seed=config.box.seed, coherent=config.box.coherent_shells,
            dtype=np.dtype(config.box.dtype),
        )

        self.shells = config.shells                       # sorted (n_z,)
        self.pixel_grid = np.arange(config.healpix.npix)
        # sampling rng separate from field rng -> reproducible independently
        self._rng = np.random.default_rng(config.box.seed + 1)

        # filled by build(), all (n_z, n_pix)
        self.delta_matrix = None      # matter contrast
        self.tracer_matrix = None     # 1 + b(z) delta   (icarogw)
        self.angular_pdf = None       # p(pix | z), rows sum 1
        self.host_matrix = None       # joint host pdf   (moca)

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ #
    @classmethod
    def load_products(cls, config, out_dir, *, verbose=True):
        """rebuild from saved products, no field regen, no camb.
        delta = (tracer - 1)/b(z); b from file attrs (baseline: 1)."""
        import h5py
        out_dir = Path(out_dir)
        p = config.products

        sim = cls(config, verbose=verbose)
        with h5py.File(out_dir / p.icarogw_density_map, "r") as f:
            shells = f["redshift_grid"][:]
            tracer = f["density_matrix"][:]
            b_gw = float(f.attrs.get("b_gw", 1.0))
            alpha_gw = float(f.attrs.get("alpha_gw", 0.0))
        sim.shells = np.asarray(shells, dtype=float)
        sim.tracer_matrix = np.asarray(tracer, dtype=float)
        b_z = (b_gw * (1.0 + sim.shells) ** alpha_gw)[:, None]
        sim.delta_matrix = (sim.tracer_matrix - 1.0) / b_z
        sim.angular_pdf = sim.tracer_matrix / sim.tracer_matrix.sum(axis=1, keepdims=True)

        host = pd.read_csv(out_dir / p.moca_probability_map)["host_probability"].to_numpy()
        sim.host_matrix = host.reshape(len(sim.shells), config.healpix.npix)

        catalog = pd.read_csv(out_dir / p.host_catalogue)
        return sim, catalog

    # ------------------------------------------------------------------ #
    # steps 1-4: p(k,z) -> field -> projection -> bare tracer
    # ------------------------------------------------------------------ #
    def build(self):
        """delta matrix + unbiased products (tracer 1+delta, host pdf)."""
        self._log("computing non-linear P(k,z) with CAMB ...")
        self.cosmo.compute_power_spectra()

        n_z, n_pix = len(self.shells), self.config.healpix.npix
        delta = np.empty((n_z, n_pix), dtype=np.float64)

        self._log(f"generating + projecting {n_z} shells ...")
        for i, z in enumerate(self.shells):
            cube = self.field.realize(self.cosmo.power_at(z), z=z)
            delta[i] = self.projector.project_shell(cube, z)
            if self.verbose and (i % max(1, n_z // 10) == 0 or i == n_z - 1):
                self._log(f"  shell {i + 1:3d}/{n_z}  z={z:5.3f}  "
                          f"sigma_delta={delta[i].std():.3f}")
        self.delta_matrix = delta

        # baseline = bare matter, T = 1 + delta, floored for icarogw's log.
        # bias/blur/noise never enter here -- that's the variants layer
        self.tracer_matrix = np.maximum(1.0 + delta, self.config.floors.tracer_floor)

        # p(pix | z): row-normalised tracer
        rows = self.tracer_matrix.sum(axis=1, keepdims=True)
        self.angular_pdf = self.tracer_matrix / rows

        # moca joint p(z, pix)
        self.host_matrix = self._build_host_matrix()
        return self

    def redshift_weights(self):
        """psi dvc/dz /(1+z) dz_shell on the shell grid, (n_z,)."""
        z = self.shells
        return self.cosmo.redshift_kernel(z) * np.gradient(z)

    def _build_host_matrix(self):
        """moca host pdf. z-marginal = rate x volume kernel, angular = tracer
        -- exactly how sample_catalog draws."""
        return host_pdf_from_tracer(self.tracer_matrix, self.redshift_weights())

    # ------------------------------------------------------------------ #
    # step 5: event sampling
    # ------------------------------------------------------------------ #
    def sample_redshifts(self, n):
        """inverse-cdf draw, p(z) ~ psi dvc/dz /(1+z)."""
        zg = np.linspace(self.config.redshift.z_min,
                         self.config.redshift.z_max,
                         self.config.redshift.n_z_sampling)
        pdf = self.cosmo.redshift_kernel(zg)
        if not np.all(np.isfinite(pdf)) or pdf.min() < 0 or pdf.sum() <= 0:
            raise RuntimeError("invalid psi*dVc/dz kernel in sample_redshifts.")
        incr = 0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg)       # trapezoid
        cdf = np.concatenate([[0.0], np.cumsum(incr)])
        cdf /= cdf[-1]
        u = self._rng.uniform(0.0, 1.0, n)
        return np.interp(u, cdf, zg)

    def _nearest_shell_index(self, z):
        """closest shell per z, vectorised."""
        z = np.asarray(z, dtype=float)
        right = np.searchsorted(self.shells, z)
        right = np.clip(right, 1, len(self.shells) - 1)
        left = right - 1
        pick_left = (z - self.shells[left]) <= (self.shells[right] - z)
        return np.where(pick_left, left, right)

    def sample_catalog(self, total_events=None):
        """truth catalogue. z ~ kernel; pixel ~ p(pix | nearest shell)."""
        if self.tracer_matrix is None:
            raise RuntimeError("call build() before sample_catalog().")
        n = (self.config.events.total_events
             if total_events is None else int(total_events))

        z = self.sample_redshifts(n)
        sidx = self._nearest_shell_index(z)
        ra_pix, dec_pix = self.projector.pixel_radec_deg()

        pix = np.empty(n, dtype=np.int64)
        for s in np.unique(sidx):
            m = sidx == s
            pix[m] = self._rng.choice(self.pixel_grid, size=int(m.sum()),
                                      p=self.angular_pdf[s])

        delta = self.delta_matrix[sidx, pix]
        catalog = pd.DataFrame({
            "event_id": np.arange(n),
            "z": z,
            "ra": ra_pix[pix],
            "dec": dec_pix[pix],
            "delta": delta,                                   # local contrast
            "comoving_distance_mpch": self.cosmo.comoving_distance(z),
            "healpix_pixel": pix,
            "z_shell": self.shells[sidx],
        })
        return catalog
