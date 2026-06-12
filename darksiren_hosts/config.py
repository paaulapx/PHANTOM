"""
config.py
config dataclasses for the mock host pipeline. 
these are used both to drive the forward model and to write out the config.json for record-keeping and reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
@dataclass
class CosmoParams:
    """planck18 flat lcdm. keep ombh2 + omch2 = Omega_m * h^2 exact."""
    H0: float = 67.66
    ombh2: float = 0.02242
    omch2: float = 0.12000
    mnu: float = 0.0
    omk: float = 0.0
    tau: float = 0.0561
    As: float = 2.1e-9
    ns: float = 0.9665
    r: float = 0.0
    Omega_m: float = 0.3111

    @property
    def h(self) -> float:
        return self.H0 / 100.0


@dataclass
class BoxParams:
    """periodic box + field controls. everything mpc/h -- one ruler for
    fft grid and shell radii."""
    size: float = 15000.0        # mpc/h
    n_grid: int = 256            # fft cells per side
    # none -> 1 voxel (tapers sinc ringing from sharp kmax). 0 = hard cutoff.
    # sub-voxel = inert (field generator warns)
    smooth_mpch: float | None = None
    kmax_factor: float = 0.5     # cutoff at this * k_nyquist
    seed: int = 42
    coherent_shells: bool = True  # one phase draw for all radii
    dtype: str = "float32"       # cube only; accumulators float64
    # camb p(k) sampling
    camb_minkh: float = 1e-4
    camb_npoints: int = 1000
    pk_spline_s: float = 0.1     # log-log spline smoothing
    pk_spline_k: int = 3

    @property
    def voxel(self) -> float:
        return self.size / self.n_grid

    def resolved_smooth(self) -> float:
        """mpc/h, none -> one voxel."""
        return self.voxel if self.smooth_mpch is None else float(self.smooth_mpch)

    @property
    def k_nyquist(self) -> float:
        return math.pi * self.n_grid / self.size

    @property
    def kmax(self) -> float:
        return self.kmax_factor * self.k_nyquist


@dataclass
class RedshiftParams:
    z_min: float = 0.05
    z_max: float = 5.0
    n_shells: int = 100          # geomspaced, finer at low z
    n_z_sampling: int = 1000     # inverse-cdf sampler grid

    @property
    def shells(self) -> list[float]:
        return np.geomspace(self.z_min, self.z_max, self.n_shells).tolist()


@dataclass
class RateParams:
    """madau-dickinson psi(z). gwtc values. moca + icarogw must share these."""
    model: str = "madau_dickinson"
    gamma: float = 4.59
    kappa: float = 2.86
    zp: float = 2.47


@dataclass
class BiasParams:
    """b(z) = b_gw * (1+z)^alpha_gw. (1, 0) = bare matter."""
    b_gw: float = 1.0
    alpha_gw: float = 0.0


@dataclass
class HealpixParams:
    nside: int = 128
    nest: bool = False           # ring, same as icarogw radec2indeces

    @property
    def npix(self) -> int:
        return 12 * self.nside * self.nside


@dataclass
class EventParams:
    total_events: int = 10000


@dataclass
class Products:
    """product file names -- the gwpipeline contract."""
    host_catalogue: str = "gw_event_catalog.csv"
    moca_probability_map: str = "full_healpix_probabilities.csv"
    icarogw_density_map: str = "density_matrix.hdf5"
    icarogw_density_map_alias: str = "host_probabilities.hdf5"
    truths: str = "gwpipeline_truths.json"
    config: str = "config.json"

@dataclass
class Floors:
    tracer_floor: float = 1e-6  # min 1+b*delta (icarogw takes log)
    pdf_eps: float = 1e-12      # row norm


# --------------------------------------------------------------------------- #
# top-level config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    cosmo: CosmoParams = field(default_factory=CosmoParams)
    box: BoxParams = field(default_factory=BoxParams)
    redshift: RedshiftParams = field(default_factory=RedshiftParams)
    rate: RateParams = field(default_factory=RateParams)
    bias: BiasParams = field(default_factory=BiasParams)
    healpix: HealpixParams = field(default_factory=HealpixParams)
    events: EventParams = field(default_factory=EventParams)
    products: Products = field(default_factory=Products)
    floors: Floors = field(default_factory=Floors)
    nonlinear: bool = True
    sky_area: float = 4.0 * math.pi   # steradian
    out_dir: Path = field(
        default_factory=lambda: Path("output") / "darksiren_hosts_run"
    )


    @classmethod
    def default(cls) -> "Config":
        return cls()

    @property
    def shells(self) -> np.ndarray:
        return np.sort(np.asarray(self.redshift.shells, dtype=float))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d

    def sanity_check(self) -> None:
        """fail fast on map-corrupting configs."""
        from .cosmology import Cosmology  # avoid camb at import time

        if self.redshift.z_min <= 0:
            raise ValueError("z_min must be > 0 (psi/dVc/dz undefined at z=0).")
        if self.redshift.z_max <= self.redshift.z_min:
            raise ValueError("z_max must exceed z_min.")
        if self.box.n_grid % 2 != 0:
            raise ValueError("n_grid should be even for the rfft layout.")
        if self.healpix.nest:
            raise ValueError(
                "icarogw radec2indeces uses ring ordering; keep nest=False."
            )
        # chi(z_max) sphere must fit in the half-box, else lightcone wraps
        chi_max = Cosmology(self).comoving_distance(self.redshift.z_max)
        if chi_max > 0.5 * self.box.size:
            raise ValueError(
                f"chi(z_max)={chi_max:.0f} Mpc/h exceeds half the box "
                f"({0.5 * self.box.size:.0f} Mpc/h); enlarge box.size."
            )
