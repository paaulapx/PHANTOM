#!/usr/bin/env python
"""
main script. map generator. edit the block and run it.
output dir auto-named: sim_H0<H0>_Om<Om>_nside<N>_z<zmin>-<zmax>_ev<N>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from darksiren_hosts import Config, MockHostPipeline

cfg = Config.default()

# cosmology: planck18. omch2 re-derived from Omega_m below
cfg.cosmo.H0      = 67.66
cfg.cosmo.Omega_m = 0.3111
cfg.cosmo.mnu     = 0.0

# grid rule: chi(z_max) < ~0.45 * box.size, else lightcone wraps (moire) chi: z=2.0 -> 3592, z=2.5 -> 4038, z=3.0 -> 4401 mpc/h
# n_grid sets voxel + kmax: 640 @ 10000 -> voxel 15.6 mpc/h, kmax 0.10 h/mpc (same bandlimit as 512 @ 8000; below kmax ~0.03 maps go near-isotropic)
cfg.healpix.nside       = 128
cfg.redshift.z_min      = 0.05
cfg.redshift.z_max      = 3.0
cfg.redshift.n_shells   = 100
cfg.events.total_events = 10000

cfg.box.size   = 10000.0
cfg.box.n_grid = 640

# bias: (1, 0) = bare matter baseline
cfg.bias.b_gw    = 1.0
cfg.bias.alpha_gw = 0.0

RUN_VARIANTS = True              # also write bias/blur/noise suite
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# keep ombh2 + omch2 = Omega_m * h^2 exact
h = cfg.cosmo.H0 / 100.0
cfg.cosmo.omch2 = round(cfg.cosmo.Omega_m * h**2 - cfg.cosmo.ombh2, 6)


def sim_name(c) -> str:
    name = (f"sim_H0{c.cosmo.H0:g}_Om{c.cosmo.Omega_m:g}"
            f"_nside{c.healpix.nside}"
            f"_z{c.redshift.z_min:g}-{c.redshift.z_max:g}"
            f"_ev{c.events.total_events}")
    if (c.bias.b_gw, c.bias.alpha_gw) != (1.0, 0.0):
        name += f"_b{c.bias.b_gw:g}a{c.bias.alpha_gw:g}"
    return name


cfg.out_dir = RESULTS_ROOT / sim_name(cfg)

if __name__ == "__main__":
    print(f"output -> {cfg.out_dir}", flush=True)
    pipe = MockHostPipeline(cfg)
    pipe.run()
    if RUN_VARIANTS:
        pipe.run_variants()
