# PHANTOM - Probabilistic Host Assignment via Nonlinear Tomographic Overdensity Maps

This pipeline forward-models a mock universe of gravitational-wave host galaxies for dark-siren cosmology. 

Code is intended to be used as a `darksiren_hosts/` package + `run_sim.py`.

## Outputs

Written to an auto-named run directory:

- `density_matrix.hdf5` — bare spatial tracer `T = 1 + b(z)·δ` (for ICAROGW; it applies
  `ψ(z)·dV_c/dz` itself, so the map must not be pre-weighted).
- `full_healpix_probabilities.csv` — joint host PDF `p(z, Ω)`.
- `gw_event_catalog.csv` — the cosmic-truth host catalogue `(z, ra, dec, δ)`.
- `gwpipeline_truths.json`, `config.json` — the universe's parameters.

An optional *variants* layer applies biased tracers (galaxies, clusters, HI, GW) and map
degradations (noise, beam) on top of the same baseline field.

## Run

```bash
cd pipeline/PHANTOM

# baseline: edit the config block at the top of run_sim.py (cosmology, z range,
# nside, n_shells, events, box size/grid), then:
python run_sim.py

# optional: bias/noise/blur variants on top of a finished run (no field regen)
python -m darksiren_hosts.experiments_run --out <run_dir>

# tests
python -m darksiren_hosts.selftest     # numpy-only unit tests
python -m darksiren_hosts.smoketest    # end-to-end offline check
```

**Requires:** `numpy`, `scipy`, `camb`, `healpy`, `h5py`, `pandas`
(`selftest.py` needs only numpy + scipy).
