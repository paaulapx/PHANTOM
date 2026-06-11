"""
from_real_map.py
convert a real LSS map (e.g. HI intensity or galaxy catalogue) into the density_matrix.hdf5 format for ICAROGW. this is a separate script from the main pipeline because it doesn't need to run camb or make the field, and it needs to load the real map instead of generating it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

_FLOOR = 1e-6          # icarogw takes log -> strictly positive
_MASK_FILL = 1.0       # unobserved -> mean density (uninformative, not a void)
_NU_21CM_MHZ = 1420.405751  # rest-frame 21cm


def freq_to_redshift(freq_mhz):
    """z = nu_21/nu_obs - 1."""
    return _NU_21CM_MHZ / np.asarray(freq_mhz, dtype=float) - 1.0


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
def write_density_map(path, redshift_grid, pixel_grid, density_matrix, nside,
                      *, content="real_map_tracer", extra_attrs=None):
    density_matrix = np.asarray(density_matrix, dtype=float)
    if density_matrix.shape != (len(redshift_grid), len(pixel_grid)):
        raise ValueError(
            f"density_matrix {density_matrix.shape} must be "
            f"(n_z={len(redshift_grid)}, n_pix={len(pixel_grid)})")
    if not np.all(np.isfinite(density_matrix)) or density_matrix.min() <= 0:
        raise ValueError("density_matrix must be finite and strictly > 0 "
                         "(icarogw takes its log on ingest).")
    with h5py.File(path, "w") as f:
        f.create_dataset("redshift_grid", data=np.asarray(redshift_grid, float))
        f.create_dataset("pixel_grid", data=np.asarray(pixel_grid))
        f.create_dataset("density_matrix", data=density_matrix, compression="gzip")
        f.attrs["nside"] = int(nside)
        f.attrs["ordering"] = "RING"
        f.attrs["content"] = content
        f.attrs["note"] = ("bare spatial tracer; icarogw applies psi(z) and "
                           "dVc/dz internally -- do not pre-weight.")
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
    return path


# --------------------------------------------------------------------------- #
# per-shell field -> positive mean-1 tracer row
# --------------------------------------------------------------------------- #
def _tracer_from_field(field, *, kind, valid=None, tmean=None, floor=_FLOOR):
    """kind='density': tracer = field/<field>. kind='contrast': tracer =
    1 + field/tmean (or 1 + field). invalid pixels -> 1, then floor."""
    field = np.asarray(field, dtype=float)
    if valid is None:
        valid = np.isfinite(field)
    out = np.full(field.shape, _MASK_FILL, dtype=float)

    if kind == "density":
        m = float(np.mean(field[valid])) if valid.any() else 0.0
        if not np.isfinite(m) or m <= 0:
            raise ValueError("density field has non-positive mean; "
                             "is it actually a contrast? use kind='contrast'.")
        out[valid] = field[valid] / m
    elif kind == "contrast":
        delta = field[valid] / tmean if tmean else field[valid]
        out[valid] = 1.0 + delta
    else:
        raise ValueError("kind must be 'density' or 'contrast'")

    return np.maximum(out, floor)


def _auto_kind(cube):
    """signed + ~zero-mean (foreground-cleaned hi) -> contrast, else density."""
    finite = cube[np.isfinite(cube)]
    if finite.size == 0:
        return "density"
    has_neg = finite.min() < 0
    near_zero_mean = abs(finite.mean()) < 0.1 * (finite.std() + 1e-30)
    return "contrast" if (has_neg and near_zero_mean) else "density"


# --------------------------------------------------------------------------- #
# hi cube -> tracer matrix
# --------------------------------------------------------------------------- #
def from_hi_cube(cube, z_centers, *, kind="auto", tmean=None, mask=None,
                 floor=_FLOOR):
    """(n_z, n_pix) brightness cube -> tracer matrix. mask: n_pix bool,
    true=observed; none -> per-shell finite pixels."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 2:
        raise ValueError(f"HI cube must be 2D (n_z, n_pix); got {cube.shape}")
    z_centers = np.asarray(z_centers, dtype=float)
    if len(z_centers) != cube.shape[0]:
        raise ValueError("z_centers length must equal cube.shape[0]")
    if kind == "auto":
        kind = _auto_kind(cube)
        print(f"[from_hi_cube] auto-detected kind='{kind}'"
              + (f" (tmean={tmean})" if kind == "contrast" else ""))

    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        valid = mask if mask is not None else np.isfinite(cube[i])
        out[i] = _tracer_from_field(cube[i], kind=kind, valid=valid,
                                    tmean=tmean, floor=floor)
    return z_centers, out


def coarsen_shells(z_centers, matrix, factor):
    """average every `factor` adjacent shells (840 -> 105 etc)."""
    if factor <= 1:
        return z_centers, matrix
    n = (len(z_centers) // factor) * factor
    z = z_centers[:n].reshape(-1, factor).mean(axis=1)
    m = matrix[:n].reshape(-1, factor, matrix.shape[1]).mean(axis=1)
    return z, m


# --------------------------------------------------------------------------- #
# galaxy catalogue -> tracer matrix
# --------------------------------------------------------------------------- #
def from_galaxy_catalog(ra_deg, dec_deg, z, *, nside, z_edges, weights=None,
                        mask=None, smooth_fwhm_deg=0.0, floor=_FLOOR):
    """(ra, dec, z) [deg] -> counts -> tracer, ring (icarogw convention).
    beam reduces shot noise; mask restricts the mean to the footprint."""
    import healpy as hp
    ra = np.deg2rad(np.asarray(ra_deg, float))
    dec = np.deg2rad(np.asarray(dec_deg, float))
    z = np.asarray(z, float)
    w = np.ones_like(z) if weights is None else np.asarray(weights, float)

    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra, nest=False)   # ring
    z_edges = np.asarray(z_edges, float)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    out = np.empty((len(z_centers), npix), dtype=float)
    fwhm = np.deg2rad(smooth_fwhm_deg) if smooth_fwhm_deg > 0 else 0.0
    for i in range(len(z_centers)):
        sel = (z >= z_edges[i]) & (z < z_edges[i + 1])
        counts = np.bincount(pix[sel], weights=w[sel], minlength=npix).astype(float)
        if fwhm > 0:
            counts = hp.smoothing(counts, fwhm=fwhm)
            counts = np.maximum(counts, 0.0)
        valid = mask if mask is not None else np.ones(npix, dtype=bool)
        out[i] = _tracer_from_field(counts, kind="density", valid=valid, floor=floor)
    return z_centers, out


# --------------------------------------------------------------------------- #
# loaders (edit load_hi_cube for your files)
# --------------------------------------------------------------------------- #
def load_hi_cube(path, *, in_nest=False):
    """hi cube -> (cube[n_z, n_pix], z_centers). handles .npy/.npz/.hdf5
    with common key names; freq axis [mhz] -> z via 21cm relation if no z
    axis; nest reordered to ring when in_nest."""
    path = Path(path)
    z, freq = None, None
    if path.suffix == ".npy":
        cube = np.load(path)
    elif path.suffix == ".npz":
        d = np.load(path)
        key = next((k for k in ("cube", "maps", "T", "density") if k in d), d.files[0])
        cube = d[key]
        z = next((d[k] for k in ("z_centers", "z") if k in d), None)
        freq = next((d[k] for k in ("freq_mhz", "freq", "frequency") if k in d), None)
    elif path.suffix in (".h5", ".hdf5"):
        with h5py.File(path, "r") as f:
            key = next((k for k in ("cube", "maps", "T", "density") if k in f), None)
            if key is None:
                key = next(k for k in f if f[k].ndim == 2)
            cube = f[key][:]
            for zk in ("z_centers", "z", "redshift"):
                if zk in f:
                    z = f[zk][:]; break
            for fk in ("freq_mhz", "freq", "frequency"):
                if fk in f:
                    freq = f[fk][:]; break
    else:
        raise ValueError(f"unrecognised HI cube format: {path.suffix}; "
                         "edit load_hi_cube() for your layout.")
    cube = np.asarray(cube, dtype=float)
    if cube.shape[0] > cube.shape[1]:          # (n_pix, n_z)? transpose
        print("[load_hi_cube] transposing to (n_z, n_pix)")
        cube = cube.T
    if in_nest:
        import healpy as hp
        nside = int(round((cube.shape[1] / 12) ** 0.5))
        cube = np.array([hp.reorder(row, n2r=True) for row in cube])
        print(f"[load_hi_cube] reordered NEST->RING (nside={nside})")
    if z is None and freq is not None:
        z = freq_to_redshift(freq)
        print(f"[load_hi_cube] converted {len(z)} freq channels -> z "
              f"(nu_21={_NU_21CM_MHZ:.3f} MHz), z=[{z.min():.3f},{z.max():.3f}]")
    if z is None:
        raise ValueError("no redshift or frequency axis found in the file; pass "
                         "--z-min/--z-max or add a 'z_centers'/'freq_mhz' dataset.")
    z = np.asarray(z, dtype=float)
    if z[0] > z[-1]:                            # freq desc -> z asc, sort
        order = np.argsort(z)
        z, cube = z[order], cube[order]
    return cube, z


def load_glade_hdf5(path):
    """(ra_deg, dec_deg, z) from glade+-style hdf5; adjust names as needed."""
    with h5py.File(path, "r") as f:
        def pick(*names):
            for n in names:
                if n in f:
                    return f[n][:]
            raise KeyError(f"none of {names} in {path}; edit load_glade_hdf5().")
        ra = pick("ra", "RA")
        dec = pick("dec", "DEC", "Dec")
        z = pick("z", "z_cmb", "redshift")
    return np.asarray(ra, float), np.asarray(dec, float), np.asarray(z, float)


# --------------------------------------------------------------------------- #
def _report(z_centers, matrix):
    means = matrix.mean(axis=1)
    print(f"[from_real_map] n_z={len(z_centers)}  n_pix={matrix.shape[1]}  "
          f"z=[{z_centers.min():.3f},{z_centers.max():.3f}]")
    print(f"[from_real_map] per-shell <tracer> in [{means.min():.3f},{means.max():.3f}]"
          f" (should hover near 1), min cell={matrix.min():.2e} (>0 ok)")


def main(argv=None):
    p = argparse.ArgumentParser(description="real LSS map -> ICAROGW density_matrix.hdf5")
    sub = p.add_subparsers(dest="mode", required=True)

    h = sub.add_parser("hi", help="MeerKLASS-style HI brightness cube")
    h.add_argument("--in", dest="inp", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--kind", choices=["auto", "density", "contrast"], default="auto")
    h.add_argument("--tmean", type=float, default=None,
                   help="mean brightness if the cube is dT (e.g. 0.15e-3 K)")
    h.add_argument("--in-nest", action="store_true", help="input is NEST-ordered")
    h.add_argument("--coarsen", type=int, default=1, help="average N adjacent shells")

    g = sub.add_parser("galaxy", help="GLADE+-style (RA,Dec,z) catalogue")
    g.add_argument("--in", dest="inp", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--nside", type=int, required=True)
    g.add_argument("--z-min", type=float, required=True)
    g.add_argument("--z-max", type=float, required=True)
    g.add_argument("--n-shells", type=int, required=True)
    g.add_argument("--smooth-deg", type=float, default=0.0)
    g.add_argument("--mask", default=None, help="HEALPix mask FITS (1=observed)")
    args = p.parse_args(argv)

    if args.mode == "hi":
        cube, z = load_hi_cube(args.inp, in_nest=args.in_nest)
        z_centers, matrix = from_hi_cube(cube, z, kind=args.kind, tmean=args.tmean)
        z_centers, matrix = coarsen_shells(z_centers, matrix, args.coarsen)
        nside = int(round((matrix.shape[1] / 12) ** 0.5))
        src = {"source": "HI_intensity_map", "kind": args.kind}
    else:
        import healpy as hp
        ra, dec, z = load_glade_hdf5(args.inp)
        mask = hp.read_map(args.mask).astype(bool) if args.mask else None
        z_edges = np.linspace(args.z_min, args.z_max, args.n_shells + 1)
        z_centers, matrix = from_galaxy_catalog(
            ra, dec, z, nside=args.nside, z_edges=z_edges, mask=mask,
            smooth_fwhm_deg=args.smooth_deg)
        nside = args.nside
        src = {"source": "galaxy_catalogue"}

    pixel_grid = np.arange(matrix.shape[1])
    _report(z_centers, matrix)
    out = write_density_map(args.out, z_centers, pixel_grid, matrix, nside,
                            extra_attrs=src)
    print(f"[from_real_map] wrote {out}")


if __name__ == "__main__":
    main()
