"""
run.py
Entry point for the dark-siren mock host pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .pipeline import MockHostPipeline


def build_config(args) -> Config:
    cfg = Config.default()
    if args.out is not None:
        cfg.out_dir = Path(args.out)
    if args.n_grid is not None:
        cfg.box.n_grid = args.n_grid
    if args.box_size is not None:
        cfg.box.size = args.box_size
    if args.seed is not None:
        cfg.box.seed = args.seed
    if args.nside is not None:
        cfg.healpix.nside = args.nside
    if args.n_shells is not None:
        cfg.redshift.n_shells = args.n_shells
    if args.events is not None:
        cfg.events.total_events = args.events
    if args.b_gw is not None:
        cfg.bias.b_gw = args.b_gw
    if args.alpha_gw is not None:
        cfg.bias.alpha_gw = args.alpha_gw
    if args.linear:
        cfg.nonlinear = False
    if args.incoherent:
        cfg.box.coherent_shells = False
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="dark-siren mock host pipeline")
    p.add_argument("--out", type=str, default=None, help="output directory")
    p.add_argument("--n-grid", type=int, default=None, help="FFT cells per side")
    p.add_argument("--box-size", type=float, default=None, help="box size [Mpc/h]")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nside", type=int, default=None, help="HEALPix nside")
    p.add_argument("--n-shells", type=int, default=None, help="number of z shells")
    p.add_argument("--events", type=int, default=None, help="number of hosts")
    p.add_argument("--b-gw", type=float, default=None, help="bias normalisation")
    p.add_argument("--alpha-gw", type=float, default=None, help="bias z-slope")
    p.add_argument("--linear", action="store_true", help="linear P(k) (no halofit)")
    p.add_argument("--incoherent", action="store_true",
                   help="independent phases per shell (default: coherent)")
    p.add_argument("--no-plots", action="store_true",
                   help="skip report figures (faster; e.g. headless without healpy)")
    p.add_argument("--variants", action="store_true",
                   help="also write the EM-tracer / blur / noise variant suite")
    args = p.parse_args(argv)

    cfg = build_config(args)
    pipe = MockHostPipeline(cfg, make_plots=not args.no_plots)
    pipe.run()
    if args.variants:
        pipe.run_variants(make_figs=not args.no_plots)


if __name__ == "__main__":
    main()
