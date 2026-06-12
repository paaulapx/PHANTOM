"""
experiments_run.py
apply bias/blur/noise experiments on top of a baseline run. this is a separate script from the main pipeline because it needs to load the baseline products instead of regenerating them, and it doesn't need to run camb or make the field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import MockHostPipeline
from .plot_run import _config_from_run
from .simulation import Simulation


def main(argv=None):
    p = argparse.ArgumentParser(
        description="apply bias/blur/noise experiments on top of a baseline run")
    p.add_argument("--out", required=True, help="baseline run directory")
    p.add_argument("--no-figs", action="store_true",
                   help="skip per-variant maps and comparison sheets")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    cfg = _config_from_run(out_dir)
    cfg.out_dir = out_dir

    pipe = MockHostPipeline(cfg, verbose=True, make_plots=False)
    # loaded baseline field, not regenerated
    pipe.sim, pipe.catalog = Simulation.load_products(cfg, out_dir)
    pipe.run_variants(make_figs=not args.no_figs)


if __name__ == "__main__":
    main()
