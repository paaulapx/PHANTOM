"""
plot_run.py
render report figures from saved products. this is a separate script to avoid importing heavy plotting dependencies in
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .simulation import Simulation
from .plots import ReportPlots


def _config_from_run(out_dir: Path) -> Config:
    """defaults + the run's saved knobs, best-effort."""
    cfg = Config.default()
    cfg.out_dir = out_dir
    meta = out_dir / cfg.products.config
    if meta.exists():
        d = json.loads(meta.read_text())
        cfg.healpix.nside = int(d["healpix"]["nside"])
        cfg.redshift.n_shells = int(d["redshift"]["n_shells"])
        cfg.redshift.z_min = float(d["redshift"]["z_min"])
        cfg.redshift.z_max = float(d["redshift"]["z_max"])
        cfg.box.size = float(d["box"]["size"])
        cfg.box.n_grid = int(d["box"]["n_grid"])
        cfg.box.seed = int(d["box"]["seed"])
        cfg.bias.b_gw = float(d["bias"]["b_gw"])
        cfg.bias.alpha_gw = float(d["bias"]["alpha_gw"])
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="render report figures from saved products")
    p.add_argument("--out", required=True, help="run directory holding the products")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    cfg = _config_from_run(out_dir)
    sim, catalog = Simulation.load_products(cfg, out_dir)
    ReportPlots(sim, catalog, out_dir / "figures").make_all()


if __name__ == "__main__":
    main()
