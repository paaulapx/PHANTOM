"""
dark-siren mock host pipeline.

two products from one field:
  icarogw <- bare tracer 1 + b(z)*delta (hdf5)
  moca    <- joint host pdf p(z, omega) (csv)
"""

from __future__ import annotations

__all__ = [
    "Config",
    "Cosmology",
    "GaussianFieldGenerator",
    "SkyProjector",
    "Simulation",
    "MockHostPipeline",
    "ReportPlots",
    "Variant",
    "tracer_library",
    "default_variants",
]

_LAZY = {
    "Config": ("config", "Config"),
    "Cosmology": ("cosmology", "Cosmology"),
    "GaussianFieldGenerator": ("field", "GaussianFieldGenerator"),
    "SkyProjector": ("projection", "SkyProjector"),
    "Simulation": ("simulation", "Simulation"),
    "MockHostPipeline": ("pipeline", "MockHostPipeline"),
    "ReportPlots": ("plots", "ReportPlots"),
    "Variant": ("perturbations", "Variant"),
    "tracer_library": ("perturbations", "tracer_library"),
    "default_variants": ("perturbations", "default_variants"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
