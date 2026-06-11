"""
field.py
gaussian field generator. the field is the most expensive part of the pipeline, so it lives in its own module. it needs the cosmology for the p(k) input, but it doesn't need camb itself.
"""

from __future__ import annotations

import warnings

import numpy as np


class GaussianFieldGenerator:
    def __init__(self, n_grid, box_size, *, smooth_mpch=0.0, kmax=None,
                 seed=0, coherent=True, dtype=np.float32):
        self.n = int(n_grid)
        self.L = float(box_size)
        self.Vcell = (self.L / self.n) ** 3
        self.coherent = bool(coherent)
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

        # rfft k-grid: kx, ky full; kz half
        kf = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.L / self.n)
        kfz = 2.0 * np.pi * np.fft.rfftfreq(self.n, d=self.L / self.n)
        kx, ky, kz = np.meshgrid(kf, kf, kfz, indexing="ij")
        self.kmag = np.sqrt(kx * kx + ky * ky + kz * kz)

        self.kmax = float(kmax) if kmax is not None else float(self.kmag.max())
        self.kmask = (self.kmag > 0.0) & (self.kmag <= self.kmax)

        # w(k) = exp(-k^2 sigma^2 / 2). kmax cutoff is already a bandlimit.
        # sub-voxel sigma -> w(kmax) ~ 1 -> no-op. warn, never silently inert.
        self.smooth_mpch = float(smooth_mpch or 0.0)
        if self.smooth_mpch > 0:
            self._window = np.exp(-0.5 * (self.kmag * self.smooth_mpch) ** 2)
            w_at_kmax = float(np.exp(-0.5 * (self.kmax * self.smooth_mpch) ** 2))
            if w_at_kmax > 0.99:
                warnings.warn(
                    f"smoothing scale {self.smooth_mpch:g} Mpc/h is ineffective "
                    f"at this resolution: W(kmax)={w_at_kmax:.3f} ~ 1. The kmax "
                    f"cutoff (lambda_min={2*np.pi/self.kmax:.0f} Mpc/h) is the "
                    f"effective bandlimit; set smooth_mpch=0 or raise it above "
                    f"~1 voxel ({self.L/self.n:.0f} Mpc/h) to actually smooth.",
                    stacklevel=2,
                )
        else:
            self._window = np.ones_like(self.kmag)

        # shared phases for the coherent lightcone
        self._W = self._draw_white_W() if self.coherent else None

    # ------------------------------------------------------------------ #
    def _draw_white_W(self):
        """rfftn of unit-variance white noise."""
        w = self._rng.standard_normal(size=(self.n, self.n, self.n))
        return np.fft.rfftn(w)

    def realize(self, power_at, z=None):
        """delta(x) cube: colour white noise with sqrt(P/V_cell), zero mean.

        power_at: k [h/mpc] -> P [(mpc/h)^3], vectorised.
        """
        W = self._W if self.coherent else self._draw_white_W()

        P = np.asarray(power_at(self.kmag), dtype=float)
        P = np.where(self.kmask, P, 0.0)
        P[0, 0, 0] = 0.0                         # dc mode off -> <delta>=0

        T = np.sqrt(P / self.Vcell) * self._window
        delta = np.fft.irfftn(W * T, s=(self.n, self.n, self.n), axes=(0, 1, 2))
        # float64 math, float32 storage; projected matrix is float64 later
        return delta.astype(self.dtype, copy=False)
