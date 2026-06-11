"""
projection.py
Project 3D density cubes onto the sky, using Healpix pixelization.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


class SkyProjector:
    def __init__(self, config, cosmology):
        import healpy as hp  # lazy import
        self._hp = hp

        self.config = config
        self.cosmo = cosmology
        self.box = config.box
        self.nside = config.healpix.nside
        self.npix = config.healpix.npix

        # pixel unit vectors, ring order
        self.pix = np.arange(self.npix)
        vx, vy, vz = hp.pix2vec(self.nside, self.pix)
        self._dirs = np.array([vx, vy, vz])           # (3, npix)

        # observer nudged off-centre (seeded): never on a grid plane, else low-z shells align with voxel boundaries
        rng = np.random.RandomState(self.box.seed)
        offset = rng.uniform(-0.1, 0.1, size=3) * self.box.size
        self._obs = np.array([self.box.size / 2.0] * 3) + offset

    # ------------------------------------------------------------------ #
    def observer(self):
        return self._obs

    def pixel_radec_deg(self):
        """ra, dec [deg] per pixel, ring order."""
        theta, phi = self._hp.pix2ang(self.nside, self.pix)
        ra = np.degrees(phi)
        dec = 90.0 - np.degrees(theta)
        return ra, dec

    def project_shell(self, delta_cube, z):
        """delta(n_hat) on the chi(z) sphere. order=1 + grid-wrap = periodic box."""
        voxel = self.box.voxel
        r = float(self.cosmo.comoving_distance(z))    # mpc/h
        coords = (self._obs[:, None] + r * self._dirs) / voxel
        return map_coordinates(delta_cube, coords, order=1, mode="grid-wrap")
