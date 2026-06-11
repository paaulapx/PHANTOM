"""
cosmology.py
cosmology + camb p(k,z) interface. the cosmology is used by both the field generator and the projection to build the lightcone shells, so it lives here in the middle.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import UnivariateSpline

_C_KM_S = 299_792.458   # speed of light, km/s


@contextmanager
def _silence_stdio():
    """mute camb chatter. no numerical effect."""
    try:
        old_out = os.dup(sys.stdout.fileno())
        old_err = os.dup(sys.stderr.fileno())
        devnull = os.open(os.devnull, os.O_WRONLY)
    except (OSError, ValueError):
        yield
        return
    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        os.close(devnull)
        os.dup2(old_out, sys.stdout.fileno()); os.close(old_out)
        os.dup2(old_err, sys.stderr.fileno()); os.close(old_err)


class Cosmology:
    """flat lcdm background + camb p(k,z)."""

    def __init__(self, config):
        self.config = config
        self.cosmo = config.cosmo
        self.box = config.box
        self.redshift = config.redshift
        self.rate = config.rate
        self.bias_params = config.bias
        self.sky_area = config.sky_area

        # caches: chi(z) table + camb products
        self._dist_z = None
        self._dist_chi = None
        self._camb_k = None      # h/mpc
        self._camb_z = None
        self._camb_pk = None     # (n_z, n_k)
        self._pk_splines = None  # {z_shell: spline(log k -> log p)}

        self._build_distance_table()

    # --------------------------------------------------------------------- #
    # background
    # --------------------------------------------------------------------- #
    @property
    def Omega_m(self) -> float:
        c = self.cosmo
        if getattr(c, "Omega_m", None) is not None:
            return float(c.Omega_m)
        return (c.ombh2 + c.omch2) / c.h ** 2

    def E(self, z):
        """h(z)/h0, flat lcdm."""
        z = np.asarray(z, dtype=float)
        om = self.Omega_m
        return np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))

    def _build_distance_table(self):
        """chi(z) table [mpc/h], fine grid, fast lookup."""
        zmax = self.redshift.z_max * 1.02
        z = np.concatenate([[0.0], np.geomspace(1e-5, zmax, 4096)])
        integrand = _C_KM_S / (self.cosmo.H0 * self.E(z))     # mpc per dz
        chi_mpc = cumulative_trapezoid(integrand, z, initial=0.0)
        self._dist_z = z
        self._dist_chi = chi_mpc * self.cosmo.h               # -> mpc/h

    def comoving_distance(self, z):
        """chi(z) [mpc/h], table interp."""
        return np.interp(np.asarray(z, dtype=float), self._dist_z, self._dist_chi)

    def dchi_dz(self, z):
        """dchi/dz [mpc/h], analytic."""
        z = np.asarray(z, dtype=float)
        return (_C_KM_S / (self.cosmo.H0 * self.E(z))) * self.cosmo.h

    def dVc_dz(self, z):
        """sky_area * chi^2 * dchi/dz [(mpc/h)^3].

        no 1/(1+z) here -- time dilation lives in redshift_kernel, this
        stays clean dvc/dz.
        """
        z = np.asarray(z, dtype=float)
        chi = self.comoving_distance(z)
        return self.sky_area * chi ** 2 * self.dchi_dz(z)

    # --------------------------------------------------------------------- #
    # rate evolution and bias
    # --------------------------------------------------------------------- #
    def madau_dickinson(self, z):
        """psi(z), norm 1 at z=0. exact same form as icarogw md_rate --
        mock and inference must share one psi."""
        z = np.asarray(z, dtype=float)
        g, k, zp = self.rate.gamma, self.rate.kappa, self.rate.zp
        norm = 1.0 + (1.0 + zp) ** (-g - k)
        return norm * (1.0 + z) ** g / (1.0 + ((1.0 + z) / (1.0 + zp)) ** (g + k))

    def bias(self, z):
        """b(z) = b_gw * (1+z)^alpha_gw."""
        z = np.asarray(z, dtype=float)
        return self.bias_params.b_gw * (1.0 + z) ** self.bias_params.alpha_gw

    def redshift_kernel(self, z):
        """p(z) ~ psi(z) dvc/dz / (1+z), un-normalised. sampler builds cdf."""
        z = np.asarray(z, dtype=float)
        return self.madau_dickinson(z) * self.dVc_dz(z) / (1.0 + z)

    # --------------------------------------------------------------------- #
    # non-linear power spectrum (camb)
    # --------------------------------------------------------------------- #
    def compute_power_spectra(self):
        """one camb run on the exact shell redshifts -> per-shell p(k, z_shell)."""
        import camb  # lazy

        c = self.cosmo
        shells = self.config.shells
        kmax = self.box.kmax

        with _silence_stdio():
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=c.H0, ombh2=c.ombh2, omch2=c.omch2,
                               mnu=c.mnu, omk=c.omk, tau=c.tau)
            pars.InitPower.set_params(As=c.As, ns=c.ns, r=c.r)
            # camb wants ascending z, returns same order
            pars.set_matter_power(redshifts=shells.tolist(), kmax=kmax,
                                  nonlinear=self.config.nonlinear)
            results = camb.get_results(pars)
            kh, z_camb, pk = results.get_matter_power_spectrum(
                minkh=self.box.camb_minkh, maxkh=kmax,
                npoints=self.box.camb_npoints)

        self._camb_k = np.asarray(kh)
        self._camb_z = np.asarray(z_camb)
        self._camb_pk = np.atleast_2d(pk)

        logk = np.log(self._camb_k)
        self._pk_splines = {}
        for z in shells:
            row = self._camb_pk[int(np.argmin(np.abs(self._camb_z - z)))]
            logp = np.log(row)
            ok = np.isfinite(logk) & np.isfinite(logp)
            self._pk_splines[float(z)] = UnivariateSpline(
                logk[ok], logp[ok], s=self.box.pk_spline_s, k=self.box.pk_spline_k)
        return self

    @property
    def k_min(self) -> float:
        return float(self._camb_k[0])

    @property
    def k_max(self) -> float:
        return float(self._camb_k[-1])

    def power_at(self, z):
        """p(k) callable for the shell nearest z. zero outside camb k-range -> field generator never extrapolates."""
        if self._pk_splines is None:
            raise RuntimeError("call compute_power_spectra() first.")
        z = float(z)
        zk = min(self._pk_splines, key=lambda zz: abs(zz - z))
        spline = self._pk_splines[zk]
        kmin, kmax = self.k_min, self.k_max

        def _pk(kmag):
            kmag = np.asarray(kmag, dtype=float)
            out = np.zeros_like(kmag)
            m = (kmag >= kmin) & (kmag <= kmax)
            out[m] = np.exp(spline(np.log(kmag[m])))
            return out

        return _pk

    def power_spectrum_grid(self):
        """raw camb (kh, z, p) for diagnostics."""
        if self._camb_pk is None:
            raise RuntimeError("call compute_power_spectra() first.")
        return self._camb_k, self._camb_z, self._camb_pk
