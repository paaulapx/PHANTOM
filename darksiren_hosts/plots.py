"""
plots.py
figures for the host population and its relation to the matter distribution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless / batch-node safe
import matplotlib.pyplot as plt
import numpy as np

_OVERDENSITY_CMAP = "inferno"       # black/purple (low) -> yellow (high)
_OVERDENSITY_SMOOTH_DEG = 4.0       # display-only angular smoothing of overdensity maps
_MOLLVIEW_XSIZE = 2000              # raster width: large -> smooth, no blocky pixels


def _display_smooth(m, fwhm_deg, *, nest=False):
    """angular gaussian smoothing of a healpix map for display only.

    cleans grid-scale sampling artefacts so the mollweide looks smooth; the
    stored science maps are untouched.  returns input unchanged if smoothing
    is disabled or healpy lacks 'smoothing'.
    """
    if not fwhm_deg or fwhm_deg <= 0:
        return np.asarray(m, dtype=float)
    import healpy as hp
    m = np.asarray(m, dtype=float)
    if not hasattr(hp, "smoothing"):
        return m
    if nest:                                   # smoothing works in ring
        m = hp.reorder(m, n2r=True)
        sm = hp.smoothing(m, fwhm=np.deg2rad(fwhm_deg))
        return hp.reorder(sm, r2n=True)
    return hp.smoothing(m, fwhm=np.deg2rad(fwhm_deg))

_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
}


class ReportPlots:
    def __init__(self, sim, catalog, out_dir, *, verbose=True):
        self.sim = sim
        self.cat = catalog
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.nside = sim.config.healpix.nside
        self.shells = np.asarray(sim.shells, dtype=float)

    # ------------------------------------------------------------------ #
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _pick_shells(self, n=3):
        """representative low/mid/high shells (indices)."""
        idx = np.unique(np.linspace(0, len(self.shells) - 1, n).astype(int))
        return idx

    # ------------------------------------------------------------------ #
    @staticmethod
    def mollweide(map_1d, path, *, title="", unit="", cmap=_OVERDENSITY_CMAP,
                  symmetric=False, nest=False, smooth_fwhm_deg=_OVERDENSITY_SMOOTH_DEG):
        """single healpix map -> report-ready mollweide png (inferno).

        angular smoothing + a large xsize give a smooth map that fills the
        whole ellipse.  symmetric=true centres the colour scale on zero (for
        a contrast map); otherwise it spans [0, p99] (for a pdf).  static so
        the variant machinery can call it without a full ReportPlots.
        """
        import healpy as hp
        m = _display_smooth(map_1d, smooth_fwhm_deg, nest=nest)
        if symmetric:
            vmax = float(np.percentile(np.abs(m), 99)) or 1e-6
            vmin = -vmax
        else:
            vmin, vmax = 0.0, float(np.percentile(m, 99)) or None
        with plt.rc_context(_RC):
            fig = plt.figure(figsize=(7.0, 4.6))
            hp.mollview(m, fig=fig.number, title=title, cmap=cmap,
                        min=vmin, max=vmax, cbar=True, unit=unit, nest=nest,
                        xsize=_MOLLVIEW_XSIZE)
            hp.graticule(dpar=30, dmer=30, color="0.6", alpha=0.4)
            fig.savefig(path); plt.close(fig)
        return path

    @staticmethod
    def comparison_sheet(maps, titles, path, *, unit="", cmap=_OVERDENSITY_CMAP,
                         symmetric=True, nest=False,
                         smooth_fwhm_deg=_OVERDENSITY_SMOOTH_DEG, ncols=3,
                         suptitle=""):
        """several healpix maps in a grid under one shared colorbar.

        every panel rendered on the same colour range (global p99 across all
        maps), so a row of variants/shells is comparable at a glance.  single
        colorbar built from a ScalarMappable, not per-panel.
        """
        import healpy as hp
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm_maps = [_display_smooth(m, smooth_fwhm_deg, nest=nest) for m in maps]
        if symmetric:
            vmax = max(float(np.percentile(np.abs(m), 99)) for m in sm_maps) or 1e-6
            vmin = -vmax
        else:
            vmax = max(float(np.percentile(m, 99)) for m in sm_maps) or 1e-6
            vmin = 0.0

        n = len(sm_maps)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        with plt.rc_context(_RC):
            fig = plt.figure(figsize=(4.3 * ncols, 2.7 * nrows + 0.8))
            for k, (m, t) in enumerate(zip(sm_maps, titles)):
                hp.mollview(m, fig=fig.number, sub=(nrows, ncols, k + 1), title=t,
                            cmap=cmap, min=vmin, max=vmax, cbar=False,
                            unit=unit, nest=nest, xsize=_MOLLVIEW_XSIZE)
                hp.graticule(dpar=30, dmer=30, color="0.6", alpha=0.4)
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=fig.get_axes(), orientation="vertical",
                              fraction=0.025, pad=0.02)
            cb.set_label(unit)
            if suptitle:
                fig.suptitle(suptitle, y=1.0, fontsize=13)
            fig.savefig(path); plt.close(fig)
        return path

    def make_all(self):
        """produce every figure; a failure in one never aborts the rest."""
        figs = {
            "power_spectrum": self.plot_power_spectrum,
            "rate_and_kernel": self.plot_rate_and_kernel,
            "redshift_distribution": self.plot_redshift_distribution,
            "overdensity_mollweide": self.plot_overdensity_mollweide,
            "host_pdf_mollweide": self.plot_host_pdf_mollweide,
            "event_sky": self.plot_event_sky,
            "density_slice": self.plot_density_slice,
            "delta_histogram": self.plot_delta_histogram,
            "angular_power_spectrum": self.plot_angular_power_spectrum,
            "sigma_delta_z": self.plot_sigma_delta_z,
            "event_delta_clustering": self.plot_event_delta_clustering,
        }
        written = {}
        with plt.rc_context(_RC):
            for name, fn in figs.items():
                try:
                    written[name] = fn()
                    self._log(f"  figure: {written[name]}")
                except Exception as e:               # noqa: BLE001
                    self._log(f"  [skip] {name}: {type(e).__name__}: {e}")
        return written

    # ------------------------------------------------------------------ #
    # 1. non-linear matter power spectrum P(k, z)
    # ------------------------------------------------------------------ #
    def plot_power_spectrum(self):
        if self.sim.cosmo._camb_pk is None:
            self.sim.cosmo.compute_power_spectra()
        kh, zc, pk = self.sim.cosmo.power_spectrum_grid()
        kmax = self.sim.config.box.kmax

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for i in self._pick_shells(4):
            z = self.shells[i]
            j = int(np.argmin(np.abs(zc - z)))
            ax.loglog(kh, pk[j], label=fr"$z={z:.2f}$")
        ax.axvline(kmax, ls="--", color="0.4",
                   label=fr"$k_{{\max}}={kmax:.3f}\,h/$Mpc")
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel(r"$P(k,z)\ [(\mathrm{Mpc}/h)^3]$")
        ax.set_title("Non-linear matter power spectrum (CAMB + halofit)")
        ax.legend(frameon=False)
        path = self.out / "power_spectrum.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 2. madau-dickinson psi(z) and the redshift kernel
    # ------------------------------------------------------------------ #
    def plot_rate_and_kernel(self):
        cosmo = self.sim.cosmo
        z = np.linspace(self.sim.config.redshift.z_min,
                        self.sim.config.redshift.z_max, 400)
        psi = cosmo.madau_dickinson(z)
        ker = cosmo.redshift_kernel(z)
        ker = ker / np.trapezoid(ker, z)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
        a1.plot(z, psi, color="crimson")
        a1.set_xlabel(r"$z$"); a1.set_ylabel(r"$\psi(z)$")
        a1.set_title(r"Madau--Dickinson rate $\psi(z)$")
        a2.plot(z, ker, color="navy")
        a2.set_xlabel(r"$z$")
        a2.set_ylabel(r"$p(z)\propto\psi(z)\,\frac{dV_c}{dz}\,\frac{1}{1+z}$")
        a2.set_title("Source redshift PDF")
        fig.tight_layout()
        path = self.out / "rate_and_kernel.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 3. sampled redshift distribution vs theory
    # ------------------------------------------------------------------ #
    def plot_redshift_distribution(self):
        cosmo = self.sim.cosmo
        z = np.linspace(self.sim.config.redshift.z_min,
                        self.sim.config.redshift.z_max, 400)
        ker = cosmo.redshift_kernel(z)
        ker = ker / np.trapezoid(ker, z)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.hist(self.cat["z"], bins=50, density=True, alpha=0.55,
                color="steelblue", label="sampled hosts")
        ax.plot(z, ker, color="k", label=r"theory $p(z)$")
        ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$p(z)$")
        ax.set_title(f"Host redshift distribution ($N={len(self.cat)}$)")
        ax.legend(frameon=False)
        path = self.out / "redshift_distribution.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 4. overdensity mollweide maps  (inferno: black/purple -> yellow)
    # ------------------------------------------------------------------ #
    def plot_overdensity_mollweide(self, smooth_fwhm_deg=_OVERDENSITY_SMOOTH_DEG):
        """smoothed, filled mollweide delta maps at low/mid/high z, one shared colorbar.

        all shells share the same symmetric colour range, so growth of
        structure (higher-z shells fainter) reads off at a glance and tiny
        high-z fluctuations stay smooth rather than saturating into grain.
        """
        nest = self.sim.config.healpix.nest
        idx = self._pick_shells(3)
        maps = [self.sim.delta_matrix[i] for i in idx]
        titles = [fr"$\delta(\hat n)$ at $z={self.shells[i]:.2f}$" for i in idx]
        return self.comparison_sheet(
            maps, titles, self.out / "overdensity_mollweide.pdf",
            unit=r"$\delta$", symmetric=True, nest=nest,
            smooth_fwhm_deg=smooth_fwhm_deg,
            suptitle=fr"Matter overdensity $\delta(\hat n)$ "
                     fr"(smoothed ${smooth_fwhm_deg:g}^\circ$, shared scale)")

    # ------------------------------------------------------------------ #
    # 5. moca host pdf mollweide at a representative shell
    # ------------------------------------------------------------------ #
    def plot_host_pdf_mollweide(self):
        import healpy as hp
        nest = self.sim.config.healpix.nest
        i = self._pick_shells(3)[1]            # mid shell
        z = self.shells[i]
        m = _display_smooth(self.sim.host_matrix[i], _OVERDENSITY_SMOOTH_DEG, nest=nest)
        fig = plt.figure(figsize=(7.0, 4.6))
        hp.mollview(m, fig=fig.number, title=fr"MOCA host PDF at $z={z:.2f}$",
                    cmap=_OVERDENSITY_CMAP, cbar=True,
                    unit=r"$p(z,\hat n)$ [per pixel]",
                    nest=nest, xsize=_MOLLVIEW_XSIZE)
        hp.graticule(dpar=30, dmer=30, color="0.6", alpha=0.4)
        path = self.out / "host_pdf_mollweide.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 6. sampled event sky (native matplotlib mollweide)
    # ------------------------------------------------------------------ #
    def plot_event_sky(self):
        ra = np.deg2rad(self.cat["ra"].to_numpy())
        dec = np.deg2rad(self.cat["dec"].to_numpy())
        ra = np.mod(ra + np.pi, 2 * np.pi) - np.pi     # wrap to [-pi, pi]

        fig = plt.figure(figsize=(7.4, 4.4))
        ax = fig.add_subplot(111, projection="mollweide")
        sc = ax.scatter(ra, dec, c=self.cat["z"], s=6, cmap="viridis",
                        alpha=0.7, edgecolors="none")
        ax.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
        cb.set_label(r"$z$")
        ax.set_title("Cosmic-truth host positions", pad=14)
        path = self.out / "event_sky.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 7. density-field slice through the box (regenerated cube)
    # ------------------------------------------------------------------ #
    def plot_density_slice(self):
        if self.sim.cosmo._pk_splines is None:
            self.sim.cosmo.compute_power_spectra()
        i = self._pick_shells(3)[1]
        z = self.shells[i]
        cube = self.sim.field.realize(self.sim.cosmo.power_at(z), z)
        sl = cube[cube.shape[0] // 2]
        L = self.sim.config.box.size
        obs = self.sim.projector.observer()
        chi = float(self.sim.cosmo.comoving_distance(z))

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        vmax = np.percentile(np.abs(sl), 99) or 1e-3
        im = ax.imshow(sl.T, origin="lower", extent=[0, L, 0, L],
                       cmap=_OVERDENSITY_CMAP, vmin=-vmax, vmax=vmax)
        ax.plot(obs[0], obs[1], "x", color="cyan", ms=9, mew=2, label="observer")
        circ = plt.Circle((obs[0], obs[1]), chi, fill=False, color="cyan",
                          lw=1.2, ls="--", label=fr"shell $\chi(z={z:.2f})$")
        ax.add_patch(circ)
        ax.set_xlabel(r"$x\ [\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$y\ [\mathrm{Mpc}/h]$")
        ax.set_title(fr"$\delta$ slice through the box ($z={z:.2f}$)")
        ax.legend(loc="upper right", framealpha=0.8)
        cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label(r"$\delta$")
        path = self.out / "density_slice.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 8. delta histogram (Gaussianity) at several shells
    # ------------------------------------------------------------------ #
    def plot_delta_histogram(self):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for i in self._pick_shells(3):
            z = self.shells[i]
            d = self.sim.delta_matrix[i]
            ax.hist(d, bins=60, density=True, histtype="step",
                    label=fr"$z={z:.2f}\ (\sigma={d.std():.3f})$")
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel("PDF")
        ax.set_title(r"Projected overdensity distribution per shell")
        ax.legend(frameon=False)
        path = self.out / "delta_histogram.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 9. angular power spectrum of the overdensity maps
    # ------------------------------------------------------------------ #
    def plot_angular_power_spectrum(self):
        import healpy as hp
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for i in self._pick_shells(3):
            z = self.shells[i]
            cl = hp.anafast(self.sim.delta_matrix[i])
            ell = np.arange(cl.size)
            ax.loglog(ell[1:], ell[1:] * (ell[1:] + 1) * cl[1:] / (2 * np.pi),
                      label=fr"$z={z:.2f}$")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
        ax.set_title("Angular power spectrum of $\\delta(\\hat n)$")
        ax.legend(frameon=False)
        path = self.out / "angular_power_spectrum.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 10. sigma_delta(z): growth check
    # ------------------------------------------------------------------ #
    def plot_sigma_delta_z(self):
        sig = self.sim.delta_matrix.std(axis=1)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(self.shells, sig, "o-", ms=3, color="darkgreen")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\sigma_\delta(z)$ (per-shell angular RMS)")
        ax.set_title("Growth of structure across shells")
        path = self.out / "sigma_delta_z.pdf"
        fig.savefig(path); plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    # 11. clustering of hosts: host-delta vs field-delta
    # ------------------------------------------------------------------ #
    def plot_event_delta_clustering(self):
        host_delta = self.cat["delta"].to_numpy()
        field_delta = self.sim.delta_matrix.reshape(-1)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.hist(field_delta, bins=80, density=True, histtype="step",
                color="0.4", label=r"all sky cells $\langle\delta\rangle\approx0$")
        ax.hist(host_delta, bins=60, density=True, alpha=0.5,
                color="orange",
                label=fr"host cells $\langle\delta\rangle={host_delta.mean():+.3f}$")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(r"$\delta$ at location"); ax.set_ylabel("PDF")
        ax.set_title("Hosts trace overdense regions (clustering bias)")
        ax.legend(frameon=False)
        path = self.out / "event_delta_clustering.pdf"
        fig.savefig(path); plt.close(fig)
        return path


    def plot_tracer_cross_correlation(self, variant_name, baseline_name="baseline_matter"):
        """compute and plot r_bar(z) and r_ell(ell) between baseline and a biased variant."""
        from .perturbations import angular_cross_spectrum
        # baseline matrix from self.sim (already loaded)
        T_base = self.sim.tracer_matrix
        # variant matrix from variants/ directory
        var_dir = self.out.parent / variant_name
        import h5py
        with h5py.File(var_dir / self.sim.config.products.icarogw_density_map, "r") as f:
            T_var = f["density_matrix"][:]
        assert T_base.shape == T_var.shape
        n_z = T_base.shape[0]
        r_bar = np.empty(n_z)
        r_l_list = []
        for i in range(n_z):
            res = angular_cross_spectrum(T_base[i], T_var[i], use_contrast=True)
            r_bar[i] = np.average(res["r_l"][2:], weights=2*res["l"][2:]+1)  # skip monopole
            r_l_list.append(res["r_l"])
        # r_bar vs z
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(self.sim.shells, r_bar, 'o-', color='#1f78b4')
        ax.axhline(1, ls='--', color='0.5')
        ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\bar r$ (cross-correlation)')
        ax.set_title(f'Tracer cross-correlation: baseline vs {variant_name}')
        fig.savefig(self.out / f'xcorr_rbar_{variant_name}.pdf'); plt.close(fig)
        # r_ell for a few shells
        idx = self._pick_shells(4)
        fig, axes = plt.subplots(1, len(idx), figsize=(5*len(idx), 4), sharey=True)
        if len(idx)==1: axes = [axes]
        for ax, i in zip(axes, idx):
            ell = np.arange(len(r_l_list[i]))
            ax.plot(ell[2:], r_l_list[i][2:], color='#e31a1c')
            ax.axhline(1, ls='--', color='0.5')
            ax.set_xlabel(r'$\ell$'); ax.set_title(f'z={self.sim.shells[i]:.2f}')
            ax.set_xscale('log')
        axes[0].set_ylabel(r'$r_\ell$')
        fig.suptitle(f'r_ell: baseline vs {variant_name}')
        fig.savefig(self.out / f'xcorr_rl_{variant_name}.pdf'); plt.close(fig)
