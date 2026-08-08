"""Forward-simulate JWST NIRISS GR150C direct and dispersed (WFSS) images.

Sources are 2D Gaussians randomly placed on the 2048x2048 detector, each with
a spectral shape drawn at random from the pystellibs Kurucz library. Traces
come from the CRDS specwcs reference files via grismagic; fluxes are weighted
by the aXe-style first-order sensitivity curves.

Intended for use from notebooks::

    from simulate import simulate
    results, catalog = simulate(n_sources=200, seed=42)
"""

import datetime
import os
import subprocess

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import interp1d
from tqdm import tqdm

from grismagic.disperse import disperse_obj
from grismagic.traces import GrismTrace

CONF_DIR = "/Users/momcheva/dev/grizli_conf/CONF"
FOV = 2048

# Highest-numbered GR150C specwcs reference per filter in CONF_DIR
SPECWCS_FILES = {
    "F115W": "jwst_niriss_specwcs_0081.GR150C.F115W.asdf",
    "F150W": "jwst_niriss_specwcs_0084.GR150C.F150W.asdf",
    "F200W": "jwst_niriss_specwcs_0090.GR150C.F200W.asdf",
}

# CRDS order name -> tag in the aXe ETC sensitivity file names.
# The 0th order has no local sensitivity curve.
ORDER_SENS_TAG = {"+1": "p1", "+2": "p2", "+3": "p3", "-1": "m1"}


def resample_to_resolution(
    λ_AA,
    f_flam,
    target_R,
    *,
    min_sampling=3.0,
    quadrature_points=32,
    λ_min_out=None,
    λ_max_out=None,
):
    """Convolve spectra to resolution R = λ/Δλ and resample onto a constant-R grid.

    λ_AA : (n,) wavelength in Angstrom, increasing. f_flam : (nspec, n) or (n,)
    flam spectra. Returns (λ_out, f_out) with the same leading shape as f_flam.
    Gaussian kernel σ = λ/(R·2.355), integrated with Gauss-Hermite quadrature.
    """
    λ = np.asarray(λ_AA, dtype=float)
    f = np.atleast_2d(f_flam).astype(float)

    const = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM -> sigma
    sqrt2 = np.sqrt(2.0)

    λ_min_use = λ[0] if λ_min_out is None else float(λ_min_out)
    λ_max_use = λ[-1] if λ_max_out is None else float(λ_max_out)

    step = 1.0 / (target_R * min_sampling * const)
    N_out = int(np.log(λ_max_use / λ_min_use) / step) + 1
    λ_out = λ_min_use * np.exp(np.arange(N_out) * step)
    if λ_out[-1] < λ_max_use:
        λ_out = np.append(λ_out, λ_max_use)
        N_out += 1

    f_out = np.zeros((f.shape[0], N_out), dtype=f.dtype)
    f_interp = interp1d(λ, f, kind="linear", fill_value=0, bounds_error=False)
    x, w = hermgauss(quadrature_points)
    factor = sqrt2 / const

    for i, λ_i in enumerate(tqdm(λ_out)):
        σ_i = λ_i / (target_R * const)
        fluxj = f_interp(λ_i + sqrt2 * σ_i * x)
        f_out[:, i] = factor * np.sum(fluxj * w, axis=1)

    if np.ndim(f_flam) == 1:
        f_out = f_out[0, :]
    return λ_out, f_out


def generate_sources(n_sources, seed, n_templates, border=100, amp_range=(0.01, 1.0), fov=FOV):
    """Random positions, amplitudes and Kurucz template indices."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(border, fov - border, n_sources)
    y = rng.uniform(border, fov - border, n_sources)
    amps = rng.uniform(amp_range[0], amp_range[1], n_sources)
    kurucz_idx = rng.integers(0, n_templates, n_sources)
    return x, y, amps, kurucz_idx


def load_kurucz_spectra(target_R=140, lam_range_um=(0.7, 2.3)):
    """Kurucz library resampled to constant R, normalized to unit mean flam.

    Returns (lam_AA, spectra[n_templates, n_lam], params Table).
    """
    from pystellibs import Kurucz

    kurucz = Kurucz()
    lam = np.asarray(kurucz.wavelength.to(u.angstrom).value)
    flux = np.asarray(kurucz.spectra)

    lam_min, lam_max = (np.array(lam_range_um) * u.um).to(u.angstrom).value
    lam_out, f_out = resample_to_resolution(
        lam, flux, target_R, λ_min_out=lam_min, λ_max_out=lam_max
    )
    # Normalize each template to unit mean flam over the resampled range
    f_out = f_out / f_out.mean(axis=1, keepdims=True)

    g = kurucz.grid.data
    params = Table(
        {name: np.asarray(g[name]) for name in ("Teff", "logT", "logg", "Z", "logz")}
    )
    return lam_out, f_out, params


def load_sensitivity(filt, order, sens_dir=CONF_DIR):
    """Interpolator for the aXe ETC sensitivity curve (Angstrom in, 0 outside)."""
    tag = ORDER_SENS_TAG.get(order)
    if tag is None:
        raise ValueError(
            f"No sensitivity curve available for order {order!r}; "
            f"supported orders: {list(ORDER_SENS_TAG)}"
        )
    path = os.path.join(sens_dir, f"NIRISS.GR150C.{filt}.{tag}.etc.sens.fits")
    with fits.open(path) as hdul:
        wave = np.asarray(hdul[1].data["WAVELENGTH"], dtype=float)
        sens = np.asarray(hdul[1].data["SENSITIVITY"], dtype=float)

    def interp(lam_AA):
        return np.interp(lam_AA, wave, sens, left=0.0, right=0.0)

    return interp, path


def unit_gaussian_stamp(fwhm=1.6, size=21):
    """Odd-sized Gaussian stamp centered at ((size-1)/2, (size-1)/2), sum=1."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2))
    return g / g.sum()


def build_direct_image(x, y, fluxes, fwhm=1.6, shape=(FOV, FOV), window=12):
    """Analytic Gaussians at float positions, each with the given total flux."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    image = np.zeros(shape)
    for xi, yi, fi in zip(x, y, fluxes):
        x0, x1 = int(xi) - window, int(xi) + window + 1
        y0, y1 = int(yi) - window, int(yi) + window + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, shape[1]), min(y1, shape[0])
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma**2))
        image[y0:y1, x0:x1] += fi * g / g.sum()
    return image


def global_offset_grid(trace, order, fov=FOV):
    """Integer offset grid covering the trace extent anywhere on the detector."""
    lo, hi = np.inf, -np.inf
    for xs in (4, fov / 2, fov - 4):
        for ys in (4, fov / 2, fov - 4):
            r = trace.offset_range(order, x=xs, y=ys)
            lo, hi = min(lo, r[0]), max(hi, r[1])
    return np.arange(np.floor(lo), np.ceil(hi) + 1)


def compute_trace_weights(lam_um, spec_lam_AA, spec_flux, amp, sens_interp):
    """Per-trace-point counts: amp * flam * sensitivity * dlam. NaN-safe."""
    lam_AA = np.asarray(lam_um) * 1e4
    good = np.isfinite(lam_AA)
    lam_filled = np.where(good, lam_AA, 0.0)
    dlam = np.abs(np.gradient(np.where(good, lam_AA, np.nan)))
    flam = np.interp(lam_filled, spec_lam_AA, spec_flux, left=0.0, right=0.0)
    w = amp * flam * sens_interp(lam_filled) * np.where(good, dlam, 0.0)
    return np.where(np.isfinite(w), w, 0.0)


def build_dispersed_image(
    trace, order, x, y, amps, spec_lam_AA, source_spectra, sens_interp,
    stamp, shape=(FOV, FOV),
):
    """Disperse all sources for one order into a single image.

    Returns (image, per-source total counts).
    """
    offsets = global_offset_grid(trace, order, fov=shape[0])
    disperse_jit = jax.jit(disperse_obj, static_argnames=("chunk_size",))
    output = jnp.zeros(shape)
    stamp_j = jnp.asarray(stamp)
    fluxes = np.zeros(len(x))

    for i, (xi, yi, ai) in enumerate(zip(x, y, amps)):
        xt, yt, lam = trace.get_trace(xi, yi, order, offset=offsets)
        w = compute_trace_weights(lam, spec_lam_AA, source_spectra[i], ai, sens_interp)
        good = np.isfinite(xt) & np.isfinite(yt)
        xt = np.where(good, xt, 0.0)
        yt = np.where(good, yt, 0.0)
        w = np.where(good, w, 0.0)
        fluxes[i] = w.sum()
        output = disperse_jit(
            stamp_j, xi, yi, jnp.asarray(xt), jnp.asarray(yt), jnp.asarray(w), output
        )
    return np.asarray(output), fluxes


def write_fits(path, image, header_cards):
    hdu = fits.PrimaryHDU(np.asarray(image, dtype=np.float32))
    for key, value in header_cards.items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)


def simulate(
    n_sources=200,
    seed=42,
    filters=("F115W", "F150W", "F200W"),
    orders=("+1",),
    fwhm=1.6,
    border=100,
    amp_range=(0.01, 1.0),
    stamp_size=21,
    target_R=140,
    conf_dir=CONF_DIR,
    specwcs_files=None,
    outdir="output",
    write=True,
):
    """Run the full simulation. Returns (results dict, truth catalog Table).

    results[filt]['direct'] and results[filt]['dispersed'] are 2048x2048 arrays;
    the dispersed image is the sum over the requested orders.
    """
    specwcs = dict(SPECWCS_FILES)
    if specwcs_files:
        specwcs.update(specwcs_files)

    spec_lam_AA, spectra, params = load_kurucz_spectra(target_R=target_R)
    x, y, amps, kurucz_idx = generate_sources(
        n_sources, seed, len(spectra), border=border, amp_range=amp_range
    )
    source_spectra = spectra[kurucz_idx]
    stamp = unit_gaussian_stamp(fwhm, stamp_size)

    catalog = Table(
        {
            "id": np.arange(n_sources),
            "x": x,
            "y": y,
            "amplitude": amps,
            "fwhm_pix": np.full(n_sources, fwhm),
            "kurucz_index": kurucz_idx,
            "teff": params["Teff"][kurucz_idx],
            "logg": params["logg"][kurucz_idx],
            "Z": params["Z"][kurucz_idx],
            "logz": params["logz"][kurucz_idx],
        }
    )

    results = {}
    sens_paths = {}
    for filt in filters:
        specwcs_path = os.path.join(conf_dir, specwcs[filt])
        trace = GrismTrace.from_crds(specwcs_path, filter_name=filt)

        dispersed = np.zeros((FOV, FOV))
        for order in orders:
            sens_interp, sens_path = load_sensitivity(filt, order, conf_dir)
            sens_paths[f"{filt}_{order}"] = sens_path
            img, fluxes = build_dispersed_image(
                trace, order, x, y, amps, spec_lam_AA, source_spectra,
                sens_interp, stamp,
            )
            dispersed += img
            tag = ORDER_SENS_TAG[order]
            col = f"flux_{filt}" if order == "+1" else f"flux_{filt}_{tag}"
            catalog[col] = fluxes

        # Direct-image fluxes always follow the +1 sensitivity so the direct
        # image is defined even when +1 is not among the simulated orders.
        if f"flux_{filt}" not in catalog.colnames:
            sens_interp, sens_path = load_sensitivity(filt, "+1", conf_dir)
            fluxes = np.zeros(n_sources)
            offsets = global_offset_grid(trace, "+1")
            for i, (xi, yi, ai) in enumerate(zip(x, y, amps)):
                _, _, lam = trace.get_trace(xi, yi, "+1", offset=offsets)
                fluxes[i] = compute_trace_weights(
                    lam, spec_lam_AA, source_spectra[i], ai, sens_interp
                ).sum()
            catalog[f"flux_{filt}"] = fluxes

        direct = build_direct_image(x, y, catalog[f"flux_{filt}"], fwhm=fwhm)
        results[filt] = {"direct": direct, "dispersed": dispersed, "trace": trace}

    try:
        grismagic_hash = subprocess.run(
            ["git", "-C", os.path.expanduser("~/dev/grismagic"), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
    except OSError:
        grismagic_hash = "unknown"

    catalog.meta.update(
        {
            "seed": seed,
            "n_sources": n_sources,
            "border": border,
            "amp_range": list(amp_range),
            "fwhm_pix": fwhm,
            "stamp_size": stamp_size,
            "target_R": target_R,
            "grism": "GR150C",
            "orders": list(orders),
            "filters": list(filters),
            "specwcs_files": {f: os.path.join(conf_dir, specwcs[f]) for f in filters},
            "sensitivity_files": sens_paths,
            "flux_convention": (
                "flux_<filt> = amplitude * sum(flam_norm * sens_p1 * dlam) over the "
                "+1 trace = total counts of the source in the direct image and in "
                "the +1 dispersed spectrum. flam_norm is the Kurucz spectrum "
                "normalized to unit mean flam over 0.7-2.3 um at R=target_R."
            ),
            "grismagic_git_hash": grismagic_hash,
            "created": datetime.datetime.now().isoformat(),
        }
    )

    if write:
        os.makedirs(outdir, exist_ok=True)
        for filt in filters:
            cards = {
                "FILTER": filt,
                "GRISM": "GR150C",
                "SEED": seed,
                "NSOURCES": n_sources,
                "FWHM": fwhm,
                "ORDERS": ",".join(orders),
                "SPECWCS": specwcs[filt],
            }
            write_fits(
                os.path.join(outdir, f"sim_GR150C_{filt}_direct.fits"),
                results[filt]["direct"], {**cards, "IMTYPE": "DIRECT"},
            )
            write_fits(
                os.path.join(outdir, f"sim_GR150C_{filt}_grism.fits"),
                results[filt]["dispersed"], {**cards, "IMTYPE": "GRISM"},
            )
        catalog.write(
            os.path.join(outdir, "truth_catalog.ecsv"),
            format="ascii.ecsv", overwrite=True,
        )

    return results, catalog
