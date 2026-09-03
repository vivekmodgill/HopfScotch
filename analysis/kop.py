"""
kop_metastability_spectral_entropy.py

Computes and plots the Kuramoto-order-parameter-derived metrics reported in
Fig. 4 of the manuscript: synchrony, metastability, and alpha-band spectral
entropy, compared between the Rest and Perturbed windows for each of the
five stimulation waveforms.

Pipeline:
    1. compute_kop_and_spectral_entropy()  -- load raw simulation .npy files,
       compute per-epoch KOP(t) and spectral entropy, save to a single .npz
    2. load_results()                       -- load the .npz back into a
       results dict keyed by waveform shape
    3. summary_statistics()                 -- per-shape, per-metric
       Rest-vs-Perturbed statistics (Wilcoxon/Mann-Whitney), saved to CSV
    4. plot_figure4()                       -- reproduce Fig. 4

Usage:
    python kop_metastability_spectral_entropy.py

Expects simulation output files named:
    signal_C{K}_md{MD}_dt{dt}_shape{0..4}.npy
in DATA_DIR, as produced by src/run_sl_tACS.py (shape indices 0-4 correspond
to Sinusoidal, Square, Triangular, Pulsed, Sawtooth respectively).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats

# ============================================================
# Config
# ============================================================
FS = 1000                                  # Hz, effective sampling rate (1 ms storage resolution)
ALPHA_BAND = (8, 13)                       # Hz
EPOCH_SEC = 5                               # epoch length for per-epoch metrics
TRANSIENT_SEC = 60                          # initial transient discarded (raw simulation time)
PHASES_SEC = {"Rest": (0, 120), "Perturbed": (180, 300)}  # post-transient windows
SHAPE_LABELS = {0: "Sinusoidal", 1: "Square", 2: "Triangular", 3: "Pulsed", 4: "Sawtooth"}
SHAPES = list(SHAPE_LABELS.values())

DATA_DIR = "simulations/"                   # directory containing signal_*.npy files
NPZ_PATH = "alpha_kop_se.npz"               # cached intermediate results
STATS_CSV = "kop_se_stats.csv"


# ============================================================
# Signal-processing helpers
# ============================================================
def bandpass(x, fs, f1, f2, order=4):
    b, a = signal.butter(order, [f1, f2], btype="band", fs=fs)
    return signal.filtfilt(b, a, np.real(x), axis=-1)


def epochs_from(x, fs, epoch_sec):
    """Segment (channels, samples) into (epochs, channels, samples)."""
    ep_len = int(epoch_sec * fs)
    n = x.shape[-1] // ep_len
    if n == 0:
        return np.empty((0, x.shape[0], ep_len))
    trimmed = x[:, : n * ep_len]
    ep = trimmed.reshape(x.shape[0], n, ep_len)
    return np.moveaxis(ep, 1, 0)


def kop_series_band_epoch(x, fs, band):
    """Instantaneous Kuramoto order parameter R(t) for one epoch (channels, samples)."""
    xb = bandpass(x, fs, band[0], band[1])
    analytic = signal.hilbert(xb, axis=-1)
    phases = analytic / np.abs(analytic)
    return np.abs(np.nanmean(phases, axis=0))


def spectral_entropy_band_epoch(x, fs, band, nperseg=None, nfft=None):
    """Shannon spectral entropy within `band`, averaged across channels, for one epoch."""
    if nperseg is None:
        nperseg = 2 * fs
    if nfft is None:
        nfft = 5 * fs
    f, Pxx = signal.welch(np.real(x), fs=fs, nperseg=nperseg, nfft=nfft, axis=-1)
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return np.nan
    Pxx = np.maximum(Pxx[..., mask], 1e-12)
    P = Pxx / Pxx.sum(axis=-1, keepdims=True)
    se_per_channel = -np.sum(P * np.log(P), axis=-1)
    return float(np.nanmean(se_per_channel))


def remove_outliers(a, method="iqr", thresh=1.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0 or method == "none":
        return a, 0
    if method == "iqr":
        q1, q3 = np.percentile(a, [25, 75])
        iqr = q3 - q1
        low, high = q1 - thresh * iqr, q3 + thresh * iqr
        keep = (a >= low) & (a <= high)
        return a[keep], int((~keep).sum())
    raise ValueError("outlier method must be 'none' or 'iqr'")


def pval_to_star(p):
    if p is None or np.isnan(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def bootstrap_ci(data, n_boot=5000, seed=1234):
    rng = np.random.default_rng(seed)
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan, (np.nan, np.nan)
    boots = rng.choice(data, size=(n_boot, data.size), replace=True)
    mean = data.mean()
    lo, hi = np.percentile(boots.mean(axis=1), [2.5, 97.5])
    return mean, (lo, hi)


def compute_p_value(before, during):
    before = np.asarray(before, float)
    during = np.asarray(during, float)
    before = before[np.isfinite(before)]
    during = during[np.isfinite(during)]
    if before.size == 0 or during.size == 0:
        return np.nan
    if before.size == during.size:
        try:
            _, p = stats.wilcoxon(before, during, zero_method="wilcox", correction=True)
        except Exception:
            p = np.nan
    else:
        _, p = stats.mannwhitneyu(before, during, alternative="two-sided")
    return p


# ============================================================
# Step 1: compute KOP(t) and spectral entropy per epoch, save .npz
# ============================================================
def compute_kop_and_spectral_entropy(data_dir=DATA_DIR, out_path=NPZ_PATH):
    """
    For each waveform, load the raw simulation, discard the initial transient,
    and compute per-epoch KOP(t) and alpha-band spectral entropy for the Rest
    and Perturbed windows. Results are saved to a single .npz file.
    """
    flat = {}
    for sid, sname in SHAPE_LABELS.items():
        fpath = os.path.join(data_dir, f"signal_C11.92_md3_dt0.1_shape{sid}.npy")
        if not os.path.exists(fpath):
            print(f"[WARN] Missing: {fpath}")
            continue

        sig = np.load(fpath)
        sig = np.asarray(np.real(sig)[:, int(TRANSIENT_SEC * FS):], dtype=float)
        sig = stats.zscore(sig, axis=-1)

        for phase, (t0, t1) in PHASES_SEC.items():
            window = sig[:, t0 * FS: t1 * FS]
            eps = epochs_from(window, FS, EPOCH_SEC)

            kop_list, se_list = [], []
            for ep in eps:
                if ep.size == 0:
                    continue
                kop_list.append(kop_series_band_epoch(ep, FS, ALPHA_BAND))
                se_list.append(spectral_entropy_band_epoch(ep, FS, ALPHA_BAND))

            kop_arr = np.vstack(kop_list) if kop_list else np.empty((0, int(EPOCH_SEC * FS)))
            se_arr = np.asarray(se_list, float)

            flat[f"{sname}_kop_{phase}"] = kop_arr
            flat[f"{sname}_spectral_entropy_{phase}"] = se_arr

        print(f"[OK] {sname}: computed KOP/SE for Rest and Perturbed windows")

    np.savez(out_path, **flat)
    print(f"Saved: {out_path}")


# ============================================================
# Step 2: load results back into a convenient dict
# ============================================================
def load_results(path=NPZ_PATH, shapes=SHAPES):
    z = np.load(path, allow_pickle=True)
    results = {}
    for sh in shapes:
        results[sh] = {
            "kop": {
                "Rest": z[f"{sh}_kop_Rest"],
                "Perturbed": z[f"{sh}_kop_Perturbed"],
            },
            "spectral_entropy": {
                "Rest": z[f"{sh}_spectral_entropy_Rest"],
                "Perturbed": z[f"{sh}_spectral_entropy_Perturbed"],
            },
        }
    return results


def extract_metric(results, shape, metric_key):
    """metric_key in {'metastability', 'synchrony', 'spectral_entropy'}."""
    if metric_key == "metastability":
        B = np.nanstd(results[shape]["kop"]["Rest"], axis=1)
        D = np.nanstd(results[shape]["kop"]["Perturbed"], axis=1)
    elif metric_key == "synchrony":
        B = np.nanmean(results[shape]["kop"]["Rest"], axis=1)
        D = np.nanmean(results[shape]["kop"]["Perturbed"], axis=1)
    elif metric_key == "spectral_entropy":
        B = results[shape]["spectral_entropy"]["Rest"]
        D = results[shape]["spectral_entropy"]["Perturbed"]
    else:
        raise ValueError(f"Unknown metric: {metric_key}")
    return np.asarray(B, float), np.asarray(D, float)


# ============================================================
# Step 3: summary statistics table
# ============================================================
def summary_statistics(results, out_csv=STATS_CSV, outlier_method="iqr", outlier_thresh=1.5):
    metrics = ["metastability", "synchrony", "spectral_entropy"]
    records = []
    for shape in SHAPES:
        for metric in metrics:
            B_raw, D_raw = extract_metric(results, shape, metric)
            B, nB_rm = remove_outliers(B_raw, outlier_method, outlier_thresh)
            D, nD_rm = remove_outliers(D_raw, outlier_method, outlier_thresh)

            if B.size == 0 or D.size == 0:
                records.append({"Shape": shape, "Metric": metric, "N_Rest": 0, "N_Perturbed": 0})
                continue

            p = compute_p_value(B, D)
            test_name = "Wilcoxon (paired, corr)" if B.size == D.size else "Mann-Whitney (unpaired)"
            delta = float(np.nanmean(D - B)) if B.size == D.size else np.nan

            records.append({
                "Shape": shape, "Metric": metric,
                "N_Rest": int(B.size), "N_Perturbed": int(D.size),
                "Mean_Rest": float(np.nanmean(B)), "SD_Rest": float(np.nanstd(B)),
                "Mean_Perturbed": float(np.nanmean(D)), "SD_Perturbed": float(np.nanstd(D)),
                "Delta": delta, "Test": test_name,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "Stars": pval_to_star(p),
                "Removed_Rest": nB_rm, "Removed_Perturbed": nD_rm,
            })

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    return df


# ============================================================
# Step 4: Figure 4
# ============================================================
def plot_figure4(results, filename="Figure4_KOP_metrics.pdf", save_dir="figures_out"):
    os.makedirs(save_dir, exist_ok=True)
    color_rest, color_pert = "#4C72B0", "#DD8452"

    metrics = [("metastability", "Metastability"), ("synchrony", "Synchrony"),
               ("spectral_entropy", "Spectral entropy")]

    def global_xlims(metric_key, pad_frac=0.20):
        vals = []
        for shape in SHAPES:
            B, D = extract_metric(results, shape, metric_key)
            vals.extend(B[np.isfinite(B)])
            vals.extend(D[np.isfinite(D)])
        vals = np.asarray(vals, float)
        if vals.size == 0:
            return (0, 1)
        lo, hi = np.percentile(vals, [2, 98])
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        span = hi - lo
        return max(0, lo - pad_frac * span), hi + pad_frac * span

    xlims = {mkey: global_xlims(mkey) for mkey, _ in metrics}

    fig, ax = plt.subplots(nrows=len(SHAPES), ncols=len(metrics),
                            figsize=(16.5, 8.5), dpi=200, sharey="row")
    bar_h = 0.35
    y_off = np.array([bar_h / 2, -bar_h / 2])

    for i, shape in enumerate(SHAPES):
        for j, (metric_key, metric_label) in enumerate(metrics):
            a = ax[i, j]
            B, D = extract_metric(results, shape, metric_key)
            mean_B, ci_B = bootstrap_ci(B)
            mean_D, ci_D = bootstrap_ci(D)

            a.barh(y_off[0], mean_B, xerr=[[mean_B - ci_B[0]], [ci_B[1] - mean_B]],
                   height=bar_h, color=color_rest, edgecolor="black", capsize=4, zorder=2)
            a.barh(y_off[1], mean_D, xerr=[[mean_D - ci_D[0]], [ci_D[1] - mean_D]],
                   height=bar_h, color=color_pert, edgecolor="black", capsize=4, zorder=2)

            star = pval_to_star(compute_p_value(B, D))
            a.text(0.985, 0.5, star, transform=a.transAxes, ha="center", va="center",
                   rotation=90, fontsize=12, fontweight="bold")

            a.set_xlim(xlims[metric_key])
            a.set_yticks([])
            a.axvline(0, color=(0, 0, 0, 0.25), lw=0.8)
            a.grid(True, axis="x", linestyle=":", alpha=0.35)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            if i < len(SHAPES) - 1:
                a.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                a.tick_params(axis="x", labelsize=12)
            if i == 0:
                a.set_title(metric_label, fontsize=15)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_rest, edgecolor="black"),
               plt.Rectangle((0, 0), 1, 1, facecolor=color_pert, edgecolor="black")]
    fig.legend(handles, ["Rest", "Perturbed"], loc="upper right", frameon=False, fontsize=13)
    fig.suptitle("Waveform-Dependent Modulation of Global Brain Dynamics", fontsize=18, y=0.97)
    fig.tight_layout(rect=[0.12, 0.0, 0.95, 0.95])

    for i, shape in enumerate(SHAPES):
        bbox = ax[i, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1) - 0.01
        fig.text(0.06, y_center, shape, ha="left", va="center", fontsize=15.5)

    outpath = os.path.join(save_dir, filename)
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(NPZ_PATH):
        compute_kop_and_spectral_entropy()
    results = load_results()
    summary_statistics(results)
    plot_figure4(results)
