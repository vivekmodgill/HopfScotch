"""
metastable_oscillatory_modes.py

Computes and plots metastable oscillatory mode (MOM) statistics reported in
Fig. 5 of the manuscript: burst occupancy, burst count, and burst duration,
compared between the Rest and Perturbed windows for each of the five
stimulation waveforms. Thresholds are estimated from the Rest window only
and held fixed across both windows, following Cabral et al. (2022).

Pipeline:
    1. run_mom_for_shape() / compute_all_shapes() -- load raw simulation,
       compute alpha-band envelope, threshold, and MOM metrics
    2. plot_figure5()                              -- reproduce Fig. 5
    3. friedman_posthoc()                          -- omnibus + pairwise
       comparison of waveform effects (Friedman test, Holm-Bonferroni)
    4. per_shape_statistics()                       -- Rest-vs-Perturbed
       statistics per shape and metric, saved to CSV

Usage:
    python metastable_oscillatory_modes.py

Expects simulation output files named:
    signal_C{K}_md{MD}_dt{dt}_shape{0..4}.npy
in DATA_DIR, as produced by src/run_sl_tACS.py.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
from itertools import combinations

# ============================================================
# Config
# ============================================================
FS = 1000
ALPHA_BAND = (8, 13)
TRANSIENT_SEC = 60
PHASES_SEC = {"Rest": (0, 120), "Perturbed": (180, 300)}
SHAPE_LABELS = {0: "Sinusoidal", 1: "Square", 2: "Triangular", 3: "Pulsed", 4: "Sawtooth"}
SHAPES = list(SHAPE_LABELS.values())

DATA_DIR = "simulations/"
PKL_PATH = "mom_results.pkl"
STATS_CSV = "mom_stats.csv"

# Threshold estimation
THRESH_MODE = "std-mult"          # "std-mult" or "percentile"
BASE_OCC_RANGE = (0.01, 0.15)     # target baseline occupancy for std-mult search
PERCENTILE_Q = 95.0                # used if THRESH_MODE == "percentile"
_center_alpha = 0.5 * (ALPHA_BAND[0] + ALPHA_BAND[1])
MIN_BURST_SEC = 2.0 / _center_alpha   # minimum burst duration: two alpha cycles


# ============================================================
# Signal-processing helpers
# ============================================================
def bandpass(x, fs, f1, f2, order=4):
    b, a = butter(order, [f1, f2], btype="band", fs=fs)
    return filtfilt(b, a, np.real(x), axis=-1)


def envelope(x):
    return np.abs(hilbert(x, axis=-1))


def time_slice(sig, fs, start_s, end_s):
    i0, i1 = int(round(start_s * fs)), int(round(end_s * fs))
    return sig[:, i0:i1]


def contiguous_true_regions(x_bool_1d):
    x = x_bool_1d.astype(np.int8)
    dx = np.diff(x, prepend=0, append=0)
    starts = np.where(dx == 1)[0]
    ends = np.where(dx == -1)[0]
    return list(zip(starts, ends))


def apply_min_duration(T_bool, fs, min_sec):
    if min_sec <= 0:
        return T_bool
    min_len = int(round(min_sec * fs))
    out = T_bool.copy()
    for ch in range(out.shape[0]):
        for s, e in contiguous_true_regions(out[ch]):
            if (e - s) < min_len:
                out[ch, s:e] = False
    return out


def thresholds_from_rest(env_full, fs, rest_win, mode="std-mult",
                          base_occ_range=(0.01, 0.15), q=95.0):
    """Estimate per-channel MOM thresholds from the Rest window only."""
    env_R = time_slice(env_full, fs, *rest_win)

    if mode == "percentile":
        thr = np.percentile(env_R, q, axis=1)
        occ_rest = np.mean(env_R > thr[:, None], axis=1)
        return thr, occ_rest

    std_R = np.std(env_R, axis=1, ddof=0)
    std_R = np.where(std_R > 0, std_R, np.median(std_R[std_R > 0]) if np.any(std_R > 0) else 1.0)

    lo, hi = base_occ_range
    grid = np.logspace(-2, 1, 60)
    best_k, best_diff = grid[0], np.inf
    for k in grid:
        occ = np.mean(env_R > (std_R * k)[:, None])
        diff = 0 if (lo <= occ <= hi) else min(abs(occ - lo), abs(occ - hi))
        if diff < best_diff:
            best_diff, best_k = diff, k
            if diff == 0:
                break

    thr = std_R * best_k
    occ_rest = np.mean(env_R > thr[:, None], axis=1)
    return thr, occ_rest


def compute_mom_metrics(T_bool, fs, min_burst_sec=0.0):
    """Occupancy, burst count, and pooled burst duration from a thresholded envelope."""
    T = apply_min_duration(T_bool, fs, min_burst_sec)
    occ = np.mean(T, axis=1)

    burst_counts, pooled_durs = [], []
    for ch in range(T.shape[0]):
        regs = contiguous_true_regions(T[ch])
        burst_counts.append(len(regs))
        pooled_durs.extend((e - s) / fs for s, e in regs)

    return {
        "occupancy": occ,
        "burst_count": np.asarray(burst_counts, int),
        "burst_duration": np.asarray(pooled_durs, float),
    }


# ============================================================
# Step 1: run MOM pipeline for one shape / all shapes
# ============================================================
def run_mom_for_shape(fpath, fs=FS, band=ALPHA_BAND, phases=PHASES_SEC,
                       thr_mode=THRESH_MODE, base_occ_range=BASE_OCC_RANGE,
                       q=PERCENTILE_Q, min_burst_sec=MIN_BURST_SEC, zscore=True):
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)

    sig = np.load(fpath)
    sig = np.real(sig)[:, int(TRANSIENT_SEC * fs):]
    if zscore:
        mu = sig.mean(axis=1, keepdims=True)
        sd = sig.std(axis=1, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        sig = (sig - mu) / sd

    sig_b = bandpass(sig, fs, band[0], band[1])
    env = envelope(sig_b)

    thr, occ_rest_ch = thresholds_from_rest(
        env, fs, phases["Rest"], mode=thr_mode, base_occ_range=base_occ_range, q=q
    )

    env_R = time_slice(env, fs, *phases["Rest"])
    env_P = time_slice(env, fs, *phases["Perturbed"])
    T_R = env_R > thr[:, None]
    T_P = env_P > thr[:, None]

    met_R = compute_mom_metrics(T_R, fs, min_burst_sec=min_burst_sec)
    met_P = compute_mom_metrics(T_P, fs, min_burst_sec=min_burst_sec)

    return {
        "thresholds": thr,
        "rest_occ_per_channel": occ_rest_ch,
        "T_rest": T_R, "T_perturbed": T_P,
        "metrics_rest": met_R, "metrics_perturbed": met_P,
    }


def compute_all_shapes(data_dir=DATA_DIR, save_pkl=PKL_PATH):
    results = {}
    for sid, sname in SHAPE_LABELS.items():
        fpath = os.path.join(data_dir, f"signal_C11.92_md3_dt0.1_shape{sid}.npy")
        results[sname] = run_mom_for_shape(fpath)
        print(f"[OK] {sname}: MOM metrics computed")

    if save_pkl:
        with open(save_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved: {save_pkl}")
    return results


def load_results(path=PKL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Stats helpers
# ============================================================
def remove_outliers(a, method="iqr", thresh=1.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0 or method == "none":
        return a, 0
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    low, high = q1 - thresh * iqr, q3 + thresh * iqr
    keep = (a >= low) & (a <= high)
    return a[keep], int((~keep).sum())


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


# ============================================================
# Step 2: Figure 5
# ============================================================
def plot_figure5(results, filename="Figure5_MOMs.pdf", save_dir="figures_out"):
    os.makedirs(save_dir, exist_ok=True)
    color_rest, color_pert = "#4C72B0", "#DD8452"
    metrics = [("occupancy", "Occupancy"), ("burst_count", "Burst count"),
               ("burst_duration", "Burst duration (s)")]

    def global_xlims(metric_key, pad_frac=0.20):
        raw_vals, ci_uppers = [], []
        for shape in SHAPES:
            for cond in ["metrics_rest", "metrics_perturbed"]:
                data = np.asarray(results[shape][cond][metric_key], float)
                data = data[np.isfinite(data)]
                if data.size == 0:
                    continue
                raw_vals.extend(data)
                _, (_, ci_hi) = bootstrap_ci(data)
                if np.isfinite(ci_hi):
                    ci_uppers.append(ci_hi)
        raw_vals = np.asarray(raw_vals, float)
        if raw_vals.size == 0:
            return (0, 1)
        lo, hi = np.percentile(raw_vals, [2, 98])
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        hi_final = max(hi, np.nanmax(ci_uppers) if ci_uppers else hi)
        span = hi_final - lo
        xmin, xmax = max(0, lo - pad_frac * span), hi_final + pad_frac * span
        if metric_key == "burst_count":
            xmax = min(xmax, 100)
        return xmin, xmax

    xlims = {mkey: global_xlims(mkey) for mkey, _ in metrics}

    fig, ax = plt.subplots(nrows=len(SHAPES), ncols=len(metrics),
                            figsize=(16.5, 8.5), dpi=200, sharey="row")
    bar_h = 0.35
    y_off = np.array([bar_h / 2, -bar_h / 2])

    for i, shape in enumerate(SHAPES):
        for j, (metric_key, metric_label) in enumerate(metrics):
            a = ax[i, j]
            B = np.asarray(results[shape]["metrics_rest"][metric_key], float)
            D = np.asarray(results[shape]["metrics_perturbed"][metric_key], float)
            mean_B, ci_B = bootstrap_ci(B)
            mean_D, ci_D = bootstrap_ci(D)

            a.barh(y_off[0], mean_B, xerr=[[mean_B - ci_B[0]], [ci_B[1] - mean_B]],
                   height=bar_h, color=color_rest, edgecolor="black", linewidth=0.8,
                   capsize=4, zorder=2)
            a.barh(y_off[1], mean_D, xerr=[[mean_D - ci_D[0]], [ci_D[1] - mean_D]],
                   height=bar_h, color=color_pert, edgecolor="black", linewidth=0.8,
                   capsize=4, zorder=2)

            star = pval_to_star(compute_p_value(B, D))
            a.text(0.985, 0.5, star, transform=a.transAxes, ha="center", va="center",
                   rotation=90, fontsize=12, fontweight="bold", zorder=5)

            a.axvline(0, color=(0, 0, 0, 0.25), lw=0.8)
            a.set_xlim(xlims[metric_key])
            a.set_yticks([])
            a.grid(True, axis="x", linestyle=":", alpha=0.35)
            a.set_axisbelow(True)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            if i < len(SHAPES) - 1:
                a.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                a.tick_params(axis="x", labelsize=12)
            if i == 0:
                a.set_title(metric_label, fontsize=15)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_rest, edgecolor="black", linewidth=0.8),
               plt.Rectangle((0, 0), 1, 1, facecolor=color_pert, edgecolor="black", linewidth=0.8)]
    fig.legend(handles, ["Rest", "Perturbed"], loc="upper right", frameon=False, fontsize=13)
    fig.suptitle("Stimulation Waveform Shapes Metastable Oscillatory Modes (MOMs)", fontsize=18, y=0.97)
    fig.tight_layout(rect=[0.12, 0.0, 0.95, 0.95])

    for i, shape in enumerate(SHAPES):
        bbox = ax[i, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1) - 0.01
        fig.text(0.06, y_center, shape, ha="left", va="center", fontsize=15)

    outpath = os.path.join(save_dir, filename)
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.show()


# ============================================================
# Step 3: omnibus + pairwise waveform comparison (Friedman/Holm)
# ============================================================
def deltas_by_shape(results, metric_key):
    out = {}
    for shape in SHAPES:
        B = np.asarray(results[shape]["metrics_rest"][metric_key], float)
        D = np.asarray(results[shape]["metrics_perturbed"][metric_key], float)
        keep = np.isfinite(B) & np.isfinite(D)
        out[shape] = D[keep] - B[keep]
    return out


def holm_bonferroni(pvals):
    pvals = np.asarray(pvals, float)
    m = pvals.size
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    for k, idx in enumerate(order):
        adj[idx] = min((m - k) * pvals[idx], 1.0)
    for k in range(1, m):
        adj[order[k]] = max(adj[order[k]], adj[order[k - 1]])
    return adj


def friedman_posthoc(results, metric_key):
    """Omnibus Friedman test across waveforms, with Holm-corrected pairwise Wilcoxon posthoc."""
    deltas = deltas_by_shape(results, metric_key)
    X = np.vstack([deltas[s] for s in SHAPES]).T
    fried_stat, fried_p = stats.friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])

    pairs = list(combinations(range(len(SHAPES)), 2))
    raw_p = []
    for i, j in pairs:
        try:
            _, p = stats.wilcoxon(X[:, i], X[:, j], zero_method="wilcox", correction=True)
        except Exception:
            p = np.nan
        raw_p.append(p)
    raw_p = np.asarray(raw_p, float)
    adj_p = holm_bonferroni(raw_p)
    pair_names = [(SHAPES[i], SHAPES[j]) for i, j in pairs]

    print(f"\n[{metric_key}] Friedman: chi2={fried_stat:.3f}, p={fried_p:.3e}")
    for (a, b), p0, p1 in zip(pair_names, raw_p, adj_p):
        print(f"  {a} vs {b}: p={p0:.3e}, Holm p={p1:.3e}")
    return fried_stat, fried_p, pair_names, raw_p, adj_p


# ============================================================
# Step 4: per-shape Rest-vs-Perturbed statistics table
# ============================================================
def per_shape_statistics(results, out_csv=STATS_CSV, outlier_method="iqr", outlier_thresh=1.5):
    rows = []
    for shape, r in results.items():
        for metric in ("occupancy", "burst_count", "burst_duration"):
            B = np.asarray(r["metrics_rest"][metric], float)
            D = np.asarray(r["metrics_perturbed"][metric], float)
            B, D = B[np.isfinite(B)], D[np.isfinite(D)]
            Bc, nB = remove_outliers(B, outlier_method, outlier_thresh)
            Dc, nD = remove_outliers(D, outlier_method, outlier_thresh)

            # burst_duration is a pooled, unpaired distribution across events;
            # occupancy/burst_count are per-channel and paired if counts match
            paired_ok = (metric != "burst_duration") and Bc.size and (Bc.size == Dc.size)

            if Bc.size and Dc.size:
                if paired_ok:
                    try:
                        _, p = stats.wilcoxon(Bc, Dc, zero_method="wilcox", correction=True)
                        test, delta = "Wilcoxon (paired, corr)", float(np.mean(Dc - Bc))
                    except Exception:
                        p, test, delta = np.nan, "Wilcoxon (paired, corr)", np.nan
                else:
                    _, p = stats.mannwhitneyu(Bc, Dc, alternative="two-sided")
                    test, delta = "Mann-Whitney (unpaired)", np.nan
            else:
                p, test, delta = np.nan, "n/a", np.nan

            rows.append({
                "Shape": shape, "MOM_Metric": metric,
                "N_Rest": int(Bc.size), "N_Perturbed": int(Dc.size),
                "Mean_Rest": float(np.mean(Bc)) if Bc.size else np.nan,
                "SD_Rest": float(np.std(Bc)) if Bc.size else np.nan,
                "Mean_Perturbed": float(np.mean(Dc)) if Dc.size else np.nan,
                "SD_Perturbed": float(np.std(Dc)) if Dc.size else np.nan,
                "Delta": delta, "Test": test,
                "p_value": float(p) if p is not None else np.nan,
                "Stars": pval_to_star(p),
                "Removed_Rest": int(nB), "Removed_Perturbed": int(nD),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return df


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if os.path.exists(PKL_PATH):
        results = load_results()
    else:
        results = compute_all_shapes()

    plot_figure5(results)
    per_shape_statistics(results)
    for metric in ("occupancy", "burst_count", "burst_duration"):
        friedman_posthoc(results, metric)
