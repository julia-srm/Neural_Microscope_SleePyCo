"""
Bandleistungs-Analyse für Schlaf-EEG (.npz mit x[n_epochs, n_samples], y, fs).

Aufruf:
    python band_analysis.py /pfad/datensatz1 /pfad/datensatz2 --names ANPHY DOD-H --out ergebnisse

Zwei Datensätze (Ordner) werden eingelesen und in denselben Abbildungen verglichen.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from specparam import SpectralModel
from scipy.signal import welch
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind   # oben bei den Imports ergänzen

BANDS = {"Delta/K (1-3 Hz)": (1, 3), "Theta (4-7 Hz)": (4, 7), "Alpha (8-13 Hz)": (8, 13),
         "Spindel (12-14 Hz)": (12, 14), "Beta (13-40 Hz)": (13, 40)}
TOTAL = (0.3, 45)                       # Referenzbereich für relative Leistung
WHITENING_RANGE = (0.3, 45)
STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}   # Label 6 etc. wird verworfen
PALETTE = ["#1f77b4", "#e08a1e"]


def band_power(f, P, lo, hi):
    m = (f >= lo) & (f <= hi)
    return P[:, m].sum(1) * (f[1] - f[0])


def load_file(path, win_s=4):
    z = np.load(path, allow_pickle=False)            # kein Pickle: untrusted data
    x, y, fs = z["x"], z["y"], int(z["fs"])
    f, P = welch(x, fs=fs, nperseg=win_s * fs, noverlap=win_s * fs // 2, window="hann", axis=1)
    df = pd.DataFrame({"y": y})
    df["total"] = band_power(f, P, *TOTAL)
    for name, (lo, hi) in BANDS.items():
        df[name] = band_power(f, P, lo, hi)
    return df, f, P, y


def plot_whitened_psd(f, psd, groups, stages, colors, output_path):
    if f.max() < WHITENING_RANGE[1] or not np.any(
        (f >= WHITENING_RANGE[0]) & (f <= WHITENING_RANGE[1])
    ):
        raise ValueError(
            f"Whitening requires PSD data covering {WHITENING_RANGE[0]}-"
            f"{WHITENING_RANGE[1]} Hz; available range is {f[0]}-{f[-1]} Hz."
        )

    fig, axes = plt.subplots(2, len(stages), figsize=(3.4 * len(stages), 6),
                             sharex=True, sharey="row")
    for i, stage in enumerate(stages):
        raw_ax, flat_ax = axes[:, i]
        for ax in (raw_ax, flat_ax):
            for lo, hi in BANDS.values():
                ax.axvspan(lo, hi, color="grey", alpha=.06, zorder=0)

        for group in groups:
            subject_psds = psd.get((group, stage))
            if not subject_psds:
                continue
            mean_psd = np.mean(subject_psds, axis=0)
            model = SpectralModel(
                peak_width_limits=[2, 8],
                max_n_peaks=6,
                min_peak_height=0.1,
                aperiodic_mode="fixed",
                verbose=False,
            )
            model.fit(f, mean_psd, list(WHITENING_RANGE))

            freqs_used = model.data.freqs
            power_log = model.data.power_spectrum
            aperiodic_fit = model.results.model.get_component("aperiodic")
            raw_ax.semilogy(freqs_used, 10 ** power_log, color=colors[group],
                            label=f"{group} PSD")
            raw_ax.semilogy(freqs_used, 10 ** aperiodic_fit, color=colors[group],
                            linestyle="--", linewidth=1, label=f"{group} aperiodic fit")
            flat_ax.plot(freqs_used, power_log - aperiodic_fit, color=colors[group],
                         label=group)

        raw_ax.set_title(stage)
        raw_ax.set_xlim(*WHITENING_RANGE)
        raw_ax.spines[["top", "right"]].set_visible(False)
        flat_ax.axhline(0, color="black", linewidth=.7, linestyle=":")
        flat_ax.set_xlabel("Frequency (Hz)")
        flat_ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("PSD (µV²/Hz)")
    axes[1, 0].set_ylabel("Whitened log10 power\n(aperiodic removed)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Whitened spectra and aperiodic fits by sleep stage")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder1"); ap.add_argument("folder2")
    ap.add_argument("--names", nargs=2, default=None, help="Namen der beiden Datensätze")
    ap.add_argument("--out", default="band_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    folders = [a.folder1, a.folder2]
    names = a.names or [os.path.basename(os.path.normpath(p)) for p in folders]
    if names[0] == names[1]:
        names = [f"{names[0]}_1", f"{names[1]}_2"]

    rows, psd, f = [], {}, None
    for grp, folder in zip(names, folders):
        for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
            fname_clean = re.sub(r"^[0-9a-f]{8}-\d+_", "", os.path.basename(path))  # Upload-Präfix entfernen
            df, f, P, y = load_file(path)
            df["group"], df["subject"] = grp, f"{grp}/{fname_clean[:-4]}"
            rows.append(df)
            for s, sn in STAGES.items():                          # mittlere PSD pro Subjekt/Stadium
                if (y == s).any():
                    psd.setdefault((grp, sn), []).append(P[y == s].mean(0))
            print(f"{fname_clean}: {len(df)} Epochen -> {grp}")
    E = pd.concat(rows, ignore_index=True)
    E = E[E.y.isin(STAGES)].copy()
    E["stage"] = E.y.map(STAGES)
    for b in BANDS:
        E["rel_" + b] = E[b] / E.total

    # Subjekt-Median pro Stadium (Subjekt = statistische Einheit)
    rel = ["rel_" + b for b in BANDS]
    S = E.groupby(["group", "subject", "stage"])[rel].median().reset_index()
    S.to_csv(f"{a.out}/subject_stage_medians.csv", index=False)
    S.groupby(["group", "stage"])[rel].agg(["median", "std", "count"]).to_csv(f"{a.out}/group_summary.csv")


    # Vergleich der beiden Datensätze: Welch-t-Test + Bonferroni
    r = []
    for st in STAGES.values():
        for b in BANDS:
            v1 = S[(S.group == names[0]) & (S.stage == st)]["rel_" + b]
            v2 = S[(S.group == names[1]) & (S.stage == st)]["rel_" + b]
            if len(v1) > 1 and len(v2) > 1:
                t = ttest_ind(v1, v2, equal_var=False)          # Welch: ungleiche Varianzen/Gruppengrößen erlaubt
                sd_pool = np.sqrt((v1.var(ddof=1) + v2.var(ddof=1)) / 2)
                r.append(dict(stage=st, band=b,
                            **{f"{names[0]}_mean": v1.mean(), f"{names[1]}_mean": v2.mean()},
                            t=t.statistic, p=t.pvalue, cohens_d=(v1.mean() - v2.mean()) / sd_pool))
    res = pd.DataFrame(r)
    res["p_bonf"] = (res.p * len(res)).clip(upper=1)            # Bonferroni: p × Anzahl Tests (hier 5 Stadien × 5 Bänder = 25)
    res["signifikant"] = res.p_bonf < 0.05
    res.to_csv(f"{a.out}/group_comparison.csv", index=False)
    col = {g: PALETTE[i] for i, g in enumerate(names)}
    order = list(STAGES.values())
    band_list = list(BANDS)
    band_labels = [b.replace(" (", "\n(") for b in band_list]

    # Abb. 1: ein Panel pro Schlafstadium, x = Bänder, Boxen = Datensätze, gemeinsame y-Achse
    ymax = S[rel].max().max() * 1.05
    fig, ax = plt.subplots(1, len(order), figsize=(3.6 * len(order), 4), sharey=True)
    rng = np.random.default_rng(0)
    for a_, st in zip(ax, order):
        for i, b in enumerate(band_list):
            for j, g in enumerate(names):
                v = S[(S.group == g) & (S.stage == st)]["rel_" + b].values
                if not len(v): continue
                x0 = i + (j - (len(names) - 1) / 2) * 0.38
                a_.boxplot(v, positions=[x0], widths=.32, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=col[g], alpha=.45), medianprops=dict(color="k"))
                a_.scatter(x0 + rng.uniform(-.06, .06, len(v)), v, s=8, color=col[g], zorder=3)
            p = res.set_index(["stage", "band"]).p_bonf.get((st, b), np.nan)
            stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            if stars:
                top = S[(S.stage == st)]["rel_" + b].max()
                a_.text(i, top + ymax * 0.02, stars, ha="center", va="bottom", fontsize=12)
        a_.set_xticks(range(len(band_list))); a_.set_xticklabels(band_labels, fontsize=7)
        a_.set_title(st, fontsize=10); a_.set_ylim(0, ymax)
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel(f"relative Leistung (Anteil an {TOTAL[0]}-{TOTAL[1]} Hz)")
    for g in names:
        ax[0].plot([], [], color=col[g], lw=6, alpha=.5, label=f"{g} (n={S[S.group == g].subject.nunique()})")
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); fig.text(0.99, 0.01, "* p<0.05, ** p<0.01, *** p<0.001 (Welch-t, Bonferroni)", ha="right", fontsize=7); plt.savefig(f"{a.out}/band_power_boxplots.png", dpi=200); plt.close()

    # Abb. 2: mittlere PSD pro Stadium (log-log, Mittel über Subjekte, Bänder schattiert)
    fig, ax = plt.subplots(1, len(order), figsize=(3.4 * len(order), 3.4), sharey=True)
    for a_, st in zip(ax, order):
        for lo, hi in BANDS.values():
            a_.axvspan(lo, hi, color="grey", alpha=.05)
        for g in names:
            if (g, st) in psd:
                m = np.mean(psd[(g, st)], axis=0)
                a_.loglog(f[(f >= .5) & (f <= 40)], m[(f >= .5) & (f <= 40)], color=col[g], label=g)
        a_.set_title(st); a_.set_xlabel("Hz"); a_.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("PSD (µV²/Hz)"); ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.out}/mean_psd_by_stage.png", dpi=200); plt.close()
    plot_whitened_psd(f, psd, names, order, col, f"{a.out}/whitened_psd_by_stage.png")
    print(f"Fertig. Ergebnisse in {a.out}/")


if __name__ == "__main__":
    main()