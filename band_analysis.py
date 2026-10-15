"""
Bandleistungs-Analyse für Schlaf-EEG (.npz mit x[n_epochs, n_samples], y, fs).

Aufruf:
    python band_analysis.py /pfad/zu/ordner --out ergebnisse
    python band_analysis.py /pfad/zu/ordner --groups "ANPHY=EPCTL*" "DOD-H=*" --out ergebnisse

--groups: Name=Glob-Muster (Dateiname). Wird ein Muster zuerst gefunden, gewinnt es
          (Reihenfolge beachten: spezifische Muster zuerst, "*" zuletzt).
          Ohne --groups wird eine einzige Gruppe "all" gebildet.
"""
import argparse, fnmatch, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import mannwhitneyu

BANDS = {"Delta/K (1-3 Hz)": (1, 3), "Theta (4-7 Hz)": (4, 7), "Alpha (8-13 Hz)": (8, 13),
         "Spindel (12-14 Hz)": (12, 14), "Beta (13-40 Hz)": (13, 40)}
TOTAL = (1, 40)                       # Referenzbereich für relative Leistung
STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}   # Label 6 etc. wird verworfen
PALETTE = ["#1f77b4", "#e08a1e", "#2a9d8f", "#9b5de5"]


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


def group_of(fname, groups):
    for g, pat in groups:
        if fnmatch.fnmatch(fname, pat):
            return g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--out", default="band_out")
    ap.add_argument("--groups", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups = [tuple(g.split("=", 1)) for g in a.groups] if a.groups else [("all", "*")]

    rows, psd = [], {}
    for path in sorted(glob.glob(os.path.join(a.folder, "*.npz"))):
        fname = os.path.basename(path)
        fname_clean = re.sub(r"^[0-9a-f]{8}-\d+_", "", fname)   # Upload-Präfix entfernen
        grp = group_of(fname_clean, groups)
        if grp is None:
            continue
        df, f, P, y = load_file(path)
        df["group"], df["subject"] = grp, fname_clean[:-4]
        rows.append(df)
        for s, sn in STAGES.items():                              # mittlere PSD pro Subjekt/Stadium
            if (y == s).any():
                psd.setdefault((grp, sn), []).append(P[y == s].mean(0))
        print(f"{fname_clean}: {len(df)} Epochen -> Gruppe {grp}")
    E = pd.concat(rows, ignore_index=True)
    E = E[E.y.isin(STAGES)].copy()
    E["stage"] = E.y.map(STAGES)
    for b in BANDS:
        E["rel_" + b] = E[b] / E.total

    # Subjekt-Median pro Stadium (Subjekt = statistische Einheit)
    rel = ["rel_" + b for b in BANDS]
    S = E.groupby(["group", "subject", "stage"])[rel].median().reset_index()
    S.to_csv(f"{a.out}/subject_stage_medians.csv", index=False)
    summ = S.groupby(["group", "stage"])[rel].agg(["median", "std", "count"])
    summ.to_csv(f"{a.out}/group_summary.csv")

    # Gruppenvergleich (nur bei genau 2 Gruppen)
    names = [g for g, _ in groups]
    if len(names) == 2:
        r = []
        for st in STAGES.values():
            for b in BANDS:
                v1 = S[(S.group == names[0]) & (S.stage == st)]["rel_" + b]
                v2 = S[(S.group == names[1]) & (S.stage == st)]["rel_" + b]
                if len(v1) and len(v2):
                    u = mannwhitneyu(v1, v2)
                    r.append(dict(stage=st, band=b, **{f"{names[0]}_median": v1.median(), f"{names[1]}_median": v2.median()},
                                  p=u.pvalue, cliffs_delta=2 * u.statistic / (len(v1) * len(v2)) - 1))
        pd.DataFrame(r).to_csv(f"{a.out}/group_comparison.csv", index=False)

    col = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(names)}
    order = list(STAGES.values())

    # Abb. 1: Boxplots relative Bandleistung (Punkte = Subjekte)
    fig, ax = plt.subplots(1, len(BANDS), figsize=(3.4 * len(BANDS), 3.8))
    rng = np.random.default_rng(0)
    for a_, b in zip(ax, BANDS):
        for i, st in enumerate(order):
            for j, g in enumerate(names):
                v = S[(S.group == g) & (S.stage == st)]["rel_" + b].values
                if not len(v): continue
                x0 = i + (j - (len(names) - 1) / 2) * 0.38
                a_.boxplot(v, positions=[x0], widths=.32, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=col[g], alpha=.45), medianprops=dict(color="k"))
                a_.scatter(x0 + rng.uniform(-.06, .06, len(v)), v, s=8, color=col[g], zorder=3)
        a_.set_xticks(range(len(order))); a_.set_xticklabels(order); a_.set_title(b, fontsize=10)
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel(f"relative Leistung (Anteil an {TOTAL[0]}-{TOTAL[1]} Hz)")
    for g in names:
        ax[0].plot([], [], color=col[g], lw=6, alpha=.5, label=f"{g} (n={S[S.group==g].subject.nunique()})")
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.out}/band_power_boxplots.png", dpi=200); plt.close()

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
    print(f"Fertig. Ergebnisse in {a.out}/")


if __name__ == "__main__":
    main()