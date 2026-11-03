#!/usr/bin/env python3
"""
Cross-Dataset-Analyse von Konfusionsmatrizen für ein auf A trainiertes Modell.

Input:  zwei .npz-Dateien mit Konfusionsmatrizen PRO PROBAND (Key "cm_per_subject",
        Form (S, K, K), Zeilen = wahres Label, Spalten = vorhergesagtes Label),
        erzeugt von cross_eval_test.py (<name>_persubject.npz).
          - A -> A : Training auf A, Test auf A
          - A -> B : Training auf A, Test auf B
        (Alternativ eine .npy mit einem (S,K,K)-Array.)

Unsicherheit: CLUSTER-BOOTSTRAP ÜBER PROBANDEN. In jeder Wiederholung werden die
Probanden je Datensatz unabhängig mit Zurücklegen gezogen, ihre Matrizen summiert
und erst danach normalisiert / Metriken berechnet. Die Probanden (nicht die
Epochen) sind die unabhängigen Einheiten.

Signifikanz:
  - Zell-Differenzen (Heatmap): zweiseitiger Bootstrap-p-Wert, Holm-Korrektur über
    alle K x K Zellen. Stern = Holm-adjustiertes p < alpha.
  - Dumbbell-Plot: Bootstrap-KIs je Datensatz; Stern am Delta, wenn Holm-adjustiertes
    p (über Klassen x Metriken) < alpha.
  - Die in den Balkenplots gezeigten KIs sind UNKORRIGIERTE Perzentil-KIs.
  - TVD ist rein deskriptiv (bei endlichen Stichproben nach oben verzerrt).

Erzeugte Dateien (PNG + CSV):
  1_heatmaps.png, 2_top_changes.png, 3_dumbbell_metrics.png, 4_error_profiles.png,
  5_tvd.png, per_class_metrics.csv, cell_changes.csv, overall_metrics.csv

Beispiel:
  python visualizations_cde.py \
      --cm-aa results/anphy_to_anphy_persubject.npz \
      --cm-ab results/anphy_to_dodh_persubject.npz \
      --name-a ANPHY --name-b DOD-H --n-boot 10000 --seed 0 \
      --out-dir results/cde_cluster --top-n 10
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

C_A, C_B = "#2471a3", "#e67e22"


# ----------------------------------------------------------------------------
# Laden
# ----------------------------------------------------------------------------
def load_ps(path):
    """Lädt Per-Subject-Konfusionsmatrizen, Form (S, K, K)."""
    d = np.load(path, allow_pickle=True)
    M = d["cm_per_subject"] if hasattr(d, "files") else d
    M = np.asarray(M)
    if M.ndim != 3 or M.shape[1] != M.shape[2]:
        raise ValueError(f"{path}: erwarte Form (S,K,K), erhalten {M.shape}. "
                         "Für gepoolte (K,K)-Matrizen ist kein Cluster-Bootstrap möglich.")
    return M


# ----------------------------------------------------------------------------
# Kernberechnungen (alle funktionieren für (K,K) und (B,K,K))
# ----------------------------------------------------------------------------
def row_normalize(cm):
    """Zeilenweise normalisieren. Zeilen mit Support 0 -> NaN."""
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rows > 0, cm / rows, np.nan)


def class_metrics(cm):
    """Recall, Precision, F1, Support je Klasse (Achse -1 = Klasse)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, np.nan)
    return {"recall": recall, "precision": precision, "f1": f1, "support": support}


def overall_metrics(cm):
    """Accuracy, Macro-F1, Cohens Kappa (auch für (B,K,K))."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum(axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.trace(cm, axis1=-2, axis2=-1) / n
        pe = (cm.sum(axis=-1) * cm.sum(axis=-2)).sum(axis=-1) / n ** 2
        kappa = (acc - pe) / (1 - pe)
    macro_f1 = np.nanmean(class_metrics(cm)["f1"], axis=-1)
    return {"accuracy": acc, "macro_f1": macro_f1, "kappa": kappa}


def tvd_per_class(norm_a, norm_b):
    """Total-Variation-Distanz zwischen den Prediction-Profilen je wahrer Klasse."""
    return 0.5 * np.abs(norm_a - norm_b).sum(axis=-1)


def cluster_bootstrap(M_aa, M_ab, n_boot=10000, seed=0):
    """
    Cluster-Bootstrap über Probanden. Beide Datensätze enthalten verschiedene
    Personen und werden unabhängig gezogen. Gibt die summierten Bootstrap-
    Matrizen (B,K,K) zurück.
    """
    rng = np.random.default_rng(seed)

    def draw(M):
        s = M.shape[0]
        w = rng.multinomial(s, np.full(s, 1.0 / s), size=n_boot)   # (B,S) Ziehungshäufigkeiten
        return np.einsum("bs,skl->bkl", w, M.astype(float))

    return draw(M_aa), draw(M_ab)


def ci(samples, alpha=0.05):
    """Perzentil-KI entlang Achse 0 (NaN-tolerant)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanpercentile(samples, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(samples, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def boot_p(samples):
    """Zweiseitiger Bootstrap-p-Wert für H0: Differenz = 0 (Achse 0 = Wiederholungen)."""
    s = np.asarray(samples, dtype=float)
    valid = np.isfinite(s)
    n = valid.sum(axis=0)
    le = ((s <= 0) & valid).sum(axis=0)
    ge = ((s >= 0) & valid).sum(axis=0)
    p = (2 * np.minimum(le, ge) + 1) / (n + 1)
    return np.where(n > 0, np.minimum(p, 1.0), np.nan)


def holm(p):
    """Holm-Adjustierung; NaNs werden ignoriert. Gibt Array gleicher Form zurück."""
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    out = np.full(flat.shape, np.nan)
    idx = np.where(np.isfinite(flat))[0]
    m = len(idx)
    if m:
        order = idx[np.argsort(flat[idx])]
        adj = np.minimum(1.0, (m - np.arange(m)) * flat[order])
        out[order] = np.maximum.accumulate(adj)
    return out.reshape(p.shape)


# ----------------------------------------------------------------------------
# Plot-Helfer
# ----------------------------------------------------------------------------
def _text_color(value, vmin, vmax):
    if np.isnan(value):
        return "black"
    return "white" if (value - vmin) / (vmax - vmin + 1e-12) > 0.6 else "black"


def heatmap_fractions(ax, norm, counts, labels, title, cmap="Blues"):
    k = len(labels)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if np.isnan(norm[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8)
                continue
            ax.text(j, i, f"{norm[i, j] * 100:.2f}%\n({int(counts[i, j])})",
                    ha="center", va="center", fontsize=8,
                    color=_text_color(norm[i, j], 0, 1))
    ax.set_xticks(range(k), labels)
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title, fontsize=10)
    return im


def heatmap_diff(ax, diff, sig, labels, title):
    k = len(labels)
    finite = diff[np.isfinite(diff)]
    lim = max(0.05, float(np.max(np.abs(finite))) if finite.size else 0.05)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    im = ax.imshow(diff, cmap="RdBu_r", norm=norm)
    for i in range(k):
        for j in range(k):
            if np.isnan(diff[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8)
                continue
            star = "*" if sig[i, j] else ""
            ax.text(j, i, f"{diff[i, j] * 100:+.1f}{star}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if sig[i, j] else "normal",
                    color="white" if abs(diff[i, j]) > 0.6 * lim else "black")
    ax.set_xticks(range(k), labels)
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title, fontsize=10)
    return im


# ----------------------------------------------------------------------------
# Abbildungen
# ----------------------------------------------------------------------------
def fig_heatmaps(cm_aa, cm_ab, norm_aa, norm_ab, diff, sig, labels, a, b, n_a, n_b, alpha, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    im1 = heatmap_fractions(axes[0], norm_aa, cm_aa, labels,
                            f"Training {a} → Test {a} (n = {n_a} subjects)\n"
                            "(Proportion per true class, epochs in parentheses)")
    heatmap_fractions(axes[1], norm_ab, cm_ab, labels,
                      f"Training {a} → Test {b} (n = {n_b} subjects)\n"
                      "(Proportion per true class, epochs in parentheses)")
    im3 = heatmap_diff(axes[2], diff, sig, labels,
                       f"Difference (Test {b} − Test {a}) in Percent Points\n"
                       f"* = Holm-adjusted subject-cluster-bootstrap p < {alpha:g}")
    fig.colorbar(im1, ax=axes[:2], fraction=0.025, pad=0.02, label="Proportion")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label=f"Δ Proportion (Test {b} − Test {a})")
    fig.suptitle(f"Model trained on {a}: Confusion Matrices in Comparison", fontsize=13)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_top_changes(diff, lo, hi, sig, labels, a, b, top_n, alpha, path):
    k = len(labels)
    diag, off = [], []
    for i in range(k):
        for j in range(k):
            if np.isnan(diff[i, j]):
                continue
            rec = dict(name=f"True {labels[i]} → Pred {labels[j]}",
                       delta=diff[i, j] * 100, lo=lo[i, j] * 100, hi=hi[i, j] * 100,
                       sig=bool(sig[i, j]))
            (diag if i == j else off).append(rec)

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * top_n + 2.5))
    panels = [
        (axes[0], diag, "Change of Correct Classifications (Recall, diagonal)"),
        (axes[1], off, "Change of Misclassification Rates (off-diagonal)"),
    ]
    for ax, items, title in panels:
        items = sorted(items, key=lambda r: abs(r["delta"]), reverse=True)[:top_n][::-1]
        if not items:
            ax.set_visible(False)
            continue
        y = np.arange(len(items))
        vals = np.array([r["delta"] for r in items])
        err = np.array([[max(r["delta"] - r["lo"], 0) for r in items],
                        [max(r["hi"] - r["delta"], 0) for r in items]])
        colors = ["#c0392b" if v > 0 else "#2471a3" for v in vals]
        ax.barh(y, vals, xerr=err, color=colors, ecolor="gray", capsize=3)
        ax.set_yticks(y, [r["name"] + (" *" if r["sig"] else "") for r in items])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(f"Δ Percent Points (Test {b} − Test {a}); bars: uncorrected 95-% CI")
        ax.set_title(title, fontsize=10)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
    fig.suptitle(f"Top-{top_n} biggest Changes (Model trained on {a}); "
                 f"* = Holm-adjusted p < {alpha:g}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dumbbell(pt_a, pt_b, ci_a, ci_b, delta_sig, support_a, support_b,
                 labels, a, b, alpha, path):
    k = len(labels)
    y = np.arange(k)
    keys = ["recall", "precision", "f1"]
    titles = ["Recall", "Precision", "F1"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.6 * k + 2.8), sharey=True)
    for ax, key, title in zip(axes, keys, titles):
        v_a, v_b = pt_a[key], pt_b[key]
        for i in range(k):
            if np.isnan(v_a[i]) or np.isnan(v_b[i]):
                continue
            ax.plot([v_a[i], v_b[i]], [i, i], color="gray", lw=2, zorder=1)
        for v, (lo, hi), col, lab in [(v_a, ci_a[key], C_A, f"Test {a} (in-domain)"),
                                      (v_b, ci_b[key], C_B, f"Test {b}")]:
            ax.errorbar(v, y, xerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)],
                        fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=2, alpha=0.8)
            ax.scatter(v, y, color=col, s=70, zorder=3, label=lab)
        for i in range(k):
            if np.isnan(v_a[i]) or np.isnan(v_b[i]):
                continue
            top = max(np.nan_to_num(ci_a[key][1][i]), np.nan_to_num(ci_b[key][1][i]),
                      v_a[i], v_b[i])
            star = "*" if delta_sig[key][i] else ""
            ax.text(min(top, 1.0) + 0.02, i, f"Δ {v_b[i] - v_a[i]:+.2f}{star}",
                    va="center", fontsize=8,
                    fontweight="bold" if star else "normal")
        ax.set_xlim(0, 1.2)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [f"Class {l}\n(n={int(support_a[i])} | {int(support_b[i])})"
                           for i, l in enumerate(labels)])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Class-wise Generalization Gap – Model trained on {a}\n"
                 f"(n = epochs Test {a} | Test {b}; bars: 95-% subject-bootstrap CI; "
                 f"* = Holm-adjusted p < {alpha:g})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_error_profiles(norm_aa, norm_ab, labels, a, b, path):
    k = len(labels)
    ncols = min(k, 3)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             sharey=True, squeeze=False)
    x = np.arange(k)
    w = 0.38
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= k:
            ax.set_visible(False)
            continue
        ax.bar(x - w / 2, np.nan_to_num(norm_aa[i]), w, color=C_A, label=f"Test {a}")
        ax.bar(x + w / 2, np.nan_to_num(norm_ab[i]), w, color=C_B, label=f"Test {b}")
        ax.set_xticks(x, labels)
        ax.set_title(f"True Class {labels[i]}", fontsize=10)
        ax.set_xlabel("Predicted Class")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Proportion")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Error Profiles per True Class – Model trained on {a}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_tvd(tvd, lo, hi, labels, a, b, path):
    k = len(labels)
    fig, ax = plt.subplots(figsize=(1.2 * k + 3, 4.5))
    x = np.arange(k)
    err = np.array([np.clip(np.nan_to_num(tvd - lo), 0, None),
                    np.clip(np.nan_to_num(hi - tvd), 0, None)])
    ax.bar(x, np.nan_to_num(tvd), color="#8e44ad", yerr=err, ecolor="gray", capsize=4)
    for xi, v in zip(x, tvd):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"Class {l}" for l in labels])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Total-Variation-Distance (TVD)")
    ax.set_title(f"Change of Prediction Profile per True Class (descriptive)\n"
                 f"Test {a} vs. Test {b} (Model trained on {a}; bars: 95-% subject-bootstrap CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Hauptfunktion
# ----------------------------------------------------------------------------
def analyze(cm_aa_path, cm_ab_path, name_a="A", name_b="B", out_dir="cm_analysis",
            class_labels=None, top_n=10, n_boot=10000, seed=0, alpha=0.05):
    M_aa, M_ab = load_ps(cm_aa_path), load_ps(cm_ab_path)
    if M_aa.shape[1:] != M_ab.shape[1:]:
        raise ValueError(f"Beide Datensätze brauchen dieselbe Klassenanzahl: "
                         f"{M_aa.shape[1:]} vs. {M_ab.shape[1:]}.")
    n_a, n_b = M_aa.shape[0], M_ab.shape[0]
    cm_aa, cm_ab = M_aa.sum(0), M_ab.sum(0)          # gepoolt für Punktschätzungen
    k = cm_aa.shape[0]
    labels = [str(l) for l in (class_labels if class_labels is not None else range(k))]
    if len(labels) != k:
        raise ValueError(f"{len(labels)} Klassennamen übergeben, aber {k} Klassen in der Matrix.")
    if min(n_a, n_b) < 10:
        warnings.warn(f"Nur {min(n_a, n_b)} Probanden in einem Datensatz: Bootstrap-Intervalle sind unzuverlässig.")

    os.makedirs(out_dir, exist_ok=True)

    # Punktschätzungen (gepoolt)
    norm_aa, norm_ab = row_normalize(cm_aa), row_normalize(cm_ab)
    diff = norm_ab - norm_aa
    pt_a, pt_b = class_metrics(cm_aa), class_metrics(cm_ab)
    ov_a, ov_b = overall_metrics(cm_aa), overall_metrics(cm_ab)
    tvd = tvd_per_class(norm_aa, norm_ab)

    # Cluster-Bootstrap über Probanden
    B_aa, B_ab = cluster_bootstrap(M_aa, M_ab, n_boot=n_boot, seed=seed)
    bn_aa, bn_ab = row_normalize(B_aa), row_normalize(B_ab)
    boot_diff = bn_ab - bn_aa
    boot_tvd = tvd_per_class(bn_aa, bn_ab)
    bm_a, bm_b = class_metrics(B_aa), class_metrics(B_ab)
    bo_a, bo_b = overall_metrics(B_aa), overall_metrics(B_ab)

    # Zellen: KI, p, Holm
    d_lo, d_hi = ci(boot_diff, alpha)
    p_cell = boot_p(boot_diff)
    p_cell_holm = holm(p_cell)
    sig = np.nan_to_num(p_cell_holm, nan=1.0) < alpha
    t_lo, t_hi = ci(boot_tvd, alpha)

    # Klassenmetriken: KI je Datensatz, Delta-p mit Holm über Klassen x Metriken
    keys = ["recall", "precision", "f1"]
    ci_a = {key: ci(bm_a[key], alpha) for key in keys}
    ci_b = {key: ci(bm_b[key], alpha) for key in keys}
    dm_lo, dm_hi, dm_p = {}, {}, {}
    for key in keys:
        d = bm_b[key] - bm_a[key]
        dm_lo[key], dm_hi[key] = ci(d, alpha)
        dm_p[key] = boot_p(d)
    all_p = holm(np.stack([dm_p[key] for key in keys]))           # (3,K)
    dm_p_holm = {key: all_p[i] for i, key in enumerate(keys)}
    delta_sig = {key: np.nan_to_num(dm_p_holm[key], nan=1.0) < alpha for key in keys}

    # Abbildungen
    fig_heatmaps(cm_aa, cm_ab, norm_aa, norm_ab, diff, sig, labels, name_a, name_b,
                 n_a, n_b, alpha, os.path.join(out_dir, "1_heatmaps.png"))
    fig_top_changes(diff, d_lo, d_hi, sig, labels, name_a, name_b, top_n, alpha,
                    os.path.join(out_dir, "2_top_changes.png"))
    fig_dumbbell(pt_a, pt_b, ci_a, ci_b, delta_sig, pt_a["support"], pt_b["support"],
                 labels, name_a, name_b, alpha, os.path.join(out_dir, "3_dumbbell_metrics.png"))
    fig_error_profiles(norm_aa, norm_ab, labels, name_a, name_b,
                       os.path.join(out_dir, "4_error_profiles.png"))
    fig_tvd(tvd, t_lo, t_hi, labels, name_a, name_b, os.path.join(out_dir, "5_tvd.png"))

    # CSV: je Klasse
    cols = {"class": labels,
            f"support_{name_a}": pt_a["support"].astype(int),
            f"support_{name_b}": pt_b["support"].astype(int)}
    for key in keys:
        cols.update({
            f"{key}_{name_a}": pt_a[key], f"{key}_{name_a}_ci_low": ci_a[key][0],
            f"{key}_{name_a}_ci_high": ci_a[key][1],
            f"{key}_{name_b}": pt_b[key], f"{key}_{name_b}_ci_low": ci_b[key][0],
            f"{key}_{name_b}_ci_high": ci_b[key][1],
            f"{key}_delta": pt_b[key] - pt_a[key],
            f"{key}_delta_ci_low": dm_lo[key], f"{key}_delta_ci_high": dm_hi[key],
            f"{key}_delta_p": dm_p[key], f"{key}_delta_p_holm": dm_p_holm[key],
        })
    cols.update({"tvd": tvd, "tvd_ci_low": t_lo, "tvd_ci_high": t_hi})
    per_class = pd.DataFrame(cols)
    per_class.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    # CSV: je Zelle
    rows = []
    for i in range(k):
        for j in range(k):
            rows.append({
                "true": labels[i], "pred": labels[j],
                f"count_{name_a}": int(cm_aa[i, j]), f"count_{name_b}": int(cm_ab[i, j]),
                f"frac_{name_a}": norm_aa[i, j], f"frac_{name_b}": norm_ab[i, j],
                "delta": diff[i, j], "ci_low": d_lo[i, j], "ci_high": d_hi[i, j],
                "p_boot": p_cell[i, j], "p_holm": p_cell_holm[i, j],
                "ci_excludes_0_uncorrected": bool(d_lo[i, j] > 0 or d_hi[i, j] < 0),
                "significant_holm": bool(sig[i, j]),
            })
    pd.DataFrame(rows).sort_values("delta", key=np.abs, ascending=False).to_csv(
        os.path.join(out_dir, "cell_changes.csv"), index=False)

    # CSV: gesamt
    ov_rows = []
    for key in ["accuracy", "macro_f1", "kappa"]:
        lo_a, hi_a = ci(bo_a[key], alpha)
        lo_b, hi_b = ci(bo_b[key], alpha)
        d = bo_b[key] - bo_a[key]
        lo_d, hi_d = ci(d, alpha)
        ov_rows.append({"metric": key,
                        name_a: float(ov_a[key]), f"{name_a}_ci_low": lo_a, f"{name_a}_ci_high": hi_a,
                        name_b: float(ov_b[key]), f"{name_b}_ci_low": lo_b, f"{name_b}_ci_high": hi_b,
                        "delta": float(ov_b[key] - ov_a[key]), "delta_ci_low": lo_d,
                        "delta_ci_high": hi_d, "delta_p_boot": float(boot_p(d))})
    overall = pd.DataFrame(ov_rows)
    overall.to_csv(os.path.join(out_dir, "overall_metrics.csv"), index=False)

    # Kurzübersicht
    print(f"Probanden: {name_a} n={n_a}, {name_b} n={n_b} | Bootstrap: {n_boot} x, seed {seed}")
    print(overall.round(3).to_string(index=False))
    print(per_class[["class", f"support_{name_a}", f"support_{name_b}"] +
                    [f"{key}_delta" for key in keys] + [f"{key}_delta_p_holm" for key in keys] +
                    ["tvd"]].round(3).to_string(index=False))
    print(f"\nSignifikante Zellen (Holm, alpha={alpha:g}): {int(sig.sum())} von {int(np.isfinite(p_cell).sum())}")
    print(f"Results saved in: {os.path.abspath(out_dir)}")
    return per_class


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cm-aa", required=True,
                   help="Pfad zur Per-Subject-.npz Training A -> Test A (Key 'cm_per_subject', (S,K,K))")
    p.add_argument("--cm-ab", required=True,
                   help="Pfad zur Per-Subject-.npz Training A -> Test B (Key 'cm_per_subject', (S,K,K))")
    p.add_argument("--name-a", default="A", help="Name von Datensatz A")
    p.add_argument("--name-b", default="B", help="Name von Datensatz B")
    p.add_argument("--out-dir", default="cm_analysis", help="Ausgabeordner")
    p.add_argument("--class-labels", nargs="*", default=None, help="Optionale Klassennamen (Standard: 0..K-1)")
    p.add_argument("--top-n", type=int, default=10, help="Anzahl Einträge im Top-N-Balkenplot")
    p.add_argument("--n-boot", type=int, default=10000, help="Anzahl Bootstrap-Wiederholungen")
    p.add_argument("--alpha", type=float, default=0.05, help="Signifikanzniveau (Holm) bzw. 1 - KI-Niveau")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    analyze(args.cm_aa, args.cm_ab, args.name_a, args.name_b, args.out_dir,
            args.class_labels, args.top_n, args.n_boot, args.seed, args.alpha)


if __name__ == "__main__":
    main()