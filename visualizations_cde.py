#!/usr/bin/env python3
"""
Cross-Dataset-Analyse von Konfusionsmatrizen für ein auf A trainiertes Modell.

Input:  zwei .npy-Dateien mit ABSOLUTEN Konfusionsmatrizen
        (Zeilen = wahres Label, Spalten = vorhergesagtes Label)
          - A -> A : Training auf A, Test auf A
          - A -> B : Training auf A, Test auf B

Erzeugte Abbildungen (PNG) + CSV-Zusammenfassungen:
  1. Drei Heatmaps: A->A | A->B | Differenz (zeilennormalisiert, mit Signifikanzmarkierung)
  2. Top-N veränderte True->Predicted-Paare (Diagonale und Off-Diagonale getrennt)
  3. Dumbbell-Plot je Klasse: Recall / Precision / F1 (A->A vs. A->B)
  4. Fehlerprofile pro wahrer Klasse (Verteilung der Vorhersagen)
  5. Total-Variation-Distanz je Klasse (mit Bootstrap-KI)

Unsicherheit: Bootstrap über Testfälle. Da nur die Konfusionsmatrix vorliegt, wird
jede Matrix multinomial aus ihren Zellhäufigkeiten neu gezogen (entspricht dem
Ziehen von Testfällen mit Zurücklegen).

Beispiel:
  python cross_dataset_confusion_analysis.py \
      --cm-aa cm_A_to_A.npy --cm-ab cm_A_to_B.npy \
      --name-a "Datensatz A" --name-b "Datensatz B" \
      --out-dir results --top-n 10
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


# ----------------------------------------------------------------------------
# Kernberechnungen
# ----------------------------------------------------------------------------
def row_normalize(cm):
    """Zeilenweise normalisieren. Zeilen mit Support 0 -> NaN. Funktioniert für (K,K) und (B,K,K)."""
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rows > 0, cm / rows, np.nan)
    return out


def class_metrics(cm):
    """Recall, Precision, F1, Support pro Klasse aus absoluter Konfusionsmatrix."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, np.nan)
    return {"recall": recall, "precision": precision, "f1": f1, "support": support}


def tvd_per_class(norm_a, norm_b):
    """Total-Variation-Distanz zwischen den Prediction-Profilen je wahrer Klasse."""
    return 0.5 * np.abs(norm_a - norm_b).sum(axis=-1)


def bootstrap(cm_aa, cm_ab, n_boot=2000, seed=0):
    """
    Multinomialer Bootstrap beider Matrizen (unabhängige Datensätze).
    Gibt Arrays der Bootstrap-Stichproben zurück: diff (B,K,K), tvd (B,K).
    """
    rng = np.random.default_rng(seed)
    k = cm_aa.shape[0]

    def draw(cm):
        n = int(cm.sum())
        p = cm.ravel().astype(float) / cm.sum()
        return rng.multinomial(n, p, size=n_boot).reshape(n_boot, k, k)

    boot_aa = row_normalize(draw(cm_aa))
    boot_ab = row_normalize(draw(cm_ab))
    diff = boot_ab - boot_aa
    tvd = tvd_per_class(boot_aa, boot_ab)
    return diff, tvd


def ci(samples, alpha=0.05):
    """Perzentil-Konfidenzintervall entlang Achse 0 (NaN-tolerant)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanpercentile(samples, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(samples, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# ----------------------------------------------------------------------------
# Plot-Helfer
# ----------------------------------------------------------------------------
def _text_color(value, vmin, vmax):
    """Weiß auf dunklem, schwarz auf hellem Hintergrund (für sequentielle Skala)."""
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
            ax.text(
                j, i, f"{norm[i, j] * 100:.2f}%\n({int(counts[i, j])})",
                ha="center", va="center", fontsize=8,
                color=_text_color(norm[i, j], 0, 1),
            )
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
            weight = "bold" if sig[i, j] else "normal"
            ax.text(
                j, i, f"{diff[i, j] * 100:+.1f}{star}",
                ha="center", va="center", fontsize=8, fontweight=weight,
                color="white" if abs(diff[i, j]) > 0.6 * lim else "black",
            )
    ax.set_xticks(range(k), labels)
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title, fontsize=10)
    return im


# ----------------------------------------------------------------------------
# Abbildungen
# ----------------------------------------------------------------------------
def fig_heatmaps(cm_aa, cm_ab, norm_aa, norm_ab, diff, sig, labels, a, b, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    im1 = heatmap_fractions(axes[0], norm_aa, cm_aa, labels,
                            f"Training {a} → Test {a}\n(Proportion per true class, number in parentheses)")
    heatmap_fractions(axes[1], norm_ab, cm_ab, labels,
                      f"Training {a} → Test {b}\n(Proportion per true class, number in parentheses)")
    im3 = heatmap_diff(axes[2], diff, sig, labels,
                       f"Difference (Test {b} − Test {a}) in Percent Points\n"
                       f"* = 95-%-Bootstrap-CI does not include 0")
    fig.colorbar(im1, ax=axes[:2], fraction=0.025, pad=0.02, label="Proportion")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="Δ Proportion (Test B − Test A)")
    fig.suptitle(f"Model trained on {a}: Confusion Matrices in Comparison", fontsize=13)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_top_changes(diff, lo, hi, support_ab, support_aa, labels, a, b, top_n, path):
    k = len(labels)
    diag, off = [], []
    for i in range(k):
        for j in range(k):
            if np.isnan(diff[i, j]):
                continue
            rec = dict(
                name=f"True {labels[i]} → Pred {labels[j]}",
                delta=diff[i, j] * 100, lo=lo[i, j] * 100, hi=hi[i, j] * 100,
            )
            (diag if i == j else off).append(rec)

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.5 * top_n + 2.5))
    panels = [
        (axes[0], diag, "Change of Correct Classifications (Recall)"),
        (axes[1], off, "Change of Specific Misclassifications (Precision)"),
    ]
    for ax, items, title in panels:
        items = sorted(items, key=lambda r: abs(r["delta"]), reverse=True)[:top_n][::-1]
        if not items:
            ax.set_visible(False)
            continue
        y = np.arange(len(items))
        vals = np.array([r["delta"] for r in items])
        err = np.array([[r["delta"] - r["lo"] for r in items],
                        [r["hi"] - r["delta"] for r in items]])
        colors = ["#c0392b" if v > 0 else "#2471a3" for v in vals]
        ax.barh(y, vals, xerr=np.abs(err), color=colors, ecolor="gray", capsize=3)
        ax.set_yticks(y, [r["name"] for r in items])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(f"Δ Percent Points (Test {b} − Test {a}), with 95-%-CI")
        ax.set_title(title, fontsize=10)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
    fig.suptitle(f"Top-{top_n} biggest Changes (Model trained on {a})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dumbbell(m_aa, m_ab, labels, a, b, path):
    k = len(labels)
    y = np.arange(k)
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.6 * k + 2.5), sharey=True)
    for ax, key, title in zip(axes, ["recall", "precision", "f1"],
                              ["Recall", "Precision", "F1"]):
        v_a, v_b = m_aa[key], m_ab[key]
        for i in range(k):
            if np.isnan(v_a[i]) or np.isnan(v_b[i]):
                continue
            ax.plot([v_a[i], v_b[i]], [i, i], color="gray", lw=2, zorder=1)
        ax.scatter(v_a, y, color="#2471a3", s=70, zorder=2, label=f"Test {a} (in-domain)")
        ax.scatter(v_b, y, color="#e67e22", s=70, zorder=2, label=f"Test {b}")
        for i in range(k):
            if np.isnan(v_a[i]) or np.isnan(v_b[i]):
                continue
            ax.text(max(v_a[i], v_b[i]) + 0.02, i, f"Δ {v_b[i] - v_a[i]:+.2f}",
                    va="center", fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [f"Class {l}\n(n={int(m_aa['support'][i])} | {int(m_ab['support'][i])})"
                           for i, l in enumerate(labels)])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Class-wise Generalization Gap – Model trained on {a}\n"
                 f"(n = Support Test {a} | Test {b})", fontsize=12)
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
        ax.bar(x - w / 2, np.nan_to_num(norm_aa[i]), w, color="#2471a3", label=f"Test {a}")
        ax.bar(x + w / 2, np.nan_to_num(norm_ab[i]), w, color="#e67e22", label=f"Test {b}")
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
    err = np.array([np.nan_to_num(tvd - lo), np.nan_to_num(hi - tvd)])
    ax.bar(x, np.nan_to_num(tvd), color="#8e44ad", yerr=np.abs(err), ecolor="gray", capsize=4)
    for xi, v in zip(x, tvd):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"Class {l}" for l in labels])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Total-Variation-Distance (TVD)")
    ax.set_title(f"Change of Prediction Profile per True Class\n"
                 f"Test {a} vs. Test {b} (Model trained on {a}; Error bars: 95%-CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Hauptfunktion
# ----------------------------------------------------------------------------
def analyze(cm_aa_path, cm_ab_path, name_a="A", name_b="B", out_dir="cm_analysis",
            class_labels=None, top_n=10, n_boot=2000, seed=0):
    cm_aa = np.load(cm_aa_path)
    cm_ab = np.load(cm_ab_path)

    for nm, cm in [("A->A", cm_aa), ("A->B", cm_ab)]:
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f"Matrix {nm} muss quadratisch sein, hat aber Shape {cm.shape}.")
    if cm_aa.shape != cm_ab.shape:
        raise ValueError(f"Beide Matrizen brauchen dieselbe Klassenanzahl: {cm_aa.shape} vs. {cm_ab.shape}.")

    k = cm_aa.shape[0]
    labels = [str(l) for l in (class_labels if class_labels is not None else range(k))]
    if len(labels) != k:
        raise ValueError(f"{len(labels)} Klassennamen übergeben, aber {k} Klassen in der Matrix.")

    os.makedirs(out_dir, exist_ok=True)

    # Zeilennormalisierung, Differenz, Metriken
    norm_aa, norm_ab = row_normalize(cm_aa), row_normalize(cm_ab)
    diff = norm_ab - norm_aa
    m_aa, m_ab = class_metrics(cm_aa), class_metrics(cm_ab)
    tvd = tvd_per_class(norm_aa, norm_ab)

    # Bootstrap
    boot_diff, boot_tvd = bootstrap(cm_aa, cm_ab, n_boot=n_boot, seed=seed)
    d_lo, d_hi = ci(boot_diff)
    t_lo, t_hi = ci(boot_tvd)
    sig = (d_lo > 0) | (d_hi < 0)

    # Abbildungen
    fig_heatmaps(cm_aa, cm_ab, norm_aa, norm_ab, diff, sig, labels, name_a, name_b,
                 os.path.join(out_dir, "1_heatmaps.png"))
    fig_top_changes(diff, d_lo, d_hi, m_ab["support"], m_aa["support"], labels,
                    name_a, name_b, top_n, os.path.join(out_dir, "2_top_changes.png"))
    fig_dumbbell(m_aa, m_ab, labels, name_a, name_b, os.path.join(out_dir, "3_dumbbell_metrics.png"))
    fig_error_profiles(norm_aa, norm_ab, labels, name_a, name_b,
                       os.path.join(out_dir, "4_error_profiles.png"))
    fig_tvd(tvd, t_lo, t_hi, labels, name_a, name_b, os.path.join(out_dir, "5_tvd.png"))

    # CSV-Zusammenfassungen
    per_class = pd.DataFrame({
        "class": labels,
        f"support_{name_a}": m_aa["support"].astype(int),
        f"support_{name_b}": m_ab["support"].astype(int),
        f"recall_{name_a}": m_aa["recall"], f"recall_{name_b}": m_ab["recall"],
        "recall_delta": m_ab["recall"] - m_aa["recall"],
        f"precision_{name_a}": m_aa["precision"], f"precision_{name_b}": m_ab["precision"],
        "precision_delta": m_ab["precision"] - m_aa["precision"],
        f"f1_{name_a}": m_aa["f1"], f"f1_{name_b}": m_ab["f1"],
        "f1_delta": m_ab["f1"] - m_aa["f1"],
        "tvd": tvd, "tvd_ci_low": t_lo, "tvd_ci_high": t_hi,
    })
    per_class.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    rows = []
    for i in range(k):
        for j in range(k):
            rows.append({
                "true": labels[i], "pred": labels[j],
                f"count_{name_a}": int(cm_aa[i, j]), f"count_{name_b}": int(cm_ab[i, j]),
                f"frac_{name_a}": norm_aa[i, j], f"frac_{name_b}": norm_ab[i, j],
                "delta": diff[i, j], "ci_low": d_lo[i, j], "ci_high": d_hi[i, j],
                "significant": bool(sig[i, j]),
            })
    pd.DataFrame(rows).sort_values("delta", key=np.abs, ascending=False).to_csv(
        os.path.join(out_dir, "cell_changes.csv"), index=False)

    # Kurzübersicht
    acc_aa = np.trace(cm_aa) / cm_aa.sum()
    acc_ab = np.trace(cm_ab) / cm_ab.sum()
    print(f"Accuracy Test {name_a}: {acc_aa:.3f} | Test {name_b}: {acc_ab:.3f} | Δ {acc_ab - acc_aa:+.3f}")
    print(per_class.round(3).to_string(index=False))
    print(f"\nResults saved in: {os.path.abspath(out_dir)}")
    return per_class


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cm-aa", required=True, help="Pfad zur .npy-Matrix Training A -> Test A (absolute Counts)")
    p.add_argument("--cm-ab", required=True, help="Pfad zur .npy-Matrix Training A -> Test B (absolute Counts)")
    p.add_argument("--name-a", default="A", help="Name von Datensatz A (für Titel/Beschriftungen)")
    p.add_argument("--name-b", default="B", help="Name von Datensatz B (für Titel/Beschriftungen)")
    p.add_argument("--out-dir", default="cm_analysis", help="Ausgabeordner")
    p.add_argument("--class-labels", nargs="*", default=None, help="Optionale Klassennamen (Standard: 0..K-1)")
    p.add_argument("--top-n", type=int, default=10, help="Anzahl Einträge im Top-N-Balkenplot")
    p.add_argument("--n-boot", type=int, default=2000, help="Anzahl Bootstrap-Wiederholungen")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    analyze(args.cm_aa, args.cm_ab, args.name_a, args.name_b, args.out_dir,
            args.class_labels, args.top_n, args.n_boot, args.seed)


if __name__ == "__main__":
    main()