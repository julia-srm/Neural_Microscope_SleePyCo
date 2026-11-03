#!/usr/bin/env python3
"""
Gelernte Faltungskerne zweier Modelle (z. B. ANPHY-5-Klassen und DOD-H-5-Klassen) visualisieren
und per normalisierter Kreuzkorrelation (NCC) vergleichen.

Die Gewichte werden direkt aus den Checkpoints gelesen (kein Modellcode nötig).

Normalisierte Kreuzkorrelation zweier Kerne a, b (über alle Eingangskanäle summiert):
    NCC(a, b)[tau] = sum_t a[t + tau] * b[t] / (||a|| * ||b||)      in [-1, 1]
Pro Kernpaar wird das Maximum über die Verschiebung tau (|tau| <= --max-lag) genommen.
So gelten zeitlich verschobene Kopien desselben Filters als ähnlich.
Mit --abs zählt auch ein vorzeichenumgekehrter Filter als gleich (Beträge).

Ausgaben (in --out-dir):
  1_gallery_<A>.png / 1_gallery_<B>.png : alle Kerne als Kurven, sortiert nach Spitzenfrequenz
  2_spectra.png                          : Amplitudenspektren aller Kerne + Histogramm der Spitzenfrequenzen
  3_xcorr_matrix.png                     : NCC-Matrix A x B, Spalten nach optimaler Zuordnung sortiert
  4_matched_pairs.png                    : beste und schlechteste zugeordnete Paare, lag-ausgerichtet
  5_reference.png                        : Ähnlichkeit zugeordneter Kerne: A<->B, A<->A (andere Folds),
                                           B<->B (andere Folds), zufällige Kerne
  matches_fold<k>.csv, reference.csv

Beispiel (erste Faltungsschicht, Eingabe mit 100 Hz):
  python kernel_xcorr.py --list-layers --ckpt-a checkpoints/anphy
  python kernel_xcorr.py --ckpt-a checkpoints/anphy --ckpt-b checkpoints/dodh \
      --name-a ANPHY --name-b DOD-H --out-dir results/kernels

      --layer "layer name" --sr 0,100 (sample rate)
"""

import argparse
import glob
import itertools
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_A, C_B = '#2471a3', '#e67e22'


# ----------------------------------------------------------------------------
# Checkpoints / Kerne laden
# ----------------------------------------------------------------------------
def load_state(path: str) -> Dict[str, np.ndarray]:
    import torch
    try:
        raw = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        raw = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(raw, dict):
        for key in ('state_dict', 'model_state_dict'):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    out = {}
    for k, v in raw.items():
        if hasattr(v, 'detach'):
            out[k[7:] if k.startswith('module.') else k] = v.detach().cpu().numpy()
    return out


def conv_layers(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Alle Faltungsgewichte als (out, in, k)-Arrays."""
    res = {}
    for k, v in state.items():
        if not k.endswith('weight'):
            continue
        if v.ndim == 3:
            res[k] = v
        elif v.ndim == 4 and 1 in v.shape[2:]:
            res[k] = v.reshape(v.shape[0], v.shape[1], -1)
    return res


def pick_layer(layers: Dict[str, np.ndarray], layer: str = None) -> str:
    if not layers:
        raise ValueError('Keine Conv-Gewichte im Checkpoint gefunden.')
    if layer is None:
        return next(iter(layers))
    if layer in layers:
        return layer
    hits = [k for k in layers if layer in k]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f'Layer "{layer}" nicht eindeutig/gefunden. Treffer: {hits or list(layers)}')


def find_checkpoints(spec: str) -> Dict[int, str]:
    if os.path.isdir(spec):
        paths = sorted(glob.glob(os.path.join(spec, 'ckpt_fold-*.pth')))
    elif any(ch in spec for ch in '*?['):
        paths = sorted(glob.glob(spec))
    else:
        paths = [spec]
    out = {}
    for p in paths:
        m = re.search(r'fold[-_]?(\d+)', os.path.basename(p))
        if not m:
            raise ValueError(f'Fold-Nummer im Dateinamen nicht erkennbar: {p}')
        out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f'Keine Checkpoints gefunden für {spec}')
    return out


# ----------------------------------------------------------------------------
# Kreuzkorrelation
# ----------------------------------------------------------------------------
def xcorr_matrix(A: np.ndarray, B: np.ndarray, max_lag: int = None, use_abs: bool = False):
    """Normalisierte Kreuzkorrelation aller Kernpaare.

    A: (Ca, I, ka), B: (Cb, I, kb). Rückgabe (jeweils (Ca, Cb)):
      score : maximale NCC über |tau| <= max_lag (bei use_abs: maximaler Betrag)
      lag   : zugehöriges tau (A ist gegenüber B um tau Samples nach links verschoben, d. h. a[t+tau] ~ b[t])
      sign  : +1 / -1 (Vorzeichen der NCC am Maximum)
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f'Eingangskanäle verschieden: {A.shape[1]} vs. {B.shape[1]}')
    ka, kb = A.shape[-1], B.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(ka + kb - 1)))
    FA, FB = np.fft.rfft(A, nfft, axis=-1), np.fft.rfft(B, nfft, axis=-1)
    cross = np.einsum('aif,bif->abf', FA, np.conj(FB))
    cc = np.fft.irfft(cross, nfft, axis=-1)                 # Index = tau mod nfft
    lags = np.arange(-(kb - 1), ka)
    if max_lag is None:
        max_lag = max(ka, kb) // 4
    keep = np.abs(lags) <= max_lag
    lags = lags[keep]
    cc = cc[..., lags % nfft]

    na = np.sqrt((A ** 2).sum(axis=(1, 2)))
    nb = np.sqrt((B ** 2).sum(axis=(1, 2)))
    den = na[:, None, None] * nb[None, :, None]
    ncc = np.divide(cc, den, out=np.zeros_like(cc), where=den > 0)

    key = np.abs(ncc) if use_abs else ncc
    best = key.argmax(axis=-1)
    take = np.take_along_axis(ncc, best[..., None], axis=-1)[..., 0]
    score = np.abs(take) if use_abs else take
    return score, lags[best], np.where(take >= 0, 1, -1)


def match_kernels(score: np.ndarray):
    """Optimale 1:1-Zuordnung (ungarisch); Rückgabe (Zeilenindizes, Spaltenindizes)."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(-score)
    except ImportError:
        rows, cols, used_r, used_c = [], [], set(), set()
        for flat in np.argsort(-score, axis=None):
            i, j = np.unravel_index(flat, score.shape)
            if i in used_r or j in used_c:
                continue
            rows.append(i); cols.append(j); used_r.add(i); used_c.add(j)
        return np.array(rows), np.array(cols)


def matched_mean(A, B, max_lag, use_abs):
    score, _, _ = xcorr_matrix(A, B, max_lag, use_abs)
    r, c = match_kernels(score)
    return float(score[r, c].mean())


# ----------------------------------------------------------------------------
# Spektrum
# ----------------------------------------------------------------------------
def amplitude_spectrum(K: np.ndarray, n: int = 256):
    return np.abs(np.fft.rfft(K, n=n, axis=-1)).mean(axis=1)   # (C, F), über Eingangskanäle gemittelt


def peak_frequency(K: np.ndarray, sr: float, n: int = 256):
    S = amplitude_spectrum(K, n)
    return np.fft.rfftfreq(n, d=1.0 / sr)[S.argmax(axis=1)]


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_gallery(K, peak, name, out_path, ncols=8, color=C_A, layer=''):
    order = np.argsort(peak)
    C = len(K)
    nrows = int(np.ceil(C / ncols))
    w = K.mean(axis=1)                                           # über Eingangskanäle gemittelt
    lim = np.abs(w).max() * 1.1
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.2 * nrows + 0.8), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for pos, idx in enumerate(order):
        ax = axes[pos // ncols][pos % ncols]
        ax.axis('on')
        ax.axhline(0, color='0.8', lw=0.5)
        ax.plot(w[idx], color=color, lw=1)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'#{idx} ({peak[idx]:.1f})', fontsize=6, pad=1)
    fig.suptitle(f'{name}: {C} Kerne ({layer}), sortiert nach Spitzenfrequenz (in Klammern)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spectra(Ka, Kb, name_a, name_b, sr, out_path, nfft=256):
    unit = 'Hz' if sr != 1 else 'Zyklen/Sample'
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for K, name, col in [(Ka, name_a, C_A), (Kb, name_b, C_B)]:
        S = amplitude_spectrum(K, nfft)
        S = S / np.maximum(S.max(axis=1, keepdims=True), 1e-12)
        axes[0].plot(freqs, S.T, color=col, alpha=0.15, lw=0.8)
        axes[0].plot(freqs, S.mean(axis=0), color=col, lw=2.5, label=f'{name} (Mittel)')
        axes[1].hist(peak_frequency(K, sr, nfft), bins=np.linspace(0, freqs[-1], 26), alpha=0.55, color=col,
                     label=name)
    axes[0].set_xlabel(f'Frequenz [{unit}]'); axes[0].set_ylabel('Amplitude (je Kern auf 1 normiert)')
    axes[0].set_title('Amplitudenspektren der Kerne'); axes[0].legend(frameon=False)
    axes[1].set_xlabel(f'Spitzenfrequenz [{unit}]'); axes[1].set_ylabel('Anzahl Kerne')
    axes[1].set_title('Verteilung der Spitzenfrequenzen'); axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_xcorr_matrix(score, r, c, peak_a, name_a, name_b, use_abs, out_path):
    idx = np.argsort(peak_a[r])                                   # A nach Spitzenfrequenz, B folgt der Zuordnung
    rows = np.concatenate([r[idx], np.setdiff1d(np.arange(score.shape[0]), r)]).astype(int)
    cols = np.concatenate([c[idx], np.setdiff1d(np.arange(score.shape[1]), c)]).astype(int)
    M = score[np.ix_(rows, cols)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap='viridis' if use_abs else 'RdBu_r', vmin=0 if use_abs else -1, vmax=1, aspect='auto')
    ax.set_xlabel(f'{name_b}-Kern (nach Zuordnung sortiert)')
    ax.set_ylabel(f'{name_a}-Kern (nach Spitzenfrequenz sortiert)')
    ax.set_title(f'Max. normalisierte Kreuzkorrelation{" (Betrag)" if use_abs else ""}\n'
                 f'Diagonale = optimal zugeordnete Paare, mittlere Ähnlichkeit {score[r, c].mean():.2f}',
                 fontsize=10)
    ax.set_xticks(np.arange(len(cols))[::max(1, len(cols) // 16)])
    ax.set_xticklabels(cols[::max(1, len(cols) // 16)], fontsize=7)
    ax.set_yticks(np.arange(len(rows))[::max(1, len(rows) // 16)])
    ax.set_yticklabels(rows[::max(1, len(rows) // 16)], fontsize=7)
    fig.colorbar(im, ax=ax, label='NCC')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_matched_pairs(Ka, Kb, score, lag, sign, r, c, name_a, name_b, n_show, out_path):
    wa, wb = Ka.mean(axis=1), Kb.mean(axis=1)
    order = np.argsort(-score[r, c])
    n_show = min(n_show, len(order) // 2) or 1
    picks = [('beste', order[:n_show]), ('schlechteste', order[::-1][:n_show][::-1])]
    fig, axes = plt.subplots(2, n_show, figsize=(2.6 * n_show, 5), squeeze=False)
    for row, (label, sel) in enumerate(picks):
        for col, p in enumerate(sel):
            i, j = r[p], c[p]
            ax = axes[row][col]
            ax.plot(np.arange(wa.shape[1]), wa[i], color=C_A, lw=1.8, label=f'{name_a} #{i}')
            ax.plot(np.arange(wb.shape[1]) + lag[i, j], sign[i, j] * wb[j], color=C_B, lw=1.8, ls='--',
                    label=f'{name_b} #{j}' + (' (×−1)' if sign[i, j] < 0 else ''))
            ax.axhline(0, color='0.85', lw=0.5)
            ax.set_title(f'{label}: NCC {score[i, j]:.2f}, Lag {lag[i, j]:+d}', fontsize=8)
            ax.legend(fontsize=6, frameon=False)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Zugeordnete Kernpaare ({name_b} um den Lag verschoben und ggf. vorzeichenkorrigiert)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reference(ref: pd.DataFrame, out_path):
    groups = list(dict.fromkeys(ref['comparison']))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        v = ref.loc[ref['comparison'] == g, 'mean_matched_ncc'].values
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False)
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=14, alpha=0.7, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel('mittlere NCC zugeordneter Kerne')
    ax.set_title('Ähnlichkeit der Kerne: zwischen Datensätzen vs. zwischen Folds desselben Datensatzes', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Hauptanalyse
# ----------------------------------------------------------------------------
def analyze(kern_a: Dict[int, np.ndarray], kern_b: Dict[int, np.ndarray], name_a, name_b, out_dir,
            fold_show=None, sr=100.0, max_lag=None, use_abs=False, n_show=6, layer='', seed=0):
    os.makedirs(out_dir, exist_ok=True)
    fold = fold_show if fold_show is not None else sorted(set(kern_a) & set(kern_b))[0]
    Ka, Kb = kern_a[fold], kern_b[fold]
    sr = sr if sr and sr > 0 else 1.0
    pa, pb = peak_frequency(Ka, sr), peak_frequency(Kb, sr)

    tag = lambda s: re.sub(r'[^A-Za-z0-9]+', '-', s).strip('-')
    plot_gallery(Ka, pa, f'{name_a} (Fold {fold})', os.path.join(out_dir, f'1_gallery_{tag(name_a)}.png'), color=C_A, layer=layer)
    plot_gallery(Kb, pb, f'{name_b} (Fold {fold})', os.path.join(out_dir, f'1_gallery_{tag(name_b)}.png'), color=C_B, layer=layer)
    plot_spectra(Ka, Kb, name_a, name_b, sr, os.path.join(out_dir, '2_spectra.png'))

    score, lag, sign = xcorr_matrix(Ka, Kb, max_lag, use_abs)
    r, c = match_kernels(score)
    plot_xcorr_matrix(score, r, c, pa, name_a, name_b, use_abs, os.path.join(out_dir, '3_xcorr_matrix.png'))
    plot_matched_pairs(Ka, Kb, score, lag, sign, r, c, name_a, name_b, n_show, os.path.join(out_dir, '4_matched_pairs.png'))
    pd.DataFrame({'kernel_a': r, 'kernel_b': c, 'ncc': score[r, c], 'lag': lag[r, c], 'sign': sign[r, c],
                  'peak_freq_a': pa[r], 'peak_freq_b': pb[c]}).sort_values('ncc', ascending=False).to_csv(
        os.path.join(out_dir, f'matches_fold{fold}.csv'), index=False)

    # Referenzverteilung über alle Fold-Kombinationen
    rows = []
    for fa, fb in itertools.product(sorted(kern_a), sorted(kern_b)):
        rows.append({'comparison': f'{name_a} <-> {name_b}', 'fold_1': fa, 'fold_2': fb,
                     'mean_matched_ncc': matched_mean(kern_a[fa], kern_b[fb], max_lag, use_abs)})
    for name, kd in [(name_a, kern_a), (name_b, kern_b)]:
        for f1, f2 in itertools.combinations(sorted(kd), 2):
            rows.append({'comparison': f'{name} <-> {name} (andere Folds)', 'fold_1': f1, 'fold_2': f2,
                         'mean_matched_ncc': matched_mean(kd[f1], kd[f2], max_lag, use_abs)})
    rng = np.random.default_rng(seed)
    for _ in range(10):
        g1, g2 = rng.normal(size=Ka.shape), rng.normal(size=Kb.shape)
        rows.append({'comparison': 'zufällige Kerne', 'fold_1': -1, 'fold_2': -1,
                     'mean_matched_ncc': matched_mean(g1, g2, max_lag, use_abs)})
    ref = pd.DataFrame(rows)
    ref.to_csv(os.path.join(out_dir, 'reference.csv'), index=False)
    plot_reference(ref, os.path.join(out_dir, '5_reference.png'))
    print(ref.groupby('comparison')['mean_matched_ncc'].agg(['mean', 'std', 'count']).round(3).to_string())
    return ref


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt-a', required=True, help='Ordner mit ckpt_fold-XX.pth (Modell A), Glob oder Datei')
    p.add_argument('--ckpt-b', default=None, help='wie --ckpt-a, für Modell B')
    p.add_argument('--name-a', default='A')
    p.add_argument('--name-b', default='B')
    p.add_argument('--layer', default=None, help='Conv-Schicht (Schlüssel oder eindeutiger Teilstring); Standard: erste')
    p.add_argument('--list-layers', action='store_true', help='Conv-Schichten von Modell A auflisten und beenden')
    p.add_argument('--fold', type=int, default=None, help='Fold für die Einzelabbildungen (Standard: kleinster gemeinsamer)')
    p.add_argument('--sr', type=float, default=100.0,
                   help='Abtastrate der Eingabe der gewählten Schicht in Hz (nur für die erste Conv-Schicht sinnvoll; 0 = Zyklen/Sample)')
    p.add_argument('--max-lag', type=int, default=None, help='max. Verschiebung in Samples (Standard: Kernlänge // 4)')
    p.add_argument('--abs', dest='use_abs', action='store_true', help='Vorzeichen ignorieren (|NCC|)')
    p.add_argument('--n-show', type=int, default=6, help='Paare je Reihe in 4_matched_pairs.png')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='results/kernels')
    args = p.parse_args()

    cka = find_checkpoints(args.ckpt_a)
    if args.list_layers:
        first = load_state(cka[min(cka)])
        for k, v in conv_layers(first).items():
            print(f'{k:60s} (out, in, k) = {tuple(v.shape)}')
        return
    if not args.ckpt_b:
        raise SystemExit('--ckpt-b fehlt.')
    ckb = find_checkpoints(args.ckpt_b)

    first_a = load_state(cka[min(cka)])
    layer = pick_layer(conv_layers(first_a), args.layer)
    print(f'[INFO] Schicht: {layer}')

    def collect(cks):
        out = {}
        for f, path in cks.items():
            layers = conv_layers(load_state(path))
            if layer not in layers:
                raise KeyError(f'{layer} nicht in {path}. Vorhanden: {list(layers)[:5]} ...')
            out[f] = layers[layer].astype(float)
        return out

    analyze(collect(cka), collect(ckb), args.name_a, args.name_b, args.out_dir, fold_show=args.fold, sr=args.sr,
            max_lag=args.max_lag, use_abs=args.use_abs, n_show=args.n_show, layer=layer, seed=args.seed)


if __name__ == '__main__':
    main()
