#!/usr/bin/env python
"""
RSA zwischen zwei SleePyCo-Modellen (z.B. DOD-H vs. ANPHY), vereinfachte Fassung von rsa.py.

    python rsa_simple.py --config-a configs/<DOD-H>.json --ckpt-a checkpoints/<DOD-H>/ \
                         --config-b configs/<ANPHY>.json --ckpt-b checkpoints/<ANPHY>/ \
                         --n-samples 300 --n-per-class 60 --out results/rsa

Pro Fold (Folds werden ueber die Nummer in 'ckpt_fold-XX.pth' gepaart) und pro Eingabemenge
(Testset A, Testset B, gemeinsame klassenbalancierte Menge) wird berechnet:
  1. Aktivierungen beider Modelle an denselben Eingaben, je Schicht
     (init_layer, layer1-4 und die Pyramidenausgaenge p5/p4/p3 = Eingaenge des Transformers)
  2. je Modell und Schicht EINMAL die RDM (1 - Kosinus-Aehnlichkeit)
  3. Korrelation (Pearson, optional Spearman) der RDM-Oberdreiecke -> Matrix (Schichten A x Schichten B)
Ausgabe: <out>/rsa_matrices.npz (alle Matrizen + Schichtnamen), common_subset_fold-XX.npz, Heatmaps (png).

Zeit-Pooling: --pool 0 (Standard) flacht (Kanaele x Zeit) ab, wie in rsa.py. Dann haengt die RDM an der exakten
Zeitlage (ein um wenige Samples verschobenes Muster ist "anders") und der Speicher waechst mit
N x Kanaele x Laenge (init_layer bei SL-10: 7.7 MB pro Sample). --pool K mittelt vor der RDM auf K Zeit-Bins
(K=1 = reine Kanalprofile), das ist kleiner und weniger zeitlagenabhaengig.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loader import EEGDataLoader
from models.main_model import MainModel

BACKBONE_LAYERS = ['init_layer', 'layer1', 'layer2', 'layer3', 'layer4']
PYRAMID_NAMES = ['p5', 'p4', 'p3']  # Reihenfolge der Ausgabeliste von SleePyCoBackbone.forward


# ------------------------------------------------------------------------------------------ Modell
def load_model(config, ckpt, device):
    model = MainModel(config)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd.get('model_state_dict', sd))
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}  # DataParallel
    model.load_state_dict(sd)  # strict: fehlende/falsche Gewichte sollen laut scheitern
    return model.to(device).eval()


@torch.no_grad()
def layer_features(model, x, device, batch_size, pool):
    """Aktivierungen je Schicht, (N, D) float32 auf der CPU. x: (N, 1, L) numpy."""
    store = {}
    hooks = [getattr(model.feature, n).register_forward_hook(lambda m, i, o, n=n: store.__setitem__(n, o))
             for n in BACKBONE_LAYERS]
    out = {}
    try:
        for s in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[s:s + batch_size], dtype=torch.float32, device=device)
            pyramid = model.feature(xb)  # echter Forward, kein nachgebauter Backbone
            acts = dict(store)
            acts.update(zip(PYRAMID_NAMES, pyramid))
            for name, h in acts.items():
                h = F.adaptive_avg_pool1d(h, pool) if pool > 0 else h
                out.setdefault(name, []).append(h.flatten(1).float().cpu())
    finally:
        for h in hooks:
            h.remove()
    names = BACKBONE_LAYERS + PYRAMID_NAMES
    return {n: torch.cat(out[n]).numpy() for n in names}


# ------------------------------------------------------------------------------------------ RSA
def rdm_triu(X):
    """Oberdreieck der Kosinus-Distanz-RDM, (N*(N-1)/2,)."""
    X = torch.as_tensor(X, dtype=torch.float64)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rdm = 1.0 - X @ X.T
    i, j = torch.triu_indices(len(X), len(X), offset=1)
    return rdm[i, j].numpy()


def rsa_matrix(feats_a, feats_b, spearman=False):
    """Matrix (Schichten A x Schichten B) der Korrelationen der RDM-Oberdreiecke."""
    names = list(feats_a)
    def z(feats):
        v = np.stack([rdm_triu(feats[n]) for n in names])
        if spearman:
            v = v.argsort(1).argsort(1).astype(float)  # Raenge (Bindungen vernachlaessigbar bei Float-RDMs)
        v = v - v.mean(1, keepdims=True)
        sd = np.sqrt((v ** 2).mean(1, keepdims=True))
        return np.divide(v, sd, out=np.zeros_like(v), where=sd > 0)  # konstante RDM -> 0, wie rsa.py
    za, zb = z(feats_a), z(feats_b)
    return names, za @ zb.T / za.shape[1]


# ------------------------------------------------------------------------------------------ Daten
def window_table(ds):
    """Fuer jedes Fenster von ds: (Fenster-Index, Datei, Label der Zielepoche). Nur NICHT ueberlappende
    Fenster (Schrittweite seq_len je Datei), weil benachbarte Stride-1-Fenster fast identisch sind."""
    keep, labels = [], []
    for k, (f, i, sl) in enumerate(ds.epochs):
        if i % sl == 0:
            keep.append(k)
            labels.append(int(ds.labels[f][i + sl - 1]))
    return np.array(keep), np.array(labels)


def draw(keep, labels, rng, n=None, per_class=None, classes=None):
    """Zufallsauswahl aus (keep, labels): n Fenster ODER je Klasse per_class Fenster."""
    if per_class is None:
        return np.sort(rng.choice(keep, size=min(n, len(keep)) if n else len(keep), replace=False))
    sel = [rng.choice(keep[labels == c], size=per_class, replace=False) for c in classes]
    return np.concatenate(sel)


def load_windows(ds, idx, batch_size=32):
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False)
    xs = [x.numpy() for x, _ in loader]
    return np.concatenate(xs)


# ------------------------------------------------------------------------------------------ Plot
def save_heatmap(M, names, path, title, label_a, label_b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap='viridis', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel(label_b)
    ax.set_ylabel(label_a)
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='w' if v < 0.5 else 'k')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main
def fold_of(path):
    return int(re.search(r'fold[-_](\d+)', os.path.basename(path)).group(1))


def ckpts_by_fold(arg):
    paths = []
    for a in arg:
        paths += sorted(glob.glob(os.path.join(a, 'ckpt_fold-*.pth'))) if os.path.isdir(a) else [a]
    return {fold_of(p): p for p in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config-a', required=True)
    ap.add_argument('--config-b', required=True)
    ap.add_argument('--ckpt-a', nargs='+', required=True, help='Ordner mit ckpt_fold-XX.pth oder einzelne Dateien')
    ap.add_argument('--ckpt-b', nargs='+', required=True)
    ap.add_argument('--n-samples', type=int, default=300, help='Fenster je Testset (nicht ueberlappend, zufaellig)')
    ap.add_argument('--n-per-class', type=int, default=60, help='Fenster je Klasse und Datensatz fuer die gemeinsame Menge')
    ap.add_argument('--pool', type=int, default=0, help='Zeit-Bins vor der RDM; 0 = abflachen wie rsa.py')
    ap.add_argument('--spearman', action='store_true', help='Rangkorrelation der RDMs statt Pearson')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/rsa')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = [json.load(open(args.config_a)), json.load(open(args.config_b))]
    name = [c['dataset']['name'] for c in cfg]
    ck = [ckpts_by_fold(args.ckpt_a), ckpts_by_fold(args.ckpt_b)]
    folds = sorted(set(ck[0]) & set(ck[1]))
    if not folds:
        raise SystemExit(f'Keine gemeinsamen Fold-Nummern: A={sorted(ck[0])}, B={sorted(ck[1])}')
    if set(ck[0]) != set(ck[1]):
        print(f'[WARN] nur Folds {folds} in beiden Ordnern vorhanden (A={sorted(ck[0])}, B={sorted(ck[1])})')

    analyses = [f'{name[0]}_test', f'{name[1]}_test', 'common']
    result = {a: [] for a in analyses}
    layer_names = None
    for fold in folds:
        rng = np.random.default_rng(args.seed + fold)
        models = [load_model(c, ck[i][fold], device) for i, c in enumerate(cfg)]
        ds = [EEGDataLoader(c, fold, set='test') for c in cfg]
        tab = [window_table(d) for d in ds]

        # gemeinsame Menge: je gemeinsamer Klasse gleich viele Fenster aus A und B (A-Fenster zuerst, dann B-Fenster)
        classes = sorted(set(tab[0][1]) & set(tab[1][1]))
        k = min(args.n_per_class, *[int((t[1] == c).sum()) for t in tab for c in classes])
        idx_c = [draw(*t, rng, per_class=k, classes=classes) for t in tab]
        np.savez(os.path.join(args.out, f'common_subset_fold-{fold:02d}.npz'), classes=classes, n_per_class=k,
                 a_window_idx=idx_c[0], b_window_idx=idx_c[1],
                 a_labels=tab[0][1][np.searchsorted(tab[0][0], idx_c[0])],
                 b_labels=tab[1][1][np.searchsorted(tab[1][0], idx_c[1])])
        x_common = np.concatenate([load_windows(ds[0], idx_c[0]), load_windows(ds[1], idx_c[1])])
        inputs = {analyses[0]: load_windows(ds[0], draw(*tab[0], rng, n=args.n_samples)),
                  analyses[1]: load_windows(ds[1], draw(*tab[1], rng, n=args.n_samples)),
                  'common': x_common}

        for a, x in inputs.items():
            fa, fb = [layer_features(m, x, device, args.batch_size, args.pool) for m in models]
            layer_names, M = rsa_matrix(fa, fb, args.spearman)
            result[a].append(M)
            print(f'[INFO] fold {fold:02d} {a}: N={len(x)}, Diagonale (gleiche Schicht) = '
                  + ' '.join(f'{M[i, i]:.2f}' for i in range(len(layer_names))))
            if not args.no_plots:
                save_heatmap(M, layer_names, os.path.join(args.out, f'rsa_fold-{fold:02d}_{a}.png'),
                             f'RSA fold {fold}: {a}', f'{name[0]} (A)', f'{name[1]} (B)')

    np.savez(os.path.join(args.out, 'rsa_matrices.npz'), layers=layer_names, folds=folds, pool=args.pool,
             **{a: np.stack(v) for a, v in result.items()})
    if not args.no_plots:
        for a, v in result.items():
            for agg in (np.mean, np.median):
                save_heatmap(agg(np.stack(v), 0), layer_names,
                             os.path.join(args.out, f'rsa_{agg.__name__}_{a}.png'),
                             f'RSA {agg.__name__} ueber {len(folds)} Fold(s): {a}', f'{name[0]} (A)', f'{name[1]} (B)')
    print(f'[INFO] fertig: {args.out}/')


if __name__ == '__main__':
    main()
