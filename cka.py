#!/usr/bin/env python3
"""
Layer-wise CKA zwischen zwei Modellen (z. B. auf DOD-H und auf ANPHY trainiert), je Fold.

Pro Fold k werden Modell A (Checkpoint fold k) und Modell B (Checkpoint fold k) auf
denselben Eingaben ausgewertet und die Layer-Aktivierungen per CKA verglichen:
  - testset_<A>   : Testsplit von Datensatz A, Fold k
  - testset_<B>   : Testsplit von Datensatz B, Fold k
  - common_subset : klassenbalancierte Mischung aus beiden Testsplits (gleich viele
                    Epochen je Klasse aus A und B)
Anschließend Mittelwert / Median / SD / Min / Max über die Folds.

Schätzer:
  --cka-variant debiased (Standard): unverzerrter CKA (Song et al. 2012; Kornblith et al. 2019).
      Der klassische lineare CKA ist bei n << d (hier: wenige tausend Epochen, aber
      10^4..10^6 Features je Layer) Richtung 1 verzerrt, selbst für unabhängige Daten.
  --cka-variant biased : klassischer linearer CKA (zum Vergleich).
Speicher: Pro Layer werden die Aktivierungen über die Zeitachse auf --pool-bins Intervalle gemittelt
(Features = Kanäle * pool-bins) und nie in voller Größe gespeichert; die Zahl der Epochen je Szenario ist
auf --max-samples begrenzt (Gram-Matrizen wachsen mit n^2).
Zusätzlich wird eine Nullbasislinie berechnet (Sample-Reihenfolge eines Modells
zufällig permutiert): Werte, die nicht deutlich darüber liegen, sind nicht interpretierbar.

Ausgaben (in --output-dir):
  cka_fold<k>_<scenario>.png, cka_mean_over_folds_<scenario>.png (mit ±SD),
  cka_median_over_folds_<scenario>.png, cka_values_long.csv, cka_summary.csv

Beispiel:
  python cka_analysis.py \
      --config-a configs/dodh.json --config-b configs/anphy.json \
      --checkpoint-a checkpoints/dodh --checkpoint-b checkpoints/anphy \
      --output-dir results/cka --seed 0

Checkpoints: Ordner (enthält ckpt_fold-XX.pth), Glob-Muster, Vorlage mit {fold:02d}
oder einzelne Dateien. Die Zuordnung A<->B erfolgt über die Fold-Nummer im Dateinamen.
"""

import argparse
import gc
import glob
import json
import os
import pickle
import re
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from loader import EEGDataLoader
from models.main_model import MainModel


DEFAULT_LABEL_KEYS = ('y', 'labels', 'label', 'targets', 'target')


# ----------------------------------------------------------------------------
# Hilfsfunktionen: Labels / gemeinsame Teilmenge
# ----------------------------------------------------------------------------
def load_array_or_archive(path: str, preferred_keys: Iterable[str] = DEFAULT_LABEL_KEYS):
    """Lädt .npy/.npz/.pkl und gibt Array bzw. dict zurück."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')

    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            return arr.item()
        return arr

    if suffix == '.npz':
        data = np.load(path, allow_pickle=True)
        for key in preferred_keys:
            if key in data.files:
                return data[key]
        if len(data.files) == 1:
            return data[data.files[0]]
        return {k: data[k] for k in data.files}

    if suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)

    raise ValueError(f'Unsupported file type: {path}')


def load_labels_from_path(path: str, label_key: str = None):
    """Label-Vektor aus Datei oder Ordner (.npy/.npz/.pkl)."""
    if os.path.isdir(path):
        labels = []
        for fname in sorted(os.listdir(path)):
            full = os.path.join(path, fname)
            if not (full.endswith('.npy') or full.endswith('.npz') or full.endswith('.pkl')):
                continue
            try:
                arr = load_array_or_archive(full)
            except Exception:
                continue
            if isinstance(arr, dict):
                candidate = None
                if label_key is not None and label_key in arr:
                    candidate = arr[label_key]
                else:
                    for key in DEFAULT_LABEL_KEYS:
                        if key in arr:
                            candidate = arr[key]
                            break
                if candidate is not None:
                    labels.append(np.asarray(candidate).reshape(-1))
            elif isinstance(arr, np.ndarray):
                if arr.ndim == 1 or (arr.ndim > 1 and arr.shape[-1] == 1):
                    labels.append(arr.reshape(-1))
        if labels:
            return np.concatenate(labels)
        raise ValueError(f'No label-like arrays found in directory: {path}')

    arr = load_array_or_archive(path)
    if isinstance(arr, dict):
        if label_key is not None and label_key in arr:
            return np.asarray(arr[label_key]).reshape(-1)
        for key in DEFAULT_LABEL_KEYS:
            if key in arr:
                return np.asarray(arr[key]).reshape(-1)
        raise KeyError(f'No label field found in archive {path}. Expected one of {DEFAULT_LABEL_KEYS}.')

    arr = np.asarray(arr)
    if arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape):
        return arr.reshape(-1)
    raise ValueError(f'Could not interpret {path} as a label vector; shape={arr.shape}')


def make_common_subset(labels_a, labels_b, n_per_class: int = None, min_class_count: int = 1, seed: int = 0):
    """Klassenbalancierte Teilmenge: je gemeinsamer Klasse gleich viele Indizes aus A und B.

    Die Indizes beider Seiten sind aufsteigend sortiert. Es besteht KEINE Paarung
    a_indices[i] <-> b_indices[i]; für CKA auf gemeinsamen Eingaben wird sie nicht benötigt.
    """
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    common_classes = sorted(set(np.unique(labels_a)).intersection(np.unique(labels_b)))
    if not common_classes:
        raise ValueError('No common classes found in labels_a and labels_b.')

    rng = np.random.default_rng(seed)
    a_idx, b_idx, counts = [], [], {}
    for c in common_classes:
        idx_a = np.where(labels_a == c)[0]
        idx_b = np.where(labels_b == c)[0]
        available = min(len(idx_a), len(idx_b))
        if available < min_class_count:
            continue
        take = available if n_per_class is None else min(n_per_class, available)
        if take < min_class_count:
            continue
        counts[int(c)] = take
        a_idx.append(rng.choice(idx_a, size=take, replace=False))
        b_idx.append(rng.choice(idx_b, size=take, replace=False))

    if not a_idx:
        raise ValueError('No valid common subset could be created. Check overlapping classes and min_class_count.')

    a_idx_all = np.sort(np.concatenate(a_idx))
    b_idx_all = np.sort(np.concatenate(b_idx))
    return {
        'classes': common_classes,
        'n_per_class': counts,
        'a_indices': a_idx_all,
        'b_indices': b_idx_all,
        'a_labels': labels_a[a_idx_all],
        'b_labels': labels_b[b_idx_all],
    }


def save_common_subset(output_dir: str, subset: Dict, prefix: str = 'common_subset'):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{prefix}.npz')
    np.savez(
        out_path,
        classes=np.asarray(sorted(subset['classes'])),
        n_per_class=np.asarray([subset['n_per_class'].get(int(c), 0) for c in sorted(subset['classes'])], dtype=int),
        a_indices=np.asarray(subset['a_indices'], dtype=int),
        b_indices=np.asarray(subset['b_indices'], dtype=int),
        a_labels=np.asarray(subset['a_labels'], dtype=int),
        b_labels=np.asarray(subset['b_labels'], dtype=int),
    )
    return out_path


def build_common_subset_from_two_datasets(dataset_a, dataset_b, output_dir, n_per_class=None, seed=0,
                                          label_key_a=None, label_key_b=None, prefix='common_subset'):
    """Gemeinsame Teilmenge aus zwei Label-Quellen (Datei oder Ordner)."""
    labels_a = load_labels_from_path(dataset_a, label_key_a)
    labels_b = load_labels_from_path(dataset_b, label_key_b)
    subset = make_common_subset(labels_a, labels_b, n_per_class=n_per_class, seed=seed)
    subset_path = save_common_subset(output_dir, subset, prefix=prefix)
    print(f'[INFO] Common subset built from {dataset_a} and {dataset_b}')
    print(f'[INFO] Classes: {subset["classes"]}')
    print(f'[INFO] n_per_class: {subset["n_per_class"]}')
    print(f'[INFO] Saved to: {subset_path}')
    return subset, subset_path


# ----------------------------------------------------------------------------
# Daten
# ----------------------------------------------------------------------------
def load_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def get_dataset_name(config: dict):
    return str(config.get('dataset', {}).get('name', 'Model'))


def safe_name(name: str):
    return re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')


def collect_test_samples(config: dict, fold: int, split: str = 'test', batch_size: int = 64):
    """Alle Eingaben und Labels eines Splits (Reihenfolge wie im Loader, kein Shuffle)."""
    ds = EEGDataLoader(config, fold, set=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x_batch, y_batch in loader:
        xs.append(x_batch.detach().cpu().numpy())
        ys.append(y_batch.detach().cpu().numpy())
    if not xs:
        raise ValueError(f'No samples found for fold {fold}, split {split}')
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).reshape(len(x), -1)
    if y.shape[1] != 1:
        raise ValueError(f'Expected one label per sample, got label shape {y.shape}.')
    return x, y[:, 0]


def random_subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    """Zufällige (nicht: erste n) Teilmenge, Reihenfolge bleibt erhalten."""
    if max_samples is None or len(x) <= max_samples:
        return x, y
    idx = np.sort(np.random.default_rng(seed).choice(len(x), size=max_samples, replace=False))
    return x[idx], y[idx]


def build_common_inputs(x_a, y_a, x_b, y_b, n_per_class=None, max_samples=None, seed=0):
    """Klassenbalancierte Eingaben aus beiden Testsets (A-Teil gefolgt von B-Teil)."""
    if max_samples is not None:
        n_classes = len(set(np.unique(y_a)).intersection(np.unique(y_b)))
        cap = max(1, max_samples // (2 * max(n_classes, 1)))
        n_per_class = cap if n_per_class is None else min(n_per_class, cap)
    subset = make_common_subset(y_a, y_b, n_per_class=n_per_class, seed=seed)
    x_common = np.concatenate([x_a[subset['a_indices']], x_b[subset['b_indices']]], axis=0)
    return subset, x_common


# ----------------------------------------------------------------------------
# Modelle / Checkpoints
# ----------------------------------------------------------------------------
def _select_state_dict(raw):
    if not isinstance(raw, dict):
        raise ValueError(f'Unexpected checkpoint format: {type(raw)}')
    for key in ('state_dict', 'model_state_dict'):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    return raw


def _strip_best_prefix(state_dict: dict, model_keys: set):
    """Wählt das Präfix ('', 'module.', 'model.') mit den meisten passenden Keys."""
    best, best_n = state_dict, -1
    for prefix in ('', 'module.', 'model.'):
        cand = {(k[len(prefix):] if prefix and k.startswith(prefix) else k): v for k, v in state_dict.items()}
        n = len(set(cand) & model_keys)
        if n > best_n:
            best, best_n = cand, n
    return best


def load_model_from_checkpoint(config: dict, checkpoint_path: str, device: torch.device, allow_partial: bool = False):
    model = MainModel(config)
    try:
        raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        print(f'[WARN] weights_only=True failed for {checkpoint_path} ({exc!r}); retrying with weights_only=False.')
        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state = model.state_dict()
    state = _strip_best_prefix(_select_state_dict(raw), set(model_state))

    filtered, shape_mismatch = {}, []
    for k, v in state.items():
        if not torch.is_tensor(v) or k not in model_state:
            continue
        if v.shape != model_state[k].shape:
            shape_mismatch.append((k, tuple(model_state[k].shape), tuple(v.shape)))
            continue
        filtered[k] = v

    missing = sorted(set(model_state) - set(filtered))
    unexpected = sorted(k for k in state if k not in model_state)
    if missing or shape_mismatch:
        msg = (f'{checkpoint_path}: {len(missing)}/{len(model_state)} Modell-Parameter nicht geladen '
               f'(z. B. {missing[:5]}), {len(shape_mismatch)} Shape-Mismatches (z. B. {shape_mismatch[:3]}).')
        if not allow_partial:
            raise RuntimeError(msg + ' Abbruch, damit keine zufällig initialisierten Layer ausgewertet werden '
                                     '(mit --allow-partial-checkpoint überschreibbar).')
        print('[WARN] ' + msg)
    if unexpected:
        print(f'[INFO] {len(unexpected)} Checkpoint-Keys ohne Gegenstück im Modell ignoriert, z. B. {unexpected[:3]}')

    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model


def resolve_checkpoints(specs, num_folds: int = 5) -> Dict[int, str]:
    """Fold-Nummer -> Checkpoint-Pfad. Akzeptiert Ordner, Glob, '{fold:02d}'-Vorlage oder Dateien."""
    if isinstance(specs, str):
        specs = [specs]
    paths = []
    for p in specs:
        p = str(p)
        if '{' in p:
            paths += [p.format(fold=f) for f in range(1, num_folds + 1)]
        elif os.path.isdir(p):
            paths += sorted(glob.glob(os.path.join(p, 'ckpt_fold-*.pth')))
        elif any(ch in p for ch in '*?['):
            paths += sorted(glob.glob(p))
        else:
            paths.append(p)

    out = {}
    for p in paths:
        m = re.search(r'fold[-_]?(\d+)', os.path.basename(p))
        if not m:
            raise ValueError(f'Fold-Nummer im Dateinamen nicht erkennbar: {p}')
        fold = int(m.group(1))
        if fold in out:
            raise ValueError(f'Fold {fold} doppelt: {out[fold]} und {p}')
        if not os.path.exists(p):
            raise FileNotFoundError(f'Checkpoint nicht gefunden: {p}')
        out[fold] = p
    if not out:
        raise FileNotFoundError(f'Keine Checkpoints gefunden für {specs}')
    return out


def pair_folds(ckpts_a: Dict[int, str], ckpts_b: Dict[int, str], requested: List[int] = None):
    """Fold-Liste, die in beiden Checkpoint-Sets vorkommt. Abweichungen sind ein Fehler (kein stilles Kürzen)."""
    if requested:
        missing = [f for f in requested if f not in ckpts_a or f not in ckpts_b]
        if missing:
            raise ValueError(f'Fold(s) {missing} fehlen in A ({sorted(ckpts_a)}) oder B ({sorted(ckpts_b)}).')
        return sorted(requested)
    if set(ckpts_a) != set(ckpts_b):
        raise ValueError(f'Fold-Mengen unterschiedlich: A={sorted(ckpts_a)}, B={sorted(ckpts_b)}. '
                         f'Mit --folds explizit die gewünschten Folds angeben.')
    return sorted(ckpts_a)


# ----------------------------------------------------------------------------
# Feature-Extraktion
# ----------------------------------------------------------------------------
def _check_input(x: np.ndarray):
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f'Erwarte Eingaben der Form (N, 1, L) wie vom EEGDataLoader, erhalten {x.shape}.')
    return x


def _pool_features(h: torch.Tensor, bins: int) -> np.ndarray:
    """(B, C, ...) -> (B, C * bins): Mittelung über die Zeitachse auf `bins` Intervalle (bins <= 0: nichts poolen)."""
    b = h.shape[0]
    if h.ndim >= 3 and bins and bins > 0:
        h = torch.nn.functional.adaptive_avg_pool1d(h.reshape(b, h.shape[1], -1), bins)
    return h.detach().cpu().numpy().reshape(b, -1)


@torch.no_grad()
def extract_layer_features(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 32,
                           pool_bins: int = 16, verbose: bool = False):
    """Layer-Aktivierungen je Epoche, (N, D) float32. Batchweise, ohne Gradienten.

    Es wird nur die gepoolte Version gespeichert (D = Kanäle * pool_bins), nie die volle Aktivierung
    aller Epochen. Das hält den Speicherbedarf bei langen Eingaben (z. B. SL-10 = 30.000 Samples) klein.
    """
    x = _check_input(np.asarray(x))
    backbone = model.feature
    stages = [('init_layer', backbone.init_layer), ('layer1', backbone.layer1), ('layer2', backbone.layer2),
              ('layer3', backbone.layer3), ('layer4', backbone.layer4)]
    chunks: Dict[str, List[np.ndarray]] = {}

    for s in range(0, len(x), batch_size):
        h = torch.as_tensor(x[s:s + batch_size], dtype=torch.float32, device=device)
        for name, fn in stages:
            h = fn(h)
            pooled = _pool_features(h, pool_bins)
            chunks.setdefault(name, []).append(pooled)
            if verbose and s == 0:
                print(f'[INFO]   {name}: Aktivierung {tuple(h.shape[1:])} = {int(np.prod(h.shape[1:]))} Werte/Epoche '
                      f'-> gespeichert {pooled.shape[1]}')
        if hasattr(backbone, 'conv_c5'):
            p5 = backbone.conv_c5(h)
            pooled = _pool_features(p5, pool_bins)
            chunks.setdefault('p5', []).append(pooled)
            if verbose and s == 0:
                print(f'[INFO]   p5: Aktivierung {tuple(p5.shape[1:])} = {int(np.prod(p5.shape[1:]))} Werte/Epoche '
                      f'-> gespeichert {pooled.shape[1]}')

    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


@torch.no_grad()
def check_forward_consistency(model, x: np.ndarray, device: torch.device, name: str, n: int = 8):
    """Prüft (best effort), ob extract_layer_features die Aktivierungen von model.feature reproduziert."""
    x = _check_input(np.asarray(x[:n]))
    mine = extract_layer_features(model, x, device, batch_size=n, pool_bins=0)   # ungepoolt, nur 8 Epochen
    try:
        ref = model.feature(torch.as_tensor(x, dtype=torch.float32, device=device))
    except Exception as exc:
        print(f'[WARN] Forward-Check ({name}) übersprungen: {exc!r}')
        return
    refs = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    matches = []
    for ri, r in enumerate(refs):
        if not torch.is_tensor(r):
            continue
        r = r.detach().cpu().numpy().reshape(len(x), -1)
        for key, v in mine.items():
            if v.shape == r.shape and np.allclose(v, r, atol=1e-4, rtol=1e-3):
                matches.append((ri, key))
    if matches:
        print(f'[INFO] Forward-Check ({name}): Ausgaben von model.feature stimmen überein mit (Index, Layer) {matches}')
    else:
        print(f'[WARN] Forward-Check ({name}): KEIN Ausgabetensor von model.feature stimmt mit den extrahierten Layern '
              f'überein. Bitte prüfen, ob extract_layer_features den echten Forward-Pfad abbildet.')


# ----------------------------------------------------------------------------
# CKA
# ----------------------------------------------------------------------------
def ensure_2d_features(X: np.ndarray):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f'Features must be 2D: {X.shape}')
    return X


def centered_gram(X, device, chunk: int = 50_000):
    """Spaltenzentrierte lineare Gram-Matrix (n x n), in Spalten-Chunks (d darf riesig sein)."""
    X = ensure_2d_features(X)
    n, d = X.shape
    K = torch.zeros(n, n, dtype=torch.float64, device=device)
    for s in range(0, d, chunk):
        c = torch.as_tensor(X[:, s:s + chunk], dtype=torch.float32, device=device)
        c = c - c.mean(dim=0, keepdim=True)
        K += (c @ c.T).double()
    return K


def center_gram_unbiased(K):
    """Unverzerrte Zentrierung (Song et al. 2012): Diagonale null, Zeilen-/Spaltenmittel mit n-2 / 2(n-1).

    Der Schätzer ist translationsinvariant; die vorherige Spaltenzentrierung der Features ändert
    das Ergebnis mathematisch nicht, verbessert aber die numerische Stabilität in float32.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError('Debiased CKA braucht mindestens 4 Samples.')
    K = K.clone()
    K.fill_diagonal_(0)
    m = K.sum(dim=0) / (n - 2)
    m = m - m.sum() / (2 * (n - 1))
    K = K - m[:, None] - m[None, :]
    K.fill_diagonal_(0)
    return K


def prepare_gram(X, device, variant: str = 'debiased'):
    K = centered_gram(X, device)
    if variant == 'debiased':
        return center_gram_unbiased(K)
    if variant == 'biased':
        return K
    raise ValueError(f'Unknown CKA variant: {variant}')


def cka_from_gram(K, L):
    den = K.norm() * L.norm()
    return float((K * L).sum() / den) if den > 0 else 0.0


def cka(X, Y, device=None, variant: str = 'debiased'):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'CKA requires same sample count. Got {X.shape} and {Y.shape}.')
    return cka_from_gram(prepare_gram(X, device, variant), prepare_gram(Y, device, variant))


def compute_grams(features: Dict[str, np.ndarray], device, variant: str):
    return {name: prepare_gram(f, device, variant) for name, f in features.items()}


def cka_matrix(grams_a: Dict, grams_b: Dict, names: List[str]):
    m = np.zeros((len(names), len(names)))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            m[i, j] = cka_from_gram(grams_a[ni], grams_b[nj])
    return m


def null_cka_matrix(grams_a: Dict, grams_b: Dict, names: List[str], n_perm: int, seed: int, device):
    """Nullbasislinie: Zuordnung der Epochen zwischen den Modellen zufällig vertauscht."""
    if n_perm <= 0:
        return np.full((len(names), len(names)), np.nan)
    gen = torch.Generator().manual_seed(seed)
    n = next(iter(grams_a.values())).shape[0]
    acc = np.zeros((len(names), len(names)))
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=gen).to(device)
        for j, nj in enumerate(names):
            Lp = grams_b[nj][perm][:, perm]
            for i, ni in enumerate(names):
                acc[i, j] += cka_from_gram(grams_a[ni], Lp)
    return acc / n_perm


def rsa(X: np.ndarray, Y: np.ndarray):
    """RSA: Korrelation der RDMs (Kosinus-Distanz)."""
    from sklearn.metrics.pairwise import cosine_similarity
    X, Y = ensure_2d_features(X), ensure_2d_features(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'RSA requires same sample count. Got {X.shape} and {Y.shape}.')

    def rdm(A):
        return 1.0 - cosine_similarity(A)

    rx, ry = rdm(X), rdm(Y)
    tx = rx[np.triu_indices(rx.shape[0], k=1)]
    ty = ry[np.triu_indices(ry.shape[0], k=1)]
    if np.std(tx) == 0.0 or np.std(ty) == 0.0:
        return 0.0
    return float(np.corrcoef(tx, ty)[0, 1])


# ----------------------------------------------------------------------------
# Auswertung / Plots
# ----------------------------------------------------------------------------
def save_cka_heatmap(matrix: np.ndarray, row_labels, col_labels, out_path: str, title: str, std: np.ndarray = None):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(list(col_labels), rotation=45, ha='right')
    ax.set_yticklabels(list(row_labels))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt = f'{matrix[i, j]:.2f}'
            if std is not None:
                txt += f'\n±{std[i, j]:.2f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7 if std is not None else 8,
                    color='white' if matrix[i, j] < 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='CKA')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'[INFO] CKA heatmap saved to {out_path}')


def summarize_folds(mats: List[np.ndarray]):
    arr = np.stack(mats, axis=0)
    return {
        'mean': arr.mean(axis=0),
        'median': np.median(arr, axis=0),
        'std': arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(arr[0]),
        'min': arr.min(axis=0),
        'max': arr.max(axis=0),
    }


def run_model_cka(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_a, cfg_b = load_config(args.config_a), load_config(args.config_b)
    name_a, name_b = get_dataset_name(cfg_a), get_dataset_name(cfg_b)
    tag_a, tag_b = safe_name(name_a), safe_name(name_b)

    ckpts_a = resolve_checkpoints(args.checkpoint_a, args.num_folds)
    ckpts_b = resolve_checkpoints(args.checkpoint_b, args.num_folds)
    folds = pair_folds(ckpts_a, ckpts_b, args.folds)
    print(f'[INFO] Folds: {folds} | CKA variant: {args.cka_variant} | device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)
    scenarios = [f'testset_{tag_a}', f'testset_{tag_b}', 'common_subset']
    per_scenario = {s: [] for s in scenarios}
    per_scenario_null = {s: [] for s in scenarios}
    rows, layer_names = [], None

    for fold in folds:
        seed = args.seed + fold
        model_a = load_model_from_checkpoint(cfg_a, ckpts_a[fold], device, args.allow_partial_checkpoint)
        model_b = load_model_from_checkpoint(cfg_b, ckpts_b[fold], device, args.allow_partial_checkpoint)

        x_a, y_a = collect_test_samples(cfg_a, fold)
        x_b, y_b = collect_test_samples(cfg_b, fold)
        print(f'[INFO] Fold {fold}: test epochs {name_a}={len(x_a)}, {name_b}={len(x_b)}')

        if not args.skip_forward_check and fold == folds[0]:
            check_forward_consistency(model_a, x_a, device, f'{name_a} model')
            check_forward_consistency(model_b, x_b, device, f'{name_b} model')

        xa_s, ya_s = random_subsample(x_a, y_a, args.max_samples, seed)
        xb_s, yb_s = random_subsample(x_b, y_b, args.max_samples, seed)
        subset, x_common = build_common_inputs(x_a, y_a, x_b, y_b, args.n_per_class, args.max_samples, seed)
        print(f'[INFO] Fold {fold}: common subset n_per_class={subset["n_per_class"]}, total={len(x_common)}')

        inputs = {scenarios[0]: xa_s, scenarios[1]: xb_s, scenarios[2]: x_common}
        n_max = max(len(v) for v in inputs.values())
        print(f'[INFO] Fold {fold}: größtes Szenario n={n_max}; Gram-Matrizen brauchen ca. '
              f'{n_max ** 2 * 8 * 12 / 1e9:.1f} GB RAM (12 Stück, float64).')
        for scen, x in inputs.items():
            first = fold == folds[0] and scen == scenarios[0]
            if first:
                print(f'[INFO] Feature-Größen ({name_a} model), pool_bins={args.pool_bins}:')
            feats_a = extract_layer_features(model_a, x, device, args.batch_size, args.pool_bins, verbose=first)
            grams_a = compute_grams(feats_a, device, args.cka_variant)
            del feats_a
            feats_b = extract_layer_features(model_b, x, device, args.batch_size, args.pool_bins)
            grams_b = compute_grams(feats_b, device, args.cka_variant)
            del feats_b

            names = [n for n in grams_a if n in grams_b]
            layer_names = layer_names or names
            mat = cka_matrix(grams_a, grams_b, names)
            null = null_cka_matrix(grams_a, grams_b, names, args.n_null, seed, device)
            per_scenario[scen].append(mat)
            per_scenario_null[scen].append(null)

            row_labels = [f'{name_a} model: {n}' for n in names]
            col_labels = [f'{name_b} model: {n}' for n in names]
            save_cka_heatmap(mat, row_labels, col_labels,
                             os.path.join(args.output_dir, f'cka_fold{fold}_{scen}.png'),
                             f'CKA fold {fold}: {scen} (n={len(x)}, {args.cka_variant})')
            for i, ni in enumerate(names):
                for j, nj in enumerate(names):
                    rows.append({'fold': fold, 'scenario': scen, 'n_epochs': len(x),
                                 'layer_model_a': ni, 'layer_model_b': nj,
                                 'cka': mat[i, j], 'cka_null': null[i, j]})
            del grams_a, grams_b
            gc.collect()

        del model_a, model_b, inputs, x_a, x_b, x_common, xa_s, xb_s
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Aggregation über Folds
    names = layer_names
    row_labels = [f'{name_a} model: {n}' for n in names]
    col_labels = [f'{name_b} model: {n}' for n in names]
    summary_rows = []
    for scen in scenarios:
        s = summarize_folds(per_scenario[scen])
        null_mean = np.nanmean(np.stack(per_scenario_null[scen]), axis=0) if args.n_null > 0 else None
        k = len(folds)
        save_cka_heatmap(s['mean'], row_labels, col_labels,
                         os.path.join(args.output_dir, f'cka_mean_over_folds_{scen}.png'),
                         f'CKA mean ± SD over {k} folds: {scen} ({args.cka_variant})', std=s['std'])
        save_cka_heatmap(s['median'], row_labels, col_labels,
                         os.path.join(args.output_dir, f'cka_median_over_folds_{scen}.png'),
                         f'CKA median over {k} folds: {scen} ({args.cka_variant})')
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                summary_rows.append({'scenario': scen, 'layer_model_a': ni, 'layer_model_b': nj,
                                     'mean': s['mean'][i, j], 'sd': s['std'][i, j], 'median': s['median'][i, j],
                                     'min': s['min'][i, j], 'max': s['max'][i, j],
                                     'null_mean': null_mean[i, j] if null_mean is not None else np.nan,
                                     'n_folds': k})
        if null_mean is not None:
            print(f'[INFO] {scen}: mittlerer CKA {s["mean"].mean():.3f} | Nullbasislinie (Permutation) {np.nanmean(null_mean):.3f}')

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'cka_values_long.csv'), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, 'cka_summary.csv'), index=False)
    print('[INFO] Finished all requested CKA folds in', args.output_dir)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Modell-CKA
    p.add_argument('--config-a', type=str, default=None, help='Config JSON von Modell A / Datensatz A.')
    p.add_argument('--config-b', type=str, default=None, help='Config JSON von Modell B / Datensatz B.')
    p.add_argument('--checkpoint-a', type=str, nargs='+', default=None,
                   help='Checkpoints von Modell A: Ordner, Glob, Vorlage mit {fold:02d} oder Dateien.')
    p.add_argument('--checkpoint-b', type=str, nargs='+', default=None, help='Wie --checkpoint-a, für Modell B.')
    p.add_argument('--folds', type=int, nargs='+', default=None,
                   help='Nur diese Folds auswerten (Fold k von A wird mit Fold k von B verglichen). Standard: alle.')
    p.add_argument('--num-folds', type=int, default=5, help='Fold-Anzahl für {fold}-Vorlagen im Checkpoint-Pfad.')
    p.add_argument('--cka-variant', choices=['debiased', 'biased'], default='debiased',
                   help='debiased (Standard, unverzerrt bei n << d) oder biased (klassischer linearer CKA).')
    p.add_argument('--n-null', type=int, default=3, help='Anzahl Permutationen für die Nullbasislinie (0 = aus).')
    p.add_argument('--batch-size', type=int, default=32, help='Batchgröße bei der Feature-Extraktion.')
    p.add_argument('--pool-bins', type=int, default=16,
                   help='Aktivierungen je Layer über die Zeitachse auf diese Zahl Intervalle mitteln '
                        '(Features pro Layer = Kanäle * pool-bins). 0 = nicht poolen (braucht sehr viel RAM).')
    p.add_argument('--max-samples', type=int, default=4000,
                   help='Obergrenze der Epochen je Szenario (zufällig bzw. klassenbalanciert gezogen). '
                        'Gram-Matrizen wachsen mit n^2; 0 = keine Grenze.')
    p.add_argument('--n-per-class', type=int, default=None, help='Max. Epochen je Klasse und Datensatz im common subset.')
    p.add_argument('--allow-partial-checkpoint', action='store_true',
                   help='Unvollständig geladene Checkpoints nur warnen statt abbrechen.')
    p.add_argument('--skip-forward-check', action='store_true', help='Konsistenzprüfung gegen model.feature überspringen.')
    # Nur Labels / Features
    p.add_argument('--dataset-a', type=str, default=None, help='Label-Quelle A (.npy/.npz/.pkl oder Ordner).')
    p.add_argument('--dataset-b', type=str, default=None, help='Label-Quelle B (.npy/.npz/.pkl oder Ordner).')
    p.add_argument('--label-key-a', type=str, default=None)
    p.add_argument('--label-key-b', type=str, default=None)
    p.add_argument('--prefix', type=str, default='common_subset', help='Dateiname-Präfix der gespeicherten Teilmenge.')
    p.add_argument('--feature-a', type=str, default=None, help='.npy mit Features A, (N, D), passend zur Teilmenge.')
    p.add_argument('--feature-b', type=str, default=None, help='.npy mit Features B, (N, D), passend zur Teilmenge.')
    p.add_argument('--seed', type=int, default=0, help='Seed (Stichproben, Permutationen).')
    p.add_argument('--output-dir', type=str, default='results/cka')
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None

    if args.config_a and args.config_b and args.checkpoint_a and args.checkpoint_b:
        run_model_cka(args)
        return

    if not (args.dataset_a and args.dataset_b):
        raise SystemExit('Entweder --config-a/--config-b/--checkpoint-a/--checkpoint-b (Modell-CKA) '
                         'oder --dataset-a/--dataset-b (gemeinsame Teilmenge) angeben.')

    subset, _ = build_common_subset_from_two_datasets(
        args.dataset_a, args.dataset_b, args.output_dir, n_per_class=args.n_per_class, seed=args.seed,
        label_key_a=args.label_key_a, label_key_b=args.label_key_b, prefix=args.prefix)

    if args.feature_a and args.feature_b:
        X, Y = np.load(args.feature_a), np.load(args.feature_b)
        if X.shape[0] != len(subset['a_indices']) or Y.shape[0] != len(subset['b_indices']):
            print('[WARN] Feature matrices do not match the selected common subset size.')
        print(f'[INFO] CKA ({args.cka_variant}) = {cka(X, Y, variant=args.cka_variant):.6f}')
        print(f'[INFO] RSA = {rsa(X, Y):.6f}')


if __name__ == '__main__':
    main()