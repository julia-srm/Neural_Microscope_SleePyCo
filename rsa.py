import argparse
import glob
import os
import pickle
from typing import Iterable, List, Dict, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional for plotting only
    plt = None

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional for model-based analysis only
    torch = None


DEFAULT_LABEL_KEYS = ('y', 'labels', 'label', 'targets', 'target')


def cosine_similarity_matrix(X: np.ndarray):
    X = ensure_2d_features(X)
    if X.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X_norm @ X_norm.T


def load_array_or_archive(path: str, preferred_keys: Iterable[str] = DEFAULT_LABEL_KEYS):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')

    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 0:
                return arr.item()
            return arr
        if isinstance(arr, dict):
            return arr
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
            data = pickle.load(f)
        if isinstance(data, dict):
            return data
        return data

    raise ValueError(f'Unsupported file type: {path}')


def load_labels_from_path(path: str, label_key: str = None):
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
                if arr.ndim == 1:
                    labels.append(arr.reshape(-1))
                elif arr.ndim > 1 and arr.shape[-1] == 1:
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
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.reshape(-1)
    raise ValueError(f'Could not interpret {path} as a label vector; shape={arr.shape}')


def make_common_subset(labels_a: np.ndarray, labels_b: np.ndarray, n_per_class: int = None, min_class_count: int = 1, seed: int = 0):
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    common_classes = sorted(set(np.unique(labels_a)).intersection(np.unique(labels_b)))

    if not common_classes:
        raise ValueError('No common classes found in labels_a and labels_b.')

    rng = np.random.default_rng(seed)
    a_idx = []
    b_idx = []
    a_labels_sel = []
    b_labels_sel = []
    counts = {}

    for c in common_classes:
        idx_a = np.where(labels_a == c)[0]
        idx_b = np.where(labels_b == c)[0]
        ccount_a = len(idx_a)
        ccount_b = len(idx_b)
        if ccount_a < min_class_count or ccount_b < min_class_count:
            continue
        available = min(ccount_a, ccount_b)
        take = available if n_per_class is None else min(n_per_class, available)
        if take < min_class_count:
            continue
        counts[int(c)] = take
        sel_a = rng.choice(idx_a, size=take, replace=False)
        sel_b = rng.choice(idx_b, size=take, replace=False)
        a_idx.append(sel_a)
        b_idx.append(sel_b)
        a_labels_sel.append(labels_a[sel_a])
        b_labels_sel.append(labels_b[sel_b])

    if not a_idx:
        raise ValueError('No valid common subset could be created. Check overlapping classes and min_class_count.')

    a_idx_all = np.concatenate(a_idx)
    b_idx_all = np.concatenate(b_idx)
    a_labels_all = np.concatenate(a_labels_sel)
    b_labels_all = np.concatenate(b_labels_sel)

    order = np.argsort(a_idx_all)
    a_idx_all = a_idx_all[order]
    a_labels_all = a_labels_all[order]

    order_b = np.argsort(b_idx_all)
    b_idx_all = b_idx_all[order_b]
    b_labels_all = b_labels_all[order_b]

    return {
        'classes': common_classes,
        'n_per_class': counts,
        'a_indices': a_idx_all,
        'b_indices': b_idx_all,
        'a_labels': a_labels_all,
        'b_labels': b_labels_all,
    }


def save_common_subset(output_dir: str, subset: Dict[str, Any], prefix: str = 'common_subset'):
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


def ensure_2d_features(X: np.ndarray):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f'Features must be 2D: {X.shape}')
    return X


def rdm_from_features(X: np.ndarray):
    X = ensure_2d_features(X)
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], X.shape[0]), dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    return 1.0 - sim


def rsa(X: np.ndarray, Y: np.ndarray):
    """Representational Similarity Analysis via correlation of RDMs."""
    X = ensure_2d_features(X)
    Y = ensure_2d_features(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'RSA requires same sample count. Got X shape={X.shape} and Y shape={Y.shape}.')

    rdm_x = rdm_from_features(X)
    rdm_y = rdm_from_features(Y)
    triu_x = rdm_x[np.triu_indices(rdm_x.shape[0], k=1)]
    triu_y = rdm_y[np.triu_indices(rdm_y.shape[0], k=1)]

    if triu_x.size == 0 or triu_y.size == 0:
        return 0.0
    if np.std(triu_x) == 0.0 or np.std(triu_y) == 0.0:
        return 0.0
    return float(np.corrcoef(triu_x, triu_y)[0, 1])


def rsa_heatmap_from_layer_features(features_a, features_b, layer_names=None, model_a_name='Model A', model_b_name='Model B'):
    layer_names = list(layer_names) if layer_names is not None else list(features_a.keys())
    common_names = [n for n in layer_names if n in features_a and n in features_b]
    matrix = np.zeros((len(common_names), len(common_names)), dtype=float)
    for i, name_i in enumerate(common_names):
        rep_i = features_a[name_i]
        for j, name_j in enumerate(common_names):
            rep_j = features_b[name_j]
            if rep_i.shape[0] != rep_j.shape[0]:
                raise ValueError(f'RSA requires same number of samples. Got {rep_i.shape[0]} and {rep_j.shape[0]} for layers {name_i} and {name_j}.')
            matrix[i, j] = rsa(rep_i, rep_j)
    row_labels = [f'{model_a_name}: {name}' for name in common_names]
    col_labels = [f'{model_b_name}: {name}' for name in common_names]
    return row_labels, col_labels, matrix


def save_rsa_heatmap(matrix: np.ndarray, row_labels: Iterable[str], col_labels: Iterable[str], out_path: str, title: str):
    if plt is None:
        raise ModuleNotFoundError('matplotlib is required to save RSA heatmaps.')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap='viridis', vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(list(col_labels), rotation=45, ha='right')
    ax.set_yticklabels(list(row_labels))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white' if abs(matrix[i, j]) < 0.5 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'[INFO] RSA heatmap saved to {out_path}')


def _strip_prefix(sd, prefixes=('module.', 'model.')):
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
                break
        out[kk] = v
    return out


def load_model_from_checkpoint(config_path: str, checkpoint_path: str, device):
    if torch is None:
        raise ModuleNotFoundError('torch is required to load model checkpoints for model-based RSA.')
    import json
    from models.main_model import MainModel
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = MainModel(config)
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw, dict):
        if 'state_dict' in raw:
            raw_state = raw['state_dict']
        elif 'model_state_dict' in raw:
            raw_state = raw['model_state_dict']
        else:
            raw_state = raw
    else:
        raise ValueError(f'Unexpected checkpoint format for {checkpoint_path}: {type(raw)}')

    state_dict = _strip_prefix(raw_state)
    filtered = {}
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v

    missing = set(model_state) - set(filtered)
    if missing:
        print(f'[WARN] {len(missing)}/{len(model_state)} tensors not loaded from {checkpoint_path}; examples: {list(sorted(missing))[:10]}')
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model


def get_dataset_name_from_config(config_path: str):
    import json
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    dataset_cfg = cfg.get('dataset', {})
    return str(dataset_cfg.get('name', 'Model'))


def collect_test_samples(config_path: str, fold: int = 1, max_samples: int = None, split: str = 'test', batch_size: int = None):
    if torch is None:
        raise ModuleNotFoundError('torch is required to collect model test samples.')
    import json
    from torch.utils.data import DataLoader
    from loader import EEGDataLoader
    with open(config_path, 'r') as f:
        config = json.load(f)
    ds = EEGDataLoader(config, fold, set=split)
    if batch_size is None:
        batch_size = config['training_params'].get('batch_size', 8)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    xs, ys = [], []
    total = 0
    for x_batch, y_batch in loader:
        xs.append(x_batch.detach().cpu().numpy())
        ys.append(y_batch.detach().cpu().numpy())
        total += x_batch.shape[0]
        if max_samples is not None and total >= max_samples:
            break
    if not xs:
        raise ValueError(f'No samples found for {config_path} fold {fold} split {split}')
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    return x, y


def extract_layer_features(model, x: np.ndarray, device, layer_names: Iterable[str] = None):
    if torch is None:
        raise ModuleNotFoundError('torch is required to extract model features.')
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if x_t.ndim == 2:
        x_t = x_t.unsqueeze(1)
    if x_t.ndim == 3 and x_t.shape[1] != 1:
        x_t = x_t.transpose(1, 2) if x_t.shape[1] not in (1, 2) else x_t
        x_t = x_t.unsqueeze(1) if x_t.ndim == 2 else x_t

    features = {}
    backbone = model.feature
    h = backbone.init_layer(x_t)
    features['init_layer'] = h.detach().cpu().numpy().reshape(h.shape[0], -1)
    h = backbone.layer1(h)
    features['layer1'] = h.detach().cpu().numpy().reshape(h.shape[0], -1)
    h = backbone.layer2(h)
    features['layer2'] = h.detach().cpu().numpy().reshape(h.shape[0], -1)
    h = backbone.layer3(h)
    features['layer3'] = h.detach().cpu().numpy().reshape(h.shape[0], -1)
    h = backbone.layer4(h)
    features['layer4'] = h.detach().cpu().numpy().reshape(h.shape[0], -1)
    if hasattr(backbone, 'conv_c5'):
        features['p5'] = backbone.conv_c5(h).detach().cpu().numpy().reshape(h.shape[0], -1)
    if layer_names is not None:
        names = list(layer_names)
        return {name: features[name] for name in names if name in features}
    return features


def compute_model_rsa_on_shared_inputs(model_a, model_b, x_shared: np.ndarray, device, model_a_name: str = 'Model A', model_b_name: str = 'Model B'):
    features_a = extract_layer_features(model_a, x_shared, device)
    features_b = extract_layer_features(model_b, x_shared, device)
    return rsa_heatmap_from_layer_features(features_a, features_b, layer_names=list(features_a.keys()), model_a_name=model_a_name, model_b_name=model_b_name)


def compute_model_rsa_on_common_subset(model_a, model_b, x_a_common: np.ndarray, x_b_common: np.ndarray, device, model_a_name: str = 'Model A', model_b_name: str = 'Model B'):
    n = min(len(x_a_common), len(x_b_common))
    x_common = np.concatenate([x_a_common[:n], x_b_common[:n]], axis=0)
    features_a = extract_layer_features(model_a, x_common, device)
    features_b = extract_layer_features(model_b, x_common, device)
    return rsa_heatmap_from_layer_features(features_a, features_b, layer_names=list(features_a.keys()), model_a_name=model_a_name, model_b_name=model_b_name)


def build_common_subset_from_configs(config_a: str, config_b: str, fold_a: int = 1, fold_b: int = 1, n_per_class: int = None, seed: int = 0, max_samples: int = None):
    x_a, y_a = collect_test_samples(config_a, fold=fold_a)
    x_b, y_b = collect_test_samples(config_b, fold=fold_b)
    subset = make_common_subset(y_a, y_b, n_per_class=n_per_class, seed=seed)
    x_a_common = x_a[subset['a_indices']]
    x_b_common = x_b[subset['b_indices']]
    if max_samples is not None:
        n = min(max_samples, len(x_a_common), len(x_b_common))
        x_a_common = x_a_common[:n]
        x_b_common = x_b_common[:n]
    return subset, x_a_common, x_b_common


def resolve_model_checkpoint_paths(model_path):
    if isinstance(model_path, (list, tuple)):
        all_paths = []
        for p in model_path:
            all_paths.extend(resolve_model_checkpoint_paths(p))
        return sorted(set(all_paths))

    path = str(model_path)
    dir_name = os.path.dirname(path)
    if dir_name:
        fold_matches = sorted(glob.glob(os.path.join(dir_name, 'ckpt_fold-*.pth')))
        if fold_matches:
            return fold_matches
    if '{fold}' in path:
        return [path.format(fold=f) for f in range(1, 6)]
    if os.path.exists(path):
        return [path]
    if '*' in path or '?' in path:
        matches = sorted(glob.glob(path))
        if matches:
            return matches
    return [path]


def aggregate_fold_matrices(matrices: List[np.ndarray], mode: str = 'mean'):
    if not matrices:
        raise ValueError('No matrices to aggregate.')
    arr = np.stack(matrices, axis=0)
    if mode == 'mean':
        return np.mean(arr, axis=0)
    if mode == 'median':
        return np.median(arr, axis=0)
    raise ValueError(f'Unsupported aggregation mode: {mode}')


def build_common_subset_from_two_datasets(dataset_a: str, dataset_b: str, output_dir: str, n_per_class: int = None, seed: int = 0,
                                          label_key_a: str = None, label_key_b: str = None, prefix: str = 'common_subset'):
    labels_a = load_labels_from_path(dataset_a, label_key_a)
    labels_b = load_labels_from_path(dataset_b, label_key_b)
    subset = make_common_subset(labels_a, labels_b, n_per_class=n_per_class, seed=seed)
    subset_path = save_common_subset(output_dir, subset, prefix=prefix)
    print(f'[INFO] Common subset built from {dataset_a} and {dataset_b}')
    print(f'[INFO] Classes: {subset["classes"]}')
    print(f'[INFO] n_per_class: {subset["n_per_class"]}')
    print(f'[INFO] Saved to: {subset_path}')
    return subset, subset_path


def parse_args():
    parser = argparse.ArgumentParser(description='Build a matched common hold-out set from two labelled datasets, and compute RSA on DOD-H, ANPHY, and the CKA common subset.')
    parser.add_argument('--dataset-a', type=str, default=None, help='Path to dataset A. Accepts .npy/.npz/.pkl or a folder containing labelled arrays.')
    parser.add_argument('--dataset-b', type=str, default=None, help='Path to dataset B. Accepts .npy/.npz/.pkl or a folder containing labelled arrays.')
    parser.add_argument('--label-key-a', type=str, default=None, help='Optional label field for dataset A, e.g. y or labels.')
    parser.add_argument('--label-key-b', type=str, default=None, help='Optional label field for dataset B, e.g. y or labels.')
    parser.add_argument('--n-per-class', type=int, default=None, help='Max common samples per class to keep. If omitted, uses the minimum available count per class.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used for balanced sampling.')
    parser.add_argument('--output-dir', type=str, default='results/rsa', help='Directory for saved RSA and subset outputs.')
    parser.add_argument('--prefix', type=str, default='common_subset', help='Filename prefix for the saved subset.')
    parser.add_argument('--feature-a', type=str, default=None, help='Optional .npy file with features for dataset A, shape (N, D), aligned with the selected common subset.')
    parser.add_argument('--feature-b', type=str, default=None, help='Optional .npy file with features for dataset B, shape (N, D), aligned with the selected common subset.')
    parser.add_argument('--config-a', type=str, default=None, help='Config JSON for model A for the 3-matrix RSA analysis.')
    parser.add_argument('--config-b', type=str, default=None, help='Config JSON for model B for the 3-matrix RSA analysis.')
    parser.add_argument('--checkpoint-a', type=str, nargs='+', default=None, help='One or more checkpoint .pth files for model A.')
    parser.add_argument('--checkpoint-b', type=str, nargs='+', default=None, help='One or more checkpoint .pth files for model B.')
    parser.add_argument('--fold-a', type=int, default=None, help='Fold index used for dataset A in the three-matrix analysis. If omitted, all available folds are processed.')
    parser.add_argument('--fold-b', type=int, default=None, help='Fold index used for dataset B in the three-matrix analysis. If omitted, all available folds are processed.')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap on samples to keep during the RSA analysis for a quick smoke test.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config_a and args.config_b and args.checkpoint_a and args.checkpoint_b:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_a_name = get_dataset_name_from_config(args.config_a)
        model_b_name = get_dataset_name_from_config(args.config_b)

        checkpoint_paths_a = resolve_model_checkpoint_paths(args.checkpoint_a)
        checkpoint_paths_b = resolve_model_checkpoint_paths(args.checkpoint_b)
        if args.fold_a is not None and args.fold_b is not None:
            checkpoint_paths_a = [p for p in checkpoint_paths_a if f'fold-{args.fold_a:02d}' in os.path.basename(p) or f'fold_{args.fold_a:02d}' in os.path.basename(p)]
            checkpoint_paths_b = [p for p in checkpoint_paths_b if f'fold-{args.fold_b:02d}' in os.path.basename(p) or f'fold_{args.fold_b:02d}' in os.path.basename(p)]

        if not checkpoint_paths_a or not checkpoint_paths_b:
            raise FileNotFoundError(f'No matching checkpoints found for {args.checkpoint_a} and {args.checkpoint_b}.')

        if len(checkpoint_paths_a) != len(checkpoint_paths_b):
            n = min(len(checkpoint_paths_a), len(checkpoint_paths_b))
            checkpoint_paths_a = checkpoint_paths_a[:n]
            checkpoint_paths_b = checkpoint_paths_b[:n]

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        fold_dodh_mats = []
        fold_anphy_mats = []
        fold_common_mats = []
        fold_labels = []

        for idx, (cp_a, cp_b) in enumerate(zip(checkpoint_paths_a, checkpoint_paths_b), start=1):
            model_a = load_model_from_checkpoint(args.config_a, cp_a, device)
            model_b = load_model_from_checkpoint(args.config_b, cp_b, device)

            fold_a = int(os.path.basename(cp_a).split('fold-')[-1].split('.pth')[0]) if 'fold-' in os.path.basename(cp_a) else args.fold_a or idx
            fold_b = int(os.path.basename(cp_b).split('fold-')[-1].split('.pth')[0]) if 'fold-' in os.path.basename(cp_b) else args.fold_b or idx

            x_a, _ = collect_test_samples(args.config_a, fold=fold_a, max_samples=args.max_samples)
            x_b, _ = collect_test_samples(args.config_b, fold=fold_b, max_samples=args.max_samples)

            row_labels, col_labels, mat_dodh = compute_model_rsa_on_shared_inputs(model_a, model_b, x_a, device, model_a_name=model_a_name, model_b_name=model_b_name)
            save_rsa_heatmap(mat_dodh, row_labels, col_labels, os.path.join(output_dir, f'rsa_heatmap_fold_{fold_a}_dodh_testset.png'), f'RSA fold {fold_a}: DOD-H test set')
            fold_dodh_mats.append(mat_dodh)

            row_labels, col_labels, mat_anphy = compute_model_rsa_on_shared_inputs(model_a, model_b, x_b, device, model_a_name=model_a_name, model_b_name=model_b_name)
            save_rsa_heatmap(mat_anphy, row_labels, col_labels, os.path.join(output_dir, f'rsa_heatmap_fold_{fold_b}_anphy_testset.png'), f'RSA fold {fold_b}: ANPHY test set')
            fold_anphy_mats.append(mat_anphy)

            subset, x_a_common, x_b_common = build_common_subset_from_configs(args.config_a, args.config_b, fold_a=fold_a, fold_b=fold_b, n_per_class=args.n_per_class, seed=args.seed, max_samples=args.max_samples)
            row_labels, col_labels, mat_common = compute_model_rsa_on_common_subset(model_a, model_b, x_a_common, x_b_common, device, model_a_name=model_a_name, model_b_name=model_b_name)
            save_rsa_heatmap(mat_common, row_labels, col_labels, os.path.join(output_dir, f'rsa_heatmap_fold_{fold_a}_common_subset.png'), f'RSA fold {fold_a}: common subset')
            fold_common_mats.append(mat_common)
            fold_labels.append((row_labels, col_labels))

            print(f'[INFO] Fold {fold_a}/{fold_b} complete: {cp_a} vs {cp_b}')

        if fold_dodh_mats:
            row_labels, col_labels = fold_labels[0]
            mean_dodh = aggregate_fold_matrices(fold_dodh_mats, mode='mean')
            median_dodh = aggregate_fold_matrices(fold_dodh_mats, mode='median')
            save_rsa_heatmap(mean_dodh, row_labels, col_labels, os.path.join(output_dir, 'rsa_heatmap_mean_over_folds_dodh_testset.png'), 'RSA mean over 5 folds: DOD-H test set')
            save_rsa_heatmap(median_dodh, row_labels, col_labels, os.path.join(output_dir, 'rsa_heatmap_median_over_folds_dodh_testset.png'), 'RSA median over 5 folds: DOD-H test set')

            mean_anphy = aggregate_fold_matrices(fold_anphy_mats, mode='mean')
            median_anphy = aggregate_fold_matrices(fold_anphy_mats, mode='median')
            save_rsa_heatmap(mean_anphy, row_labels, col_labels, os.path.join(output_dir, 'rsa_heatmap_mean_over_folds_anphy_testset.png'), 'RSA mean over 5 folds: ANPHY test set')
            save_rsa_heatmap(median_anphy, row_labels, col_labels, os.path.join(output_dir, 'rsa_heatmap_median_over_folds_anphy_testset.png'), 'RSA median over 5 folds: ANPHY test set')

            mean_common = aggregate_fold_matrices(fold_common_mats, mode='mean')
            median_common = aggregate_fold_matrices(fold_common_mats, mode='median')
            save_rsa_heatmap(mean_common, row_labels, col_labels, os.path.join(output_dir, 'rsa_heatmap_mean_over_folds_common_subset.png'), 'RSA mean over 5 folds: common subset')
            save_rsa_heatmap(median_common, row_labels, col_labels, os.path.join(output_dir, 'rsa_heatmap_median_over_folds_common_subset.png'), 'RSA median over 5 folds: common subset')

        print('[INFO] Finished all requested RSA folds in', output_dir)
        return

    subset, _ = build_common_subset_from_two_datasets(
        dataset_a=args.dataset_a,
        dataset_b=args.dataset_b,
        output_dir=args.output_dir,
        n_per_class=args.n_per_class,
        seed=args.seed,
        label_key_a=args.label_key_a,
        label_key_b=args.label_key_b,
        prefix=args.prefix,
    )

    if args.feature_a and args.feature_b:
        X = np.load(args.feature_a)
        Y = np.load(args.feature_b)
        if X.shape[0] != len(subset['a_indices']) or Y.shape[0] != len(subset['b_indices']):
            print('[WARN] Feature matrices do not match the selected common subset size; RSA will still be computed if shapes are compatible.')
        print(f'[INFO] RSA = {rsa(X, Y):.6f}')


if __name__ == '__main__':
    main()
