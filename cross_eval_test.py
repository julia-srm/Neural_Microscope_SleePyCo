import os
import json
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import summarize_result, set_random_seed
from loader import EEGDataLoader
from models.main_model import MainModel


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute the confusion matrix for a set of true and predicted labels."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return np.zeros((num_classes, num_classes), dtype=int)
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def print_confusion_matrix(cm, title='Confusion Matrix', labels=None):
    """Print a confusion matrix to stdout."""
    if labels is None:
        labels = list(range(cm.shape[0]))

    print('\n[INFO] {}\n'.format(title))
    header = '     ' + ' '.join(f'{lab:>6}' for lab in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f'{labels[i]:>3}:', ' '.join(f'{val:>6}' for val in row))


def compute_relative_performance(cm):
    """Compute the normalized/relative performance per class.

    For each true class s and predicted class t, the value is g[s, t] / g[s, s],
    where g[s, s] is the within-dataset diagonal performance. This highlights how much
    a class is over- or under-predicted relative to its own best diagonal match.
    """
    cm = np.asarray(cm, dtype=float)
    diag = np.diag(cm).copy()
    denom = np.where(np.abs(diag) > 0, diag, 1.0)
    rel = np.divide(cm, denom[:, None], out=np.zeros_like(cm, dtype=float), where=np.abs(denom[:, None]) > 0)
    return rel


def save_confusion_matrix_as_jpg(cm, out_path, title='Confusion Matrix', labels=None, normalize=False, show_relative=False):
    """Save a confusion matrix as a JPG image.

    When normalize=True, each row is normalized by the total number of samples in the
    corresponding true class so that the color intensity reflects the class-distribution
    within that row, and the displayed values are percentages of the true-class total.
    If show_relative=True, the figure also includes the normalized/relative performance
    matrix g[s, t] / g[s, s].
    """
    if labels is None:
        labels = list(range(cm.shape[0]))

    cm_raw = np.asarray(cm, dtype=float)
    cm_display = cm_raw.copy()
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm_display / row_sums

    if show_relative:
        relative = compute_relative_performance(cm_raw)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_main = axes[0]
        ax_rel = axes[1]
    else:
        fig, ax_main = plt.subplots(figsize=(7, 6))
        ax_rel = None

    im = ax_main.imshow(cm_display, cmap='Blues', vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
    ax_main.set_title(title)
    ax_main.set_xlabel('Predicted label')
    ax_main.set_ylabel('True label')
    ax_main.set_xticks(np.arange(len(labels)))
    ax_main.set_yticks(np.arange(len(labels)))
    ax_main.set_xticklabels(labels)
    ax_main.set_yticklabels(labels)

    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            if normalize:
                text = f'{val * 100:.1f}%'
                txt_color = 'black' if val < 0.7 else 'white'
            else:
                text = f'{int(cm_raw[i, j])}'
                txt_color = 'black' if val < 0.7 else 'white'
            ax_main.text(j, i, text, ha='center', va='center', color=txt_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    if normalize:
        cbar.set_label('Share of true class (%)', rotation=270, labelpad=15)

    if show_relative:
        rel_vmax = max(1.0, np.nanmax(relative)) if np.size(relative) else 1.0
        im_rel = ax_rel.imshow(relative, cmap='viridis', vmin=0.0, vmax=rel_vmax)
        ax_rel.set_title('Normalized/Relative Performance\n(g[s,t] / g[s,s])')
        ax_rel.set_xlabel('Predicted label')
        ax_rel.set_ylabel('True label')
        ax_rel.set_xticks(np.arange(len(labels)))
        ax_rel.set_yticks(np.arange(len(labels)))
        ax_rel.set_xticklabels(labels)
        ax_rel.set_yticklabels(labels)

        for i in range(relative.shape[0]):
            for j in range(relative.shape[1]):
                val = relative[i, j]
                text = f'{val:.2f}'
                txt_color = 'black' if val < rel_vmax / 2.0 else 'white'
                ax_rel.text(j, i, text, ha='center', va='center', color=txt_color, fontsize=7)

        cbar_rel = fig.colorbar(im_rel, ax=ax_rel, fraction=0.046, pad=0.04)
        cbar_rel.set_label('Relative to within-dataset diagonal', rotation=270, labelpad=18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print('[INFO] Confusion matrix image saved to {}'.format(out_path))


def compute_asymmetry_index(cm_ab, cm_ba):
    """Compute ΔAB = C(A→B) - C(B→A)."""
    cm_ab = np.asarray(cm_ab, dtype=float)
    cm_ba = np.asarray(cm_ba, dtype=float)
    if cm_ab.shape != cm_ba.shape:
        raise ValueError('Confusion matrices must have the same shape for asymmetry computation. Got {} and {}.'.format(cm_ab.shape, cm_ba.shape))
    return cm_ab - cm_ba


class CrossDatasetEvaluator:
    def __init__(self, args, fold, target_cfg, source_ckpt_dir):
        self.args = args
        self.fold = fold

        self.cfg = target_cfg
        self.ds_cfg = target_cfg['dataset']
        self.tp_cfg = target_cfg['training_params']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Target config name: {}'.format(target_cfg['name']))
        print('[INFO] Source checkpoint dir: {}'.format(source_ckpt_dir))

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = source_ckpt_dir
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)

    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def build_dataloader(self):
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.tp_cfg['batch_size'],
            shuffle=False,
            num_workers=4 * len(self.args.gpu.split(",")),
            pin_memory=True,
        )
        print('[INFO] Dataloader prepared')
        return {'test': test_loader}

    def load_cross_checkpoint(self):
        ckpt_file = os.path.join(self.ckpt_path, self.ckpt_name)
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError('[ERROR] Checkpoint not found: {}'.format(ckpt_file))

        src_state = torch.load(ckpt_file, map_location=self.device, weights_only=True)
        target_state = self.model.state_dict()

        filtered_state = {}
        skipped = []
        for k, v in src_state.items():
            if k not in target_state:
                continue
            if target_state[k].shape != v.shape:
                skipped.append((k, tuple(target_state[k].shape), tuple(v.shape)))
                continue
            filtered_state[k] = v

        if skipped:
            print('[WARN] Skipping {} layers because the source and target shapes do not match:'.format(len(skipped)))
            for k, tgt_shape, src_shape in skipped:
                print('       - {}: target={} source={}'.format(k, tgt_shape, src_shape))

        missing = set(target_state.keys()) - set(filtered_state.keys())
        if missing:
            print('[WARN] Missing {} parameters in the loaded checkpoint; they will stay at their initialized values.'.format(len(missing)))
            for k in sorted(list(missing))[:10]:
                print('       - {}'.format(k))

        self.model.load_state_dict(filtered_state, strict=False)
        print('[INFO] Loaded source checkpoint with shape-compatible parameters only.')

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]

            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()

            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

        return y_true, y_pred

    def run(self):
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.load_cross_checkpoint()
        y_true, y_pred = self.evaluate(mode='test')
        print('')
        return y_true, y_pred


def main():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--src-config', type=str, required=True, help='source config file used to train the checkpoint')
    parser.add_argument('--tgt-config', type=str, required=True, help='target dataset config used for evaluation')
    parser.add_argument('--ckpt-dir', type=str, default=None, help='optional checkpoint directory override; defaults to checkpoints/<source-config-name>')
    parser.add_argument('--compare-cm', type=str, default=None, help='optional path to a saved confusion matrix from the reverse direction; if provided, compute ΔAB = C(A→B) - C(B→A)')
    parser.add_argument('--cm-out', type=str, default=None, help='optional output file for saving the confusion matrix (.npy)')
    parser.add_argument('--cm-jpg', type=str, default=None, help='optional output path for saving a JPG visual confusion matrix')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    set_random_seed(args.seed, use_cuda=True)

    with open(args.tgt_config) as config_file:
        target_cfg = json.load(config_file)

    source_cfg_name = os.path.basename(args.src_config).replace('.json', '')
    target_cfg['name'] = os.path.basename(args.tgt_config).replace('.json', '') + '_cross_' + source_cfg_name
    target_cfg['config_path'] = args.tgt_config

    if args.ckpt_dir is None:
        ckpt_dir = os.path.join('checkpoints', source_cfg_name)
    else:
        ckpt_dir = args.ckpt_dir

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError('[ERROR] Source checkpoint directory not found: {}'.format(ckpt_dir))

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, target_cfg['classifier']['num_classes']))

    for fold in range(1, target_cfg['dataset']['num_splits'] + 1):
        evaluator = CrossDatasetEvaluator(args, fold, target_cfg, ckpt_dir)
        y_true, y_pred = evaluator.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(target_cfg, fold, Y_true, Y_pred)

    y_pred_argmax = np.argmax(Y_pred, axis=1)
    cm = compute_confusion_matrix(Y_true, y_pred_argmax, target_cfg['classifier']['num_classes'])
    print_confusion_matrix(cm, title='Cross-dataset confusion matrix (target dataset)', labels=list(range(target_cfg['classifier']['num_classes'])))

    if args.cm_out is None:
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        cm_out = os.path.join(output_dir, target_cfg['name'] + '_cm.npy')
    else:
        output_dir = os.path.dirname(args.cm_out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cm_out = args.cm_out

    np.save(cm_out, cm)
    print('[INFO] Confusion matrix saved to {}'.format(cm_out))

    if args.cm_jpg is not None:
        jpg_dir = os.path.dirname(args.cm_jpg)
        if jpg_dir:
            os.makedirs(jpg_dir, exist_ok=True)
        save_confusion_matrix_as_jpg(
            cm,
            args.cm_jpg,
            title='Cross-dataset confusion matrix (row-normalized by true class)',
            labels=list(range(target_cfg['classifier']['num_classes'])),
            normalize=True,
            show_relative=True,
        )

    if args.compare_cm is not None:
        if not os.path.exists(args.compare_cm):
            raise FileNotFoundError('[ERROR] Reverse confusion matrix not found: {}'.format(args.compare_cm))

        cm_reverse = np.load(args.compare_cm)
        delta_ab = compute_asymmetry_index(cm, cm_reverse)
        print('\n[INFO] Confusion asymmetry index ΔAB = C(A→B) - C(B→A)')
        print(delta_ab)

        delta_out = cm_out.replace('_cm.npy', '_delta_ab.npy')
        np.save(delta_out, delta_ab)
        print('[INFO] Asymmetry matrix saved to {}'.format(delta_out))


if __name__ == '__main__':
    main()
