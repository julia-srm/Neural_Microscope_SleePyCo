#!/usr/bin/env python
"""
Layer-wise Relevance Propagation (LRP) fuer SleePyCo mit Zennit.

Lege dieses Skript in das Repo-Root (neben train_mtcl.py) und starte z.B.:

    python attribute_lrp.py \
        --config configs/SleePyCo-Transformer_SL-10_numScales-3_ANPHY_scratch_5classes.json \
        --fold 1 --per-class 10 --gpu 0

Was passiert
------------
1. Das trainierte Modell (checkpoints/<config-name>/ckpt_fold-XX.pth) wird geladen.
2. Es wird ein LRP-faehiger Nachbau gebaut (gleiche Gewichte, geprueft: identische Logits).
3. Fuer eine stratifizierte Stichprobe des Testsets wird die Relevanz der *summierten*
   Logits (o3 + o4 + o5) der vorhergesagten (oder wahren) Klasse bis auf das Roh-EEG
   (30000 Samples = 10 Epochen a 30 s bei 100 Hz) zurueckpropagiert.
4. Ergebnisse (npz, csv, Plots) landen in --out.

Behandlung der nicht-LRP-konformen Stellen (siehe Klassen unten)
-----------------------------------------------------------------
* Conv -> BatchNorm        : BatchNorm wird per Canonizer in die Conv gemerged.
* ChannelGate (SE-Block)   : Gate-Faktor wird als Konstante behandelt, Relevanz geht unveraendert durch.
* Self-Attention           : Attention-Gewichte A werden als Konstante behandelt; Relevanz fliesst
                             nur ueber den Value-Pfad (konservative Propagation, Ali et al. 2022).
* LayerNorm, Pos.-Encoding : Identitaetsregel (Relevanz geht unveraendert durch).
* Residual-Additionen      : Epsilon-Regel auf einer Summe (proportionale Aufteilung).
* Attention-Pooling        : Pooling-Gewichte alpha konstant, Epsilon-Regel auf der gewichteten Summe.
* 3 Skalen (o3,o4,o5)      : Summe der Logits als Zennit-Sum-Layer (Epsilon); jede Skala bekommt eine
                             eigene Classifier-Kopie, weil Zennit-Hooks bei mehrfach aufgerufenen
                             Modulen mit unterschiedlicher Eingabelaenge sonst falsch rechnen.
"""
import argparse
import copy
import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.core import Composite
from zennit.layer import Sum
from zennit.rules import Epsilon, Gamma, Pass, ZBox
from zennit.types import Activation, Convolution

from models.main_model import MainModel
from models.sleepyco import ChannelGate

SAMPLE_RATE = 100
EPOCH_LEN = 30 * SAMPLE_RATE  # 3000 Samples
DEFAULT_CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 15), 'beta': (15, 30), 'gamma': (30, 49)}


# ----------------------------------------------------------------------------
# LRP-faehige Bausteine
# ----------------------------------------------------------------------------
class ConstMatMul(nn.Module):
    """y = A @ v mit konstantem (detached) A, das vor dem Aufruf als Attribut gesetzt wird.
    Fuer Zennit ein lineares Modul in v -> Epsilon-Regel verteilt die Relevanz proportional zu A_ij * v_j."""

    def __init__(self):
        super().__init__()
        self.A = None

    def forward(self, v):
        return torch.matmul(self.A, v)


class MeanDim(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(self.dim)


class WeightedSum(nn.Module):
    """sum_t alpha_t * x_t mit konstantem alpha (B, T, 1)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = None

    def forward(self, x):
        return (self.alpha * x).sum(self.dim)


class FrozenChannelGate(nn.Module):
    """SE-Gate aus SleePyCo (nur 'avg'-Pooling). Skalierungsfaktor wird ohne Gradient berechnet
    und als Konstante multipliziert; Zennit bekommt dafuer die Pass-Regel."""

    def __init__(self, gate):
        super().__init__()
        assert list(gate.pool_types) == ['avg'], 'nur pool_types=[avg] wird unterstuetzt'
        lin1, lin2 = gate.mlp[1], gate.mlp[3]
        self.register_buffer('w1', lin1.weight.detach().clone())
        self.register_buffer('b1', lin1.bias.detach().clone())
        self.register_buffer('w2', lin2.weight.detach().clone())
        self.register_buffer('b2', lin2.bias.detach().clone())

    def forward(self, x):
        with torch.no_grad():
            p = x.mean(dim=2)  # == avg_pool1d ueber die ganze Laenge + Flatten
            s = torch.sigmoid(F.linear(F.relu(F.linear(p, self.w1, self.b1)), self.w2, self.b2))
        return x * s.unsqueeze(2)


def replace_channel_gates(module):
    for name, child in module.named_children():
        if isinstance(child, ChannelGate):
            setattr(module, name, FrozenChannelGate(child))
        else:
            replace_channel_gates(child)


class AddPositionalEncoding(nn.Module):
    """x + pe, exakt wie in classifiers.PositionalEncoding, aber batch-first (B, T, D)."""

    def __init__(self, pos_enc):
        super().__init__()
        self.num_scales = pos_enc.num_scales
        self.max_len = pos_enc.max_len
        self.register_buffer('pe', pos_enc.pe.detach().clone())  # (max_len, 1, D)

    def forward(self, x):
        T = x.size(1)
        if self.num_scales > 1:
            hop = self.max_len // T
            pe = self.pe[hop // 2::hop, :]
        else:
            pe = self.pe
        if pe.shape[0] != T:
            pe = pe[:T, :]
        return x + pe.transpose(0, 1)


class LRPSelfAttention(nn.Module):
    """Multi-Head-Self-Attention (batch-first). A = softmax(QK^T/sqrt(d)) wird ohne Gradient
    berechnet; die Relevanz laeuft ueber Value- und Output-Projektion."""

    def __init__(self, mha):
        super().__init__()
        d, h = mha.embed_dim, mha.num_heads
        assert mha.in_proj_weight is not None, 'nur gemeinsame in_proj (kdim == vdim == embed_dim)'
        self.h, self.hd = h, d // h
        W = mha.in_proj_weight.detach().clone()
        b = mha.in_proj_bias.detach().clone() if mha.in_proj_bias is not None else torch.zeros(3 * d)
        self.register_buffer('wq', W[:d].clone())
        self.register_buffer('wk', W[d:2 * d].clone())
        self.register_buffer('bq', b[:d].clone())
        self.register_buffer('bk', b[d:2 * d].clone())
        self.v_proj = nn.Linear(d, d)
        self.v_proj.weight.data.copy_(W[2 * d:])
        self.v_proj.bias.data.copy_(b[2 * d:])
        self.mix = ConstMatMul()
        self.out_proj = nn.Linear(d, d)
        self.out_proj.weight.data.copy_(mha.out_proj.weight.detach())
        if mha.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(mha.out_proj.bias.detach())
        else:
            self.out_proj.bias.data.zero_()

    def forward(self, x):  # (B, S, D)
        B, S, D = x.shape
        with torch.no_grad():
            q = F.linear(x, self.wq, self.bq).view(B, S, self.h, self.hd).transpose(1, 2)
            k = F.linear(x, self.wk, self.bk).view(B, S, self.h, self.hd).transpose(1, 2)
            self.mix.A = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.hd), dim=-1)
        v = self.v_proj(x).view(B, S, self.h, self.hd).transpose(1, 2)
        o = self.mix(v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(o)


class LRPEncoderLayer(nn.Module):
    """Nachbau von nn.TransformerEncoderLayer (post-norm, ReLU, kein Dropout)."""

    def __init__(self, layer):
        super().__init__()
        assert not layer.norm_first, 'nur post-norm wird unterstuetzt'
        act = layer.activation
        act_name = getattr(act, '__name__', type(act).__name__).lower()
        assert 'relu' in act_name, 'nur ReLU im Feed-Forward wird unterstuetzt (gefunden: %s)' % act_name
        self.attn = LRPSelfAttention(layer.self_attn)
        self.res1 = Sum(dim=-1)
        self.norm1 = copy.deepcopy(layer.norm1)
        self.lin1 = copy.deepcopy(layer.linear1)
        self.act = nn.ReLU()
        self.lin2 = copy.deepcopy(layer.linear2)
        self.res2 = Sum(dim=-1)
        self.norm2 = copy.deepcopy(layer.norm2)

    def forward(self, x):
        x = self.norm1(self.res1(torch.stack([x, self.attn(x)], dim=-1)))
        h = self.lin2(self.act(self.lin1(x)))
        return self.norm2(self.res2(torch.stack([x, h], dim=-1)))


class LRPTransformerClassifier(nn.Module):
    """Nachbau von classifiers.Transformer.forward (Eingabe (B, T, C) wie in MainModel)."""

    def __init__(self, clf):
        super().__init__()
        assert clf.transformer.norm is None
        self.pe_fc = copy.deepcopy(clf.pos_encoding.fc)
        self.pe_act = copy.deepcopy(clf.pos_encoding.act_fn)
        self.pe_add = AddPositionalEncoding(clf.pos_encoding)
        self.layers = nn.ModuleList([LRPEncoderLayer(l) for l in clf.transformer.layers])
        self.pool = clf.pool
        if self.pool == 'attn':
            self.w_ha = copy.deepcopy(clf.w_ha)
            self.tanh = nn.Tanh()
            self.register_buffer('w_at', clf.w_at.weight.detach().clone())
            self.wsum = WeightedSum(1)
        elif self.pool == 'mean':
            self.mean = MeanDim(1)
        else:
            raise NotImplementedError('pool=%s wird nicht unterstuetzt (nur attn, mean)' % self.pool)
        self.fc = copy.deepcopy(clf.fc)

    def forward(self, x):
        x = self.pe_add(self.pe_act(self.pe_fc(x)))
        for layer in self.layers:
            x = layer(x)
        if self.pool == 'attn':
            a = self.tanh(self.w_ha(x))
            with torch.no_grad():
                self.wsum.alpha = torch.softmax(F.linear(a, self.w_at), dim=1)  # (B, T, 1)
            x = self.wsum(a)
        else:
            x = self.mean(x)
        return self.fc(x)


class LRPSleePyCo(nn.Module):
    """Backbone + eine Classifier-Kopie pro Skala + Summe der Logits."""

    def __init__(self, model):
        super().__init__()
        self.feature = copy.deepcopy(model.feature)
        replace_channel_gates(self.feature)
        n_scales = getattr(self.feature, 'num_scales', 1)
        self.classifiers = nn.ModuleList([LRPTransformerClassifier(model.classifier) for _ in range(n_scales)])
        self.branch_sum = Sum(dim=-1)

    def forward(self, x):
        feats = self.feature(x)
        outs = [clf(f.transpose(1, 2)) for clf, f in zip(self.classifiers, feats)]
        return self.branch_sum(torch.stack(outs, dim=-1))


# ----------------------------------------------------------------------------
# Composite
# ----------------------------------------------------------------------------
def build_composite(model, low, high, gamma, epsilon):
    first_conv = next(n for n, m in model.named_modules() if isinstance(m, nn.Conv1d))

    def module_map(ctx, name, module):
        if name == first_conv:
            return ZBox(low, high)
        if isinstance(module, (FrozenChannelGate, AddPositionalEncoding, nn.LayerNorm)):
            return Pass()
        if isinstance(module, Activation):  # ReLU, PReLU, Tanh, ...
            return Pass()
        if isinstance(module, Convolution):
            return Gamma(gamma=gamma)
        if isinstance(module, (nn.Linear, Sum, ConstMatMul, MeanDim, WeightedSum)):
            return Epsilon(epsilon=epsilon)
        return None

    return Composite(module_map=module_map, canonizers=[SequentialMergeBatchNorm()])


# ----------------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------------
def attribute_batch(model, lrp_model, composite, x, labels, target_mode):
    with torch.no_grad():
        branches = torch.stack(model(x), dim=1)  # (B, n_scales, C)
        logits = branches.sum(1)
        lrp_logits = lrp_model(x)
    max_diff = (logits - lrp_logits).abs().max().item()
    if max_diff > 1e-3 * max(1.0, logits.abs().max().item()):
        raise RuntimeError('LRP-Nachbau weicht vom Originalmodell ab (max |dlogit| = %.3e). '
                           'Relevanzen waeren nicht vertrauenswuerdig.' % max_diff)

    pred = logits.argmax(1)
    target = pred if target_mode == 'pred' else labels
    seed = F.one_hot(target, logits.shape[1]).to(logits.dtype) * logits
    with Gradient(model=lrp_model, composite=composite) as attributor:
        _, relevance = attributor(x, seed)
    return dict(relevance=relevance.detach(), logits=logits, branches=branches, pred=pred, target=target,
                target_logit=logits.gather(1, target[:, None])[:, 0], max_diff=max_diff)


def run_attribution(model, x_all, labels_all, args, device, low, high, meta=None):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lrp_model = LRPSleePyCo(model).to(device).eval()
    composite = build_composite(lrp_model, low, high, args.gamma, args.epsilon)

    res = {k: [] for k in ['relevance', 'signal', 'label', 'pred', 'target', 'target_logit', 'logits', 'branches']}
    n = len(x_all)
    for i in range(0, n, args.batch_size):
        x = x_all[i:i + args.batch_size].to(device)
        y = labels_all[i:i + args.batch_size].to(device)
        out = attribute_batch(model, lrp_model, composite, x, y, args.target)
        res['relevance'].append(out['relevance'][:, 0].cpu().numpy())
        res['signal'].append(x[:, 0].cpu().numpy())
        res['label'].append(y.cpu().numpy())
        for k in ['pred', 'target', 'target_logit', 'logits', 'branches']:
            res[k].append(out[k].cpu().numpy())
        print('[LRP] %d/%d  (max |dlogit| Nachbau vs. Original: %.1e)' % (min(i + args.batch_size, n), n, out['max_diff']))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    res = {k: np.concatenate(v).astype(np.float32 if k in ('relevance', 'signal', 'logits', 'branches', 'target_logit') else np.int64)
           for k, v in res.items()}
    res['rel_sum'] = res['relevance'].sum(1)
    return res


# ----------------------------------------------------------------------------
# Auswertung
# ----------------------------------------------------------------------------
def analyze(data, out_dir, class_names, plots=True):
    R = data['relevance']
    N = R.shape[0]
    n_ep = R.shape[1] // EPOCH_LEN
    R_ep = R.reshape(N, n_ep, EPOCH_LEN)
    abs_share = np.abs(R_ep).sum(2) / np.maximum(np.abs(R_ep).sum((1, 2))[:, None], 1e-12)  # (N, n_ep)
    signed = R_ep.sum(2)  # (N, n_ep)
    target = data['target']

    # --- CSV pro Sample
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'subject', 'start_epoch', 'label', 'pred', 'target', 'target_logit', 'rel_sum', 'rel_sum/logit']
                   + ['share_abs_ep%d' % (i - n_ep + 1) for i in range(n_ep)])
        for i in range(N):
            subj = data['subject'][i] if 'subject' in data else ''
            start = data['start_epoch'][i] if 'start_epoch' in data else ''
            w.writerow([i, subj, start, data['label'][i], data['pred'][i], target[i], '%.4f' % data['target_logit'][i],
                        '%.4f' % data['rel_sum'][i], '%.4f' % (data['rel_sum'][i] / (data['target_logit'][i] + 1e-12))]
                       + ['%.5f' % v for v in abs_share[i]])
    ratio = data['rel_sum'] / (data['target_logit'] + 1e-12)
    print('[INFO] Relevanzerhaltung sum(R)/Logit: Median %.3f (IQR %.3f - %.3f). Werte < 1 sind normal: '
          'Biases, Gate/Attention-Konstanten und Epsilon nehmen Relevanz auf.'
          % (np.median(ratio), np.percentile(ratio, 25), np.percentile(ratio, 75)))

    # --- Bandanteile der Relevanz im Zielepoch (letzte Epoche)
    from scipy.signal import welch
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # np.trapz wurde in NumPy 2 entfernt
    tgt_R = R_ep[:, -1, :]
    freqs, psd = welch(tgt_R, fs=SAMPLE_RATE, nperseg=300, axis=1)
    rows = []
    for c in sorted(set(target.tolist())):
        m = target == c
        tot = trapezoid(psd[m][:, (freqs >= 0.5) & (freqs <= 49)], freqs[(freqs >= 0.5) & (freqs <= 49)], axis=1)
        row = [class_names[c], int(m.sum())]
        for lo, hi in BANDS.values():
            sel = (freqs >= lo) & (freqs < hi)
            row.append('%.3f' % np.mean(trapezoid(psd[m][:, sel], freqs[sel], axis=1) / np.maximum(tot, 1e-20)))
        rows.append(row)
    with open(os.path.join(out_dir, 'band_relevance.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'n'] + list(BANDS))
        w.writerows(rows)

    if not plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    classes = sorted(set(target.tolist()))
    colors = plt.cm.tab10(np.arange(len(class_names)))

    # 1) Relevanz pro Epochenposition
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    pos = np.arange(n_ep) - n_ep + 1
    for c in classes:
        m = target == c
        ax[0].plot(pos, abs_share[m].mean(0), marker='o', color=colors[c], label='%s (n=%d)' % (class_names[c], m.sum()))
        ax[1].plot(pos, signed[m].mean(0), marker='o', color=colors[c])
    ax[0].set_title('Anteil an Summe |R| pro Epoche')
    ax[1].set_title('Mittlere signierte Relevanz pro Epoche (Logit-Einheiten)')
    for a in ax:
        a.set_xlabel('Epoche relativ zur Zielepoche (0 = Label)')
        a.axhline(0, color='k', lw=0.5)
    ax[0].legend(title='erklaerte Klasse')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'relevance_per_epoch.png'), dpi=150)
    plt.close(fig)

    # 2) Spektrum der Relevanz im Zielepoch
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in classes:
        m = target == c
        ax.semilogy(freqs, psd[m].mean(0), color=colors[c], label=class_names[c])
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, color='gray', alpha=0.08)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], name, ha='center', va='top', fontsize=7)
    ax.set_xlim(0, 49)
    ax.set_xlabel('Frequenz [Hz]')
    ax.set_ylabel('PSD der Relevanz R(t)')
    ax.set_title('Spektrum der Relevanz im Zielepoch (Welch, 3 s Fenster)')
    ax.legend(title='erklaerte Klasse')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'relevance_spectrum.png'), dpi=150)
    plt.close(fig)

    # 3) Beispiele: EEG + Relevanz
    win = np.ones(100) / 100
    for c in classes:
        idx = np.where(target == c)[0][:2]
        fig, axes = plt.subplots(2 * len(idx), 1, figsize=(14, 3.2 * len(idx)), sharex=True)
        axes = np.atleast_1d(axes)
        for j, i in enumerate(idx):
            t = np.arange(R.shape[1]) / SAMPLE_RATE
            r = np.convolve(R[i], win, mode='same')
            a_sig, a_rel = axes[2 * j], axes[2 * j + 1]
            a_sig.plot(t, data['signal'][i], color='0.25', lw=0.4)
            a_sig.set_ylabel('EEG')
            a_sig.set_title('%s: Label=%s, Pred=%s, Logit=%.2f'
                            % (class_names[c], class_names[data['label'][i]], class_names[data['pred'][i]], data['target_logit'][i]), fontsize=9)
            a_rel.fill_between(t, 0, r, where=r > 0, color='crimson', lw=0)
            a_rel.fill_between(t, 0, r, where=r < 0, color='royalblue', lw=0)
            a_rel.set_ylabel('R (1 s gegl.)')
            for a in (a_sig, a_rel):
                for e in range(1, n_ep):
                    a.axvline(e * 30, color='k', lw=0.4, alpha=0.4)
                a.axvspan((n_ep - 1) * 30, n_ep * 30, color='gold', alpha=0.12)
        axes[-1].set_xlim(0, n_ep * 30)
        axes[-1].set_xlabel('Zeit [s]  (gelb = Zielepoche)')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'examples_%s.png' % class_names[c]), dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Daten / main
# ----------------------------------------------------------------------------
def select_samples(dataset, per_class, seed):
    rng = np.random.RandomState(seed)
    labels = np.array([dataset.labels[fi][i:i + sl][dataset.target_idx] for fi, i, sl in dataset.epochs])
    chosen = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        chosen += rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist()
    return sorted(chosen)


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--config', type=str, help='Config-JSON (wie beim Training)')
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--ckpt', type=str, default=None, help='Default: checkpoints/<config-name>/ckpt_fold-XX.pth')
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--out', type=str, default='lrp_results')
    ap.add_argument('--per-class', type=int, default=10, help='Samples pro Klasse (Testset, zufaellig, stratifiziert)')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--target', choices=['pred', 'label'], default='pred', help='Welche Klasse wird erklaert')
    ap.add_argument('--gamma', type=float, default=0.25, help='Gamma-Regel in den Convs')
    ap.add_argument('--epsilon', type=float, default=1e-6, help='Epsilon-Regel in Linear/Sum')
    ap.add_argument('--box', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                    help='ZBox-Grenzen; Default: Min/Max der Test-Aufnahmen')
    ap.add_argument('--class-names', type=str, nargs='+', default=DEFAULT_CLASS_NAMES)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--analyze', type=str, default=None, help='Nur Auswertung/Plots aus vorhandener npz neu erzeugen')
    ap.add_argument('--selftest', action='store_true', help='Zufallsgewichte + Zufallssignal, ohne Daten/Checkpoint')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.analyze:
        data = dict(np.load(args.analyze, allow_pickle=True))
        analyze(data, args.out, args.class_names, plots=not args.no_plots)
        return

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = json.load(f)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    assert config['backbone']['name'] == 'SleePyCo' and config['classifier']['name'] == 'Transformer', \
        'Dieses Skript ist fuer SleePyCo + Transformer-Classifier geschrieben.'

    model = MainModel(config).to(device)
    meta = {}
    if args.selftest:
        n = 2 * len(args.class_names)
        x_all = torch.randn(n, 1, EPOCH_LEN * config['dataset']['seq_len'])
        labels_all = torch.arange(n) % len(args.class_names)
        low, high = float(x_all.min()), float(x_all.max())
    else:
        from loader import EEGDataLoader  # lazy: braucht transform.py & Daten
        ckpt = args.ckpt or os.path.join('checkpoints', config['name'], 'ckpt_fold-{0:02d}.pth'.format(args.fold))
        load_checkpoint(model, ckpt, device)
        print('[INFO] Checkpoint geladen: %s' % ckpt)
        dataset = EEGDataLoader(config, args.fold, set='test')
        idx = select_samples(dataset, args.per_class, args.seed)
        items = [dataset[i] for i in idx]
        x_all = torch.stack([it[0] for it in items])
        labels_all = torch.stack([it[1] for it in items])
        meta['subject'] = np.array([dataset.subject_ids[i] for i in idx])
        meta['start_epoch'] = np.array([dataset.epochs[i][1] for i in idx])
        if args.box:
            low, high = args.box
        else:
            low = min(float(np.min(a)) for a in dataset.inputs)
            high = max(float(np.max(a)) for a in dataset.inputs)
        print('[INFO] %d Samples ausgewaehlt, ZBox-Grenzen: [%.3f, %.3f]' % (len(idx), low, high))

    data = run_attribution(model, x_all, labels_all, args, device, low, high)
    data.update(meta)
    data['box'] = np.array([low, high])
    npz_path = os.path.join(args.out, 'lrp_fold-{0:02d}.npz'.format(args.fold))
    np.savez_compressed(npz_path, **data)
    print('[INFO] gespeichert: %s' % npz_path)
    analyze(data, args.out, args.class_names, plots=not args.no_plots)
    print('[INFO] fertig. Ergebnisse in %s/' % args.out)


if __name__ == '__main__':
    main()
