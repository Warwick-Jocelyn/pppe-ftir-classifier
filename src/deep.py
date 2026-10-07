"""PyTorch 1-D backbones with an sklearn-style wrapper (fit / predict_proba).

Augmentation is applied on the RAW native-grid spectrum (before SG/SNV) so it mimics real
acquisition variation: baseline tilt/offset (poor ATR contact, dark pigments), gain changes,
detector noise and +-N-point wavenumber shifts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from preprocess import NATIVE_WN, SpectraPreproc

N_SPEC = len(NATIVE_WN)


# ------------------------------------------------------------------------------ architectures
class MLP(nn.Module):
    def __init__(self, L, width=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(L, width), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(width, 32), nn.GELU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, L, ch=(16, 32, 64), k=9, p=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in ch:
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.GELU(), nn.MaxPool1d(2)]
            c_in = c
        self.f = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(c_in * 2, 1))

    def forward(self, x):
        h = self.f(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, c_out, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(c_out)
        self.c2 = nn.Conv1d(c_out, c_out, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(c_out)
        self.sc = nn.Identity() if (c_in == c_out and stride == 1) else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.gelu(h + self.sc(x))


class ResNet1D(nn.Module):
    def __init__(self, L, widths=(16, 32, 64), p=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, widths[0], 15, 2, 7, bias=False), nn.BatchNorm1d(widths[0]), nn.GELU())
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResBlock(c, w, stride=1 if i == 0 else 2))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(c, 1))

    def forward(self, x):
        h = self.blocks(self.stem(x)).mean(-1)
        return self.head(h).squeeze(-1)


class SpecTransformer(nn.Module):
    """Patch embedding + 2-layer Transformer encoder + CLS token (ViT-style for spectra)."""

    def __init__(self, L, patch=16, d=48, heads=4, layers=2, p=0.2):
        super().__init__()
        self.patch = patch
        self.n = L // patch
        self.embed = nn.Conv1d(1, d, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, self.n + 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, p, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        x = x[..., : self.n * self.patch]
        h = self.embed(x).transpose(1, 2)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos
        h = self.norm(self.enc(h)[:, 0])
        return self.head(h).squeeze(-1)


ARCHS = {"mlp": MLP, "cnn1d": CNN1D, "resnet1d": ResNet1D, "transformer": SpecTransformer}


# ------------------------------------------------------------------------------ wrapper
def augment_raw(X, rng, max_shift=2, tilt=0.02, gain=(0.7, 1.3), noise=0.003):
    X = X.copy()
    n, L = X.shape
    t = np.linspace(-1, 1, L)[None]
    scale = X.max(1, keepdims=True)
    X = X * rng.uniform(*gain, size=(n, 1))
    X = X + scale * (rng.normal(0, tilt, (n, 1)) * t + rng.normal(0, tilt, (n, 1)))
    X = X + rng.normal(0, 1, X.shape) * noise
    if max_shift:
        sh = rng.integers(-max_shift, max_shift + 1, n)
        for k in np.unique(sh):
            if k:
                X[sh == k] = np.roll(X[sh == k], k, axis=1)
    return X


class TorchSpectraClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, arch="cnn1d", pp=None, epochs=80, lr=2e-3, wd=1e-3, batch=32, augment=True, seed=0, n_models=1):
        self.arch = arch
        self.pp = pp
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.batch = batch
        self.augment = augment
        self.seed = seed
        self.n_models = n_models

    def _pp(self):
        return SpectraPreproc(**(self.pp or dict(region="fp+ch", step=2, smooth=11, deriv=1, norm="snv")))

    def _tensor(self, Z):
        return torch.tensor(Z, dtype=torch.float32)[:, None, :]

    def fit(self, X, y):
        torch.set_num_threads(1)
        X = np.asarray(X)[:, :N_SPEC]
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.pp_ = self._pp().fit(X)
        Z0 = self.pp_.transform(X)
        self.mu_, self.sd_ = Z0.mean(), Z0.std() + 1e-9
        self.models_ = []
        for m in range(self.n_models):
            seed = self.seed + 1000 * m
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = ARCHS[self.arch](Z0.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            steps = self.epochs * int(np.ceil(len(y) / self.batch))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
            yt = torch.tensor(y, dtype=torch.float32)
            for ep in range(self.epochs):
                Xa = augment_raw(X, rng) if self.augment else X
                Z = (self.pp_.transform(Xa) - self.mu_) / self.sd_
                Zt = self._tensor(Z)
                perm = torch.randperm(len(y))
                net.train()
                for i in range(0, len(y), self.batch):
                    idx = perm[i:i + self.batch]
                    if len(idx) < 2:
                        continue
                    ys = yt[idx] * 0.95 + 0.025  # label smoothing
                    loss = F.binary_cross_entropy_with_logits(net(Zt[idx]), ys)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            net.eval()
            self.models_.append(net)
        return self

    def decision_function(self, X):
        Z = (self.pp_.transform(np.asarray(X)[:, :N_SPEC]) - self.mu_) / self.sd_
        Zt = self._tensor(Z)
        with torch.no_grad():
            return np.mean([m(Zt).numpy() for m in self.models_], 0)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def augment_strong(X, rng, max_shift=8, noise_rel=0.04, blur_fwhm=10.0, gain=(0.3, 1.3), tilt=0.1):
    """Stronger raw-domain augmentation covering the stress-test envelope (used to harden classical models)."""
    from scipy.ndimage import gaussian_filter1d
    X = np.asarray(X, float).copy()
    n, L = X.shape
    pk = X.max(1, keepdims=True)
    # optical resolution loss (per-spectrum FWHM): blur in 3 buckets to stay vectorised
    fw = rng.uniform(0, blur_fwhm, n)
    for lo, hi in [(1.5, 5), (5, 8), (8, 20)]:
        m = (fw >= lo) & (fw < hi)
        if m.any():
            X[m] = gaussian_filter1d(X[m], sigma=(lo + hi) / 2 / 2.355 / 0.964, axis=1)
    X *= rng.uniform(*gain, size=(n, 1))
    tt = np.linspace(-1, 1, L)[None]
    X += pk * tilt * (rng.normal(0, 1, (n, 1)) * tt + rng.normal(0, 1, (n, 1)))
    X += rng.normal(0, 1, X.shape) * (rng.uniform(0, noise_rel, (n, 1)) * pk)
    sh = rng.integers(-max_shift, max_shift + 1, n)
    for k in np.unique(sh):
        if k:
            X[sh == k] = np.roll(X[sh == k], k, axis=1)
    return X
