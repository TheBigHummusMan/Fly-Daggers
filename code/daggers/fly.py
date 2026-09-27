"""
The fly that plays Devil Daggers: eyes, connectome, and a readout to the
keyboard and mouse.

    frame ──> Retina ──> Eyes (evolved genome) ──> visual projection neurons
          ──> 138k-neuron FlyWire connectome (EventBrain, never changed)
          ──> 1,409 descending + motor neurons ──> Readout (fitted) ──> keys, mouse

The readout is the fly's "body": a linear map from the brain's output neurons
to what a player's hands do, fitted by ridge regression to recordings of a
person playing. The genome is evolved so that the brain's output carries as
much of what the player did as possible (train.py). The connectome between
the input and output neurons is FlyWire's, unmodified.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain import EventBrain  # noqa: E402
from eyes import BANDED, BANDS, SIDES, Eyes, Genome  # noqa: E402
from fast_brain import DT  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NEURONS_CSV = ROOT / 'data' / 'daggers_neurons.csv'

FPS = 20                         # looks per second, when recording and playing
LOOK_MS = 50.0                   # brain time per look (= 1000 / FPS: brain in real time)
SLOW_TAU_MS = 200.0              # second, slower readout of the same neurons

# What the fly's "hands" can do. Keys are Windows virtual-key codes.
BUTTONS = {
    'w': ('key', 0x57), 'a': ('key', 0x41), 's': ('key', 0x53), 'd': ('key', 0x44),
    'jump': ('key', 0x20),                       # space
    'lmb': ('mouse', 'left'), 'rmb': ('mouse', 'right'),
}
BUTTON_NAMES = list(BUTTONS)
OUTPUTS = BUTTON_NAMES + ['mouse_x', 'mouse_y']


class DaggersFly:
    """Eyes' rates into the connectome, output-neuron firing rates out."""

    def __init__(self, weights, flyid2i, seed=None, neurons_csv=NEURONS_CSV):
        df = pd.read_csv(neurons_csv)

        def pick(mask):
            return np.array([flyid2i[f] for f in df.loc[mask, 'flywire_id']], dtype=np.int64)

        # Keys (channel, side, band); unbanded channels use band 0 only
        groups = {}
        for i, side in enumerate(SIDES):
            for ch in BANDED:
                for band in range(BANDS):
                    groups[ch, side, band] = pick((df.role == ch) & (df.side == side)
                                                  & (df.band == band))
            for ch in ('scroll', 'taste'):
                groups[ch, side, 0] = pick((df.role == ch) & (df.side == side))
            # Motion toward the fly's right is front-to-back for its right eye
            # (HS cells) and back-to-front for its left eye (H2); in the
            # connectome both reach the right DNp15 (as in Fly Screen)
            other = SIDES[1 - i]
            groups['pan', side, 0] = pick((df.role == 'pan') & (
                ((df.cell_type != 'H2') & (df.side == side))
                | ((df.cell_type == 'H2') & (df.side == other))))
        self.slices, start = {}, 0
        for key, idx in groups.items():
            self.slices[key] = slice(start, start + len(idx))
            start += len(idx)

        out = df[df.role.str.startswith('readout')]
        self.readout_names = (out.cell_type + '_' + out.side).tolist()
        self.brain = EventBrain(weights, np.concatenate(list(groups.values())),
                                pick(df.role.str.startswith('readout')), seed=seed)
        self.rates = np.zeros(len(self.brain.stim))
        self.slow_keep = math.exp(-LOOK_MS / SLOW_TAU_MS)
        self.reset()

    @property
    def n_readout(self):
        return 2 * len(self.brain.readout)

    def reset(self, seed=None):
        if seed is not None:
            self.brain.seed = seed
        self.brain.reset()
        self.slow = np.zeros(len(self.brain.readout))

    def look(self, rates, ms=LOOK_MS):
        """Drive the input neurons at rates ({channel: (2, BANDS)} Hz, from
        Eyes.rates) for ms of brain time. Returns the readout neurons' firing
        rates (Hz) over the look and, smoothed, over the last SLOW_TAU_MS,
        concatenated."""
        for (ch, side, band), sl in self.slices.items():
            self.rates[sl] = rates[ch][SIDES.index(side), band]
        self.brain.set_rates(self.rates)
        now = self.brain.run(int(round(ms / DT))) / (ms / 1000.0)
        self.slow = self.slow_keep * self.slow + (1 - self.slow_keep) * now
        return np.concatenate([now, self.slow])


# ============================================================================
# Readout: output-neuron rates to keys and mouse, by ridge regression
# ============================================================================

class Readout:
    """Linear map from readout rates to OUTPUTS.

    Columns that never vary in the fitting data (silent neurons) are dropped.
    Button outputs estimate the fraction of the look the key is held; a key is
    pressed when that is above a threshold chosen so the fly presses it as
    often as the player did. Mouse outputs are in units of the player's
    typical motion per look (see dataset.mouse_scale).
    """

    def __init__(self, cols, mean, std, coef, bias, thresholds, lam):
        self.cols = np.asarray(cols, dtype=np.int64)
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        self.coef, self.bias = np.asarray(coef), np.asarray(bias)
        self.thresholds = np.asarray(thresholds)
        self.lam = np.asarray(lam)

    @classmethod
    def fit(cls, X, Y, lam):
        """lam: ridge penalty per output (from cv_score)."""
        cols = np.flatnonzero(X.std(axis=0) > 1e-6)
        Xs, mean, std = _standardize(X[:, cols])
        path = _RidgePath(Xs, Y)
        coef = path.coef(lam)
        bias = path.y_mean
        pred = Xs @ coef + bias
        # Press threshold: the fly holds a key for the same share of looks as the player
        n_b = len(BUTTON_NAMES)
        share = (Y[:, :n_b] > 0.5).mean(axis=0)
        thr = np.array([np.quantile(pred[:, k], 1 - share[k]) if 0 < share[k] < 1
                        else (np.inf if share[k] == 0 else -np.inf) for k in range(n_b)])
        return cls(cols, mean, std, coef, bias, thr, lam)

    def predict(self, X):
        X = np.atleast_2d(X)
        return ((X[:, self.cols] - self.mean) / self.std) @ self.coef + self.bias

    def act(self, x):
        """One look's readout rates -> (buttons held {name: bool}, mouse [x, y] in scale units)."""
        y = self.predict(x)[0]
        n_b = len(BUTTON_NAMES)
        held = {name: bool(y[k] > self.thresholds[k]) for k, name in enumerate(BUTTON_NAMES)}
        return held, y[n_b:]

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return (X - mean) / std, mean, std


class _RidgePath:
    """Ridge regression of Y on centred, standardized Xs for any penalties,
    from one eigendecomposition: of Xs'Xs (features x features) or, when
    there are fewer samples than features, of Xs Xs' (samples x samples).
    The penalty is lam * n, so lam = 1 shrinks a typical column by half."""

    def __init__(self, Xs, Y):
        self.n = len(Xs)
        self.y_mean = Y.mean(axis=0)
        Yc = Y - self.y_mean
        self.Xs = Xs
        self.dual = len(Xs) < Xs.shape[1]
        if self.dual:
            self.evals, self.evecs = np.linalg.eigh(Xs @ Xs.T)
            self.proj = self.evecs.T @ Yc
        else:
            self.evals, self.evecs = np.linalg.eigh(Xs.T @ Xs)
            self.proj = self.evecs.T @ (Xs.T @ Yc)

    def coef(self, lam):
        """Coefficients (features x outputs); lam is one penalty or one per output."""
        lam = np.broadcast_to(np.asarray(lam, dtype=float), (self.proj.shape[1],))
        shrunk = self.proj / (np.maximum(self.evals, 0.0)[:, None] + lam[None, :] * self.n)
        return self.Xs.T @ (self.evecs @ shrunk) if self.dual else self.evecs @ shrunk


LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)


def cv_score(Xs, Ys, weights=None, lambdas=LAMBDAS):
    """Two-fold cross-validated R^2 of a ridge readout, averaged over outputs.

    Xs, Ys: lists of per-clip arrays; even clips fit and odd clips score, and
    then the other way round. Each output gets the penalty that suits it best;
    outputs the player never varied are skipped. In the score, an output no
    better than its mean counts as 0 rather than negative, so what can't be
    predicted at all (e.g. when a player happens to jump) adds no noise.
    Returns (score, per-output R^2, per-output penalty).
    """
    folds = [(np.concatenate(Xs[0::2]), np.concatenate(Ys[0::2])),
             (np.concatenate(Xs[1::2]), np.concatenate(Ys[1::2]))]
    y_all = np.concatenate(Ys)
    live = y_all.std(axis=0) > 1e-3
    w = np.ones(y_all.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    w = w * live

    r2 = np.zeros((len(lambdas), y_all.shape[1]))
    for (Xa, Ya), (Xb, Yb) in (folds, folds[::-1]):
        ss_tot = ((Yb - Yb.mean(axis=0)) ** 2).sum(axis=0)
        cols = np.flatnonzero(Xa.std(axis=0) > 1e-6)
        if len(cols) == 0:
            preds = [np.broadcast_to(Ya.mean(axis=0), Yb.shape)] * len(lambdas)
        else:
            Xa_s, mean, std = _standardize(Xa[:, cols])
            Xb_s = (Xb[:, cols] - mean) / std
            path = _RidgePath(Xa_s, Ya)
            preds = [Xb_s @ path.coef(lam) + path.y_mean for lam in lambdas]
        for j, pred in enumerate(preds):
            ss_res = ((Yb - pred) ** 2).sum(axis=0)
            r2[j] += np.where(ss_tot > 0, 1 - ss_res / np.maximum(ss_tot, 1e-12), 0.0) / 2
    r2 = np.maximum(r2, -1.0)
    best = np.argmax(r2, axis=0)
    r2_best = r2[best, np.arange(r2.shape[1])]
    score = float((np.maximum(r2_best, 0.0) * w).sum() / max(w.sum(), 1e-9))
    return score, r2_best, np.asarray(lambdas)[best]


# ============================================================================
# Policy: everything play.py needs
# ============================================================================

def save_policy(path, genome, scales, readout, mouse_scale, stats=None):
    Path(path).write_text(json.dumps({
        'genes': genome.as_dict(), 'feature_scales': scales, 'mouse_scale': mouse_scale,
        'readout': readout.to_dict(), 'fps': FPS, 'look_ms': LOOK_MS,
        'outputs': OUTPUTS, 'stats': stats or {},
    }, indent=1))


def latest_run(runs_dir, need):
    """The run folder under runs_dir whose `need` file (e.g. policy.json) is newest."""
    found = [p for p in Path(runs_dir).glob('*') if (p / need).exists()]
    if not found:
        raise SystemExit(f'No run with {need} in {runs_dir}')
    return max(found, key=lambda p: (p / need).stat().st_mtime)


def keep_awake():
    """On Windows, stop the computer sleeping while this process runs
    (released automatically when it exits). Elsewhere, nothing."""
    import os
    if os.name == 'nt':
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def load_policy(path):
    d = json.loads(Path(path).read_text())
    genome = Genome.from_dict(d['genes'])
    return genome, Eyes(genome, d['feature_scales']), Readout.from_dict(d['readout']), d
