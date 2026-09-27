"""Tests for Fly Daggers (code/daggers) that run without the connectome or the game.

    python -m unittest tests.test_daggers
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'daggers'))

from brain import EventBrain  # noqa: E402
from dataset import LABEL_SHIFT, clips  # noqa: E402
from eyes import FLOW_CONF, FLOW_X, FRAME_H, FRAME_W, Retina  # noqa: E402
from fast_brain import FastBrain  # noqa: E402
from fly import BUTTON_NAMES, Readout, _RidgePath, _standardize, cv_score  # noqa: E402


def random_network(n=600, density=0.04, seed=0):
    """A small excitatory-leaning network with FlyWire-like synapse counts."""
    rng = np.random.default_rng(seed)
    w = sparse.random(n, n, density=density, random_state=rng, format='csr',
                      data_rvs=lambda k: rng.integers(-8, 25, size=k).astype(np.float32))
    w.eliminate_zeros()
    return w


class EventBrainTest(unittest.TestCase):

    def test_same_spikes_as_fastbrain_until_a_rounding_graze(self):
        """Step for step the same spikes as FastBrain, until float rounding
        (float32 there, float64 here) puts a membrane on different sides of
        threshold by a hair; that first difference must be such a graze."""
        weights = random_network()
        stim = np.arange(60)
        rng = np.random.default_rng(1)
        steps = 3000
        kicks = [stim[rng.random(len(stim)) < 0.015] for _ in range(steps)]
        ref = FastBrain(weights, stim, accel=False)
        brain = EventBrain(weights, stim)
        same = 0
        for t in range(steps):
            v, g = ref.v.copy(), ref.g.copy()
            ref_spikes = set(ref.step(poisson_idx=kicks[t]).tolist())
            rec = (np.zeros(len(stim) + weights.shape[0], dtype=np.int64),
                   np.zeros(len(stim) + weights.shape[0], dtype=np.int64), np.zeros(1, dtype=np.int64))
            brain.run(1, hits=(np.full(len(kicks[t]), t), kicks[t]), record=rec)
            ev_spikes = set(rec[1][:rec[2][0]].tolist())
            if ref_spikes != ev_spikes:
                for i in ref_spikes ^ ev_spikes:
                    kick = ref.poisson_kick if i in kicks[t] else 0.0
                    v_new = (v[i] + kick) + (g[i] - ((v[i] + kick) - ref.v_rest)) * ref.mem_factor
                    self.assertAlmostEqual(float(v_new), float(ref.v_th), delta=1e-3,
                                           msg=f'step {t}, neuron {i}: not a threshold graze')
                break
            same += len(ref_spikes)
        self.assertGreater(same, 1000)                            # plenty identical before any graze

    def test_poisson_rate(self):
        weights = sparse.csr_matrix((50, 50), dtype=np.float32)   # no synapses
        brain = EventBrain(weights, np.arange(50), readout_indices=np.arange(50), seed=3)
        brain.set_rates(np.full(50, 100.0))
        counts = brain.run(20_000)                                # 2 s
        self.assertAlmostEqual(counts.mean() / 2.0, 100.0, delta=5.0)

    def test_reset_repeats_with_seed(self):
        weights = random_network(seed=2)
        brain = EventBrain(weights, np.arange(40), readout_indices=np.arange(600), seed=5)
        brain.set_rates(np.full(40, 150.0))
        first = brain.run(2000)
        brain.reset()
        self.assertTrue(np.array_equal(first, brain.run(2000)))


class ReadoutTest(unittest.TestCase):

    def test_dual_and_primal_ridge_agree(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(40, 3))
        for p in (10, 80):                                        # primal, then dual
            Xs, _, _ = _standardize(rng.normal(size=(40, p)))
            path = _RidgePath(Xs, Y)
            coef = path.coef([0.1, 1.0, 10.0])
            for k, lam in enumerate([0.1, 1.0, 10.0]):
                direct = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(p),
                                         Xs.T @ (Y[:, k] - Y[:, k].mean()))
                np.testing.assert_allclose(coef[:, k], direct, rtol=1e-6, atol=1e-8)

    def test_cv_score_finds_signal_not_noise(self):
        rng = np.random.default_rng(1)
        Xs, Ys = [], []
        for _ in range(10):
            X = rng.normal(size=(100, 30))
            y_sig = X[:, :3] @ np.array([1.0, -2.0, 0.5])
            Ys.append(np.stack([y_sig, rng.normal(size=100)], axis=1))
            Xs.append(X)
        score, r2, lam = cv_score(Xs, Ys)
        self.assertGreater(r2[0], 0.95)
        self.assertLess(r2[1], 0.05)
        self.assertAlmostEqual(score, (max(r2[0], 0) + max(r2[1], 0)) / 2)

    def test_press_share_matches_player(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(2000, 5))
        n_b = len(BUTTON_NAMES)
        Y = np.zeros((2000, n_b + 2))
        Y[:, 0] = (X[:, 0] + 0.3 * rng.normal(size=2000) > 0.8).astype(float)   # held ~20%
        readout = Readout.fit(X, Y, np.full(n_b + 2, 0.01))
        pressed = np.mean([readout.act(x)[0][BUTTON_NAMES[0]] for x in X])
        self.assertAlmostEqual(pressed, Y[:, 0].mean(), delta=0.02)
        self.assertFalse(any(readout.act(X[0])[0][b] for b in BUTTON_NAMES[1:]))   # never held


class RetinaTest(unittest.TestCase):

    def test_turning_view_is_optic_flow(self):
        rng = np.random.default_rng(0)
        world = (rng.random((FRAME_H, FRAME_W * 2, 3)) * 255).astype(np.uint8)
        retina = Retina(masks=[])
        retina.see(world[:, 10:10 + FRAME_W], 0.05)
        raw = retina.see(world[:, 16:16 + FRAME_W], 0.05)        # view turned right: content moves left
        self.assertAlmostEqual(raw[FLOW_X], -6 / FRAME_W / 0.05, places=5)
        self.assertGreater(raw[FLOW_CONF], 0.3)


class EyesTest(unittest.TestCase):

    def test_a_thing_drives_only_its_own_band(self):
        from eyes import BANDS, EYE_BINS, N_RAW, OBJ, Eyes, Genome
        scales = {ch: [0.0, 1.0] for ch in ('object', 'red', 'loom', 'bustle', 'pan', 'scroll', 'taste')}
        eyes = Eyes(Genome.default(), scales)
        raw = np.zeros(N_RAW, dtype=np.float32)
        eyes.rates(raw, 0.05)
        raw[OBJ.start + 6] = 0.5                    # a thing in the 7th of 8 strips: right eye, band 2
        obj = eyes.rates(raw, 0.05)['object']
        self.assertEqual(obj.shape, (2, BANDS))
        self.assertEqual(tuple(np.argwhere(obj > 0)[0]), (1, 2))
        self.assertEqual((obj > 0).sum(), 1)
        self.assertEqual(EYE_BINS[1, 2], 6)


class DatasetTest(unittest.TestCase):

    def test_labels_are_the_next_look(self):
        n = 200
        session = {
            'name': 's', 'raw': np.arange(n, dtype=np.float32)[:, None].repeat(27, axis=1),
            't': np.arange(n) / 20.0, 'segment': np.zeros(n, dtype=int),
            'buttons': np.zeros((n, len(BUTTON_NAMES)), dtype=np.float32),
            'mouse': np.arange(n, dtype=np.float32)[:, None].repeat(2, axis=1),
        }
        (clip,) = clips([session], mouse_scale=[1000.0, 1000.0])
        self.assertEqual(len(clip['raw']), n - LABEL_SHIFT)
        np.testing.assert_allclose(clip['y'][:, -2] * 1000.0, clip['raw'][:, 0] + LABEL_SHIFT)


if __name__ == '__main__':
    unittest.main()
