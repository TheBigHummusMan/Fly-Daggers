"""
The fly as a mouse cursor: a closed loop between a game window and the
whole-brain LIF model (fast_brain.py).

    frame around the cursor ──> two eye patches, left and right of the heading
        contrast               ──> LC10a        (small objects)
        darkening              ──> LC4 + LPLC2  (looming)
        change                 ──> LC9          (bustle)
    colours under the cursor   ──> sugar GRNs, bitter GRNs  (tasting with the legs)
                               ──> 138k-neuron FlyWire connectome
    DNp09 walking              ──> speed; above hold_thr the button is held (scratching)
    DNa01 + DNa02, left - right──> turning
    MN9 proboscis              ──> click
    Giant Fiber DNp01          ──> escape jump backwards

Everything between the input and output neurons is the unmodified connectome.
Only the Genome is evolved: how strongly each screen feature drives its
neurons, which colours taste sweet or bitter, and how firing rates become
cursor motion. It gets perceptual features only, never game state.

In the connectome, bitter GRNs silence the sugar -> MN9 pathway (MN9 falls from
~50-85 Hz to 0 when both are driven), so bitter things are not clicked.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_brain import DT, FastBrain  # noqa: E402

NEURONS_CSV = Path(__file__).resolve().parents[2] / 'data' / 'scratch_neurons.csv'

LOOK_MS = 25.0                   # brain time per look at the frame
RATE_TAU_MS = 50.0               # smoothing of output firing rates
RATE_MAX = 200.0                 # Hz Poisson input at full drive
REF_W = 1024                     # 'px' below are pixels of a 1024-wide window
CELL = 4                         # px per grid cell the eyes sample from
CLICK_GAP_MS = 250.0             # brain time between clicks
ESCAPE_GAP_MS = 300.0
SIDES = ('left', 'right')
EYE_CHANNELS = ('object', 'loom', 'bustle')
N_BINS = 16                      # taste colour bins: 12 hues + 4 greys

GENES = [
    # name, low, high
    ('eye_offset', 10.0, 150.0),     # px from the cursor to each eye patch
    ('eye_angle', 0.2, 1.4),         # rad either side of the heading
    ('eye_radius', 8.0, 80.0),       # px
    ('object_gain', 0.0, 1.0),
    ('object_sat', 0.02, 0.4),       # luminance s.d. for full LC10a drive
    ('loom_gain', 0.0, 1.0),
    ('loom_sat', 0.02, 0.4),         # mean darkening per look for full drive
    ('bustle_gain', 0.0, 1.0),
    ('bustle_sat', 0.01, 0.3),       # mean |change| per look for full drive
    ('taste_radius', 3.0, 40.0),     # px
    ('taste_gain', 0.0, 1.0),
    ('speed_gain', 0.0, 50.0),       # px/s per Hz of DNp09
    ('base_speed', 0.0, 1500.0),     # px/s (a human scratches at ~1400)
    ('turn_gain', -0.15, 0.15),      # rad/s per Hz of left - right steering
    ('turn_bias', -2.0, 2.0),        # rad/s
    ('click_thr', 5.0, 150.0),       # Hz of MN9
    ('escape_thr', 10.0, 200.0),     # Hz of DNp01
    ('jump_dist', 0.0, 300.0),       # px
    ('hold_thr', 0.0, 150.0),        # Hz of DNp09 above which the button is held
] + [(f'sugar_{i}', -1.0, 1.0) for i in range(N_BINS)] \
  + [(f'bitter_{i}', -1.0, 1.0) for i in range(N_BINS)]

GENE_NAMES = [g[0] for g in GENES]
LOW = np.array([g[1] for g in GENES])
HIGH = np.array([g[2] for g in GENES])


class Genome:
    """The evolved interface. CMA-ES works on unit (0-1) coordinates."""

    def __init__(self, values):
        self.values = np.clip(np.asarray(values, dtype=float), LOW, HIGH)

    @classmethod
    def from_unit(cls, u):
        return cls(LOW + np.clip(u, 0.0, 1.0) * (HIGH - LOW))

    def to_unit(self):
        return (self.values - LOW) / (HIGH - LOW)

    def __getitem__(self, name):
        return self.values[GENE_NAMES.index(name)]

    def vector(self, prefix):
        return np.array([self[f'{prefix}_{i}'] for i in range(N_BINS)])

    def as_dict(self):
        return {n: float(v) for n, v in zip(GENE_NAMES, self.values)}

    def save(self, path, **extra):
        Path(path).write_text(json.dumps({'genes': self.as_dict(), **extra}, indent=2))

    @classmethod
    def load(cls, path):
        genes = json.loads(Path(path).read_text())['genes']
        return cls([genes.get(n, (lo + hi) / 2) for n, lo, hi in GENES])

    @classmethod
    def start(cls):
        """The calibrated start (calibrate.py) if there is one, else the hand-set default."""
        path = Path(__file__).resolve().parents[2] / 'data' / 'scratch' / 'start_genome.json'
        return cls.load(path) if path.exists() else cls.default()

    @classmethod
    def default(cls):
        """Hand-set start: eyes on, sweet bright colours, a slow walk."""
        g = {n: (lo + hi) / 2 for n, lo, hi in GENES}
        g.update(eye_offset=40, eye_angle=0.7, eye_radius=24, object_gain=0.8, object_sat=0.15,
                 loom_gain=0.3, loom_sat=0.2, bustle_gain=0.8, bustle_sat=0.05, taste_radius=10,
                 taste_gain=0.8, speed_gain=4.0, base_speed=40.0, turn_gain=0.03, turn_bias=0.0,
                 click_thr=40.0, escape_thr=80.0, jump_dist=80.0, hold_thr=20.0)
        for i in range(N_BINS):
            g[f'sugar_{i}'] = 0.5 if i < 3 else 0.0      # reds/oranges/yellows taste sweet
            g[f'bitter_{i}'] = 0.0
        return cls([g[n] for n in GENE_NAMES])


def colour_bins(rgb):
    """Fraction of pixels in each of N_BINS colour bins (12 hues, 4 greys)."""
    hi, lo = rgb.max(axis=-1), rgb.min(axis=-1)
    chroma = hi - lo
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    c = np.maximum(chroma, 1e-6)
    hue = np.where(hi == r, ((g - b) / c) % 6, np.where(hi == g, (b - r) / c + 2, (r - g) / c + 4))
    hue_bin = np.minimum((hue * 2).astype(int), 11)                  # 12 bins of 30 deg
    grey_bin = 12 + np.minimum((hi * 4).astype(int), 3)
    bins = np.where(chroma > 0.2, hue_bin, grey_bin)
    return np.bincount(bins.ravel(), minlength=N_BINS)[:N_BINS] / max(bins.size, 1)


def _disc(n=7):
    """Unit-disc sample offsets on an n x n grid."""
    a = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(a, a)
    keep = xx ** 2 + yy ** 2 <= 1.0
    return np.stack([xx[keep], yy[keep]], axis=1)


DISC = _disc()


class CursorFly:
    """One fly: a FastBrain, its genome, and a cursor with a heading."""

    def __init__(self, weights, flyid2i, genome, seed=None, blind=False):
        df = pd.read_csv(NEURONS_CSV)

        def pick(mask):
            return np.array([flyid2i[f] for f in df.loc[mask, 'flywire_id']], dtype=np.int64)

        groups = {}
        for ch in EYE_CHANNELS:
            for side in SIDES:
                groups[ch, side] = pick((df.role == ch) & (df.side == side))
        groups['sugar', None] = pick(df.role == 'sugar')
        # The 21 sugar GRNs are all left-side, so taste uses the left bitter GRNs too
        groups['bitter', None] = pick((df.role == 'bitter') & (df.side == 'left'))

        self.slices, start = {}, 0
        for key, idx in groups.items():
            self.slices[key] = slice(start, start + len(idx))
            start += len(idx)
        self.brain = FastBrain(weights, np.concatenate(list(groups.values())), seed=seed)

        self.out_keys = [(role, side) for role in ('walk', 'turn', 'feed', 'escape') for side in SIDES]
        self.out_code = np.zeros(self.brain.n, dtype=np.int64)
        self.out_size = np.zeros(len(self.out_keys))
        for k, (role, side) in enumerate(self.out_keys):
            idx = pick((df.role == role) & (df.side == side))
            self.out_code[idx] = k + 1
            self.out_size[k] = len(idx)

        self.genome = genome
        self.blind = blind
        self.rng = np.random.default_rng(seed)
        self.smooth = 1 - math.exp(-LOOK_MS / RATE_TAU_MS)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, x=0.5, y=0.5, heading=None):
        self.brain.reset()
        self.x, self.y = x, y                    # 0-1 window coordinates
        self.heading = self.rng.uniform(0, 2 * math.pi) if heading is None else heading
        self.prev_grid = None
        self.rates = {role: np.zeros(2) for role in ('walk', 'turn', 'feed', 'escape')}
        self.drive = {}
        self.t_ms = 0.0
        self.last_click = self.last_escape = -1e9
        self.stats = dict(clicks=0, escapes=0, held_looks=0, looks=0, path_px=0.0)

    # ------------------------------------------------------------------
    def _grid(self, frame):
        """Frame -> (luminance, rgb) grid of CELL-px cells, values 0-1."""
        h, w = frame.shape[:2]
        gw = REF_W // CELL
        gh = max(8, round(gw * h / w))
        sy, sx = h // gh, w // gw
        img = frame[:gh * sy, :gw * sx].reshape(gh, sy, gw, sx, 3)
        rgb = img[:, ::max(1, sy // 2), :, ::max(1, sx // 2)].mean(axis=(1, 3)) / 255.0
        lum = rgb @ np.array([0.299, 0.587, 0.114])
        return lum, rgb

    def _sample(self, arr, cx, cy, radius):
        """Values of a grid at a disc around (cx, cy) in px."""
        gh, gw = arr.shape[:2]
        pts = DISC * radius + (cx, cy)
        cols = np.clip((pts[:, 0] / CELL).astype(int), 0, gw - 1)
        rows = np.clip((pts[:, 1] / CELL).astype(int), 0, gh - 1)
        return arr[rows, cols]

    def _senses(self, frame):
        g = self.genome
        lum, rgb = self._grid(frame)
        gh, gw = lum.shape
        W, H = gw * CELL, gh * CELL
        px, py = self.x * W, self.y * H
        diff = np.zeros_like(lum) if self.prev_grid is None or self.prev_grid.shape != lum.shape \
            else lum - self.prev_grid
        self.prev_grid = lum

        drive = {}
        for i, side in enumerate(SIDES):
            # left eye at heading - angle (screen y points down)
            a = self.heading + (-1 if side == 'left' else 1) * g['eye_angle']
            ex, ey = px + g['eye_offset'] * math.cos(a), py + g['eye_offset'] * math.sin(a)
            patch = self._sample(lum, ex, ey, g['eye_radius'])
            change = self._sample(diff, ex, ey, g['eye_radius'])
            drive['object', side] = g['object_gain'] * min(patch.std() / g['object_sat'], 1.0)
            drive['loom', side] = g['loom_gain'] * min(np.maximum(-change, 0).mean() / g['loom_sat'], 1.0)
            drive['bustle', side] = g['bustle_gain'] * min(np.abs(change).mean() / g['bustle_sat'], 1.0)

        under = self._sample(rgb, px, py, g['taste_radius'])
        bins = colour_bins(under)
        drive['sugar', None] = g['taste_gain'] * float(np.clip(g.vector('sugar') @ bins, 0, 1))
        drive['bitter', None] = g['taste_gain'] * float(np.clip(g.vector('bitter') @ bins, 0, 1))
        return drive, W, H

    # ------------------------------------------------------------------
    def look(self, frame):
        """See the frame for LOOK_MS of brain time; return (x, y, button)."""
        g = self.genome
        drive, W, H = self._senses(frame)
        self.drive = drive
        for key, sl in self.slices.items():
            self.brain.rates[sl] = 0.0 if self.blind else drive[key] * RATE_MAX

        counts = np.zeros(len(self.out_keys) + 1, dtype=np.int64)
        for _ in range(int(round(LOOK_MS / DT))):
            spikes = self.brain.step()
            if len(spikes):
                counts += np.bincount(self.out_code[spikes], minlength=len(counts))
        self.t_ms += LOOK_MS
        hz = counts[1:] / self.out_size / (LOOK_MS / 1000)
        for k, (role, side) in enumerate(self.out_keys):
            r = self.rates[role]
            i = SIDES.index(side)
            r[i] += self.smooth * (hz[k] - r[i])

        # --- motor ---
        dt = LOOK_MS / 1000
        walk = self.rates['walk'].mean()
        steer = self.rates['turn'][0] - self.rates['turn'][1]      # > 0: turn left
        self.heading -= (g['turn_bias'] + g['turn_gain'] * steer) * dt
        speed = g['base_speed'] + g['speed_gain'] * walk
        px, py = self.x * W, self.y * H
        button = 'down' if walk >= g['hold_thr'] else 'up'      # hold_thr 0: always held

        if self.rates['escape'].max() > g['escape_thr'] and self.t_ms - self.last_escape > ESCAPE_GAP_MS:
            px -= g['jump_dist'] * math.cos(self.heading)
            py -= g['jump_dist'] * math.sin(self.heading)
            self.heading += math.pi
            self.last_escape = self.t_ms
            self.stats['escapes'] += 1
            button = 'up'
        else:
            step = speed * dt
            px += step * math.cos(self.heading)
            py += step * math.sin(self.heading)
            self.stats['path_px'] += step

        # bounce off the window edges
        if not 0 <= px <= W:
            self.heading = math.pi - self.heading
            px = min(max(px, 0), W)
        if not 0 <= py <= H:
            self.heading = -self.heading
            py = min(max(py, 0), H)
        self.x, self.y = px / W, py / H

        if self.rates['feed'].max() > g['click_thr'] and self.t_ms - self.last_click > CLICK_GAP_MS:
            self.last_click = self.t_ms
            self.stats['clicks'] += 1
            button = 'click'
        self.stats['looks'] += 1
        self.stats['held_looks'] += button == 'down'
        return self.x, self.y, button

    def status(self):
        d, r = self.drive, self.rates

        def lr(x):
            return f'{x[0]:.0f}|{x[1]:.0f}'
        return (f'in Hz  LC10a {d.get(("object", "left"), 0) * RATE_MAX:.0f}|{d.get(("object", "right"), 0) * RATE_MAX:.0f}'
                f'  LC4 {d.get(("loom", "left"), 0) * RATE_MAX:.0f}|{d.get(("loom", "right"), 0) * RATE_MAX:.0f}'
                f'  LC9 {d.get(("bustle", "left"), 0) * RATE_MAX:.0f}|{d.get(("bustle", "right"), 0) * RATE_MAX:.0f}'
                f'  sugar {d.get(("sugar", None), 0) * RATE_MAX:.0f}  bitter {d.get(("bitter", None), 0) * RATE_MAX:.0f}\n'
                f'out Hz DNp09 {lr(r["walk"])}  DNa01/02 {lr(r["turn"])}  MN9 {lr(r["feed"])}  GF {lr(r["escape"])}')
