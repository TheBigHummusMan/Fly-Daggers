"""
The fly's eyes for Devil Daggers: game frames in, Poisson rates for the
visual projection neurons out.

Two stages, so the expensive one runs once per recording and not once per
genome:

    Retina (fixed)   frame ──> raw features: brightness above the local
                                 background ("things"), redness, and motion
                                 left over after the view's own shift, each in
                                 N_BINS bands across the screen, plus the
                                 view's global shift (optic flow)
    Eyes   (evolved) raw features ──> drive (0-1) per channel, side and band ──> Hz

Devil Daggers is dark, and what matters in it is bright or red (skulls,
spiders, gems, daggers). Unlike Fly Screen's retina, local motion is not
suppressed while the whole view moves: in a first-person shooter it nearly
always moves. Instead the view's shift is estimated and removed first.

The left half of the frame is the fly's left eye. Channels and neurons are
Fly Screen's (data/daggers_neurons.csv):

    object   LC10a        things, weighted by how far from the centre they are
    loom     LC4 + LPLC2  a sudden rise in how much of a strip is things
    bustle   LC9          motion that keeps going
    pan      HS + H2      the view sliding sideways (the mouse turning)
    scroll   VS           the view sliding up or down
    taste    sugar GRNs   red (gems, and some enemies); not biology, a colour sense

Object, loom and bustle are banded: each eye's BANDS strips of screen
(centre to edge) drive separate quarters of the population, chosen by
make_neurons.py for their distinct downstream pathways. The other channels
drive their whole population at one rate per side.

Usage:
    python code/daggers/eyes.py --preview data/daggers/recordings/<session>
        writes <session>/preview.png: one frame with the mask and bands
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

FRAME_W, FRAME_H = 128, 72       # recorded frame size, the fly's "ommatidia"
N_BINS = 8                       # strips across the screen, N_BINS // 2 per eye
BANDS = N_BINS // 2              # strips per eye, band 0 at the centre
BANDED = ('object', 'loom', 'bustle')
BLUR = 9                         # cells of local background for "things"
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Parts of the frame the fly never sees, as (x0, y0, x1, y1) fractions: the
# player's glowing hand (it rises to mid-screen when shooting), and the HUD's
# gem counter (top left) and timer (top centre). Check with --preview after
# recording, and prepare again (dataset.py --force) if you change it.
MASKS = [(0.33, 0.50, 0.67, 1.0), (0.0, 0.0, 0.12, 0.07), (0.42, 0.0, 0.58, 0.10)]

SIDES = ('left', 'right')
CHANNELS = ('object', 'loom', 'bustle', 'pan', 'scroll', 'taste')
RATE_MAX = {'object': 150.0, 'loom': 150.0, 'bustle': 120.0,    # Hz at full drive,
            'pan': 200.0, 'scroll': 200.0, 'taste': 150.0}      # as in Fly Screen
BUSY_TAU = 0.5                   # s, sustained motion for LC9

# Raw feature layout
OBJ = slice(0, N_BINS)
RED = slice(N_BINS, 2 * N_BINS)
MOT = slice(2 * N_BINS, 3 * N_BINS)
FLOW_X, FLOW_Y, FLOW_CONF = 3 * N_BINS, 3 * N_BINS + 1, 3 * N_BINS + 2
N_RAW = 3 * N_BINS + 3
RAW_NAMES = ([f'obj_{b}' for b in range(N_BINS)] + [f'red_{b}' for b in range(N_BINS)]
             + [f'mot_{b}' for b in range(N_BINS)] + ['flow_x', 'flow_y', 'flow_conf'])


def mask_array(h=FRAME_H, w=FRAME_W, masks=MASKS):
    """True where the fly can see."""
    visible = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in masks:
        visible[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = False
    return visible


class Retina:
    """Fixed early vision: frames (FRAME_H x FRAME_W x 3 uint8) to raw features."""

    def __init__(self, masks=MASKS):
        self.visible = mask_array(masks=masks)
        edges = np.linspace(0, FRAME_W, N_BINS + 1).astype(int)
        self.bin_of_col = np.searchsorted(edges, np.arange(FRAME_W), side='right') - 1
        self.bin_area = np.bincount(self.bin_of_col, weights=self.visible.sum(axis=0),
                                    minlength=N_BINS)
        self.window = np.outer(np.hanning(FRAME_H), np.hanning(FRAME_W)).astype(np.float32)
        self.reset()

    def reset(self):
        self.prev = None

    def _bins(self, img):
        return np.bincount(self.bin_of_col, weights=img.sum(axis=0),
                           minlength=N_BINS) / np.maximum(self.bin_area, 1)

    def see(self, frame, dt):
        """One frame, dt seconds after the last. Returns the raw feature vector."""
        rgb = frame.astype(np.float32) / 255.0
        lum = rgb @ LUMA
        lum[~self.visible] = 0.0
        out = np.zeros(N_RAW, dtype=np.float32)

        # Things: brighter than their surroundings; and redness
        things = np.maximum(lum - ndimage.uniform_filter(lum, BLUR, mode='nearest'), 0.0)
        red = np.maximum(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]), 0.0)
        red[~self.visible] = 0.0
        out[OBJ] = self._bins(things)
        out[RED] = self._bins(red)

        prev, self.prev = self.prev, lum
        if prev is not None and dt > 0:
            dy, dx, conf = self._shift(prev, lum)
            out[FLOW_X] = dx / FRAME_W / dt           # screens per second
            out[FLOW_Y] = dy / FRAME_H / dt
            out[FLOW_CONF] = conf
            # Motion the view's own shift doesn't explain
            moved = np.roll(prev, (dy, dx), axis=(0, 1))
            valid = self.visible.copy()
            valid &= np.roll(self.visible, (dy, dx), axis=(0, 1))
            if dy > 0:
                valid[:dy] = False
            elif dy < 0:
                valid[dy:] = False
            if dx > 0:
                valid[:, :dx] = False
            elif dx < 0:
                valid[:, dx:] = False
            out[MOT] = self._bins(np.abs(lum - moved) * valid) / dt   # per second
        return out

    def _shift(self, a, b):
        """Displacement (dy, dx) of frame b from frame a by phase correlation,
        and how sharp the correlation peak is (0 = no single shift, 1 = exact)."""
        fa = np.fft.rfft2((a - a.mean()) * self.window)
        fb = np.fft.rfft2((b - b.mean()) * self.window)
        cross = fb * np.conj(fa)
        corr = np.fft.irfft2(cross / (np.abs(cross) + 1e-9), s=a.shape)
        k = np.argmax(corr)
        dy, dx = np.unravel_index(k, corr.shape)
        h, w = a.shape
        conf = float(np.clip(corr.flat[k], 0.0, 1.0))
        return int(dy - h if dy > h // 2 else dy), int(dx - w if dx > w // 2 else dx), conf

    def run(self, frames, times):
        """Raw features for a sequence of frames captured at times (s)."""
        self.reset()
        dts = np.diff(times, prepend=times[0])
        return np.stack([self.see(f, dt) for f, dt in zip(frames, dts)])


# ============================================================================
# Evolved stage
# ============================================================================

GENES = [
    # name, low, high. Each feature is scaled so its median in the recordings
    # is 0 and its 90th percentile 1 (feature_scales()); a *_sat gene is the
    # log10 of the scaled value that drives the channel fully.
    ('object_gain', 0.0, 1.0),
    ('object_sat', -1.0, 1.0),
    ('object_ecc', -1.5, 1.5),       # >0: things off to the side drive harder (turn toward them)
    ('red_weight', 0.0, 3.0),        # how much redness counts as a thing
    ('loom_gain', 0.0, 1.0),
    ('loom_sat', -1.0, 1.0),
    ('loom_tau', 0.05, 2.0),         # s, how fast looming habituates
    ('bustle_gain', 0.0, 1.0),
    ('bustle_sat', -1.0, 1.0),
    ('pan_gain', 0.0, 1.0),
    ('pan_sat', -1.0, 1.0),
    ('scroll_gain', 0.0, 1.0),
    ('scroll_sat', -1.0, 1.0),
    ('taste_gain', 0.0, 1.0),
    ('taste_sat', -1.0, 1.0),
]
GENE_NAMES = [g[0] for g in GENES]
LOW = np.array([g[1] for g in GENES])
HIGH = np.array([g[2] for g in GENES])


class Genome:
    """The evolved part of the eyes. CMA-ES works on unit (0-1) coordinates."""

    def __init__(self, values):
        self.values = np.clip(np.asarray(values, dtype=float), LOW, HIGH)

    @classmethod
    def from_unit(cls, u):
        return cls(LOW + np.clip(u, 0.0, 1.0) * (HIGH - LOW))

    def to_unit(self):
        return (self.values - LOW) / (HIGH - LOW)

    def __getitem__(self, name):
        return self.values[GENE_NAMES.index(name)]

    def as_dict(self):
        return {n: float(v) for n, v in zip(GENE_NAMES, self.values)}

    @classmethod
    def from_dict(cls, genes):
        return cls([genes.get(n, (lo + hi) / 2) for n, lo, hi in GENES])

    @classmethod
    def default(cls):
        """Every channel but self-motion on at moderate gain, saturating at typical values."""
        g = {n: (lo + hi) / 2 for n, lo, hi in GENES}
        g.update({f'{ch}_gain': 0.6 for ch in CHANNELS})
        g.update({f'{ch}_sat': 0.0 for ch in CHANNELS})
        g.update({f'{ch}_gain': 0.0 for ch in SELF_MOTION})
        g.update(object_ecc=0.5, red_weight=1.0, loom_tau=0.3)
        return cls.from_dict(g)


# The view sliding as a whole is mostly the player's own turning. A readout
# fitted to a player's smooth mouse motion can learn "the view slides, so keep
# turning", which in closed loop makes the fly spin. These channels stay off
# unless training is asked to use them (train.py --self-motion).
SELF_MOTION = ('pan', 'scroll')


def feature_scales(raw):
    """Where each quantity the Eyes use usually sits in the recordings, as
    [median, 90th percentile], from an (N, N_RAW) array of raw features.

    The Eyes map the median to no drive and the 90th percentile to full drive
    (at *_sat = 0), so a textured floor that is always a little "thing"-like
    doesn't hold a channel at a constant rate.
    """
    def lohi(x):
        x = np.asarray(x, dtype=float).ravel()
        lo, hi = (float(v) for v in np.percentile(x, [50, 90])) if len(x) else (0.0, 1.0)
        return [lo, hi if hi - lo > 1e-9 else lo + max(abs(lo) * 0.1, 1e-6)]

    obj = _unit(raw[:, OBJ], *lohi(raw[:, OBJ]))
    conf = raw[:, FLOW_CONF]
    return {
        'object': lohi(raw[:, OBJ]),                    # per strip
        'red': lohi(raw[:, RED]),                       # per strip
        'loom': [0.0, lohi(np.maximum(np.diff(obj, axis=0), 0.0))[1]],
        'bustle': lohi(raw[:, MOT]),
        'pan': lohi(np.abs(raw[:, FLOW_X]) * conf),
        'scroll': lohi(np.abs(raw[:, FLOW_Y]) * conf),
        'taste': lohi(raw[:, RED].mean(1)),
    }


def _unit(x, lo, hi):
    """x on the scale median = 0, 90th percentile = 1, clipped below at 0."""
    return np.maximum((x - lo) / (hi - lo), 0.0)


# Raw strips (left to right across the screen) as [side, band], band 0 at the centre
EYE_BINS = np.stack([np.arange(BANDS)[::-1], BANDS + np.arange(BANDS)])


class Eyes:
    """Raw features to Poisson rates (Hz) per channel, side and band, for one genome."""

    def __init__(self, genome, scales):
        self.g = genome
        self.scales = scales
        ecc = (np.arange(BANDS) + 0.5) / BANDS                 # centre -> edge
        w = ecc ** genome['object_ecc']
        self.w = w / w.mean()
        self.sat = {ch: 10 ** genome[f'{ch}_sat'] for ch in CHANNELS}
        self.hz = {ch: RATE_MAX[ch] * genome[f'{ch}_gain'] for ch in CHANNELS}
        self.reset()

    def reset(self):
        self.loom_base = None
        self.busy = None

    def rates(self, raw, dt):
        """Rates (Hz) as {channel: (2, BANDS) array [side, band]}. Unbanded
        channels repeat one value across bands; pan's sides are [leftward, rightward]."""
        g, sc = self.g, self.scales
        obj = _unit(raw[OBJ], *sc['object'])
        things = (obj + g['red_weight'] * _unit(raw[RED], *sc['red'])) / (1 + g['red_weight'])
        mot = raw[MOT]

        if self.loom_base is None:
            self.loom_base, self.busy = obj.copy(), mot.copy()
        loom = np.maximum(obj - self.loom_base, 0.0)
        self.loom_base += (1 - math.exp(-dt / g['loom_tau'])) * (obj - self.loom_base)
        self.busy += (1 - math.exp(-dt / BUSY_TAU)) * (mot - self.busy)

        conf = raw[FLOW_CONF]
        pan = np.zeros(2)
        pan[int(raw[FLOW_X] > 0)] = _unit(abs(raw[FLOW_X]) * conf, *sc['pan'])
        scroll = _unit(abs(raw[FLOW_Y]) * conf, *sc['scroll'])
        taste = _unit(raw[RED].mean(), *sc['taste'])

        ones = np.ones((2, BANDS))
        drive = {'object': things[EYE_BINS] * self.w,
                 'loom': _unit(loom, *sc['loom'])[EYE_BINS],
                 'bustle': _unit(self.busy, *sc['bustle'])[EYE_BINS],
                 'pan': pan[:, None] * ones, 'scroll': scroll * ones, 'taste': taste * ones}
        return {ch: np.clip(drive[ch] / self.sat[ch], 0.0, 1.0) * self.hz[ch]
                for ch in CHANNELS}


# ============================================================================
# Preview
# ============================================================================

def preview(session):
    """Save one mid-session frame, enlarged, with the mask and bands drawn."""
    from PIL import Image, ImageDraw
    session = Path(session)
    chunks = sorted(session.glob('chunk_*.npz'))
    if not chunks:
        raise SystemExit(f'No chunk_*.npz in {session}')
    with np.load(chunks[len(chunks) // 2]) as z:
        frame = z['frames'][len(z['frames']) // 2]
    scale = 6
    img = Image.fromarray(frame).resize((FRAME_W * scale, FRAME_H * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img, 'RGBA')
    for x0, y0, x1, y1 in MASKS:
        draw.rectangle([x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height],
                       fill=(0, 120, 255, 90), outline=(0, 160, 255, 255))
    for b in range(1, N_BINS):
        x = b * img.width / N_BINS
        draw.line([x, 0, x, img.height], fill=(255, 255, 0, 200 if b == N_BINS // 2 else 80))
    out = session / 'preview.png'
    img.save(out)
    print(f'Wrote {out}: blue = masked, yellow = bands (bright line = left/right eye)')
    meta = session / 'session.json'
    if meta.exists():
        print(json.loads(meta.read_text()))


def main():
    parser = argparse.ArgumentParser(description='Preview what the fly sees')
    parser.add_argument('--preview', type=Path, required=True, metavar='SESSION_DIR')
    preview(parser.parse_args().preview)


if __name__ == '__main__':
    main()
