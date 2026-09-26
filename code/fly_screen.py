"""
Fly Screen: the emulated Drosophila brain watches your screen and texts you
what it does about it.

The whole-brain LIF model (fast_brain.py, same dynamics as run_pytorch.py)
runs in a closed loop with the screen:

    screen ──> motion, looming, optic-flow and color features
           ──> visual projection neurons (and sugar taste neurons)
           ──> 138k-neuron FlyWire connectome
           ──> descending neurons for steering, escape, head, walking, feeding
    text   <── whichever of those neurons are firing

The fly faces the screen, so the left half of the screen is its left eye.
The screen features stand in for the fly's optic lobe; in this model,
driving the photoreceptors directly reaches nothing downstream. Sending
fruit-colored pixels to sugar taste neurons is not biology either, just fun.
Everything between the input neurons and the descending neurons is the
unmodified connectome, and a message only reports which descending neurons
fire. No rule maps a screen feature straight to a message.

Neuron IDs and sides come from the FlyWire cell-type annotations (Schlegel et
al. 2024), stored in data/fly_screen_neurons.csv. The sugar neurons are the
21 used by the benchmark's sugar experiment.

Usage:
    python code/fly_screen.py               # watch the screen, messages in a floating window
    python code/fly_screen.py --no-window   # messages in the terminal only
    python code/fly_screen.py --demo        # synthetic scenes, no screen capture needed
    python code/fly_screen.py --game        # trash talk, even if no game is detected

While you play a game (on macOS, detected from the frontmost app), the fly
trash-talks you. It reacts to the same neurons; only its wording changes.

Screen capture needs mss (pip install mss). On macOS, the app you run this
from also needs Screen Recording permission (System Settings > Privacy &
Security), or the fly sees only the desktop wallpaper.
"""

import argparse
import math
import os
import plistlib
import queue
import random
import re
import subprocess
import sys
import threading
import traceback
from collections import namedtuple
from pathlib import Path
from time import perf_counter, strftime

import numpy as np
import pandas as pd
from scipy import ndimage

from fast_brain import DT, FastBrain, load_connectome

NEURONS_CSV = Path(__file__).resolve().parent.parent / 'data' / 'fly_screen_neurons.csv'

SIDES = ('left', 'right')
CHANNELS = ('object', 'loom', 'pan', 'scroll', 'bustle', 'taste')

# ============================================================================
# Eyes: screen features (luminance in 0-1, times in wall seconds)
# ============================================================================

GRID_W = 128                     # cells across the screen, the fly's "ommatidia"
CHANGE_THR = 0.08                # luminance change that counts as motion
SMALL_BLOB = 0.012               # changing blobs up to this fraction of the screen are objects
HABIT_STEP = 0.3                 # each change makes a cell this much more boring...
HABIT_RECOVER = 20.0             # ...and boredom fades with this time constant
LOOM_ADAPT_TAU = 0.5             # looming habituates, so a playing video stops startling
BUSY_TAU = 1.5                   # time constant of sustained local motion
FLOW_MIN = 0.03                  # fraction of the screen one global shift must explain
FLOW_COVER_SAT = 0.15            # moving fraction of the screen for full optic-flow drive
FLOW_SPEED_SAT = 0.6             # screens per second for full optic-flow drive
PERSIST_TAU = {'object': 0.2, 'loom': 0.2, 'pan': 0.3, 'scroll': 0.3}

# (feature value at zero drive, value at full drive)
OBJ = (1.0, 7.0)                 # novel changed cells in small blobs
LOOM = (0.08, 0.30)              # sudden rise in the big-blob fraction of a half screen
BUSY = (0.03, 0.15)              # changing fraction of a half screen, sustained
FRUIT = (0.05, 0.30)             # fraction of the screen in saturated red to yellow

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# ============================================================================
# Brain interface
# ============================================================================

LOOK_MS = 50                     # brain time per look at the screen
RATE_TAU_MS = 60.0               # smoothing of descending-neuron firing rates
RECENT_MS = 150.0                # how long an input counts as recent in a message
RATE_MAX = {                     # Hz Poisson input at full drive
    'object': 150.0,             # LC10a, small moving objects
    'loom': 150.0,               # LC4 + LPLC2, looming
    'pan': 200.0,                # HS + H2, horizontal wide-field motion
    'scroll': 200.0,             # VS, vertical wide-field motion
    'bustle': 120.0,             # LC9, lots of motion
    'taste': 150.0,              # sugar GRNs, fruit colors
}
INPUT_NAMES = {'object': 'LC10a', 'loom': 'LC4/LPLC2', 'pan': 'HS/H2', 'scroll': 'VS',
               'bustle': 'LC9', 'taste': 'sugar GRNs'}

# role: (neurons, sided, Hz to count as a behavior, Hz for full strength).
# A sided behavior is read from the left - right difference; the rest from
# the stronger side. Order is the priority when several start at once.
BEHAVIORS = {
    'escape': ('Giant Fiber DNp01', False, 50.0, 150.0),
    'turn': ('DNa02 steering', True, 35.0, 130.0),
    'feed': ('MN9 proboscis', False, 30.0, 80.0),
    'walk': ('DNp09 walking', False, 35.0, 100.0),
    'head_yaw': ('DNp15 head yaw', True, 30.0, 90.0),
    'head_tilt': ('DNp20 neck', False, 25.0, 80.0),
}

Behavior = namedtuple('Behavior', 'role side strength detail')
Message = namedtuple('Message', 'kind text detail')


def ramp(x, lo, hi):
    return min(max((x - lo) / (hi - lo), 0.0), 1.0)


def downsample(img, grid_h, grid_w):
    """Average an H x W x 3 uint8 image into a grid of RGB floats in [0, 1]."""
    h, w = img.shape[:2]
    step = max(1, min(h // grid_h, w // grid_w) // 4)     # ~4x4 samples per cell
    img = img[::step, ::step].astype(np.float32)
    h, w = img.shape[:2]
    rows = np.arange(grid_h) * h // grid_h
    cols = np.arange(grid_w) * w // grid_w
    sums = np.add.reduceat(np.add.reduceat(img, rows, axis=0), cols, axis=1)
    counts = np.outer(np.diff(np.r_[rows, h]), np.diff(np.r_[cols, w]))
    return sums / (counts[..., None] * 255.0)


class Retina:
    """Turns successive screen frames into drive (0-1) for each input channel.

    Stands in for the optic lobe: every channel gets a [left, right] pair,
    except 'pan', which is [leftward, rightward] motion.
    """

    def __init__(self, grid_h, grid_w):
        self.shape = (grid_h, grid_w)
        left = np.zeros(self.shape, dtype=bool)
        left[:, :grid_w // 2] = True
        self.halves = (left, ~left)
        self.half_cells = grid_h * grid_w / 2
        self.small_blob = max(4, int(SMALL_BLOB * grid_h * grid_w))
        self.window = np.outer(np.hanning(grid_h), np.hanning(grid_w)).astype(np.float32)
        self.reset()

    def reset(self):
        self.prev = None
        self.habit = np.zeros(self.shape, dtype=np.float32)
        self.loom_base = np.zeros(2)
        self.busy = np.zeros(2)
        self.drive = {ch: np.zeros(2) for ch in CHANNELS}

    def see(self, rgb, dt, mask=None):
        """Take in one grid frame, dt seconds after the last. Returns the drive.

        Cells under mask (the message window) are ignored.
        """
        visible = np.ones(self.shape, dtype=bool) if mask is None else ~mask
        new = {ch: np.zeros(2) for ch in CHANNELS}
        new['taste'][:] = ramp(self._fruit_fraction(rgb, visible), *FRUIT)

        lum = rgb @ LUMA
        prev = self.prev
        if prev is not None and mask is not None:
            lum[mask] = prev[mask]
        self.prev = lum
        if prev is not None:
            diff = lum - prev
            changed = np.abs(diff) > CHANGE_THR
            # LC neurons are suppressed by wide-field motion, so while the
            # whole view moves, local changes are neither objects nor looms.
            flow = self._optic_flow(prev, lum, changed, dt, new)
            self._local_motion(np.zeros_like(changed) if flow else changed, diff, dt, new)
            # Something that keeps changing in one place (a blinking caret, a
            # spinner, a clock) stops catching the fly's eye
            self.habit *= math.exp(-dt / HABIT_RECOVER)
            self.habit[changed] += HABIT_STEP * (1 - self.habit[changed])

        for ch, tau in PERSIST_TAU.items():
            new[ch] = np.maximum(new[ch], self.drive[ch] * math.exp(-dt / tau))
        self.drive = new
        return new

    def _optic_flow(self, prev, cur, changed, dt, new):
        """Scrolling and panning: the view shifting as a whole drives HS/H2 and VS."""
        n_min = FLOW_MIN * changed.size
        n_changed = changed.sum()
        if n_changed < n_min:
            return False
        dy, dx = self._shift(prev, cur)
        moved = np.roll(prev, (dy, dx), axis=(0, 1))
        valid = np.ones_like(changed)
        if dy > 0:
            valid[:dy] = False
        elif dy < 0:
            valid[dy:] = False
        if dx > 0:
            valid[:, :dx] = False
        elif dx < 0:
            valid[:, dx:] = False
        explained = (changed & valid & (np.abs(cur - moved) <= CHANGE_THR)).sum()
        if explained < max(n_min, 0.4 * n_changed):
            return False

        gain = min(explained / changed.size / FLOW_COVER_SAT, 1.0)
        vx = dx / self.shape[1] / dt                 # screens per second
        vy = dy / self.shape[0] / dt
        new['pan'][int(vx > 0)] = min(abs(vx) / FLOW_SPEED_SAT, 1.0) * gain
        new['scroll'][:] = min(abs(vy) / FLOW_SPEED_SAT, 1.0) * gain
        return True

    def _shift(self, a, b):
        """Displacement (dy, dx) of frame b from frame a by phase correlation.

        Zero shift is excluded, so a scrolling pane is found even when the
        rest of the screen is still.
        """
        fa = np.fft.rfft2((a - a.mean()) * self.window)
        fb = np.fft.rfft2((b - b.mean()) * self.window)
        cross = fb * np.conj(fa)
        corr = np.fft.irfft2(cross / (np.abs(cross) + 1e-9), s=a.shape)
        corr[0, 0] = -np.inf
        dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
        h, w = a.shape
        return int(dy - h if dy > h // 2 else dy), int(dx - w if dx > w // 2 else dx)

    def _local_motion(self, changed, diff, dt, new):
        """Small changing blobs are objects (LC10a), big sudden ones loom
        (LC4, LPLC2), and change that keeps going is bustle (LC9)."""
        labels, _ = ndimage.label(changed, structure=np.ones((3, 3)))
        small = np.bincount(labels.ravel()) <= self.small_blob
        small[0] = False
        in_small = small[labels]
        in_big = changed & ~in_small
        novelty = 1.0 - self.habit
        dark = np.where(diff < 0, 1.0, 0.7)        # looming detectors prefer darkening
        a_loom = 1 - math.exp(-dt / LOOM_ADAPT_TAU)
        a_busy = 1 - math.exp(-dt / BUSY_TAU)
        for i, half in enumerate(self.halves):
            obj = novelty[in_small & half].sum()
            big = dark[in_big & half].sum() / self.half_cells
            active = (changed & half).sum() / self.half_cells
            new['object'][i] = ramp(obj, *OBJ)
            new['loom'][i] = ramp(big - self.loom_base[i], *LOOM)
            self.loom_base[i] += a_loom * (big - self.loom_base[i])
            self.busy[i] += a_busy * (active - self.busy[i])
            new['bustle'][i] = ramp(self.busy[i], *BUSY)

    @staticmethod
    def _fruit_fraction(rgb, visible):
        """Fraction of the visible screen in saturated red, orange or yellow."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        hi, lo = rgb.max(axis=-1), rgb.min(axis=-1)
        chroma = hi - lo
        hue = (g - b) / np.maximum(chroma, 1e-6)    # hue / 60 deg when red is brightest
        fruit = ((r == hi) & (hue > -0.5) & (hue <= 1.0)
                 & (chroma > 0.45 * hi) & (hi > 0.35) & visible)
        return fruit.sum() / max(visible.sum(), 1)


# ============================================================================
# Brain: screen features in, descending-neuron firing rates out
# ============================================================================

class ScreenFly:
    """The connectome between the Retina's channels and the behavior neurons."""

    def __init__(self, weights, flyid2i, seed=None):
        df = pd.read_csv(NEURONS_CSV)

        def pick(mask):
            return np.array([flyid2i[f] for f in df.loc[mask, 'flywire_id']], dtype=np.int64)

        groups = {}
        for i, side in enumerate(SIDES):
            for ch in ('object', 'loom', 'scroll', 'bustle', 'taste'):
                groups[ch, side] = pick((df.role == ch) & (df.side == side))
            # Motion toward the fly's right is front-to-back for its right
            # eye (HS cells) and back-to-front for its left eye (H2); in the
            # connectome both reach the right DNp15.
            other = SIDES[1 - i]
            groups['pan', side] = pick((df.role == 'pan') & (
                ((df.cell_type != 'H2') & (df.side == side))
                | ((df.cell_type == 'H2') & (df.side == other))))

        self.slices, start = {}, 0
        for key, idx in groups.items():
            self.slices[key] = slice(start, start + len(idx))
            start += len(idx)
        self.brain = FastBrain(weights, np.concatenate(list(groups.values())), seed=seed)

        # 0 = not read out, k = behavior neuron group k - 1
        self.out_keys = [(role, side) for role in BEHAVIORS for side in SIDES]
        self.out_code = np.zeros(self.brain.n, dtype=np.int64)
        self.out_size = np.zeros(len(self.out_keys))
        for k, (role, side) in enumerate(self.out_keys):
            idx = pick((df.role == role) & (df.side == side))
            self.out_code[idx] = k + 1
            self.out_size[k] = len(idx)

        self.smooth = 1 - math.exp(-LOOK_MS / RATE_TAU_MS)
        self.fade = math.exp(-LOOK_MS / RECENT_MS)
        self.t_ms = 0
        self.reset()

    def reset(self):
        self.brain.reset()
        self.input_hz = {ch: np.zeros(2) for ch in CHANNELS}
        self.recent_hz = {ch: np.zeros(2) for ch in CHANNELS}
        self.rates = {role: np.zeros(2) for role in BEHAVIORS}

    def look(self, drive, ms=LOOK_MS):
        """Show the fly the screen features for ms of brain time.

        drive maps each channel to [left, right] strengths in 0-1. Returns the
        smoothed firing rates (Hz) of the behavior neurons, role -> [left, right].
        """
        for ch in CHANNELS:
            self.input_hz[ch] = drive[ch] * RATE_MAX[ch]
            self.recent_hz[ch] = np.maximum(self.input_hz[ch], self.recent_hz[ch] * self.fade)
            for i, side in enumerate(SIDES):
                self.brain.rates[self.slices[ch, side]] = self.input_hz[ch][i]

        counts = np.zeros(len(self.out_keys) + 1, dtype=np.int64)
        for _ in range(int(round(ms / DT))):
            spikes = self.brain.step()
            if len(spikes):
                counts += np.bincount(self.out_code[spikes], minlength=len(counts))
        self.t_ms += ms

        hz = counts[1:] / self.out_size / (ms / 1000)
        for k, (role, side) in enumerate(self.out_keys):
            rate = self.rates[role]
            i = SIDES.index(side)
            rate[i] += self.smooth * (hz[k] - rate[i])
        return self.rates


def read_behaviors(rates):
    """Behaviors the descending neurons are commanding, strongest first."""
    out = []
    for role, (label, sided, threshold, full) in BEHAVIORS.items():
        left, right = rates[role]
        level = abs(left - right) if sided else max(left, right)
        if level < threshold:
            continue
        if sided or (role == 'escape' and max(left, right) >= 1.3 * min(left, right)):
            side = SIDES[int(right > left)]
        else:
            side = None                          # for escape: both Giant Fibers alike
        out.append(Behavior(role, side, level / full,
                            f'{label} L {left:.0f} · R {right:.0f} Hz'))
    return sorted(out, key=lambda b: -b.strength)


# ============================================================================
# Voice: behaviors to text messages
# ============================================================================

# Lines say what the fly does, never what was on screen: any input can reach
# any of these neurons through the connectome, so the cause is not assumed.
SAY = {
    'escape': [
        'WHOA! Something is coming at me {where}. Taking off!',
        'Incoming {where}! Jumping out of here!',
        'AAH! Escape jump!',
    ],
    'turn': [
        "What's that on my {side}? Turning to look.",
        'Turning {side} to get a better look.',
        'Something caught my eye. Swiveling {side} to track it.',
    ],
    'feed': [
        'Is that... food? Sticking out my proboscis.',
        'Extending my proboscis for a taste.',
        'Mmm? Tasting the screen.',
    ],
    'walk': [
        'Walking forward to check it out.',
        'Walking closer.',
        'Marching forward!',
    ],
    'head_yaw': [
        'Whoa, the world is swinging. Turning my head {side} to keep up.',
        'Following the world with my head, off to the {side}.',
    ],
    'head_tilt': [
        'Whoa, tilting my head to keep the world level.',
        'Tilting my head... getting a bit dizzy.',
    ],
}
STILL = {
    'escape': ['Still fleeing!'],
    'turn': ['Still watching something on my {side}.', 'Keeping an eye on the {side}.'],
    'feed': ['Still tasting. Mmm.', 'Still sipping.'],
    'walk': ['Still walking forward.'],
    'head_yaw': ['Still turning my head {side}.'],
    'head_tilt': ['Still tilting my head.'],
}
CALM = [
    'All quiet now. Just sitting here.',
    'Calm again.',
    "Nothing's happening. I'll just sit here and watch.",
]
WHERE = {'left': 'from the left', 'right': 'from the right', None: 'from all sides'}

# While you play a game, the same behaviors come out as trash talk
TRASH_SAY = {
    'escape': [
        "AAH! Even I'd run from that play. Jumping out!",
        'Incoming {where}! Unlike you, I actually dodge things.',
        'Nope! Bailing out. You should too.',
    ],
    'turn': [
        "Wait, what's that on the {side}? Did you even see it?",
        'Turning {side}. My 138k neurons track better than your whole brain.',
        'Over there, on the {side}! Pay attention!',
    ],
    'feed': [
        "Snack break. Not like you're winning anyway.",
        'Tasting the screen. Tastes like a loss.',
        "Sticking out my proboscis. At least one of us is getting something done.",
    ],
    'walk': [
        'Walking closer to watch this train wreck.',
        'Marching in for a better view of you losing.',
        'Getting closer. This is too painful to miss.',
    ],
    'head_yaw': [
        'Whipping my head {side}. Pick a direction!',
        'Turning my head {side}. Even the camera is confused.',
    ],
    'head_tilt': [
        'Tilting my head... what was that move?',
        "I'm getting dizzy just watching you play.",
    ],
}
TRASH_STILL = {
    'escape': ['Still fleeing. Smart, unlike you.'],
    'turn': ["Still watching the {side}. You still haven't noticed?"],
    'feed': ['Still snacking. Wake me when you win.'],
    'walk': ['Still walking. Still waiting for you to get good.'],
    'head_yaw': ['Still turning {side}. Make up your mind.'],
    'head_tilt': ['Still tilting. Still confused by your strategy.'],
}
TRASH_CALM = [
    'Nothing happening? Are you AFK?',
    "I've seen snails play faster.",
    'Boring. Do something!',
]
GAME_START = [
    "Oh, you're playing {game}? This I've got to see.",
    '{game}, huh? Try not to embarrass yourself.',
    "{game}? I've got 138,639 neurons and I bet I'm better at it.",
]
GAME_STOP = [
    'Done with {game} already? Rage quit?',
    "Leaving {game}? Probably for the best.",
]


class Narrator:
    """Decides when the fly says something, based only on its behaviors.

    While a game is being played (self.game), the same behaviors are voiced
    as trash talk.
    """

    MIN_GAP = 1.5          # s between messages, except for escapes
    REPEAT = 10.0          # s before mentioning an ongoing behavior again
    FORGET = 5.0           # s a behavior must stop for before it is news again
    CALM_AFTER = 6.0       # s without behavior before saying so

    def __init__(self, seed=None, game=None):
        self.rng = random.Random(seed)
        self.game = game               # name of the game being played, or None
        self.reset()

    def reset(self):
        self.mentioned = {}            # key -> time last mentioned
        self.active_at = {}            # key -> time last active
        self.last_t = -math.inf
        self.quiet_since = None
        self.calm = True

    @staticmethod
    def key(b):
        # A turn to the other side is news; an escape changing side is not
        return b.role, b.side if BEHAVIORS[b.role][1] else None

    def update(self, t, behaviors):
        """Return a Message if the fly has something to say at time t (s)."""
        active = {self.key(b): b for b in behaviors}
        self.active_at.update((k, t) for k in active)
        for k in [k for k, seen in self.active_at.items() if t - seen > self.FORGET]:
            del self.active_at[k]
            self.mentioned.pop(k, None)
        if not active:
            if self.quiet_since is None:
                self.quiet_since = t
            if not self.calm and t - self.quiet_since >= self.CALM_AFTER:
                self.calm = True
                return self._post(t, [], 'calm', self.rng.choice(TRASH_CALM if self.game else CALM),
                                  'no behavior neurons above threshold')
            return None
        self.quiet_since = None

        new = [b for key, b in active.items() if key not in self.mentioned]
        escapes = [b for b in new if b.role == 'escape']
        say, still = (TRASH_SAY, TRASH_STILL) if self.game else (SAY, STILL)
        if escapes:
            said, lines = escapes[:1], say
        elif new and t - self.last_t >= self.MIN_GAP:
            said, lines = new[:2], say
        elif not new and t - self.last_t >= self.REPEAT:
            said = [min(behaviors, key=lambda b: self.mentioned[self.key(b)])]
            lines = still
        else:
            return None
        text = ' '.join(self.rng.choice(lines[b.role]).format(side=b.side, where=WHERE[b.side])
                        for b in said)
        return self._post(t, said, said[0].role, text, '   '.join(b.detail for b in said))

    def set_game(self, t, game, detail=''):
        """Switch trash talk on (game name) or off (None); returns the announcement."""
        if game == self.game:
            return None
        lines, name = (GAME_START, game) if game else (GAME_STOP, self.game)
        self.game = game
        return self._post(t, [], 'game', self.rng.choice(lines).format(game=name), detail)

    def _post(self, t, said, kind, text, detail):
        for b in said:
            self.mentioned[self.key(b)] = t
        self.last_t = t
        self.calm = kind == 'calm'
        return Message(kind, text, detail)


def with_inputs(msg, fly, top=3, min_hz=20.0):
    """Add the most strongly driven recent input neurons to a message's detail."""
    seen = []
    for ch, name in INPUT_NAMES.items():
        hz = fly.recent_hz[ch]
        if ch == 'taste':                # one group; both entries are the same
            seen.append((hz[0], name))
        else:
            seen += [(h, f'{name} {s}') for s, h in zip('LR', hz)]
    seen = [f'{name} {h:.0f}' for h, name in sorted(seen, reverse=True)[:top] if h >= min_hz]
    seen = ', '.join(seen) + ' Hz' if seen else 'none'
    return msg._replace(detail=f'{msg.detail}   |   input: {seen}')


def status_text(fly, speed):
    """Live input and output firing rates, for the window footer and --verbose."""
    i, r = fly.input_hz, fly.rates

    def lr(x):
        return f'{x[0]:.0f}|{x[1]:.0f}'

    return (f'in Hz   LC10a {lr(i["object"])}  LC4/LPLC2 {lr(i["loom"])}  LC9 {lr(i["bustle"])}\n'
            f'        HS/H2 {lr(i["pan"])}  VS {lr(i["scroll"])}  sugar {i["taste"][0]:.0f}\n'
            f'out Hz  DNa02 {lr(r["turn"])}  GF {lr(r["escape"])}  P9 {lr(r["walk"])}\n'
            f'        MN9 {lr(r["feed"])}  DNp15 {lr(r["head_yaw"])}  DNp20 {lr(r["head_tilt"])}\n'
            f'brain   {fly.t_ms / 1000:.1f} s simulated  ·  {speed:.2f}x real time')


# ============================================================================
# Watching the real screen
# ============================================================================

class ScreenCamera:
    """Grabs a monitor with mss and averages it down to the fly's grid."""

    def __init__(self, monitor=1, grid_w=GRID_W):
        try:
            import mss
        except ImportError:
            raise RuntimeError('Screen capture needs the mss package: pip install mss') from None
        self.sct = (getattr(mss, 'MSS', None) or mss.mss)()
        self.mon = self.sct.monitors[monitor]
        self.grid_w = grid_w
        self.grid_h = max(8, round(grid_w * self.mon['height'] / self.mon['width']))

    def grab(self):
        shot = self.sct.grab(self.mon)
        img = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return downsample(img[..., 2::-1], self.grid_h, self.grid_w)

    def mask(self, rect, pad=1):
        """Grid cells under rect = (x, y, w, h) in screen coordinates."""
        m = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        if rect is None:
            return m
        x, y, w, h = rect
        sx = self.grid_w / self.mon['width']
        sy = self.grid_h / self.mon['height']
        c0 = max(int((x - self.mon['left']) * sx) - pad, 0)
        c1 = max(math.ceil((x + w - self.mon['left']) * sx) + pad, 0)
        r0 = max(int((y - self.mon['top']) * sy) - pad, 0)
        r1 = max(math.ceil((y + h - self.mon['top']) * sy) + pad, 0)
        m[r0:r1, c0:c1] = True
        return m


def check_screen_permission():
    """On macOS, warn (and ask once) if windows of other apps can't be captured."""
    if sys.platform != 'darwin':
        return
    import ctypes
    try:
        cg = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
        cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        if not cg.CGPreflightScreenCaptureAccess():
            print('Screen Recording permission is missing, so the fly will only see the '
                  'desktop wallpaper.\nAllow it for this terminal in System Settings > '
                  'Privacy & Security > Screen Recording, then restart it.')
            cg.CGRequestScreenCaptureAccess()
    except (OSError, AttributeError):
        pass


GAME_NAMES = ('roblox', 'minecraft')     # games that don't declare the games category


def frontmost_app():
    """(display name, bundle path, pid) of the frontmost app on macOS, else Nones."""
    if sys.platform != 'darwin':
        return None, None, None
    try:
        asn = subprocess.run(['lsappinfo', 'front'], capture_output=True, text=True, timeout=2).stdout.strip()
        info = subprocess.run(['lsappinfo', 'info', '-only', 'name', '-only', 'bundlepath', '-only', 'pid', asn],
                              capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    fields = dict(re.findall(r'"(\w+)"=("[^"]*"|\d+)', info))
    pid = fields.get('pid')
    return (fields.get('LSDisplayName', '').strip('"') or None,
            fields.get('LSBundlePath', '').strip('"') or None,
            int(pid) if pid else None)


class GameDetector:
    """Tells whether the frontmost app is a game (macOS; elsewhere use --game).

    An app counts as a game if its Info.plist says so, it lives in a Steam
    library, or its name is in GAME_NAMES.
    """

    POLL_S = 2.0           # how often to check the frontmost app
    GRACE_S = 6.0          # a game must be out of front this long to count as stopped

    def __init__(self):
        self.known = {}                # bundle path or name -> is a game
        self.game = None
        self.last_poll = -math.inf
        self.last_seen = -math.inf
        self.app = None

    def poll(self, now):
        """Name of the game being played, or None. Cheap to call every look."""
        if now - self.last_poll < self.POLL_S:
            return self.game
        self.last_poll = now
        name, path, pid = frontmost_app()
        if pid == os.getpid():         # clicking the fly's own window isn't quitting
            self.last_seen = now if self.game else self.last_seen
            return self.game
        self.app = name
        if name and self._is_game(name, path):
            self.game, self.last_seen = name, now
        elif now - self.last_seen >= self.GRACE_S:
            self.game = None
        return self.game

    def _is_game(self, name, path):
        key = path or name
        if key not in self.known:
            category = ''
            if path:
                try:
                    with open(Path(path) / 'Contents' / 'Info.plist', 'rb') as f:
                        category = str(plistlib.load(f).get('LSApplicationCategoryType', ''))
                except (OSError, plistlib.InvalidFileException, ValueError):
                    pass
            self.known[key] = ('games' in category or '/steamapps/' in (path or '')
                               or any(g in name.lower() for g in GAME_NAMES))
        return self.known[key]


def watch_screen(args, emit, stop, get_rect=lambda: None):
    """Screen -> fly -> messages until stop is set. emit(kind, payload) reports."""
    camera = ScreenCamera(args.monitor)
    emit('system', 'Loading the fly brain (138,639 neurons)...')
    weights, flyid2i = load_connectome()
    fly = ScreenFly(weights, flyid2i, seed=args.seed)
    retina = Retina(camera.grid_h, camera.grid_w)
    narrator = Narrator(seed=args.seed, game='this game' if args.game else None)
    games = None if args.game else GameDetector()
    emit('system', "Hi! I'm a fruit fly brain, wired like the FlyWire connectome. "
                   'Show me something.')

    start = last = perf_counter()
    speed = 0.0
    while not stop.is_set():
        frame = camera.grab()
        now = perf_counter()
        if games:
            game = games.poll(now)
            msg = narrator.set_game(now - start, game, f'frontmost app: {game or games.app}')
            if msg:
                emit('message', msg)
        drive = retina.see(frame, max(now - last, 1e-3), camera.mask(get_rect()))
        rates = fly.look(drive)
        msg = narrator.update(now - start, read_behaviors(rates))
        done = perf_counter()
        speed += 0.2 * (LOOK_MS / 1000 / (done - last) - speed)
        last = now
        if msg:
            emit('message', with_inputs(msg, fly))
        emit('status', status_text(fly, speed))


def print_message(msg, clock=None):
    clock = clock or strftime('%H:%M:%S')
    print(f'[{clock}] Fly: {msg.text}')
    if msg.detail:
        print(f'{"":11s}({msg.detail})')


def run_terminal(args):
    check_screen_permission()
    stop = threading.Event()

    def emit(kind, payload):
        if kind == 'message':
            print_message(payload)
        elif kind == 'system':
            print(payload)
        elif kind == 'status' and args.verbose:
            print(payload + '\n')

    try:
        watch_screen(args, emit, stop)
    except KeyboardInterrupt:
        print('\nBye!')
    except RuntimeError as e:
        sys.exit(str(e))


# ============================================================================
# Message window
# ============================================================================

BG, FG, MUTED, LINE = '#0e1116', '#e6edf3', '#7d8590', '#30363d'
FLY_COLOR = '#f2b134'
BUBBLE = {'escape': '#6e2b2b', 'calm': '#161b22', 'game': '#3d2a5c'}
BUBBLE_DEFAULT = '#21262d'


class MessageWindow:
    """A small always-on-top chat window where the fly posts its reactions.

    The window hides itself from the fly: its area is masked out of every
    frame, so the fly doesn't react to its own messages.
    """

    WIDTH, HEIGHT = 400, 600
    KEEP = 8                    # bubbles kept on screen
    POLL_MS = 100

    def __init__(self, root, args):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.args = args
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.rect = None
        self.bubbles = []

        root.title('Fly')
        root.configure(bg=BG)
        root.attributes('-topmost', True)
        x = root.winfo_screenwidth() - self.WIDTH - 24
        y = max(root.winfo_screenheight() - self.HEIGHT - 110, 30)
        root.geometry(f'{self.WIDTH}x{self.HEIGHT}+{x}+{y}')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Escape>', lambda e: self.close())

        header = tk.Frame(root, bg=BG)
        header.pack(fill='x', padx=14, pady=(12, 6))
        icon = tk.Canvas(header, width=34, height=28, bg=BG, highlightthickness=0)
        icon.create_oval(4, 2, 22, 13, fill='#c9d1d9', outline='')       # wings
        icon.create_oval(4, 15, 22, 26, fill='#c9d1d9', outline='')
        icon.create_oval(2, 10, 26, 18, fill='#8b5a2b', outline='')      # body
        icon.create_oval(23, 9, 33, 19, fill='#d1242f', outline='')      # head
        icon.pack(side='left')
        tk.Label(header, text='Fly', bg=BG, fg=FG, font=('Helvetica', 16, 'bold')).pack(side='left', padx=(8, 0))
        self.state = tk.Label(header, text='waking up...', bg=BG, fg=MUTED, font=('Helvetica', 11))
        self.state.pack(side='left', padx=(10, 0))
        tk.Frame(root, bg=LINE, height=1).pack(fill='x')

        self.footer = tk.Label(root, bg=BG, fg=MUTED, font=('Menlo', 10), justify='left', anchor='w')
        self.footer.pack(side='bottom', fill='x', padx=14, pady=(4, 10))
        tk.Frame(root, bg=LINE, height=1).pack(side='bottom', fill='x')
        self.log = tk.Frame(root, bg=BG)
        self.log.pack(fill='both', expand=True, padx=14, pady=4)

        threading.Thread(target=self._watch, daemon=True).start()
        root.after(self.POLL_MS, self._poll)

    def _watch(self):
        try:
            watch_screen(self.args, lambda kind, p: self.events.put((kind, p)),
                         self.stop, lambda: self.rect)
        except Exception:
            self.events.put(('error', traceback.format_exc()))

    def _poll(self):
        r = self.root
        top = 32                  # the title bar sits above the window's content
        self.rect = (r.winfo_rootx(), r.winfo_rooty() - top, r.winfo_width(), r.winfo_height() + top)
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == 'message':
                    print_message(payload)
                    self._bubble(payload.text, f'{strftime("%H:%M:%S")}  ·  {payload.detail}',
                                 BUBBLE.get(payload.kind, BUBBLE_DEFAULT))
                elif kind == 'system':
                    print(payload)
                    self._bubble(payload, '', BG, fg=MUTED)
                elif kind == 'status':
                    self.state.configure(text='watching your screen')
                    self.footer.configure(text=payload)
                elif kind == 'error':
                    print(payload, file=sys.stderr)
                    self.state.configure(text='stopped: error')
                    self._bubble(payload.strip().splitlines()[-1], 'see the terminal for details', BUBBLE['escape'])
        except queue.Empty:
            pass
        self.root.after(self.POLL_MS, self._poll)

    def _bubble(self, text, detail, bg, fg=FG):
        tk = self.tk
        frame = tk.Frame(self.log, bg=BG)
        tk.Label(frame, text=text, bg=bg, fg=fg, font=('Helvetica', 13), wraplength=self.WIDTH - 60,
                 justify='left', anchor='w', padx=10, pady=7).pack(anchor='w')
        if detail:
            tk.Label(frame, text=detail, bg=BG, fg=MUTED, font=('Helvetica', 10),
                     wraplength=self.WIDTH - 40, justify='left').pack(anchor='w', pady=(2, 0))
        # Newest at the bottom; when the log overflows, the oldest are clipped
        kw = {'before': self.bubbles[-1]} if self.bubbles else {}
        frame.pack(side='bottom', anchor='w', fill='x', pady=5, **kw)
        self.bubbles.append(frame)
        while len(self.bubbles) > self.KEEP:
            self.bubbles.pop(0).destroy()

    def close(self):
        self.stop.set()
        self.root.destroy()


def run_window(args):
    import tkinter as tk
    check_screen_permission()
    root = tk.Tk()
    MessageWindow(root, args)
    root.mainloop()


# ============================================================================
# Demo: synthetic scenes, no screen capture
# ============================================================================

DEMO_H, DEMO_W = 400, 640
DEMO_DT = 0.25                   # nominal seconds between frames
DEMO_LOOKS = 16


def _canvas(value=210):
    return np.full((DEMO_H, DEMO_W, 3), value, dtype=np.uint8)


def _disk(img, cx, cy, r, color):
    yy, xx = np.ogrid[:DEMO_H, :DEMO_W]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = color
    return img


def demo_scenes(rng):
    """(description, frame k -> H x W x 3 uint8) for each demo scene."""
    # A tall page of "text" lines and a few pictures, for scrolling
    page = np.full((DEMO_H * 8, DEMO_W, 3), 245, dtype=np.uint8)
    for y in range(20, page.shape[0] - 20, 22):
        x = 40
        while x < DEMO_W - 60:
            w = int(rng.integers(15, 70))
            page[y:y + 9, x:min(x + w, DEMO_W - 40)] = 40
            x += w + 10
    for y in range(150, page.shape[0] - 200, 520):
        page[y:y + 160, 60:320] = rng.integers(0, 256, 3)

    # A wide landscape of soft blobs, for panning
    land = np.full((DEMO_H, DEMO_W * 6, 3), 120, dtype=np.uint8)
    yy, xx = np.ogrid[:DEMO_H, :land.shape[1]]
    for _ in range(160):
        cx, cy, r = rng.integers(0, land.shape[1]), rng.integers(0, DEMO_H), rng.integers(15, 60)
        land[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = rng.integers(20, 236)

    fruit = _canvas(0)
    fruit[:] = (40, 90, 45)
    for _ in range(14):
        color = [(255, 140, 0), (220, 30, 30), (250, 215, 40)][int(rng.integers(3))]
        _disk(fruit, rng.integers(60, DEMO_W - 60), rng.integers(60, DEMO_H - 60), rng.integers(45, 90), color)

    def video(k):
        img = _canvas(180)
        clip = rng.integers(0, 256, (7, 7, 3), dtype=np.uint8)
        img[60:340, 40:292] = np.kron(clip, np.ones((40, 36, 1), dtype=np.uint8))
        return img

    return [
        ('A plain gray screen', lambda k: _canvas()),
        ('A small dot darting around on the left',
         lambda k: _disk(_canvas(), 160 + 100 * math.sin(1.7 * k), 200 + 120 * math.sin(1.1 * k + 1), 9, 0)),
        ('A small dot darting around on the right',
         lambda k: _disk(_canvas(), 480 + 100 * math.sin(1.7 * k), 200 + 120 * math.sin(1.1 * k + 1), 9, 0)),
        ('A dark shape rushing at the fly from the right',
         lambda k: _disk(_canvas(), 500, 200, min(8 * 1.5 ** k, 800), 20)),
        ('A web page scrolling down',
         lambda k: page[k * 48:k * 48 + DEMO_H]),
        ('The whole scene sliding to the right',
         lambda k: land[:, DEMO_W * 4 - k * 40:DEMO_W * 5 - k * 40]),
        ('A picture of fruit', lambda k: fruit),
        ('A video playing on the left', video),
    ]


def run_demo(args):
    print('Loading the fly brain (138,639 neurons)...')
    weights, flyid2i = load_connectome()
    fly = ScreenFly(weights, flyid2i, seed=args.seed)
    retina = Retina(80, GRID_W)
    narrator = Narrator(seed=args.seed, game='this game' if args.game else None)
    rng = np.random.default_rng(0 if args.seed is None else args.seed)

    for title, frame in demo_scenes(rng):
        print(f'\n== {title}')
        fly.reset()
        retina.reset()
        narrator.reset()
        peak = {role: np.zeros(2) for role in BEHAVIORS}
        t0 = perf_counter()
        for k in range(DEMO_LOOKS):
            drive = retina.see(downsample(frame(k), 80, GRID_W), DEMO_DT)
            rates = fly.look(drive)
            for role in BEHAVIORS:
                np.maximum(peak[role], rates[role], out=peak[role])
            if args.verbose:
                print(status_text(fly, 0.0) + '\n')
            msg = narrator.update(k * DEMO_DT, read_behaviors(rates))
            if msg:
                print_message(with_inputs(msg, fly), clock=f'{k * DEMO_DT:5.2f} s ')
        wall = perf_counter() - t0
        print('   peak Hz: ' + '  '.join(f'{role} {p[0]:.0f}|{p[1]:.0f}' for role, p in peak.items())
              + f'   ({DEMO_LOOKS * LOOK_MS} ms brain in {wall:.1f} s)')


def main():
    parser = argparse.ArgumentParser(description='The emulated fly brain watches your screen')
    parser.add_argument('--demo', action='store_true',
                        help='Show the fly synthetic scenes instead of the screen')
    parser.add_argument('--no-window', action='store_true',
                        help='Print the messages in the terminal instead of a window')
    parser.add_argument('--game', action='store_true',
                        help='Trash-talk as if you were playing a game (default: detect games, macOS)')
    parser.add_argument('--monitor', type=int, default=1,
                        help='Screen to watch, numbered as in mss (default: 1, the main screen)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print input and output firing rates for every look (demo, --no-window)')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.demo:
        run_demo(args)
    elif args.no_window:
        run_terminal(args)
    else:
        run_window(args)


if __name__ == '__main__':
    main()
