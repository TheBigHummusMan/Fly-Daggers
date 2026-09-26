"""
ScratchSim: a small, fast stand-in for Scritchy Scratchy, so the fly's
interface can be evolved over thousands of episodes (the real game runs in
real time, one copy at a time).

Same interface as game_io.RealGame:
    env.reset(seed) -> frame     env.frame()     env.act(x, y, button)     env.money()
plus env.advance(ms) to move sim time on (the real game moves on by itself).

Frames are rendered straight at the fly's grid resolution (256 px wide, one
pixel per CursorFly cell), on a backdrop taken from a real screenshot, so the
fly sees the same layout in both.

VERSION 0: the ticket rules below (RULES) are placeholders until they are
calibrated from recordings (recorder.py). Upgrades, the Scratch Bot, loans and
prestige are not modelled yet.

Usage:
    python code/scratch/sim.py --render            # zigzag bot, saves a few frames
"""

import argparse
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'scratch'
BACKDROP = DATA_DIR / 'backdrop.png'   # a clean screenshot of the empty table
GW, GH = 256, 167                      # frame size = CursorFly grid for a 1024 x 668 window

# 0-1 window coordinates, measured on the 1024 x 668 window
TABLE = (0.15, 0.13, 0.86, 0.88)
SHOP = [                               # (name, rect of its button in the ticket list)
    ('Day Job', (0.02, 0.175, 0.16, 0.255)),
    ('Two Win', (0.02, 0.335, 0.16, 0.415)),
]
FORBIDDEN = [(0.0, 0.0, 1.0, 0.045), (0.945, 0.04, 1.0, 0.1)]   # title bar, settings

# Placeholder rules, to calibrate from recordings
RULES = {
    'start_money': 8.0,
    'start_tickets': 1,                # free Day Jobs on the table at the start, so scratching pays at once
    'max_tickets': 4,                  # on the table at once
    'coin_radius': 0.016,              # scratch radius, fraction of window width
    'reveal_frac': 0.85,               # a cell counts as revealed when this much foil is gone
    'claim_ms': 600.0,                 # a fully revealed ticket pays out and vanishes after this
    'tickets': {
        # price, foil cells (cols x rows), P(win symbol), win value, P(penalty), penalty
        'Day Job': dict(price=1.0, grid=(3, 2), p_win=0.35, win=1.0, p_bad=0.0, bad=0.0,
                        size=(0.10, 0.20), colour=(0.93, 0.93, 0.88)),
        'Two Win': dict(price=10.0, grid=(3, 3), p_win=0.25, win=6.0, p_bad=0.08, bad=10.0,
                        size=(0.12, 0.24), colour=(0.30, 0.75, 0.45)),
    },
}
FOIL = np.array([0.72, 0.72, 0.76])
WIN_COLOUR = np.array([0.20, 0.80, 0.25])
BAD_COLOUR = np.array([0.55, 0.10, 0.60])
BLANK_COLOUR = np.array([0.95, 0.95, 0.95])


def load_backdrop():
    """The empty table from a real screenshot, averaged down to GW x GH."""
    try:
        from PIL import Image
        img = Image.open(BACKDROP if BACKDROP.exists() else DATA_DIR / 'look.png').convert('RGB')
        return np.asarray(img.resize((GW, GH), Image.BOX), dtype=np.float32) / 255.0
    except (OSError, ImportError):
        bg = np.full((GH, GW, 3), (0.16, 0.17, 0.22), dtype=np.float32)
        x0, y0, x1, y1 = (np.array(TABLE) * [GW, GH, GW, GH]).astype(int)
        bg[y0:y1, x0:x1] = (0.45, 0.30, 0.20)
        return bg


class Ticket:
    def __init__(self, kind, spec, x, y, rng):
        self.kind, self.spec = kind, spec
        w, h = spec['size']
        self.rect = (x, y, x + w, y + h)
        cols, rows = spec['grid']
        self.cells = []                 # (rect, symbol, value)
        fx0, fy0 = x + 0.1 * w, y + 0.35 * h
        cw, ch = 0.8 * w / cols, 0.55 * h / rows
        for r in range(rows):
            for c in range(cols):
                u = rng.random()
                if u < spec['p_bad']:
                    sym, val = 'bad', -spec['bad']
                elif u < spec['p_bad'] + spec['p_win']:
                    sym, val = 'win', spec['win']
                else:
                    sym, val = 'blank', 0.0
                self.cells.append(((fx0 + c * cw, fy0 + r * ch, fx0 + (c + 1) * cw, fy0 + (r + 1) * ch), sym, val))
        self.revealed = [False] * len(self.cells)
        self.foil_total = 0
        self.done_at = None             # sim ms when fully revealed
        # foil mask at grid resolution, True = still covered
        x0, y0, x1, y1 = self.px(self.rect)
        self.mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for (cr, _, _) in self.cells:
            a0, b0, a1, b1 = self.px(cr)
            self.mask[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = True
        self.foil_total = max(int(self.mask.sum()), 1)

    def cleared(self):
        """Fraction of this ticket's foil scratched off."""
        return 1.0 - self.mask.sum() / self.foil_total

    @staticmethod
    def px(rect):
        x0, y0, x1, y1 = rect
        return int(x0 * GW), int(y0 * GH), max(int(x1 * GW), int(x0 * GW) + 1), max(int(y1 * GH), int(y0 * GH) + 1)


class ScratchSim:
    def __init__(self, rules=RULES, randomize=True):
        self.rules = rules
        self.randomize = randomize
        self.backdrop = load_backdrop()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t_ms = 0.0
        self._money = self.rules['start_money']
        self.tickets = []
        self.cursor = None
        self.down = False
        self.stats = dict(bought=0, paid=0.0, won=0.0, penalties=0.0, forbidden_clicks=0, clicks=0,
                          claimed=0)
        # domain randomization: lighting and colour jitter
        self.gain = self.rng.uniform(0.9, 1.1, 3) if self.randomize else np.ones(3)
        for _ in range(self.rules['start_tickets']):
            self._spawn('Day Job')
        return self.frame()

    def start_position(self):
        """Where the fly starts: on the free ticket half the time (so scratching
        pays off at once), otherwise anywhere on the table (so it must find one)."""
        x0, y0, x1, y1 = self.tickets[0].rect if self.tickets and self.rng.random() < 0.5 else TABLE
        return self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)

    # ------------------------------------------------------------------
    def money(self):
        return self._money

    def foil_cleared(self):
        """Tickets' worth of foil scratched off so far (a training signal, not seen by the fly)."""
        return self.stats['claimed'] + sum(t.cleared() for t in self.tickets)

    def frame(self):
        img = self.backdrop.copy()
        for t in self.tickets:
            x0, y0, x1, y1 = t.px(t.rect)
            img[y0:y1, x0:x1] = t.spec['colour']
            for (cr, sym, _) in t.cells:
                a0, b0, a1, b1 = t.px(cr)
                img[b0:b1, a0:a1] = {'win': WIN_COLOUR, 'bad': BAD_COLOUR, 'blank': BLANK_COLOUR}[sym]
            covered = t.mask
            region = img[y0:y1, x0:x1]
            region[covered] = FOIL
        img = np.clip(img * self.gain, 0, 1)
        return (img * 255).astype(np.uint8)

    def act(self, x, y, button='up'):
        if button == 'click':
            self.stats['clicks'] += 1
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in FORBIDDEN):
                self.stats['forbidden_clicks'] += 1
                return False
            self._click(x, y)
            self.down = False
        else:
            if button == 'down' and self.cursor is not None:
                self._scratch(self.cursor, (x, y))
            self.down = button == 'down'
        self.cursor = (x, y)
        return True

    def advance(self, ms):
        self.t_ms += ms
        keep = []
        for t in self.tickets:
            if t.done_at is not None and self.t_ms - t.done_at >= self.rules['claim_ms']:
                won = sum(v for (_, s, v), r in zip(t.cells, t.revealed) if r and s == 'win')
                self._money += won
                self.stats['won'] += won
                self.stats['claimed'] += 1
            else:
                keep.append(t)
        self.tickets = keep

    # ------------------------------------------------------------------
    def _click(self, x, y):
        for name, (x0, y0, x1, y1) in SHOP:
            if x0 <= x <= x1 and y0 <= y <= y1:
                spec = self.rules['tickets'][name]
                if self._money >= spec['price'] and len(self.tickets) < self.rules['max_tickets']:
                    self._money -= spec['price']
                    self.stats['bought'] += 1
                    self.stats['paid'] += spec['price']
                    self._spawn(name)
                return

    def _spawn(self, name):
        spec = self.rules['tickets'][name]
        w, h = spec['size']
        tx0, ty0, tx1, ty1 = TABLE
        self.tickets.append(Ticket(name, spec, self.rng.uniform(tx0, tx1 - w),
                                   self.rng.uniform(ty0, ty1 - h), self.rng))

    def _scratch(self, a, b):
        """Remove foil along the segment a -> b with the coin."""
        r = self.rules['coin_radius'] * GW
        ax, ay, bx, by = a[0] * GW, a[1] * GH, b[0] * GW, b[1] * GH
        n = max(1, int(math.hypot(bx - ax, by - ay) / max(r / 2, 0.5)))
        pts = [(ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(n + 1)]
        for t in self.tickets:
            x0, y0, x1, y1 = t.px(t.rect)
            hit = False
            yy, xx = np.ogrid[y0:y1, x0:x1]
            for (px, py) in pts:
                if x0 - r <= px <= x1 + r and y0 - r <= py <= y1 + r:
                    t.mask &= (xx - px) ** 2 + (yy - py) ** 2 > r * r
                    hit = True
            if hit:
                self._update_reveals(t)

    def _update_reveals(self, t):
        x0, y0, _, _ = t.px(t.rect)
        for i, (cr, sym, val) in enumerate(t.cells):
            if t.revealed[i]:
                continue
            a0, b0, a1, b1 = t.px(cr)
            cell = t.mask[b0 - y0:b1 - y0, a0 - x0:a1 - x0]
            if cell.size and 1 - cell.mean() >= self.rules['reveal_frac']:
                t.revealed[i] = True
                if sym == 'bad':
                    self._money += val          # val is negative
                    self.stats['penalties'] -= val
        if all(t.revealed) and t.done_at is None:
            t.done_at = self.t_ms


def zigzag(env, seconds=30.0, look_ms=25.0, seed=0):
    """Scripted bot: buy a Day Job whenever possible, then zigzag over tickets."""
    env.reset(seed)
    rng = np.random.default_rng(seed)
    frames = []
    k = 0
    while env.t_ms < seconds * 1000:
        if not env.tickets:
            x0, y0, x1, y1 = SHOP[0][1]
            env.act((x0 + x1) / 2, (y0 + y1) / 2, 'click')
        else:
            t = env.tickets[0]
            x0, y0, x1, y1 = t.rect
            ph = (k % 40) / 40
            x = x0 + (x1 - x0) * (0.5 + 0.45 * math.sin(2 * math.pi * 3 * ph))
            y = y0 + (y1 - y0) * (0.3 + 0.7 * ph)
            env.act(x, y, 'down')
        env.advance(look_ms)
        k += 1
        if k % 200 == 1:
            frames.append(env.frame())
    return frames, rng


def main():
    parser = argparse.ArgumentParser(description='ScratchSim, the fast stand-in for Scritchy Scratchy')
    parser.add_argument('--render', action='store_true', help='run the zigzag bot and save frames')
    parser.add_argument('--seconds', type=float, default=30.0)
    args = parser.parse_args()
    env = ScratchSim()
    frames, _ = zigzag(env, args.seconds)
    print(f'zigzag bot, {args.seconds:.0f} s: money {RULES["start_money"]:.0f} -> {env.money():.1f}  {env.stats}')
    if args.render:
        from PIL import Image
        for i, f in enumerate(frames[:4]):
            path = DATA_DIR / f'sim_{i}.png'
            Image.fromarray(f).resize((GW * 4, GH * 4), Image.NEAREST).save(path)
            print(f'saved {path}')


if __name__ == '__main__':
    main()
