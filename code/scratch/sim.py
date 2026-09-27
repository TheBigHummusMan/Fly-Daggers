"""
ScratchSim: a small, fast stand-in for Scritchy Scratchy, so the fly's
interface can be evolved over thousands of episodes (the real game runs in
real time, one copy at a time).

Same interface as game_io.RealGame:
    env.reset(seed) -> frame     env.frame()     env.act(x, y, button)     env.money()
plus env.advance(ms) to move sim time on (the real game moves on by itself).

Frames are drawn at the fly's grid resolution (256 px wide, one pixel per
CursorFly cell) from sprites cut out of recordings (sprites.py), at the
places the real game draws them. The game loop, as recorded:

    click a shop entry (money >= price) -> a small item appears on the table
    click it                            -> it zooms to the centre, with its info panel
    hold the button and rub it          -> dirt / foil comes off (sponge or coin cursor)
    all revealed                        -> a claim button shows the payout; click it
    earnings fill the goal bar          -> unlocks: 3 Trash Can, 10 Two Win,
                                           50 Upgrades, 100 Mini Scratch (phone call),
                                           1000 Scratch Bot (call), 2000 Apple Tree (call)
    a call: click the ringing phone, then click through the dialogue and the popup

Prices, odds, payouts, cell layouts, and the unlock order come from the
recordings. Anything marked ASSUMED in RULES was not observed and is a guess.

Usage:
    python code/scratch/sim.py --render            # scripted player, saves frames
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'scratch'
SPRITE_DIR = DATA_DIR / 'sprites'
GW, GH = 256, 167                      # frame size = CursorFly grid for a 1024 x 668 window

# 0-1 window coordinates, measured on the 1024 x 668 window
TABLE = (0.17, 0.15, 0.84, 0.85)
TABS = {'tickets': (0.01, 0.13, 0.085, 0.165), 'gadgets': (0.09, 0.13, 0.165, 0.165)}
ENTRY_X = (0.01, 0.165)
TICKET_ENTRIES = {'dayjob': 0.175, 'twowin': 0.335, 'mini': 0.415, 'apple': 0.495}   # top y
GADGET_ENTRIES = {'trash': 0.175, 'bot': 0.255, 'bot_speed': 0.335, 'bot_capacity': 0.415,
                  'bot_strength': 0.495}
ENTRY_H = 0.08
UPGRADE_ENTRIES = {'luck': (0.845, 0.195, 0.99, 0.27), 'size': (0.845, 0.275, 0.99, 0.35),
                   'coin': (0.845, 0.355, 0.99, 0.43)}
PHONE = (0.69, 0.08, 0.87, 0.23)
TRASH = (0.90, 0.70, 1.0, 1.0)
BOT = (0.745, 0.655, 0.855, 0.905)
FORBIDDEN = [(0.0, 0.0, 1.0, 0.045), (0.945, 0.04, 1.0, 0.1)]   # title bar (close!), settings

ITEM_SPRITES = {                       # zoomed sprite names, info panel, claim button
    'dayjob': ('plate_dirty', 'plate_clean', 'info_dayjob0', 'claim_plate'),
    'twowin': ('twowin_covered', 'twowin_revealed', 'info_twowin', 'claim_ticket'),
    'mini': ('mini_covered', 'mini_revealed', 'info_mini', 'claim_ticket'),
    'apple': ('mini_covered', 'mini_revealed', 'info_mini', 'claim_ticket'),    # ASSUMED look
}

RULES = {
    'start_money': 1.0,
    'start_zoomed': 0.5,               # training aid: P(episode starts with a Day Job already zoomed)
    'max_on_table': 10,
    'goals': [                         # (earnings needed, unlock, comes with a phone call)
        (3, 'trash', False), (10, 'twowin', False), (50, 'upgrades', False),
        (100, 'mini', True), (1000, 'gadgets', True), (2000, 'apple', True),
        (10000, None, False),          # later goals: ASSUMED x5 each, nothing unlocked
    ],
    'call_lines': 3,                   # dialogue lines to click through on a call
    'tickets': {
        # price, cover hardness, cells, matches needed, [(symbol, chance, payout)]
        'dayjob': dict(price=1, hardness=0, cells=1, match=1, reveal=0.90,
                       levels={0: [('plate', 1.0, 2)], 1: [('plate', 0.9, 5), ('broken', 0.1, -5)]}),
        'twowin': dict(price=10, hardness=0, cells=3, match=2, reveal=0.95,
                       symbols=[('goldcoin', 0.5, 10), ('cash', 0.4, 25), ('pile', 0.1, 50)]),
        'mini': dict(price=100, hardness=1, cells=5, match=3, reveal=0.9,     # reveal fitted by validate.py
                     symbols=[('redcoin', 0.3, 100), ('pile', 0.3, 200), ('cash', 0.3, 500),
                              ('goldstack', 0.1, 1000)]),
        'apple': dict(price=2000, hardness=2, cells=5, match=3, reveal=0.95,          # ASSUMED rules
                      symbols=[('redcoin', 0.3, 2000), ('pile', 0.3, 4000), ('cash', 0.3, 10000),
                               ('goldstack', 0.1, 20000)]),
    },
    'level_mult': 1.3,                 # payouts per ticket level (Two Win 10/25/50 -> 13/32/65)
    'xp_per_level': 5,                 # ASSUMED; a full bar makes the shop entry a Level Up button
    'max_level': 3,                    # ASSUMED (the star after a ticket's name)
    'sponge_radius': 0.009,            # Day Job sponge, fraction of window width (fitted by validate.py)
    'coin_radius': 0.010,              # ticket coin, fraction of window width (fitted by validate.py)
    'coin_scale': 1.7,                 # radius x this per coin tier (Tin, Aluminum), fitted by validate.py
    'upgrades': {
        'luck': [200, 1000, 5000],     # each level: better symbols 1.3x likelier per rank (ASSUMED)
        'size': {'base': [20, 200], 'tin': [2000, 20000], 'aluminum': [50000, 500000]},  # tin+ ASSUMED
        'coins': [('base', 1, 0), ('tin', 2, 1000), ('aluminum', 4, 200000)],  # name, strength, price
        'size_mult': 1.3,              # ASSUMED
    },
    'gadgets': {'trash': 2, 'bot': 1000, 'bot_speed': 5000, 'bot_capacity': 2500, 'bot_strength': 2000},
    'bot_seconds': 8.0,                # per ticket, x0.7 per speed level (ASSUMED)
    'bot_capacity': 3,
}
LIGHT_CELL = {'twowin': (0.78, 0.84, 0.86), 'mini': (0.62, 0.78, 0.93), 'apple': (0.62, 0.78, 0.93)}


def inside(x, y, rect):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def gx(x):
    return int(round(x * GW))


def gy(y):
    return int(round(y * GH))


class Sprites:
    """Sprites from sprites.py, resized to the grid at their window places."""

    def __init__(self):
        from PIL import Image
        meta = json.loads((SPRITE_DIR / 'sprites.json').read_text())
        self.rect = {n: tuple(s['rect']) for n, s in meta['sprites'].items()}
        self.cells = meta['cells']
        self.img = {}
        for name, rect in self.rect.items():
            im = Image.open(SPRITE_DIR / f'{name}.png').convert('RGB')
            w, h = max(1, gx(rect[2]) - gx(rect[0])), max(1, gy(rect[3]) - gy(rect[1]))
            self.img[name] = np.asarray(im.resize((w, h), Image.BOX), dtype=np.float32) / 255
        self.icons = {k: Image.open(SPRITE_DIR / v).convert('RGB') for k, v in meta['icons'].items()}
        self._icon_cache = {}

    def icon(self, name, size):
        key = (name, size)
        if key not in self._icon_cache:
            from PIL import Image
            self._icon_cache[key] = np.asarray(self.icons[name].resize((size, size), Image.BOX),
                                               dtype=np.float32) / 255
        return self._icon_cache[key]

    def small(self, name, width):
        """A table-sized version of a zoomed sprite."""
        key = (name, 'small', width)
        if key not in self._icon_cache:
            from PIL import Image
            src = Image.fromarray((self.img[name] * 255).astype(np.uint8))
            h = max(1, round(width * src.height / src.width))
            self._icon_cache[key] = np.asarray(src.resize((width, h), Image.BOX), dtype=np.float32) / 255
        return self._icon_cache[key]


def paste(img, sprite, x0, y0):
    """Paste a grid-resolution sprite with its top-left at grid (x0, y0), clipped."""
    h, w = sprite.shape[:2]
    ax, ay = max(x0, 0), max(y0, 0)
    bx, by = min(x0 + w, GW), min(y0 + h, GH)
    if ax < bx and ay < by:
        img[ay:by, ax:bx] = sprite[ay - y0:by - y0, ax - x0:bx - x0]


class Zoomed:
    """The item being scratched, in the middle of the table."""

    def __init__(self, kind, sim):
        self.kind = kind
        spec = sim.rules['tickets'][kind]
        cov_name, rev_name, self.info, self.claim_sprite = ITEM_SPRITES[kind]
        sp = sim.sprites
        self.rect = sp.rect[cov_name]
        self.x0, self.y0 = gx(self.rect[0]), gy(self.rect[1])
        self.cov = sp.img[cov_name].copy()
        h, w = self.cov.shape[:2]
        self.hardness = spec['hardness']
        self.reveal_frac = spec['reveal']
        level = sim.levels[kind]
        mult = sim.rules['level_mult'] ** max(level - (1 if kind == 'dayjob' else 0), 0)
        if kind == 'dayjob':
            if level >= 1:
                self.info = 'info_dayjob1'
            table = spec['levels'][min(level, 1)]
            sym, _, value = sim.draw(table)
            self.symbols = [sym]
            self.payout = round(value * mult)
            self.rev = sp.img[rev_name].copy()
            dirt = np.abs(self.cov - self.rev).sum(axis=-1) > 0.12
            self.cells = [dirt]
        else:
            self.symbols, values = [], {}
            for _ in range(spec['cells']):
                sym, _, value = sim.draw(spec['symbols'], luck=sim.upgrades['luck'])
                self.symbols.append(sym)
                values[sym] = round(value * mult)
            counts = {s: self.symbols.count(s) for s in set(self.symbols)}
            won = [values[s] for s, c in counts.items() if c >= spec['match']]
            self.payout = max(won) if won else 0         # a loser has no claim button: trash it
            self.rev = self.cov.copy()
            yy, xx = np.mgrid[0:h, 0:w]
            self.cells = []
            for (cx, cy, r), sym in zip(sp.cells[kind if kind != 'apple' else 'mini'], self.symbols):
                px, py, pr = cx * w, cy * h, r * w
                disc = (xx - px) ** 2 + (yy - py) ** 2 <= pr ** 2
                self.rev[disc] = LIGHT_CELL[kind]
                size = max(2, int(pr * 1.6))
                paste_local(self.rev, sp.icon(sym, size), int(px - size / 2), int(py - size / 2))
                self.cells.append(disc)
        self.mask = np.zeros((h, w), dtype=bool)          # True = still covered
        for c in self.cells:
            self.mask |= c
        self.cell_px = [max(int(c.sum()), 1) for c in self.cells]
        self.revealed = [False] * len(self.cells)
        self.done = False

    def cleared(self):
        return sum(1 - (self.mask & c).sum() / n for c, n in zip(self.cells, self.cell_px)) / len(self.cells)

    def image(self):
        out = self.rev.copy()
        out[self.mask] = self.cov[self.mask]
        return out


def paste_local(img, sprite, x0, y0):
    h, w = sprite.shape[:2]
    H, W = img.shape[:2]
    ax, ay, bx, by = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if ax < bx and ay < by:
        img[ay:by, ax:bx] = sprite[ay - y0:by - y0, ax - x0:bx - x0]


class ScratchSim:
    def __init__(self, rules=RULES, randomize=True):
        self.rules = rules
        self.randomize = randomize
        self.sprites = Sprites()

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        r = self.rules
        self.rng = np.random.default_rng(seed)
        self.t_ms = 0.0
        self._money = float(r['start_money'])
        self.goal_i = 0
        self.progress = 0.0
        self.unlocked = {'dayjob'}
        self.levels = {k: 0 for k in r['tickets']}
        self.xp = {k: 0 for k in r['tickets']}
        self.upgrades = {'luck': 0, 'size': 0, 'coin': 0}
        self.gadgets = set()
        self.bot_level = {'bot_speed': 0, 'bot_capacity': 0, 'bot_strength': 0}
        self.bot_queue, self.bot_t = [], 0.0
        self.table = []                  # small items: [kind, x, y]
        self.zoomed = None
        self.tab = 'tickets'
        self.ringing = []                # unlocks waiting for the phone to be answered
        self.modal = []                  # steps to click through: 'line' or ('popup', unlock)
        self.down = False
        self.cursor = (0.5, 0.5)
        self.drag = None                 # [item, press x, press y]
        self.stats = dict(bought=0, paid=0.0, won=0.0, finished=0, claimed=0, forbidden_clicks=0, clicks=0,
                          upgrades=0, level_ups=0, unlocks=0, bot_done=0)
        self.gain = self.rng.uniform(0.9, 1.1, 3) if self.randomize else np.ones(3)
        if self.rng.random() < r['start_zoomed']:
            self._money -= r['tickets']['dayjob']['price']
            self.zoomed = Zoomed('dayjob', self)
        return self.frame()

    def start_position(self):
        """Where the fly starts: on the zoomed item if there is one, else anywhere on the table."""
        x0, y0, x1, y1 = self.zoomed.rect if self.zoomed else TABLE
        return self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)

    def money(self):
        return self._money

    def foil_cleared(self):
        """Items' worth of dirt/foil removed (a training signal, not seen by the fly)."""
        return self.stats['claimed'] + (self.zoomed.cleared() if self.zoomed else 0.0)

    # ------------------------------------------------------------------
    def draw(self, table, luck=0):
        """(symbol, chance, payout) drawn from table; luck favours the better ranks."""
        p = np.array([c for _, c, _ in table], dtype=float)
        p *= (1.3 ** luck) ** np.arange(len(p))
        p /= p.sum()
        return table[int(self.rng.choice(len(table), p=p))]

    def stage(self):
        n = self.goal_i
        return 'stage' + {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '4'}.get(n, '6')

    # ------------------------------------------------------------------
    def frame(self):
        sp = self.sprites
        img = sp.img[self.stage()].copy()
        if self.tab == 'gadgets' and 'trash' in self.unlocked:
            name = 'gadgets_stage5' if 'gadgets' in self.unlocked else 'gadgets_stage1'
            paste(img, sp.img[name], gx(sp.rect[name][0]), gy(sp.rect[name][1]))
        if 'bot' in self.gadgets:
            paste(img, sp.img['bot'], gx(BOT[0]), gy(BOT[1]))
        if self.ringing and int(self.t_ms / 250) % 2:          # ASSUMED look of a ringing phone
            x0, y0, x1, y1 = gx(PHONE[0]), gy(PHONE[1]), gx(PHONE[2]), gy(PHONE[3])
            img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1] * 1.35, 0, 1)
        for kind, x, y in self.table:
            name = ITEM_SPRITES[kind][0]
            s = sp.small(name, 12 if kind == 'dayjob' else 11)
            paste(img, s, gx(x) - s.shape[1] // 2, gy(y) - s.shape[0] // 2)
        z = self.zoomed
        if z is not None:
            paste(img, z.image(), z.x0, z.y0)
            paste(img, sp.img[z.info], gx(sp.rect[z.info][0]), gy(sp.rect[z.info][1]))
            if z.done and (z.payout != 0 or z.kind == 'dayjob'):
                paste(img, sp.img[z.claim_sprite], gx(sp.rect[z.claim_sprite][0]), gy(sp.rect[z.claim_sprite][1]))
        if self.drag:
            kind = self.drag[0][0]
            s = sp.small(ITEM_SPRITES[kind][0], 12)
            paste(img, s, gx(self.cursor[0]) - s.shape[1] // 2, gy(self.cursor[1]) - s.shape[0] // 2)
        if self.modal:
            img *= 0.45
            if self.modal[0] == 'line':
                paste(img, sp.img['dialogue'], gx(sp.rect['dialogue'][0]), gy(sp.rect['dialogue'][1]))
            else:
                paste(img, sp.img['unlock_popup'], gx(sp.rect['unlock_popup'][0]), gy(sp.rect['unlock_popup'][1]))
        self._cursor(img)
        return (np.clip(img * self.gain, 0, 1) * 255).astype(np.uint8)

    def _cursor(self, img):
        """The game draws its own cursor: a sponge on the plate, a coin on a ticket, else an arrow."""
        x, y = self.cursor
        z = self.zoomed
        if z is not None and inside(x, y, z.rect) and not self.modal:
            s = self.sprites.img['sponge' if z.kind == 'dayjob' else 'coin']
            paste(img, s, gx(x) - s.shape[1] // 2, gy(y) - s.shape[0] // 2)
        else:
            px, py = gx(x), gy(y)
            for k in range(4):                                   # small white arrow, dark edge
                if 0 <= py + k < GH:
                    img[py + k, max(px - 1, 0):min(px + k + 2, GW)] = 0.1
                    img[py + k, max(px, 0):min(px + k + 1, GW)] = 1.0

    # ------------------------------------------------------------------
    def act(self, x, y, button='up'):
        if button == 'click':
            self.stats['clicks'] += 1
            if any(inside(x, y, r) for r in FORBIDDEN):
                self.stats['forbidden_clicks'] += 1
                return False
            self.cursor = (x, y)
            self._press(x, y)
            self._release(x, y)
            self.down = False
            return True
        if button == 'down':
            if not self.down:
                self.down = True
                if any(inside(x, y, r) for r in FORBIDDEN):
                    self.stats['forbidden_clicks'] += 1
                else:
                    self._press(x, y)
            else:
                self._move(self.cursor, (x, y))
        elif self.down:
            self._release(x, y)
            self.down = False
        self.cursor = (x, y)
        return True

    def advance(self, ms):
        self.t_ms += ms
        if self.bot_queue:
            self.bot_t += ms
            need = self.rules['bot_seconds'] * 1000 * 0.7 ** self.bot_level['bot_speed']
            if self.bot_t >= need:
                self.bot_t = 0.0
                kind = self.bot_queue.pop(0)
                item = Zoomed(kind, self)                     # the bot reveals everything
                self._pay(kind, item.payout)
                self.stats['bot_done'] += 1

    # ------------------------------------------------------------------
    def _press(self, x, y):
        if self.modal:
            step = self.modal.pop(0)
            if step != 'line':
                self._unlock(step[1])
            return
        z = self.zoomed
        if z is not None and z.done and (z.payout != 0 or z.kind == 'dayjob') \
                and inside(x, y, self.sprites.rect[z.claim_sprite]):
            self._pay(z.kind, z.payout)
            self.stats['claimed'] += 1
            self.zoomed = None
            return
        if self.ringing and inside(x, y, PHONE):
            unlock = self.ringing.pop(0)
            self.modal = ['line'] * self.rules['call_lines'] + [('popup', unlock)]
            return
        for tab, rect in TABS.items():
            if inside(x, y, rect) and (tab == 'tickets' or 'trash' in self.unlocked):
                self.tab = tab
                return
        if ENTRY_X[0] <= x <= ENTRY_X[1]:
            entries = TICKET_ENTRIES if self.tab == 'tickets' else GADGET_ENTRIES
            for name, top in entries.items():
                if top <= y <= top + ENTRY_H:
                    self._shop(name)
                    return
        if 'upgrades' in self.unlocked:
            for name, rect in UPGRADE_ENTRIES.items():
                if inside(x, y, rect):
                    self._upgrade(name)
                    return
        if z is not None and z.done and z.payout == 0 and z.kind != 'dayjob' and inside(x, y, z.rect):
            self._loser = dict(z.__dict__)
            self.drag = [[z.kind, x, y], x, y, 'loser']     # pick up the losing ticket
            self.zoomed = None
            return
        for item in reversed(self.table):
            if abs(x - item[1]) < 0.03 and abs(y - item[2]) < 0.04:
                self.drag = [item, x, y]
                self.table.remove(item)
                return
        if z is not None and inside(x, y, z.rect):
            self._scratch((x, y), (x, y))

    def _move(self, a, b):
        if self.drag is None and self.zoomed is not None and not self.modal:
            self._scratch(a, b)

    def _release(self, x, y):
        if self.drag is None:
            return
        item, loser = self.drag[0], len(self.drag) > 3
        self.drag = None
        kind = item[0]
        if 'trash' in self.gadgets and inside(x, y, TRASH):
            if loser:
                self.xp[kind] += 1
                self.stats['claimed'] += 1
            return                                           # thrown away
        if loser:                                            # dropped elsewhere: it comes back
            self.zoomed = Zoomed.__new__(Zoomed)
            self.zoomed.__dict__.update(self._loser)
            return
        if 'bot' in self.gadgets and inside(x, y, BOT):
            cap = self.rules['bot_capacity'] + self.bot_level['bot_capacity']
            if len(self.bot_queue) < cap:
                self.bot_queue.append(kind)
                return
        if self.zoomed is None:
            self.zoomed = Zoomed(kind, self)                 # clicking (or dropping) it zooms in
        else:
            self.table.append([kind, min(max(x, TABLE[0]), TABLE[2]), min(max(y, TABLE[1]), TABLE[3])])

    def _shop(self, name):
        r = self.rules
        if self.tab == 'tickets':
            if name not in self.unlocked:
                return
            if self.xp[name] >= r['xp_per_level'] and self.levels[name] < r['max_level']:
                self.xp[name] = 0
                self.levels[name] += 1
                self.stats['level_ups'] += 1
                return
            price = r['tickets'][name]['price']
            if self._money >= price and len(self.table) < r['max_on_table']:
                self._money -= price
                self.stats['bought'] += 1
                self.stats['paid'] += price
                self.table.append([name, self.rng.uniform(0.45, 0.6), self.rng.uniform(0.25, 0.35)])
        else:
            if name == 'trash' and 'trash' in self.unlocked:
                self._buy_gadget('trash', r['gadgets']['trash'])
            elif name == 'bot' and 'gadgets' in self.unlocked:
                self._buy_gadget('bot', r['gadgets']['bot'])
            elif name in self.bot_level and 'bot' in self.gadgets:
                price = r['gadgets'][name] * (2 ** self.bot_level[name])     # ASSUMED doubling
                if self._money >= price:
                    self._money -= price
                    self.bot_level[name] += 1
                    self.stats['upgrades'] += 1

    def _buy_gadget(self, name, price):
        if name not in self.gadgets and self._money >= price:
            self._money -= price
            self.gadgets.add(name)
            self.stats['upgrades'] += 1

    def _upgrade(self, name):
        u = self.rules['upgrades']
        if name == 'luck':
            prices = u['luck']
            level = self.upgrades['luck']
        elif name == 'size':
            prices = u['size'][u['coins'][self.upgrades['coin']][0]]
            level = self.upgrades['size']
        else:
            nxt = self.upgrades['coin'] + 1
            prices = [c[2] for c in u['coins'][1:]]
            level = nxt - 1
        if level >= len(prices) or self._money < prices[level]:
            return
        self._money -= prices[level]
        self.stats['upgrades'] += 1
        if name == 'coin':
            self.upgrades['coin'] += 1
            self.upgrades['size'] = 0                        # size upgrades are per coin
        else:
            self.upgrades[name] += 1

    def _pay(self, kind, payout):
        self._money += payout
        self.stats['won'] += payout
        self.xp[kind] += 1
        if payout > 0:
            self.progress += payout
            goals = self.rules['goals']
            while True:
                need = goals[self.goal_i][0] if self.goal_i < len(goals) else goals[-1][0] * 5 ** (self.goal_i - len(goals) + 1)
                if self.progress < need:
                    break
                self.progress = 0.0                          # the bar starts over at each goal
                unlock, call = (goals[self.goal_i][1], goals[self.goal_i][2]) if self.goal_i < len(goals) else (None, False)
                self.goal_i += 1
                self.stats['unlocks'] += 1
                if unlock is None:
                    continue
                if call:
                    self.ringing.append(unlock)
                else:
                    self.modal.append(('popup', unlock))

    def _unlock(self, unlock):
        self.unlocked.add(unlock)

    def _scratch(self, a, b):
        """Rub the zoomed item along a -> b with the sponge or coin."""
        z = self.zoomed
        if z is None or z.done:
            return
        u = self.rules['upgrades']
        coin, strength, _ = u['coins'][self.upgrades['coin']]
        base = self.rules['sponge_radius'] if z.kind == 'dayjob' else self.rules['coin_radius']
        r = base * GW * self.rules['coin_scale'] ** self.upgrades['coin'] * u['size_mult'] ** self.upgrades['size']
        ax, ay, bx, by = a[0] * GW - z.x0, a[1] * GH - z.y0, b[0] * GW - z.x0, b[1] * GH - z.y0
        n = max(1, int(math.hypot(bx - ax, by - ay) / max(r / 2, 0.5)))
        h, w = z.mask.shape
        yy, xx = np.ogrid[0:h, 0:w]
        swept = np.zeros((h, w), dtype=bool)
        for k in range(n + 1):
            px, py = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            if -r <= px <= w + r and -r <= py <= h + r:
                swept |= (xx - px) ** 2 + (yy - py) ** 2 <= r * r
        swept &= z.mask
        if not swept.any():
            return
        if strength <= z.hardness:                           # too soft a coin: partial wear (ASSUMED)
            swept &= self.rng.random((h, w)) < strength / (z.hardness + 1)
        z.mask &= ~swept
        for i, (c, npx) in enumerate(zip(z.cells, z.cell_px)):
            if not z.revealed[i] and 1 - (z.mask & c).sum() / npx >= z.reveal_frac:
                z.revealed[i] = True
                z.mask &= ~c                                 # the rest of a revealed cell falls away
        if all(z.revealed):
            z.done = True
            self.stats['finished'] += 1


# ============================================================================
# A scripted player, to check the sim plays like the real game
# ============================================================================

def scripted(env, seconds=120.0, look_ms=25.0, seed=0, every=None):
    """Plays like a simple human: answer calls, claim, buy the best ticket it
    can afford, zoom it, scrub it in a zigzag, buy upgrades when rich."""
    env.reset(seed)
    frames, k, target = [], 0, None
    rng = np.random.default_rng(seed)

    def click(x, y):
        env.act(x, y, 'click')

    while env.t_ms < seconds * 1000:
        z = env.zoomed
        if env.modal:
            click(0.5, 0.5)
        elif z is not None and z.done and z.payout == 0 and z.kind != 'dayjob':
            x0, y0, x1, y1 = z.rect                          # a loser: drag it into the trash
            env.act((x0 + x1) / 2, (y0 + y1) / 2, 'up')
            env.act((x0 + x1) / 2, (y0 + y1) / 2, 'down')
            env.act(0.95, 0.85, 'down')
            env.act(0.95, 0.85, 'up')
        elif z is not None and z.done:
            r = env.sprites.rect[z.claim_sprite]
            click((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        elif env.ringing:
            click((PHONE[0] + PHONE[2]) / 2, (PHONE[1] + PHONE[3]) / 2)
        elif z is not None:
            # rub towards a random still-covered spot, like a person chasing the last bits
            h, w = z.mask.shape
            if k % 8 == 0 or target is None:
                ys, xs = np.nonzero(z.mask)
                j = rng.integers(len(ys)) if len(ys) else 0
                target = ((z.x0 + (xs[j] if len(xs) else w / 2)) / GW, (z.y0 + (ys[j] if len(ys) else h / 2)) / GH)
            cx, cy = env.cursor
            env.act(cx + 0.35 * (target[0] - cx), cy + 0.35 * (target[1] - cy), 'down')
        elif env.table:
            env.act(0.5, 0.5, 'up')
            _, x, y = env.table[0]
            click(x, y)
        else:
            env.act(0.5, 0.5, 'up')
            if 'trash' in env.unlocked and 'trash' not in env.gadgets and env.money() >= 2:
                click(0.12, 0.15)                            # Gadgets tab, buy the trash can
                click(0.08, GADGET_ENTRIES['trash'] + ENTRY_H / 2)
            if env.tab != 'tickets':
                click(0.04, 0.15)
            if 'upgrades' in env.unlocked and env.money() > 300 and rng.random() < 0.3:
                click(0.9, rng.choice([0.23, 0.31, 0.39]))
            best = None
            for name in ('apple', 'mini', 'twowin', 'dayjob'):
                if name in env.unlocked and env.money() >= env.rules['tickets'][name]['price']:
                    best = name
                    break
            if best:
                click(0.08, TICKET_ENTRIES[best] + ENTRY_H / 2)
        env.advance(look_ms)
        k += 1
        if every and k % every == 1:
            frames.append(env.frame())
    return frames


def main():
    parser = argparse.ArgumentParser(description='ScratchSim, the fast stand-in for Scritchy Scratchy')
    parser.add_argument('--render', action='store_true', help='save a few frames of the scripted player')
    parser.add_argument('--seconds', type=float, default=300.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    env = ScratchSim(randomize=False)
    import time
    t0 = time.time()
    frames = scripted(env, args.seconds, seed=args.seed, every=400 if args.render else None)
    print(f'scripted player, {args.seconds:.0f} s game time in {time.time() - t0:.1f} s: '
          f'money ${env.money():,.0f}, goal {env.goal_i}, unlocked {sorted(env.unlocked)}\n  {env.stats}')
    if args.render:
        from PIL import Image
        for i, f in enumerate(frames[:8]):
            path = DATA_DIR / f'sim_{i}.png'
            Image.fromarray(f).resize((GW * 3, GH * 3), Image.NEAREST).save(path)
        print(f'saved {min(len(frames), 8)} frames to {DATA_DIR}/sim_*.png')


if __name__ == '__main__':
    main()
