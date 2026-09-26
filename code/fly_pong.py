"""
Fly Pong: the emulated Drosophila brain plays Pong.

The left paddle is controlled by the whole-brain LIF model (fast_brain.py,
same dynamics as run_pytorch.py) in a closed sensorimotor loop:

    ball bearing ──> LC10a visual projection neurons (small-object tracking)
                     left eye if the ball is to the fly's left, right if right
                 ──> 138k-neuron FlyWire connectome
                 ──> DNa01 + DNa02 steering descending neurons
    paddle speed <── left − right steering firing rate

The fly sits on its paddle facing the opponent, so its left is up on screen
and a left turn moves the paddle up. Nothing is trained: whether the fly
turns toward the ball is decided by the connectome's wiring. Neuron IDs and
sides come from the FlyWire cell-type annotations (Schlegel et al. 2024),
stored in data/fly_pong_neurons.csv.

The game runs in brain time -- one physics tick per simulated millisecond --
so it plays in slow motion at whatever speed the CPU can simulate the brain.

Usage:
    python code/fly_pong.py                     # you (arrow keys) vs the fly
    python code/fly_pong.py --autopilot         # CPU vs the fly
    python code/fly_pong.py --headless --points 20   # no window, print stats
    python code/fly_pong.py --headless --blind  # control: fly sees nothing

Keys: Up/Down or W/S move your paddle, A toggles autopilot, Space pauses,
Esc quits.
"""

import argparse
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from fast_brain import DT, FastBrain, load_connectome

NEURONS_CSV = Path(__file__).resolve().parent.parent / 'data' / 'fly_pong_neurons.csv'

# ============================================================================
# Game constants (distances in px, times in brain seconds)
# ============================================================================

FIELD_W, FIELD_H = 900, 540
PADDLE_W, PADDLE_H = 12, 90
PADDLE_MARGIN = 40
BALL_R = 8
BALL_SPEED0 = 450.0              # px/s at serve
BALL_SPEEDUP = 1.05              # per paddle hit
BALL_SPEED_MAX = 900.0
MAX_BOUNCE = math.radians(50)    # bounce angle when hitting a paddle's edge
SERVE_DELAY = 0.5                # s
FLY_SPEED = 600.0                # px/s at saturated steering
PLAYER_SPEED = 550.0
CPU_SPEED = 380.0

# ============================================================================
# Sensorimotor interface
# ============================================================================

EYE_RATE_MAX = 200.0             # Hz Poisson input to LC10a at saturation
EYE_DEAD = math.radians(1.5)     # ball this close to straight ahead: no input
EYE_SAT = math.radians(20)       # bearing at which the input saturates
STEER_TAU_MS = 25.0              # smoothing of the steering firing rate
STEER_HALF = 60.0                # L-R Hz difference giving tanh(1) of FLY_SPEED


class FlyController:
    """Closed loop between the Pong ball and the connectome."""

    def __init__(self, weights, flyid2i, seed=None, blind=False):
        df = pd.read_csv(NEURONS_CSV)

        def pick(role, side):
            ids = df.loc[(df.role == role) & (df.side == side), 'flywire_id']
            return np.array([flyid2i[f] for f in ids])

        eye_l, eye_r = pick('eye', 'left'), pick('eye', 'right')
        self.n_eye_l = len(eye_l)
        self.brain = FastBrain(weights, np.concatenate([eye_l, eye_r]), seed=seed)

        # 0 = other, 1 = left steering DN, 2 = right steering DN
        self.steer_side = np.zeros(self.brain.n, dtype=np.int8)
        self.steer_side[pick('steering', 'left')] = 1
        self.steer_side[pick('steering', 'right')] = 2

        self.steps_per_ms = int(round(1.0 / DT))
        self.decay = math.exp(-1.0 / STEER_TAU_MS)
        self.blind = blind

        self.eye_rate = [0.0, 0.0]       # Hz into left / right LC10a
        self.steer_rate = [0.0, 0.0]     # Hz, smoothed left / right DNa01+DNa02
        self.last_spike = np.full(self.brain.n, -np.inf)
        self.spikes_total = 0
        self.t_ms = 0

    def step_ms(self, bearing):
        """Show the fly the ball for 1 ms of brain time.

        bearing is the ball's angle from straight ahead in radians, positive
        to the fly's left. Returns a steering command in [-1, 1], positive for
        a left turn.
        """
        drive = (abs(bearing) - EYE_DEAD) / (EYE_SAT - EYE_DEAD)
        drive = 0.0 if self.blind else min(max(drive, 0.0), 1.0) * EYE_RATE_MAX
        self.eye_rate = [drive, 0.0] if bearing > 0 else [0.0, drive]
        self.brain.rates[:self.n_eye_l] = self.eye_rate[0]
        self.brain.rates[self.n_eye_l:] = self.eye_rate[1]

        counts = np.zeros(3, dtype=np.int64)
        for _ in range(self.steps_per_ms):
            spikes = self.brain.step()
            if len(spikes):
                counts += np.bincount(self.steer_side[spikes], minlength=3)
                self.last_spike[spikes] = self.t_ms
                self.spikes_total += len(spikes)
        self.t_ms += 1

        for side in (0, 1):
            instant = counts[side + 1] * 1000.0
            self.steer_rate[side] = instant + (self.steer_rate[side] - instant) * self.decay
        return math.tanh((self.steer_rate[0] - self.steer_rate[1]) / STEER_HALF)

    def active_neurons(self, window_ms=100):
        """Number of distinct neurons that spiked in the last window_ms."""
        return int((self.last_spike > self.t_ms - window_ms).sum())


def clamp_paddle(y):
    return min(max(y, PADDLE_H / 2), FIELD_H - PADDLE_H / 2)


class Pong:
    """Pong physics, advanced one brain millisecond per tick."""

    def __init__(self, fly, autopilot=False, seed=None):
        self.fly = fly
        self.autopilot = autopilot
        self.rng = random.Random(seed)
        self.fly_x = PADDLE_MARGIN                          # paddle left edges
        self.opp_x = FIELD_W - PADDLE_MARGIN - PADDLE_W
        self.fly_y = self.opp_y = FIELD_H / 2               # paddle centers
        self.score = [0, 0]                                 # fly, opponent
        self.fly_returns = 0
        self.player_dir = 0                                 # -1 up, +1 down
        self.cpu_aim = 0.0                                  # autopilot's hit offset
        self.steer = 0.0
        self.serve(toward_fly=True)

    def serve(self, toward_fly):
        self.ball_x, self.ball_y = FIELD_W / 2, FIELD_H / 2
        angle = self.rng.uniform(-math.radians(30), math.radians(30))
        direction = -1 if toward_fly else 1
        self.ball_vx = direction * BALL_SPEED0 * math.cos(angle)
        self.ball_vy = BALL_SPEED0 * math.sin(angle)
        self.serve_timer = SERVE_DELAY
        self._pick_cpu_aim()

    def _pick_cpu_aim(self):
        # The autopilot hits with a random part of its paddle, so its returns
        # come back at an angle instead of straight at the fly.
        self.cpu_aim = self.rng.uniform(-0.8, 0.8) * PADDLE_H / 2

    def tick(self):
        dt = 0.001

        # Fly: bearing of the ball from the front of its paddle, + = its left
        eye_x = self.fly_x + PADDLE_W
        bearing = math.atan2(self.fly_y - self.ball_y, self.ball_x - eye_x)
        self.steer = self.fly.step_ms(bearing)
        self.fly_y = clamp_paddle(self.fly_y - FLY_SPEED * self.steer * dt)

        # Opponent
        if self.autopilot:
            # Like classic Pong AI, it only reacts once the ball is in its half
            incoming = self.ball_vx > 0 and self.ball_x > FIELD_W / 2
            target = self.ball_y - self.cpu_aim if incoming else FIELD_H / 2
            v = max(-CPU_SPEED, min(CPU_SPEED, (target - self.opp_y) * 10))
        else:
            v = PLAYER_SPEED * self.player_dir
        self.opp_y = clamp_paddle(self.opp_y + v * dt)

        if self.serve_timer > 0:
            self.serve_timer -= dt
            return

        # Ball
        prev_x = self.ball_x
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt
        if self.ball_y < BALL_R:
            self.ball_y, self.ball_vy = BALL_R, abs(self.ball_vy)
        elif self.ball_y > FIELD_H - BALL_R:
            self.ball_y, self.ball_vy = FIELD_H - BALL_R, -abs(self.ball_vy)

        fly_face = self.fly_x + PADDLE_W
        if (self.ball_vx < 0 and prev_x - BALL_R >= fly_face > self.ball_x - BALL_R
                and abs(self.ball_y - self.fly_y) <= PADDLE_H / 2 + BALL_R):
            self._bounce(self.fly_y, direction=1)
            self.fly_returns += 1
            self._pick_cpu_aim()
        elif (self.ball_vx > 0 and prev_x + BALL_R <= self.opp_x < self.ball_x + BALL_R
                and abs(self.ball_y - self.opp_y) <= PADDLE_H / 2 + BALL_R):
            self._bounce(self.opp_y, direction=-1)

        if self.ball_x < -BALL_R:
            self.score[1] += 1
            self.serve(toward_fly=True)
        elif self.ball_x > FIELD_W + BALL_R:
            self.score[0] += 1
            self.serve(toward_fly=False)

    def _bounce(self, paddle_y, direction):
        offset = (self.ball_y - paddle_y) / (PADDLE_H / 2 + BALL_R)
        speed = min(math.hypot(self.ball_vx, self.ball_vy) * BALL_SPEEDUP, BALL_SPEED_MAX)
        angle = offset * MAX_BOUNCE
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)


# ============================================================================
# Headless mode
# ============================================================================

def run_headless(args):
    print('Loading connectome...')
    weights, flyid2i = load_connectome()
    fly = FlyController(weights, flyid2i, seed=args.seed, blind=args.blind)
    game = Pong(fly, autopilot=True, seed=args.seed)

    print(f'Fly{" (blind)" if args.blind else ""} vs CPU, {args.points} points')
    t0 = perf_counter()
    points = 0
    while points < args.points:
        game.tick()
        if sum(game.score) != points:
            points = sum(game.score)
            print(f'  {fly.t_ms / 1000:7.2f}s brain | fly {game.score[0]} - {game.score[1]} cpu'
                  f' | fly returns so far: {game.fly_returns}')
        elif fly.t_ms % 10000 == 0:
            print(f'  {fly.t_ms / 1000:7.2f}s brain | rally in progress'
                  f' | fly returns so far: {game.fly_returns}')
    wall = perf_counter() - t0

    faced = game.fly_returns + game.score[1]
    print(f'\nFly returned {game.fly_returns} of {faced} balls '
          f'({100 * game.fly_returns / max(faced, 1):.0f}%)')
    print(f'Brain time {fly.t_ms / 1000:.1f}s in {wall:.1f}s wall '
          f'({fly.t_ms / 1000 / wall:.2f}x real time)')


# ============================================================================
# Window
# ============================================================================

HUD_H = 110
BG, FG, MUTED, LINE = '#0e1116', '#e6edf3', '#7d8590', '#30363d'
FLY_COLOR, OPP_COLOR = '#f2b134', '#58a6ff'


class PongWindow:
    FPS = 30
    BUDGET = 0.024           # wall seconds of simulation per frame

    def __init__(self, root, args):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.args = args
        root.title('Fly Pong')
        root.configure(bg=BG)
        root.resizable(False, False)
        self.canvas = tk.Canvas(root, width=FIELD_W, height=FIELD_H + HUD_H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(FIELD_W / 2, FIELD_H / 2, fill=FG, tags='loading',
                                font=('Helvetica', 18),
                                text='Loading the fly brain (138,639 neurons)...')
        self.keys = set()
        self.paused = False
        self.speed = 0.0
        root.after(50, self._load)

    def _load(self):
        weights, flyid2i = load_connectome()
        self.fly = FlyController(weights, flyid2i, seed=self.args.seed, blind=self.args.blind)
        self.game = Pong(self.fly, autopilot=self.args.autopilot, seed=self.args.seed)
        self.canvas.delete('loading')
        self._build_scene()
        self.root.bind('<KeyPress>', self._key_down)
        self.root.bind('<KeyRelease>', lambda e: self.keys.discard(e.keysym.lower()))
        self.last_frame = perf_counter()
        self._frame()

    # ---- scene -------------------------------------------------------------

    def _build_scene(self):
        c, g = self.canvas, self.game
        for y in range(0, FIELD_H, 24):
            c.create_line(FIELD_W / 2, y, FIELD_W / 2, y + 12, fill=LINE, width=2)
        c.create_line(0, FIELD_H, FIELD_W, FIELD_H, fill=LINE)

        self.score_fly = c.create_text(FIELD_W / 2 - 60, 50, fill=FLY_COLOR, font=('Helvetica', 40, 'bold'))
        self.score_opp = c.create_text(FIELD_W / 2 + 60, 50, fill=OPP_COLOR, font=('Helvetica', 40, 'bold'))
        c.create_text(FIELD_W / 2 - 60, 84, text='FLY', fill=MUTED, font=('Helvetica', 11))
        self.opp_label = c.create_text(FIELD_W / 2 + 60, 84, fill=MUTED, font=('Helvetica', 11))
        self.pause_text = c.create_text(FIELD_W / 2, FIELD_H / 2 - 40, fill=FG, font=('Helvetica', 20))

        self.fly_paddle = c.create_rectangle(0, 0, 0, 0, fill=FLY_COLOR, outline='')
        self.opp_paddle = c.create_rectangle(0, 0, 0, 0, fill=OPP_COLOR, outline='')
        self.ball = c.create_oval(0, 0, 0, 0, fill=FG, outline='')

        # The fly riding its paddle, facing right; its left wing is on top
        x = g.fly_x - 4
        self.fly_parts = [
            c.create_oval(x - 22, -11, x - 8, -2, fill='#c9d1d9', outline='', stipple='gray50'),
            c.create_oval(x - 22, 2, x - 8, 11, fill='#c9d1d9', outline='', stipple='gray50'),
            c.create_oval(x - 26, -4, x - 7, 4, fill='#8b5a2b', outline=''),
            c.create_oval(x - 9, -4, x - 1, 4, fill='#d1242f', outline=''),
        ]
        self.fly_drawn_y = 0.0

        # Brain monitor
        y0 = FIELD_H + 24
        self.bars = {}
        for row, (label, key, vmax) in enumerate([
                ('EYES  ·  LC10a input', 'eye', EYE_RATE_MAX),
                ('STEERING  ·  DNa01 + DNa02', 'steer', 300.0)]):
            y = y0 + row * 30
            c.create_text(20, y, text=label, anchor='w', fill=FG, font=('Helvetica', 12, 'bold'))
            for side, x in (('left', 290), ('right', 590)):
                c.create_text(x, y, text=side, anchor='e', fill=MUTED, font=('Helvetica', 11))
                c.create_rectangle(x + 8, y - 7, x + 208, y + 7, outline=LINE)
                bar = c.create_rectangle(x + 8, y - 7, x + 8, y + 7, fill=FLY_COLOR, outline='')
                val = c.create_text(x + 216, y, anchor='w', fill=MUTED, font=('Helvetica', 11))
                self.bars[key, side] = (bar, val, x + 8, y, vmax)
        self.stats = c.create_text(20, y0 + 64, anchor='w', fill=MUTED, font=('Helvetica', 11))

    def _draw(self):
        c, g, f = self.canvas, self.game, self.fly
        c.coords(self.fly_paddle, g.fly_x, g.fly_y - PADDLE_H / 2, g.fly_x + PADDLE_W, g.fly_y + PADDLE_H / 2)
        c.coords(self.opp_paddle, g.opp_x, g.opp_y - PADDLE_H / 2, g.opp_x + PADDLE_W, g.opp_y + PADDLE_H / 2)
        c.coords(self.ball, g.ball_x - BALL_R, g.ball_y - BALL_R, g.ball_x + BALL_R, g.ball_y + BALL_R)
        for part in self.fly_parts:
            c.move(part, 0, g.fly_y - self.fly_drawn_y)
        self.fly_drawn_y = g.fly_y

        c.itemconfigure(self.score_fly, text=str(g.score[0]))
        c.itemconfigure(self.score_opp, text=str(g.score[1]))
        c.itemconfigure(self.opp_label, text='CPU' if g.autopilot else 'YOU')
        c.itemconfigure(self.pause_text, text='PAUSED' if self.paused else '')

        for key, rates in (('eye', f.eye_rate), ('steer', f.steer_rate)):
            for side, rate in zip(('left', 'right'), rates):
                bar, val, x, y, vmax = self.bars[key, side]
                c.coords(bar, x, y - 7, x + 200 * min(rate / vmax, 1.0), y + 7)
                c.itemconfigure(val, text=f'{rate:4.0f} Hz')

        c.itemconfigure(self.stats, text=(
            f'brain time {f.t_ms / 1000:6.1f} s   ·   {self.speed:.2f}× real time   ·   '
            f'{f.active_neurons()} neurons active in the last 100 ms   ·   '
            f'{f.spikes_total:,} spikes total'
            + ('   ·   BLIND' if f.blind else '')))

    # ---- loop --------------------------------------------------------------

    def _key_down(self, event):
        key = event.keysym.lower()
        if key == 'space':
            self.paused = not self.paused
        elif key == 'a':
            self.game.autopilot = not self.game.autopilot
        elif key in ('escape', 'q'):
            self.root.destroy()
        else:
            self.keys.add(key)

    def _frame(self):
        start = perf_counter()
        g = self.game
        g.player_dir = (('down' in self.keys or 's' in self.keys)
                        - ('up' in self.keys or 'w' in self.keys))
        ticks = 0
        if not self.paused:
            while perf_counter() - start < self.BUDGET:
                g.tick()
                ticks += 1
        self._draw()

        now = perf_counter()
        if ticks:
            self.speed += 0.1 * (ticks / 1000 / (now - self.last_frame) - self.speed)
        self.last_frame = now
        delay = 1000 / self.FPS - (now - start) * 1000
        self.root.after(max(1, int(delay)), self._frame)


def main():
    parser = argparse.ArgumentParser(description='The emulated fly brain plays Pong')
    parser.add_argument('--autopilot', action='store_true',
                        help='CPU plays the right paddle (toggle in-game with A)')
    parser.add_argument('--headless', action='store_true',
                        help='No window: fly vs CPU, print how often the fly returns the ball')
    parser.add_argument('--points', type=int, default=10,
                        help='Points to play in headless mode (default: 10)')
    parser.add_argument('--blind', action='store_true',
                        help='Control: give the fly no visual input')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.headless:
        run_headless(args)
    else:
        import tkinter as tk
        root = tk.Tk()
        PongWindow(root, args)
        root.mainloop()


if __name__ == '__main__':
    main()
