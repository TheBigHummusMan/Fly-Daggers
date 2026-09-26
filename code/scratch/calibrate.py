"""
Set the fly's starting genome from recordings of a human playing.

It uses only what the fly itself would perceive:
    taste  - colours under the cursor when the human clicked, compared with
             colours anywhere on screen, become sweet (sugar weights); colours
             that were clicked less often than they appear become slightly unsweet
    speed  - the human's median cursor speed while moving
    button - held while walking if the human scratched a lot (hold_thr = 0)

The result, data/scratch/start_genome.json, is the default for play.py and
evolve.py. Evolution then starts from it rather than from the hand-set genome.

Usage:
    python code/scratch/calibrate.py                     # all recordings
    python code/scratch/calibrate.py data/scratch/recordings/<run> ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cursor_fly import CursorFly, Genome, N_BINS, REF_W, colour_bins  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'scratch'
START_GENOME = DATA_DIR / 'start_genome.json'
MAX_LAG_S = 0.15          # a click is matched to a frame at most this far away in time
TASTE_RADIUS = 10.0       # px, as in the hand-set genome
EPS = 0.01                # smoothing for colour-frequency ratios


def taste_bins(frame, x, y, radius=TASTE_RADIUS):
    """Colour bins under window point (x, y), sampled exactly as CursorFly tastes."""
    lum, rgb = CursorFly._grid(None, frame)
    gh, gw = lum.shape
    return colour_bins(CursorFly._sample(None, rgb, x * gw * 4, y * gh * 4, radius))


def analyse(rec, rng):
    frames = pd.read_csv(rec / 'frames.csv')
    frames = frames[frames.front == 1].reset_index(drop=True)
    mouse = pd.read_csv(rec / 'mouse.csv')
    mouse = mouse[(mouse.x.between(0, 1)) & (mouse.y.between(0, 1))]
    t = mouse.t.to_numpy()

    # clicks: button going down
    down = mouse.down.to_numpy()
    starts = mouse.iloc[np.flatnonzero((down[1:] == 1) & (down[:-1] == 0)) + 1]

    clicked, background = [], []
    cache = {}

    def frame_at(ts):
        i = int(np.abs(frames.t.to_numpy() - ts).argmin())
        if abs(frames.t.iloc[i] - ts) > MAX_LAG_S:
            return None
        name = frames.file.iloc[i]
        if name not in cache:
            cache.clear()
            cache[name] = np.asarray(Image.open(rec / 'frames' / name).convert('RGB'))
        return cache[name]

    for _, c in starts.iterrows():
        f = frame_at(c.t)
        if f is not None:
            clicked.append(taste_bins(f, c.x, c.y))
    for name in frames.file.sample(min(200, len(frames)), random_state=0):
        f = np.asarray(Image.open(rec / 'frames' / name).convert('RGB'))
        for _ in range(10):
            background.append(taste_bins(f, rng.uniform(0.02, 0.98), rng.uniform(0.05, 0.98)))

    # cursor speed in px of a 1024-wide window, while moving
    w = json.loads((rec / 'meta.json').read_text())['window']
    dx = np.diff(mouse.x.to_numpy()) * REF_W
    dy = np.diff(mouse.y.to_numpy()) * REF_W * w['h'] / w['w']
    dt = np.maximum(np.diff(t), 1e-3)
    speed = np.hypot(dx, dy) / dt
    moving = speed > 20
    return dict(clicked=clicked, background=background, clicks=len(starts),
                speed_down=speed[moving & (down[1:] == 1)], speed_up=speed[moving & (down[1:] == 0)],
                held=float(down.mean()), minutes=(t[-1] - t[0]) / 60)


def main():
    parser = argparse.ArgumentParser(description='Starting genome from human recordings')
    parser.add_argument('recordings', nargs='*', type=Path)
    args = parser.parse_args()
    recs = args.recordings or sorted((DATA_DIR / 'recordings').glob('*/'))
    if not recs:
        sys.exit('No recordings yet. Run code/scratch/recorder.py first.')

    rng = np.random.default_rng(0)
    res = [analyse(r, rng) for r in recs]
    clicked = np.array(sum((r['clicked'] for r in res), []))
    background = np.array(sum((r['background'] for r in res), []))
    speed_down = np.concatenate([r['speed_down'] for r in res])
    held = np.average([r['held'] for r in res], weights=[r['minutes'] for r in res])
    minutes = sum(r['minutes'] for r in res)

    p_click = clicked.mean(axis=0)
    p_bg = background.mean(axis=0)
    ratio = np.log((p_click + EPS) / (p_bg + EPS))
    sugar = np.clip(ratio / 2.0, -1, 1)          # 7x more often under clicks -> fully sweet
    sugar[12:] = np.minimum(sugar[12:], 0.0)     # greys never sweet: the game's cursor is grey

    g = Genome.default().as_dict()
    for i in range(N_BINS):
        g[f'sugar_{i}'] = float(sugar[i])
        g[f'bitter_{i}'] = 0.0                   # nothing in the recordings says what to avoid
    g['taste_radius'] = TASTE_RADIUS
    g['base_speed'] = float(np.median(speed_down)) if len(speed_down) else g['base_speed']
    g['hold_thr'] = 0.0 if held > 0.3 else g['hold_thr']
    genome = Genome([g[n] for n in g])           # clipped to the gene bounds
    genome.save(START_GENOME, source=[str(r) for r in recs], minutes=round(minutes, 1),
                clicks=len(clicked), held_fraction=round(held, 3))

    names = [f'hue {30 * i}-{30 * i + 30}' for i in range(12)] + ['grey dark', 'grey mid', 'grey light', 'white']
    print(f'{len(recs)} recording(s), {minutes:.1f} min, {len(clicked)} clicks matched to frames')
    print(f'{"colour":16s} {"under clicks":>12s} {"on screen":>10s} {"sugar":>7s}')
    for i in range(N_BINS):
        print(f'{names[i]:16s} {p_click[i]:12.3f} {p_bg[i]:10.3f} {genome[f"sugar_{i}"]:+7.2f}')
    print(f'cursor speed while scratching: median {np.median(speed_down):.0f} px/s '
          f'-> base_speed {genome["base_speed"]:.0f}')
    print(f'button held {held:.0%} of the time -> hold_thr {genome["hold_thr"]:.0f} Hz')
    print(f'Saved {START_GENOME}')


if __name__ == '__main__':
    main()
