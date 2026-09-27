"""
Cut the simulator's sprites out of recorded frames of the real game.

Writes data/scratch/sprites/*.png at half the window's resolution (the sim
renders at a quarter) plus sprites.json with each sprite's source and its
place in the window (0-1 coordinates), so the sim draws everything where the
real game does. The sprites are small and committed; recordings are not.

Run once after recording (recorder.py). The frame numbers below were picked by
hand from the first two recordings; point SOURCES at other recordings to redo.

Usage:
    python code/scratch/sprites.py            # write sprites + sprites_sheet.png to check
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'scratch'
REC_DIR = DATA_DIR / 'recordings'
OUT_DIR = DATA_DIR / 'sprites'
R1, R2 = '20260926-134122', '20260926-140745'
SCALE = 0.5                     # sprites are stored at half the window's pixels

# name: (recording, [frame numbers; several are median-combined], rect in the window)
SOURCES = {
    # empty table at each unlock stage (full window)
    'stage0': (R2, [48], (0, 0, 1, 1)),           # Day Job only
    'stage1': (R2, [94], (0, 0, 1, 1)),           # + Trash Can
    'stage2': (R2, [205], (0, 0, 1, 1)),          # + Catalog #1 (Two Win)
    'stage3': (R2, [421], (0, 0, 1, 1)),          # + Upgrades panel
    'stage4': (R1, [349], (0, 0, 1, 1)),          # + Mini Scratch
    'stage6': (R1, [891], (0, 0, 1, 1)),          # + Apple Tree
    'gadgets_stage1': (R2, [72], (0.0, 0.12, 0.17, 0.98)),
    'gadgets_stage5': (R1, [535], (0.0, 0.12, 0.17, 0.98)),
    # zoomed items; the median of several frames removes the cursor sprite
    'plate_dirty': (R2, [99, 8, 102], (0.19, 0.17, 0.615, 0.87)),
    'plate_clean': (R2, [93], (0.19, 0.17, 0.615, 0.87)),
    'twowin_covered': (R1, [167], (0.29, 0.35, 0.52, 0.705)),
    'twowin_revealed': (R1, [136], (0.29, 0.35, 0.52, 0.705)),
    'mini_covered': (R1, [310], (0.22, 0.365, 0.585, 0.72)),
    'mini_revealed': (R1, [332], (0.22, 0.365, 0.585, 0.72)),
    'info_dayjob0': (R2, [99], (0.655, 0.405, 0.86, 0.635)),
    'info_dayjob1': (R1, [209], (0.655, 0.36, 0.86, 0.66)),
    'info_twowin': (R1, [167], (0.53, 0.355, 0.725, 0.685)),
    'info_mini': (R1, [332], (0.595, 0.335, 0.795, 0.70)),
    'claim_plate': (R2, [92], (0.33, 0.885, 0.49, 0.965)),
    'claim_ticket': (R1, [136], (0.335, 0.715, 0.48, 0.79)),
    # overlays and gadgets
    'unlock_popup': (R2, [72], (0.30, 0.30, 0.58, 0.72)),
    'dialogue': (R1, [191], (0.20, 0.74, 0.90, 0.92)),
    'bot': (R1, [1001], (0.745, 0.655, 0.855, 0.905)),
    # cursors
    'sponge': (R2, [99], (0.315, 0.355, 0.375, 0.445)),
    'coin': (R1, [310], (0.415, 0.49, 0.465, 0.575)),
}

# Cells of each ticket, as (cx, cy, r) in the ticket sprite's 0-1 coordinates
CELLS = {
    'twowin': [(0.27, 0.73, 0.085), (0.50, 0.73, 0.085), (0.73, 0.73, 0.085)],
    'mini': [(0.18, 0.40, 0.075), (0.375, 0.40, 0.075), (0.575, 0.40, 0.075),
             (0.28, 0.63, 0.075), (0.475, 0.63, 0.075)],
}
# Symbol icons: (sprite, cell index) where a revealed frame shows that symbol
ICONS = {
    'cash': ('twowin_revealed', 0), 'goldcoin': ('twowin_revealed', 1),
    'redcoin': ('mini_revealed', 1), 'goldstack': ('mini_revealed', 2), 'pile': ('mini_revealed', 3),
}


def load(rec, i):
    return np.asarray(Image.open(REC_DIR / rec / 'frames' / f'{i:06d}.jpg').convert('RGB'))


def crop(img, rect):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'scale': SCALE, 'window': [1024, 668], 'sprites': {}, 'cells': CELLS, 'icons': {}}
    arrays = {}
    for name, (rec, frames, rect) in SOURCES.items():
        stack = np.stack([crop(load(rec, i), rect) for i in frames])
        arrays[name] = np.median(stack, axis=0).astype(np.uint8)

    # Mini Scratch: the coin cursor covers cell 2 of the covered ticket; copy cell 1 over it
    m = arrays['mini_covered']
    h, w = m.shape[:2]
    (ax, ay, r), (bx, by, _) = CELLS['mini'][1], CELLS['mini'][2]
    half = int(r * w * 1.3)
    src = m[int(ay * h) - half:int(ay * h) + half, int(ax * w) - half:int(ax * w) + half].copy()
    m[int(by * h) - half:int(by * h) + half, int(bx * w) - half:int(bx * w) + half] = src

    for name, img in arrays.items():
        out = Image.fromarray(img)
        out = out.resize((max(1, int(out.width * SCALE)), max(1, int(out.height * SCALE))), Image.BOX)
        out.save(OUT_DIR / f'{name}.png')
        manifest['sprites'][name] = {'rect': list(SOURCES[name][2]), 'source': f'{SOURCES[name][0]}/{SOURCES[name][1]}'}

    for icon, (sprite, k) in ICONS.items():
        img = arrays[sprite]
        h, w = img.shape[:2]
        cx, cy, r = CELLS[sprite.split('_')[0]][k]
        c = img[int((cy - r * w / h) * h):int((cy + r * w / h) * h), int((cx - r) * w):int((cx + r) * w)]
        out = Image.fromarray(c)
        out.resize((max(1, int(out.width * SCALE)), max(1, int(out.height * SCALE))), Image.BOX).save(OUT_DIR / f'icon_{icon}.png')
        manifest['icons'][icon] = f'icon_{icon}.png'

    (OUT_DIR / 'sprites.json').write_text(json.dumps(manifest, indent=2))

    # contact sheet to check by eye
    names = [n for n in SOURCES if not n.startswith('stage')] + [f'icon_{i}' for i in ICONS]
    tiles = [Image.open(OUT_DIR / f'{n}.png') for n in names]
    sheet = Image.new('RGB', (1200, 700), (40, 40, 40))
    x = y = rowh = 0
    for t in tiles:
        t = t.copy()
        t.thumbnail((240, 240))
        if x + t.width > 1200:
            x, y, rowh = 0, y + rowh + 6, 0
        sheet.paste(t, (x, y))
        x += t.width + 6
        rowh = max(rowh, t.height)
    sheet.save(DATA_DIR / 'sprites_sheet.png')
    total = sum(p.stat().st_size for p in OUT_DIR.glob('*.png'))
    print(f'Wrote {len(list(OUT_DIR.glob("*.png")))} sprites ({total / 1e6:.1f} MB) to {OUT_DIR}')


if __name__ == '__main__':
    main()
