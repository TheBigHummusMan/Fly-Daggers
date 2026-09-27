"""
Is ScratchSim good enough to train on? Compare it with recordings of the real
game (recorder.py), and optionally with the fly playing the real game.

    A  senses   For sampled real frames, detect the game state (unlock stage,
                which item is zoomed, how much of it is scratched), render the
                same state in the sim, and compare what the fly would sense:
                pixels, taste (colour bins under the cursor), eye contrast.
    B  scratch  Replay the human's own mouse strokes on each zoomed item in the
                sim and compare when the item is finished with when the real
                claim button appeared.
    C  payouts  Money change at each real claim vs the payouts the sim allows
                for that item and level.
    D  transfer (--transfer, needs the game) Score several genomes in the sim
                and on the real game; the sim is good enough when the two
                rankings agree.

Upgrade and ticket levels at each moment come from the recording's save.json
snapshots, so the sim scratches with the same coin the human had.

Usage:
    python code/scratch/validate.py                       # A-C on all recordings
    python code/scratch/validate.py --show 6              # also save side-by-side frames
    python code/scratch/validate.py --transfer data/scratch/evolve/*/best.json --save my-progress
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sim as S  # noqa: E402
from cursor_fly import CursorFly, colour_bins  # noqa: E402

DATA_DIR = S.DATA_DIR
PASS = {'pixels': 0.06, 'taste': 0.30, 'eyes': 0.05, 'scratch': (0.7, 1.4), 'payout': 0.8}
KINDS = ('dayjob', 'twowin', 'mini')
STAGES = ('stage0', 'stage1', 'stage2', 'stage3', 'stage4', 'stage6')
STAGE_GOAL = {'stage0': 0, 'stage1': 1, 'stage2': 2, 'stage3': 3, 'stage4': 4, 'stage6': 6}
PANEL_RECTS = [(0.0, 0.12, 0.17, 0.60), (0.84, 0.12, 1.0, 0.45)]
SAVE_KIND = {'Day Job': 'dayjob', 'Two Win': 'twowin', 'Mini Scratch': 'mini', 'Apple Tree': 'apple'}


def grid(img):
    return np.asarray(Image.fromarray(img).resize((S.GW, S.GH), Image.BOX), dtype=np.float32) / 255


def region(a, rect):
    x0, y0, x1, y1 = rect
    return a[S.gy(y0):S.gy(y1), S.gx(x0):S.gx(x1)]


class Detector:
    """Reads the sim-relevant state off a real frame, using the sim's own sprites."""

    def __init__(self, env):
        self.env = env
        self.sp = env.sprites
        self.items = {}
        for kind in KINDS:
            cov, rev, _, claim = S.ITEM_SPRITES[kind]
            self.items[kind] = (self.sp.img[cov], self.sp.img[rev], self.sp.rect[cov], claim)

    def stage(self, g):
        best, err = None, 1e9
        for st in STAGES:
            e = np.mean([np.abs(region(g, r) - region(self.sp.img[st], r)).mean() for r in PANEL_RECTS])
            if e < err:
                best, err = st, e
        return best

    def zoomed(self, g):
        """(kind, per-pixel 'looks revealed' mask) of the zoomed item, or (None, None)."""
        best, err, rev_mask = None, 1e9, None
        for kind, (cov, rev, rect, _) in self.items.items():
            r = region(g, rect)
            h, w = min(r.shape[0], cov.shape[0]), min(r.shape[1], cov.shape[1])
            r, c, v = r[:h, :w], cov[:h, :w], rev[:h, :w]
            dc, dv = np.abs(r - c).sum(-1), np.abs(r - v).sum(-1)
            e = np.minimum(dc, dv).mean() / 3
            if e < err:
                best, err, rev_mask = kind, e, dv < dc
        return (best, rev_mask) if err < 0.07 else (None, None)

    def claim_visible(self, g, kind):
        name = self.items[kind][3]
        ref = self.sp.img[name]
        r = region(g, self.sp.rect[name])
        h, w = min(r.shape[0], ref.shape[0]), min(r.shape[1], ref.shape[1])
        return np.abs(r[:h, :w] - ref[:h, :w]).mean() < 0.12


def save_states(rec):
    out = []
    for p in sorted((rec / 'saves').glob('*.json')):
        try:
            out.append((float(p.stem), json.loads(p.read_text())))
        except (ValueError, OSError):
            pass
    return out


def apply_save(env, states, t):
    """Set the sim's levels and upgrades from the last save before time t."""
    before = [s for ts, s in states if ts <= t] or [s for _, s in states[:1]]
    if not before:
        return
    L = before[-1]['layerOne']
    for name, d in L.get('ticketProgressionDict', {}).items():
        if name in SAVE_KIND:
            env.levels[SAVE_KIND[name]] = int(d.get('level', 0))
    up = {k: int(v.get('buyCount', 0)) for k, v in L.get('upgradeDataDict', {}).items()}
    coin = 2 if up.get('Aluminum Coin') else 1 if up.get('Tin Coin') else 0
    coin_name = ['Base Coin', 'Tin Coin', 'Aluminum Coin'][coin]
    env.upgrades.update(luck=up.get('Scratch Luck', 0), coin=coin, size=up.get(f'Scratch Size_{coin_name}', 0))


def senses(fly_grid_frame, points):
    """Taste bins and eye-patch contrast at points, as CursorFly would sense them."""
    lum = fly_grid_frame @ np.array([0.299, 0.587, 0.114])
    taste, contrast = [], []
    for x, y in points:
        px, py = x * S.GW * 4, y * S.GH * 4
        taste.append(colour_bins(CursorFly._sample(None, fly_grid_frame, px, py, 10)))
        contrast.append(CursorFly._sample(None, lum, px, py, 24).std())
    return np.array(taste), np.array(contrast)


# ============================================================================

def check_senses(env, det, recs, n_frames, show, rng):
    rows, shown = [], []
    for rec in recs:
        f = pd.read_csv(rec / 'frames.csv')
        f = f[f.front == 1]
        mouse = pd.read_csv(rec / 'mouse.csv') if (rec / 'mouse.csv').exists() else None
        states = save_states(rec)
        for _, row in f.sample(min(n_frames, len(f)), random_state=0).iterrows():
            real = grid(np.asarray(Image.open(rec / 'frames' / row.file).convert('RGB')))
            env.reset(0)
            env.gain[:] = 1
            env.zoomed = None
            st = det.stage(real)
            env.goal_i = STAGE_GOAL[st]
            kind, rev_mask = det.zoomed(real)
            if kind:
                apply_save(env, states, row.t)
                z = S.Zoomed(kind, env)
                h, w = min(z.mask.shape[0], rev_mask.shape[0]), min(z.mask.shape[1], rev_mask.shape[1])
                z.mask[:h, :w] &= ~rev_mask[:h, :w]
                env.zoomed = z
            if mouse is not None:
                m = mouse.iloc[int((mouse.t - row.t).abs().idxmin())]
                env.cursor = (float(np.clip(m.x, 0, 1)), float(np.clip(m.y, 0, 1)))
            sim = env.frame().astype(np.float32) / 255
            pts = rng.uniform([0.02, 0.05], [0.98, 0.98], size=(30, 2))
            tr, cr = senses(real, pts)
            ts, cs = senses(sim, pts)
            rows.append(dict(rec=rec.name, file=row.file, stage=st, item=kind or '-',
                             pixels=float(np.abs(real - sim).mean()),
                             taste=float(0.5 * np.abs(tr - ts).sum(axis=1).mean()),
                             eyes=float(np.abs(cr - cs).mean())))
            shown.append((rows[-1]['pixels'], real, sim, f'{rec.name[-6:]}/{row.file} {st} {kind or "-"}'))
    df = pd.DataFrame(rows)
    if show:
        shown.sort(key=lambda s: -s[0])
        picks = shown[:show // 2] + shown[len(shown) // 2:len(shown) // 2 + show - show // 2]
        sheet = Image.new('RGB', (S.GW * 4, S.GH * 2 * len(picks)))
        for i, (_, real, sim, label) in enumerate(picks):
            pair = np.concatenate([real, sim], axis=1)
            img = Image.fromarray((pair * 255).astype(np.uint8)).resize((S.GW * 4, S.GH * 2), Image.NEAREST)
            sheet.paste(img, (0, i * S.GH * 2))
        path = DATA_DIR / 'validate_senses.png'
        sheet.save(path)
        print(f'  real (left) vs sim (right), worst and typical frames: {path}')
    return df


def zoom_segments(det, rec):
    """Consecutive frames with the same zoomed item: (kind, t_start, t_claim or None, t_end)."""
    f = pd.read_csv(rec / 'frames.csv')
    f = f[f.front == 1].reset_index(drop=True)
    segs, cur = [], None
    for _, row in f.iterrows():
        g = grid(np.asarray(Image.open(rec / 'frames' / row.file).convert('RGB')))
        kind, _ = det.zoomed(g)
        claim = kind is not None and det.claim_visible(g, kind)
        if cur and (kind != cur[0] or row.t - cur[3] > 3.0):
            segs.append(cur)
            cur = None
        if kind:
            if cur is None:
                cur = [kind, row.t, None, row.t]
            cur[3] = row.t
            if claim and cur[2] is None:
                cur[2] = row.t
                cur.append(rec / 'frames' / row.file)
    if cur:
        segs.append(cur)
    return segs


def claim_amount(path, env, kind):
    """The payout printed on the real claim button (OCR)."""
    from game_io import parse_money, read_text
    img = np.asarray(Image.open(path).convert('RGB'))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = env.sprites.rect[S.ITEM_SPRITES[kind][3]]
    crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    crop = np.pad(np.kron(crop, np.ones((2, 2, 1), dtype=np.uint8)), ((20, 20), (20, 20), (0, 0)), mode='edge')
    for text, _, _ in read_text(crop):
        v = parse_money(text.replace('S', '$'), need_dollar=False)
        if v is not None:
            return -v if '-' in text else v
    return math.nan


segments = {}


def check_scratch(env, det, recs, payouts=True):
    rows = []
    for rec in recs:
        if not (rec / 'mouse.csv').exists():
            continue
        mouse = pd.read_csv(rec / 'mouse.csv')
        states = save_states(rec)
        for seg in segments.setdefault(rec, zoom_segments(det, rec)):
            kind, t0, t_claim, t1 = seg[:4]
            if t_claim is None or t_claim - t0 < 1.0:
                continue
            env.reset(0)
            apply_save(env, states, t0)
            z = env.zoomed = S.Zoomed(kind, env)
            m = mouse[(mouse.t >= t0 - 0.5) & (mouse.t <= t0 + 60)]
            prev, t_done = None, None
            for _, s in m.iterrows():
                cur = (float(s.x), float(s.y))
                if s.down and prev is not None:
                    env._scratch(prev, cur)
                    if z.done:
                        t_done = s.t
                        break
                prev = cur if s.down else None
            real_pay = claim_amount(seg[4], env, kind) if payouts else math.nan
            spec = env.rules['tickets'][kind]
            level = env.levels[kind]
            allowed = set()
            # saves are written every 2 min, so the ticket may have levelled up since
            for lv in range(max(level - 1, 0), level + 3):
                mult = env.rules['level_mult'] ** max(lv - (1 if kind == 'dayjob' else 0), 0)
                table = spec['levels'][min(lv, 1)] if kind == 'dayjob' else spec['symbols']
                allowed |= {round(v * mult) for _, _, v in table}
            rows.append(dict(rec=rec.name, kind=kind, level=level, real_s=t_claim - t0,
                             sim_s=(t_done - t0) if t_done else math.nan, real_pay=real_pay,
                             pay_ok=math.nan if math.isnan(real_pay) else
                             float(any(abs(real_pay - a) <= max(1, 0.05 * abs(a)) for a in allowed))))
    return pd.DataFrame(rows)


def check_transfer(args):
    from fast_brain import load_connectome
    from cursor_fly import Genome
    from evolve import run_episode, signed_log
    from game_io import RealGame, check_accessibility, check_screen_permission
    check_screen_permission()
    check_accessibility()
    weights, flyid2i = load_connectome()
    rows = []
    for path in args.transfer:
        g = Genome.load(path)
        sim_scores = []
        for seed in range(args.n_seeds):
            fly = CursorFly(weights, flyid2i, g, seed=seed)
            gained, _ = run_episode(S.ScratchSim(), fly, args.seconds, seed=10_000 + seed)
            sim_scores.append(signed_log(gained))
        fly = CursorFly(weights, flyid2i, g, seed=0)
        gained, _ = run_episode(RealGame(), fly, args.seconds, save=args.save)
        rows.append(dict(genome=path, sim=float(np.mean(sim_scores)), real=signed_log(gained)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    rho = df.sim.rank().corr(df.real.rank())
    print(df.to_string(index=False))
    print(f'Rank agreement (Spearman) between sim and real: {rho:+.2f}  '
          f'({"good enough" if rho >= 0.7 else "not yet"}; 1 = same order)')


def main():
    parser = argparse.ArgumentParser(description='Compare ScratchSim with recordings of the real game')
    parser.add_argument('recordings', nargs='*', type=Path)
    parser.add_argument('--frames', type=int, default=60, help='frames sampled per recording for check A')
    parser.add_argument('--show', type=int, default=0, help='save this many real/sim frame pairs')
    parser.add_argument('--fit', action='store_true', help='fit sponge and coin sizes to the human strokes')
    parser.add_argument('--transfer', nargs='+', help='genome jsons to score in sim and real game (check D)')
    parser.add_argument('--save', help='(D) snapshot to restore before each real episode')
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--n-seeds', type=int, default=5)
    args = parser.parse_args()
    if args.transfer:
        return check_transfer(args)

    recs = args.recordings or sorted((DATA_DIR / 'recordings').glob('*/'))
    env = S.ScratchSim(randomize=False)
    det = Detector(env)
    if args.fit:
        for key, kinds in (('sponge_radius', ['dayjob']), ('coin_radius', ['twowin', 'mini'])):
            best = None
            for r in (0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.024):
                env.rules[key] = r
                b = check_scratch(env, det, recs, payouts=False)
                b = b[b.kind.isin(kinds)]
                ratio = (b.sim_s / b.real_s).median()
                unfinished = b.sim_s.isna().mean()
                score = abs(math.log(ratio)) if unfinished < 0.2 and ratio == ratio else 9
                print(f'  {key} {r:.3f}: sim/real {ratio:.2f}, unfinished {unfinished:.0%}')
                if best is None or score < best[0]:
                    best = (score, r)
            print(f'best {key} = {best[1]}  (set it in sim.RULES)')
        return
    rng = np.random.default_rng(0)
    verdict = []

    print('A. What the fly senses: real frame vs the same state in the sim')
    a = check_senses(env, det, recs, args.frames, args.show, rng)
    by = a.groupby('item')[['pixels', 'taste', 'eyes']].median().round(3)
    by['frames'] = a.groupby('item').size()
    print(by.to_string())
    for key in ('pixels', 'taste', 'eyes'):
        med = a[key].median()
        ok = med <= PASS[key]
        verdict.append(ok)
        print(f'  {key:7s} median {med:.3f}  (pass <= {PASS[key]})  {"PASS" if ok else "FAIL"}')

    print('\nB/C. Human strokes replayed in the sim, and payouts')
    b = check_scratch(env, det, recs)
    if len(b):
        b['ratio'] = b.sim_s / b.real_s
        print(b.groupby('kind').agg(items=('ratio', 'size'), real_s=('real_s', 'median'), sim_s=('sim_s', 'median'),
                                    ratio=('ratio', 'median'), unfinished=('sim_s', lambda s: int(s.isna().sum())),
                                    payouts_ok=('pay_ok', 'mean')).round(2).to_string())
        ratio = b.ratio.median()
        lo, hi = PASS['scratch']
        ok = lo <= ratio <= hi and b.sim_s.isna().mean() < 0.2
        verdict.append(ok)
        print(f'  scratch time sim/real median {ratio:.2f}  (pass {lo}-{hi}, <20% unfinished)  {"PASS" if ok else "FAIL"}')
        pay = b.pay_ok.mean()                             # unreadable claim buttons (NaN) are skipped
        verdict.append(pay >= PASS['payout'])
        print(f'  payouts matching the sim tables {pay:.0%}  (pass >= {PASS["payout"]:.0%})  '
              f'{"PASS" if pay >= PASS["payout"] else "FAIL"}')
    else:
        print('  no complete zoomed items found in the recordings')
    print(f'\nOverall: {sum(verdict)}/{len(verdict)} checks pass. '
          f'Check D (--transfer) needs the game and is the final word.')


if __name__ == '__main__':
    main()
