"""
Recordings of a person playing Devil Daggers, turned into training clips.

record.py writes one folder per session under data/daggers/recordings/:

    session.json        fps, frame size, button names, window size
    chunk_0000.npz      frames  (N, FRAME_H, FRAME_W, 3) uint8, what the game showed
                        t       (N,) seconds since the session started, at capture
                        buttons (N, len(BUTTON_NAMES)) fraction of [t_i, t_i+1) each was held
                        mouse   (N, 2) raw mouse counts moved in [t_i, t_i+1)

Preparing runs the fixed Retina over every frame once and keeps only its raw
features next to the inputs, a few MB per hour of play. That is all training
needs, so it's all that goes to the cloud. Frames stay on this machine.

    data/daggers/prepared/<session>.npz   raw (N, N_RAW), t, buttons, mouse, segment
    data/daggers/prepared/stats.json      feature scales and mouse scale over all sessions

(FLY_DAGGERS_DATA=<dir> moves data/daggers/ elsewhere.)

Usage:
    python code/daggers/dataset.py            # prepare new or changed sessions
    python code/daggers/dataset.py --force    # prepare everything again (after changing eyes.MASKS)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import eyes
from eyes import Retina, feature_scales
from fly import BUTTON_NAMES, FPS

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('FLY_DAGGERS_DATA', ROOT / 'data' / 'daggers'))
RECORDINGS = DATA / 'recordings'
PREPARED = DATA / 'prepared'

GAP_FRAMES = 2.5                 # a pause longer than this many frames starts a new segment
CLIP_S = 10.0                    # training clip length
MIN_CLIP_S = 4.0                 # shorter leftovers are dropped
WARM_S = 1.0                     # brain settling time at the start of a clip, not scored
LABEL_SHIFT = 1                  # the fly's action lands a look after the frame it saw
MOUSE_CLIP = 3.0                 # mouse targets clipped at this many mouse_scales
FROZEN_S = 2.0                   # a picture this still for this long is failed capture...
FROZEN_DIFF = 0.5                # ...(mean change per frame, 0-255 scale)


def load_session(folder):
    """All chunks of a recorded session, concatenated in order."""
    folder = Path(folder)
    parts = {k: [] for k in ('frames', 't', 'buttons', 'mouse')}
    for chunk in sorted(folder.glob('chunk_*.npz')):
        with np.load(chunk) as z:
            for k in parts:
                parts[k].append(z[k])
    if not parts['t']:
        return None
    return {k: np.concatenate(v) for k, v in parts.items()}


def segments(t, fps=FPS):
    """Segment id per frame: a new one after every pause in recording."""
    gaps = np.diff(t, prepend=t[0]) > GAP_FRAMES / fps
    return np.cumsum(gaps)


def frozen(frames, fps=FPS):
    """Frames in a stretch of FROZEN_S or more where the picture barely
    changes. Devil Daggers never holds still in play, so this is capture that
    failed: exclusive fullscreen, where the capture shows whatever window is
    behind the game."""
    diff = np.zeros(len(frames))
    for k in range(1, len(frames)):                  # one pair at a time: frames are big
        diff[k] = np.abs(frames[k].astype(np.int16) - frames[k - 1]).mean()
    n = max(int(FROZEN_S * fps), 1)
    still = np.convolve(diff, np.ones(n) / n, mode='same') < FROZEN_DIFF
    # Grow each still stretch by half a window on both sides, to its full extent
    return np.convolve(still, np.ones(n), mode='same') > 0


def prepare(force=False):
    PREPARED.mkdir(parents=True, exist_ok=True)
    retina = Retina()
    for folder in sorted(p for p in RECORDINGS.glob('*') if p.is_dir()):
        out = PREPARED / f'{folder.name}.npz'
        chunks = sorted(folder.glob('chunk_*.npz'))
        if not chunks:
            continue
        if out.exists() and not force and out.stat().st_mtime > max(c.stat().st_mtime for c in chunks):
            continue
        s = load_session(folder)
        bad = frozen(s['frames'])
        if bad.any():
            print(f'{folder.name}: dropping {bad.sum()} frames ({bad.sum() / FPS / 60:.1f} min) '
                  'where the capture was frozen (exclusive fullscreen?)')
            s = {k: v[~bad] for k, v in s.items()}
        seg = segments(s['t'])
        raw = np.zeros((len(seg), eyes.N_RAW), dtype=np.float32)
        for k in np.unique(seg):
            idx = np.flatnonzero(seg == k)
            raw[idx] = retina.run(s['frames'][idx], s['t'][idx])
        np.savez_compressed(out, raw=raw, t=s['t'], buttons=s['buttons'].astype(np.float32),
                            mouse=s['mouse'].astype(np.float32), segment=seg,
                            masks=np.array(eyes.MASKS, dtype=np.float32))
        print(f'{folder.name}: {len(seg)} frames ({len(seg) / FPS / 60:.1f} min), '
              f'{len(np.unique(seg))} segments -> {out}')
    write_stats()


def load_prepared(folder=PREPARED):
    sessions = []
    for path in sorted(Path(folder).glob('*.npz')):
        with np.load(path) as z:
            sessions.append({k: z[k] for k in z.files} | {'name': path.stem})
    return sessions


def write_stats(folder=PREPARED):
    sessions = load_prepared(folder)
    if not sessions:
        raise SystemExit(f'Nothing recorded yet: no sessions under {RECORDINGS}')
    raw = np.concatenate([s['raw'] for s in sessions])
    mouse = np.concatenate([s['mouse'] for s in sessions])
    buttons = np.concatenate([s['buttons'] for s in sessions])
    stats = {
        'feature_scales': feature_scales(raw),
        # Typical mouse motion per look while the mouse is moving
        'mouse_scale': [max(float(np.percentile(np.abs(m[m != 0]), 90)), 1.0) if np.any(m) else 1.0
                        for m in mouse.T],
        'minutes': len(raw) / FPS / 60,
        'button_share': dict(zip(BUTTON_NAMES, (buttons > 0.5).mean(axis=0).round(3).tolist())),
    }
    (Path(folder) / 'stats.json').write_text(json.dumps(stats, indent=1))
    print(f"{stats['minutes']:.1f} min of play; buttons held (share of looks): {stats['button_share']}")
    return stats


def load_stats(folder=PREPARED):
    return json.loads((Path(folder) / 'stats.json').read_text())


def clips(sessions, mouse_scale, clip_s=CLIP_S, fps=FPS):
    """Cut sessions into clips of (raw, dt, Y). Y[i] is what the player did
    in the look LABEL_SHIFT after frame i: buttons, then mouse in scale units."""
    n_clip, n_min = int(clip_s * fps), int(MIN_CLIP_S * fps)
    scale = np.asarray(mouse_scale, dtype=np.float32)
    out = []
    for s in sessions:
        seg = s['segment']
        for k in np.unique(seg):
            idx = np.flatnonzero(seg == k)
            raw, t = s['raw'][idx], s['t'][idx]
            dt = np.diff(t, prepend=t[0]).astype(np.float32)
            y = np.concatenate([s['buttons'][idx],
                                np.clip(s['mouse'][idx] / scale, -MOUSE_CLIP, MOUSE_CLIP)], axis=1)
            raw, dt, y = raw[:-LABEL_SHIFT], dt[:-LABEL_SHIFT], y[LABEL_SHIFT:]
            for a in range(0, len(raw), n_clip):
                if len(raw) - a >= n_min:
                    out.append({'raw': raw[a:a + n_clip], 'dt': dt[a:a + n_clip],
                                'y': y[a:a + n_clip], 'name': f"{s['name']}/{k}/{a}"})
    return out


def split(all_clips, every=5):
    """Every `every`-th clip is held out for validation; the rest train."""
    val = all_clips[every - 1::every]
    train = [c for i, c in enumerate(all_clips) if i % every != every - 1]
    return train, val


def main():
    parser = argparse.ArgumentParser(description='Prepare recordings for training')
    parser.add_argument('--force', action='store_true', help='prepare every session again')
    prepare(parser.parse_args().force)


if __name__ == '__main__':
    main()
