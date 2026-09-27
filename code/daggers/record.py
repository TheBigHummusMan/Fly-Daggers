"""
Record yourself playing Devil Daggers: what the game shows and what your
hands do, for the fly to learn from.

FPS times a second, the game window is captured, shrunk to the fly's
resolution (eyes.FRAME_W x FRAME_H), and saved with the share of the look
each button was held and how far the mouse moved until the next frame.
Recording runs only while the game has focus and is in play (cursor hidden),
so menus, the death screen and alt-tabbing are left out.

Usage:
    python code/daggers/record.py                     # record until F10
    python code/daggers/record.py --minutes 30
    python code/daggers/record.py --no-cursor-check   # if it never sees you in play

F9 pauses and resumes, F10 stops. Play in windowed or borderless mode: in
exclusive fullscreen, capture shows whatever window is behind the game (the
status line then says FROZEN CAPTURE). Then:
    python code/daggers/eyes.py --preview data/daggers/recordings/<session>
    python code/daggers/dataset.py
"""

import argparse
import json
import queue
import threading
import time
from datetime import datetime

import numpy as np

from dataset import FROZEN_DIFF, RECORDINGS
from eyes import FRAME_H, FRAME_W
from fly import BUTTON_NAMES, BUTTONS, FPS
from winio import VK_F9, VK_F10, GameCamera, GameState, InputListener, make_dpi_aware

CHUNK_S = 30.0                   # seconds of play per saved file


class ChunkWriter:
    """Saves chunks on a background thread, so capture never waits on the disk."""

    def __init__(self, folder):
        self.folder = folder
        self.n = 0
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, name='chunk-writer', daemon=True)
        self.thread.start()

    def put(self, rows):
        if rows:
            self.q.put((self.n, rows))
            self.n += 1

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            n, rows = item
            frames, t, buttons, mouse = zip(*rows)
            np.savez_compressed(self.folder / f'chunk_{n:04d}.npz', frames=np.stack(frames),
                                t=np.array(t), buttons=np.stack(buttons).astype(np.float32),
                                mouse=np.stack(mouse).astype(np.float32))

    def close(self):
        self.q.put(None)
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description='Record Devil Daggers play for the fly')
    parser.add_argument('--minutes', type=float, default=None, help='stop after this much play')
    parser.add_argument('--no-cursor-check', action='store_true',
                        help='record whenever the game has focus, even with the cursor showing')
    args = parser.parse_args()

    make_dpi_aware()
    folder = RECORDINGS / datetime.now().strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    listener = InputListener(BUTTONS)
    state = GameState(check_cursor=not args.no_cursor_check)
    camera = GameCamera((FRAME_W, FRAME_H))
    writer = ChunkWriter(folder)
    meta = {'fps': FPS, 'frame_w': FRAME_W, 'frame_h': FRAME_H, 'buttons': BUTTON_NAMES,
            'check_cursor': not args.no_cursor_check, 'started': datetime.now().isoformat()}

    print(f'Recording to {folder}. Switch to Devil Daggers and play; F9 pauses, F10 stops.')
    period = 1.0 / FPS
    t0 = time.perf_counter()
    next_t = t0
    rows, pending = [], None
    pending_prev, still = None, 0         # frames in a row the picture hasn't changed
    recorded = 0
    paused = False
    last_status = 0.0
    try:
        while True:
            while not listener.hotkeys.empty():
                key = listener.hotkeys.get()
                if key == VK_F9:
                    paused = not paused
                elif key == VK_F10:
                    raise KeyboardInterrupt
            now = time.perf_counter()
            if next_t > now:
                time.sleep(next_t - now)
            next_t = max(next_t + period, time.perf_counter() - period)

            hwnd, focused, in_play = state.poll()
            recording = in_play and not paused
            # What the hands did since the last frame belongs to that frame
            held, mouse = listener.take()
            if pending is not None:
                rows.append(pending + (held, mouse))
                recorded += 1
                pending = None
            if recording:
                t = time.perf_counter() - t0
                frame, rect = camera.grab(hwnd)
                if frame is not None:
                    if pending_prev is not None:
                        change = np.abs(frame.astype(np.int16) - pending_prev).mean()
                        still = still + 1 if change < FROZEN_DIFF else 0
                    pending_prev = frame
                    pending = (frame, t)
                    meta.setdefault('window', [rect['width'], rect['height']])
            if len(rows) >= CHUNK_S * FPS:
                writer.put(rows)
                rows = []

            if args.minutes and recorded >= args.minutes * 60 * FPS:
                break
            if time.perf_counter() - last_status > 1.0:
                last_status = time.perf_counter()
                what = ('PAUSED (F9)' if paused
                        else 'FROZEN CAPTURE: use borderless window' if recording and still > 3 * FPS
                        else 'recording' if recording
                        else 'waiting: game not focused' if hwnd and not focused
                        else 'waiting: menu or cursor showing' if hwnd
                        else 'waiting: Devil Daggers window not found')
                print(f'\r{what:<42} {recorded / FPS / 60:5.1f} min recorded', end='', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        held, mouse = listener.take()
        if pending is not None:
            rows.append(pending + (held, mouse))
            recorded += 1
        writer.put(rows)
        writer.close()
        listener.close()
        camera.close()
        meta['frames'] = recorded
        (folder / 'session.json').write_text(json.dumps(meta, indent=1))
        print(f'\nSaved {recorded} frames ({recorded / FPS / 60:.1f} min) in {writer.n} chunks to {folder}')
        if 'window' in meta and abs(meta['window'][0] / meta['window'][1] - FRAME_W / FRAME_H) > 0.05:
            print(f"Note: the game window is {meta['window'][0]}x{meta['window'][1]}, not 16:9; "
                  'frames are stretched to fit.')


if __name__ == '__main__':
    main()
