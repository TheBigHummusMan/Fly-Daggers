"""
Record a human playing Scritchy Scratchy, to learn the game's rules for the
simulator.

Writes data/scratch/recordings/<time>/ with:
    frames/NNNNNN.jpg   the game window, --fps times per second
    frames.csv          t, frame file, game in front, window x/y/w/h
    mouse.csv           t, x, y (0-1 window coordinates), left button down, ~60 Hz
    money.csv           t, money on the counter (OCR, every --money-every s)
    saves/<t>.json      every new version of the game's save.json (its full state)
    meta.json           settings and layout

Usage:
    python code/scratch/recorder.py                 # record until Ctrl-C
    python code/scratch/recorder.py --minutes 30
"""

import argparse
import csv
import json
import shutil
import threading
import time
from pathlib import Path

from PIL import Image

from game_io import (DATA_DIR, SAVE_DIR, RealGame, check_screen_permission, find_window,
                     game_in_front, load_layout)

MOUSE_HZ = 60


def sample_mouse(win_ref, rows, stop):
    """Log the mouse in window coordinates until stop is set."""
    import Quartz
    state = Quartz.kCGEventSourceStateHIDSystemState
    while not stop.is_set():
        p = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        down = Quartz.CGEventSourceButtonState(state, Quartz.kCGMouseButtonLeft)
        w = win_ref[0]
        rows.append((round(time.time(), 4), round((p.x - w['x']) / w['w'], 5),
                     round((p.y - w['y']) / w['h'], 5), int(bool(down))))
        time.sleep(1 / MOUSE_HZ)


def main():
    parser = argparse.ArgumentParser(description='Record a Scritchy Scratchy play session')
    parser.add_argument('--minutes', type=float, default=None, help='stop after this long (default: Ctrl-C)')
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--money-every', type=float, default=2.0, help='seconds between money OCRs')
    args = parser.parse_args()

    check_screen_permission()
    win = find_window()
    if win is None:
        raise SystemExit('Start Scritchy Scratchy first.')
    out = DATA_DIR / 'recordings' / time.strftime('%Y%m%d-%H%M%S')
    (out / 'frames').mkdir(parents=True)
    (out / 'saves').mkdir()
    (out / 'meta.json').write_text(json.dumps(
        {'fps': args.fps, 'window': win, 'layout': load_layout(), 'started': time.time()}, indent=2))

    env = RealGame()
    env._attach()
    win_ref = [env.win]
    mouse_rows, stop = [], threading.Event()
    threading.Thread(target=sample_mouse, args=(win_ref, mouse_rows, stop), daemon=True).start()

    frames_f = open(out / 'frames.csv', 'w', newline='')
    money_f = open(out / 'money.csv', 'w', newline='')
    frames_w, money_w = csv.writer(frames_f), csv.writer(money_f)
    frames_w.writerow(['t', 'file', 'front', 'x', 'y', 'w', 'h'])
    money_w.writerow(['t', 'money'])

    save_path = SAVE_DIR / 'save.json'
    last_save_mtime = None
    last_money = last_win = -1e9
    t_end = time.time() + args.minutes * 60 if args.minutes else float('inf')
    n = 0
    print(f'Recording to {out}\nPlay normally. Press Ctrl-C to stop.')
    try:
        while time.time() < t_end:
            t0 = time.time()
            if t0 - last_win > 5.0:            # follow the window if it moves
                win = find_window() or env.win
                env.win = win_ref[0] = win
                last_win = t0
            front = game_in_front()
            name = f'{n:06d}.jpg'
            Image.fromarray(env.frame()).save(out / 'frames' / name, quality=85)
            w = env.win
            frames_w.writerow([round(t0, 3), name, int(front), w['x'], w['y'], w['w'], w['h']])
            n += 1

            if front and t0 - last_money >= args.money_every:
                money_w.writerow([round(t0, 3), env.money(max_age=0)])
                last_money = t0

            try:
                mtime = save_path.stat().st_mtime
                if mtime != last_save_mtime:
                    shutil.copy2(save_path, out / 'saves' / f'{mtime:.3f}.json')
                    last_save_mtime = mtime
            except OSError:
                pass

            if n % 60 == 0:
                frames_f.flush()
                money_f.flush()
                print(f'  {n} frames, {len(mouse_rows)} mouse samples')
            time.sleep(max(0.0, 1 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    stop.set()
    time.sleep(0.1)
    with open(out / 'mouse.csv', 'w', newline='') as f:
        mw = csv.writer(f)
        mw.writerow(['t', 'x', 'y', 'down'])
        mw.writerows(mouse_rows)
    frames_f.close()
    money_f.close()
    print(f'\nSaved {n} frames and {len(mouse_rows)} mouse samples to {out}')


if __name__ == '__main__':
    main()
