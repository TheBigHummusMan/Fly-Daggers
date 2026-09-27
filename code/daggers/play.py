"""
The trained fly plays Devil Daggers on this machine.

FPS times a second: capture the game, Retina -> Eyes (the evolved genome) ->
connectome for LOOK_MS of brain time -> Readout -> keys held and mouse
velocity, sent with SendInput.

Safety:
- The fly only touches the controls while Devil Daggers has focus and is in
  play (cursor hidden). In menus, on the death screen, or alt-tabbed, it lets
  go of everything.
- Move the mouse or press a game key yourself and it lets go for
  TAKEOVER_S seconds: you take over.
- F9 pauses and resumes the fly, F10 stops it.
- When the fly dies, it presses R to start the next run (--no-restart turns
  this off). A death is the cursor appearing while the fly was playing; if you
  pressed a key just before (Esc to pause, say), it's you, and nothing is pressed.

Usage:
    python code/daggers/play.py                    # the newest policy
    python code/daggers/play.py --policy data/daggers/runs/<run>/policy.json
    python code/daggers/play.py --dry-run          # show what it would do, touch nothing
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from dataset import DATA, WARM_S
from eyes import BANDS, CHANNELS, FRAME_H, FRAME_W, Retina
from fly import BUTTON_NAMES, BUTTONS, DaggersFly, keep_awake, latest_run, load_policy
from winio import VK_F9, VK_F10, VK_R, Controller, FrameGrabber, InputListener, make_dpi_aware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_brain import load_connectome  # noqa: E402

TAKEOVER_S = 2.0
RESTART_DELAY_S = 1.5            # after dying, before the first R (death screen settles)
RESTART_RETRY_S = 1.5            # between tries, if the run didn't start
RESTART_TRIES = 4


def main():
    parser = argparse.ArgumentParser(description='The fly plays Devil Daggers')
    parser.add_argument('--policy', default='latest',
                        help="policy.json from train.py export (default: the newest one)")
    parser.add_argument('--dry-run', action='store_true', help='print what the fly would do; send nothing')
    parser.add_argument('--no-cursor-check', action='store_true',
                        help='treat the game as in play whenever it has focus')
    parser.add_argument('--no-restart', action='store_true',
                        help="don't press R to start a new run after the fly dies")
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.policy == 'latest':
        args.policy = latest_run(DATA / 'runs', 'policy.json') / 'policy.json'
    make_dpi_aware()
    keep_awake()
    genome, eyes, readout, meta = load_policy(args.policy)
    fps, look_ms = meta['fps'], meta['look_ms']
    period = 1.0 / fps
    mouse_scale = np.asarray(meta['mouse_scale'])
    print(f"Policy {args.policy}: held-out R^2 {meta['stats'].get('heldout_r2_per_output', 'n/a')}")

    t = time.perf_counter()
    weights, flyid2i = load_connectome()
    fly = DaggersFly(weights, flyid2i, seed=args.seed)
    fly.look({ch: np.zeros((2, BANDS)) for ch in CHANNELS})            # compile the brain kernels now
    print(f'Connectome loaded in {time.perf_counter() - t:.1f}s')
    retina = Retina()
    listener = InputListener(BUTTONS)
    controller = None if args.dry_run else Controller(BUTTONS)
    grabber = FrameGrabber((FRAME_W, FRAME_H), fps, check_cursor=not args.no_cursor_check)

    print('Switch to Devil Daggers and start a run. F9 pauses the fly, F10 stops it.'
          + (' (dry run: no input is sent)' if args.dry_run else ''))
    last_frame_t = None
    started = run_start = None                  # when the fly's current stretch of play began
    in_control = False
    paused = False
    brain_s, looks, last_status = 0.0, 0, time.perf_counter()
    held, velocity = {}, np.zeros(2)
    last_control_t = None           # when the fly last had the controls
    death = None                    # restart in progress: {'next': time of next R, 'tries': n}
    deaths, survived = 0, []

    def let_go():
        nonlocal in_control
        if controller and in_control:
            controller.release_all()
        in_control = False

    try:
        while True:
            while not listener.hotkeys.empty():
                key = listener.hotkeys.get()
                if key == VK_F9:
                    paused = not paused
                    print(f"\n{'Paused' if paused else 'Resumed'} (F9)")
                elif key == VK_F10:
                    raise KeyboardInterrupt
            look = grabber.latest()
            if look is None:
                continue
            frame, t_frame, hwnd, focused, in_play = look
            if frame is None or paused:
                let_go()
                # Did the fly just die? The cursor appeared while it was playing, and
                # you didn't press anything (Esc to pause, say) just before
                if (death is None and last_control_t is not None and not paused and focused
                        and t_frame - last_control_t < 1.0
                        and listener.last_user_key < last_control_t - 0.5):
                    deaths += 1
                    survived.append(last_control_t - run_start)
                    print(f'\nDied after {survived[-1]:.1f} s (run {deaths}; '
                          f'mean {np.mean(survived):.1f} s)' + ('' if args.no_restart else '; restarting'))
                    death = None if args.no_restart else {'next': t_frame + RESTART_DELAY_S, 'tries': 0}
                last_control_t = None
                if death is not None and (paused or time.perf_counter() - listener.last_user_key < 1.0):
                    death = None                         # you're doing something; leave it to you
                if death is not None and focused and t_frame >= death['next']:
                    if death['tries'] < RESTART_TRIES:
                        if controller:
                            controller.tap(VK_R)
                        death['tries'] += 1
                        death['next'] = t_frame + RESTART_RETRY_S
                    else:
                        print('\nThe run did not restart after pressing R; start it yourself.')
                        death = None
                if last_frame_t is not None:            # left play: start fresh next time
                    retina.reset()
                    eyes.reset()
                    fly.reset()
                    last_frame_t = started = None
                status = ('paused (F9)' if paused
                          else 'died: restarting' + (' (dry run)' if args.dry_run else '') if death
                          else 'waiting: menu or cursor showing' if focused
                          else 'waiting: game not focused' if hwnd else 'waiting: game window not found')
            else:
                death = None
                dt = 0.0 if last_frame_t is None else t_frame - last_frame_t
                last_frame_t = t_frame
                if started is None:
                    started = run_start = t_frame

                raw = retina.see(frame, dt)
                t_brain = time.perf_counter()
                x = fly.look(eyes.rates(raw, dt), look_ms)
                brain_s += time.perf_counter() - t_brain
                looks += 1
                held, mouse = readout.act(x)
                velocity = mouse * mouse_scale / period          # counts per look -> per second

                user_active = time.perf_counter() - max(listener.last_user_motion,
                                                        listener.last_user_button) < TAKEOVER_S
                if t_frame - started < WARM_S or user_active:
                    let_go()                                     # brain settling, or you're playing
                    status = 'you have control' if user_active else 'fly warming up'
                else:
                    if controller:
                        controller.apply(held, velocity)
                    in_control = True
                    last_control_t = t_frame
                    status = 'fly playing' + (' (dry run)' if args.dry_run else '')

            if time.perf_counter() - last_status > 1.0:
                speed = looks * look_ms / 1000 / brain_s if brain_s else float('nan')
                keys = ''.join(n[0].upper() if held.get(n) else '.' for n in BUTTON_NAMES)
                print(f'\r{status:<32} keys {keys}  mouse {velocity[0]:+6.0f},{velocity[1]:+6.0f}/s  '
                      f'brain {speed:4.2f}x real time', end='', flush=True)
                last_status = time.perf_counter()
                brain_s, looks = 0.0, 0
    except KeyboardInterrupt:
        pass
    finally:
        if controller:
            controller.close()
        grabber.close()
        listener.close()
        print('\nStopped; every key released.')


if __name__ == '__main__':
    main()
