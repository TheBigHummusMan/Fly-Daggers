"""
Let the fly play the real Scritchy Scratchy with an evolved (or hand-set)
genome. The fly owns the mouse while it plays: move the mouse yourself, or
into a screen corner, to stop it.

Usage:
    python code/scratch/play.py                                   # hand-set genome
    python code/scratch/play.py --genome data/scratch/evolve/<run>/best.json
    python code/scratch/play.py --genome best.json --blind        # control: fly sees and tastes nothing
    python code/scratch/play.py --save start --seconds 120        # restore a snapshot first, then 120 brain-s
    python code/scratch/play.py --sim --genome best.json          # same, in ScratchSim (no mouse)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_brain import load_connectome  # noqa: E402
from cursor_fly import CursorFly, Genome  # noqa: E402
from evolve import run_episode  # noqa: E402
from game_io import FlyStopped, RealGame, check_accessibility, check_screen_permission  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='The fly plays Scritchy Scratchy')
    parser.add_argument('--genome', help='genome json (default: data/scratch/start_genome.json from calibrate.py)')
    parser.add_argument('--seconds', type=float, default=3600.0, help='brain seconds to play')
    parser.add_argument('--blind', action='store_true', help='no sensory input (control)')
    parser.add_argument('--save', help='snapshot to restore before playing (see game_io.py --snapshot)')
    parser.add_argument('--sim', action='store_true', help='play ScratchSim instead of the real game')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    genome = Genome.load(args.genome) if args.genome else Genome.start()
    print('Loading the fly brain (138,639 neurons)...')
    weights, flyid2i = load_connectome()
    fly = CursorFly(weights, flyid2i, genome, seed=args.seed, blind=args.blind)

    if args.sim:
        from sim import ScratchSim
        env = ScratchSim()
    else:
        check_screen_permission()
        if not check_accessibility():
            sys.exit(1)
        env = RealGame()
        print('The fly takes the mouse in 3 s. Move the mouse (or slam it into a corner) to stop it.')
        time.sleep(3)

    t_wall, last = time.time(), [0.0]

    def on_look(fly, env):
        now = time.time()
        if now - last[0] >= 2.0:
            last[0] = now
            m = env.money()
            speed = fly.t_ms / 1000 / max(now - t_wall, 1e-6)
            print(f'[{fly.t_ms / 1000:7.1f} s brain, {speed:.2f}x]  money {m}  '
                  f'clicks {fly.stats["clicks"]}  at ({fly.x:.2f}, {fly.y:.2f})\n  '
                  + fly.status().replace('\n', '\n  '), flush=True)

    try:
        if args.sim:
            gained, stats = run_episode(env, fly, args.seconds, seed=args.seed, on_look=on_look)
        else:
            gained, stats = run_episode(env, fly, args.seconds, save=args.save, on_look=on_look)
        print(f'\nMoney gained: {gained:+.1f}   {stats}')
    except KeyboardInterrupt:
        print('\nStopped.')
    except FlyStopped as e:
        print(f'\nStopped: {e}')
    finally:
        if hasattr(env, 'close'):
            env.close()


if __name__ == '__main__':
    main()
