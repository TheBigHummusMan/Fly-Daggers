"""
Evolve the fly's interface (cursor_fly.Genome) with CMA-ES so the fixed
connectome earns money in Scritchy Scratchy.

Each generation, every genome plays the same few seeded ScratchSim episodes;
fitness is the mean signed log of the money gained, minus a penalty for
clicks in forbidden places (title bar, settings). Workers are forked after the
connectome is loaded, so they share its memory.

Runs anywhere with numpy/scipy/pandas/pyarrow/cma/pillow (the Mac, a Vultr
CPU box, the GPU PC); --env real plays the real game on the Mac instead,
one episode at a time, restoring a save before each.

Usage:
    python code/scratch/evolve.py --workers 3 --hours 0.5 --seconds 10     # smoke run
    python code/scratch/evolve.py --workers 30 --hours 8                   # Vultr run
    python code/scratch/evolve.py --resume data/scratch/evolve/<run>       # carry on
    python code/scratch/evolve.py --eval best.json --controls              # vs blind / shuffled brain
    python code/scratch/evolve.py --env real --save start --start best.json --popsize 6 --seconds 60
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_brain import load_connectome  # noqa: E402
from cursor_fly import LOOK_MS, CursorFly, Genome, GENE_NAMES  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parents[2] / 'data' / 'scratch' / 'evolve'
FORBIDDEN_PENALTY = 0.5          # fitness per forbidden click
SHAPING = 0.5                    # fitness per ticket's worth of foil scratched (sim only)

# Filled in the parent before forking, shared copy-on-write by the workers
WEIGHTS = {}


def signed_log(d):
    return math.copysign(math.log1p(abs(d)), d)


def shuffled(weights, seed=0):
    """Same synapses, targets relabelled at random: a brain with no FlyWire wiring."""
    w = weights.copy()
    w.indices = np.random.default_rng(seed).permutation(w.shape[0])[w.indices].astype(w.indices.dtype)
    return w


def run_episode(env, fly, seconds, seed=None, save=None, on_look=None):
    """One episode; returns (money gained, stats)."""
    if hasattr(env, 'advance'):
        frame = env.reset(seed)
    else:
        frame = env.reset(save)
    x, y = env.start_position() if hasattr(env, 'start_position') else (0.5, 0.5)
    fly.reset(x=x, y=y)
    m0 = env.money()
    while fly.t_ms < seconds * 1000:
        x, y, button = fly.look(frame)
        env.act(x, y, button)
        if hasattr(env, 'advance'):
            env.advance(LOOK_MS)
        frame = env.frame()
        if on_look:
            on_look(fly, env)
    m1 = env.money()
    stats = dict(fly.stats, **getattr(env, 'stats', {}))
    if hasattr(env, 'foil_cleared'):
        stats['foil_cleared'] = env.foil_cleared()
    if m0 is None or m1 is None:
        return float('nan'), stats
    return m1 - m0, stats


def evaluate(task):
    """Worker: play one sim episode with one genome."""
    from sim import ScratchSim
    unit, seed, seconds, variant = task
    weights = WEIGHTS['shuffled' if variant == 'shuffled' else 'real']
    fly = CursorFly(weights, WEIGHTS['flyid2i'], Genome.from_unit(np.asarray(unit)),
                    seed=seed, blind=variant == 'blind')
    gained, stats = run_episode(ScratchSim(), fly, seconds, seed=seed)
    fit = (signed_log(gained) + SHAPING * stats.get('foil_cleared', 0.0)
           - FORBIDDEN_PENALTY * stats.get('forbidden_clicks', 0))
    return fit, gained, stats


def load_brain(controls=False):
    t0 = time.time()
    weights, flyid2i = load_connectome()
    WEIGHTS.update(real=weights, flyid2i=flyid2i)
    if controls:
        WEIGHTS['shuffled'] = shuffled(weights)
    print(f'Connectome loaded in {time.time() - t0:.1f}s ({weights.nnz:,} synapses)')


def cma_options(args):
    # Early generations often tie (nobody earns anything yet); don't let
    # CMA-ES stop on a flat fitness landscape
    return {'bounds': [0, 1], 'popsize': args.popsize, 'seed': args.seed, 'verbose': -9,
            'tolflatfitness': 10 ** 9, 'tolfun': 0, 'tolfunhist': 0}


def make_pool(workers):
    # fork, so workers inherit WEIGHTS without copying or reloading
    return mp.get_context('fork').Pool(workers)


# ============================================================================
# Evolution
# ============================================================================

def evolve(args):
    import cma
    if args.resume:
        run_dir = Path(args.resume)
        es = pickle.loads((run_dir / 'es.pkl').read_bytes())
        meta = json.loads((run_dir / 'meta.json').read_text())
        gen = meta['generation']
        print(f'Resuming {run_dir} at generation {gen}')
    else:
        run_dir = RUNS_DIR / time.strftime('%Y%m%d-%H%M%S')
        run_dir.mkdir(parents=True)
        x0 = (Genome.load(args.start) if args.start else Genome.start()).to_unit()
        es = cma.CMAEvolutionStrategy(x0, args.sigma, cma_options(args))
        gen = 0
        meta = {'args': vars(args), 'genes': GENE_NAMES, 'generation': 0, 'best_fitness': None}
        with open(run_dir / 'log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(['generation', 'best', 'mean', 'median', 'best_ever',
                                    'mean_gain', 'mean_clicks', 'mean_bought', 'mean_foil', 'seconds'])
    print(f'Run directory: {run_dir}')

    load_brain()
    pool = make_pool(args.workers) if args.workers > 1 else None
    t_stop = time.time() + args.hours * 3600
    best_ever = meta.get('best_fitness')
    while time.time() < t_stop and not es.stop():
        t0 = time.time()
        pop = es.ask()
        seeds = [args.seed * 100000 + gen * 100 + k for k in range(args.episodes)]   # shared by all genomes
        tasks = [(list(u), s, args.seconds, 'normal') for u in pop for s in seeds]
        results = pool.map(evaluate, tasks, chunksize=1) if pool else list(map(evaluate, tasks))
        fits = np.array([r[0] for r in results]).reshape(len(pop), len(seeds)).mean(axis=1)
        gains = np.array([r[1] for r in results])
        es.tell(pop, list(-fits))
        gen += 1

        i = int(np.argmax(fits))
        if best_ever is None or fits[i] > best_ever:
            best_ever = float(fits[i])
            Genome.from_unit(np.asarray(pop[i])).save(run_dir / 'best.json', fitness=best_ever, generation=gen)
        Genome.from_unit(es.result.xfavorite).save(run_dir / 'mean.json', generation=gen)
        (run_dir / 'es.pkl').write_bytes(pickle.dumps(es))
        meta.update(generation=gen, best_fitness=best_ever)
        (run_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
        stats = [r[2] for r in results]
        row = [gen, round(float(fits.max()), 4), round(float(fits.mean()), 4), round(float(np.median(fits)), 4),
               round(best_ever, 4), round(float(gains.mean()), 3),
               round(float(np.mean([s.get('clicks', 0) for s in stats])), 1),
               round(float(np.mean([s.get('bought', 0) for s in stats])), 1),
               round(float(np.mean([s.get('foil_cleared', 0) for s in stats])), 2), round(time.time() - t0, 1)]
        with open(run_dir / 'log.csv', 'a', newline='') as f:
            csv.writer(f).writerow(row)
        print(f'gen {gen:4d}  best {row[1]:+.3f}  mean {row[2]:+.3f}  best ever {row[4]:+.3f}  '
              f'gain ${row[5]:.1f}  clicks {row[6]}  bought {row[7]}  foil {row[8]}  ({row[9]:.0f}s)', flush=True)
    if pool:
        pool.close()
    print(f'Done. Best genome: {run_dir / "best.json"}')


# ============================================================================
# Controls
# ============================================================================

def eval_controls(args):
    load_brain(controls=args.controls)
    unit = list(Genome.load(args.eval).to_unit())
    variants = ['normal'] + (['blind', 'shuffled'] if args.controls else [])
    seeds = [10_000 + k for k in range(args.n_seeds)]
    tasks = [(unit, s, args.seconds, v) for v in variants for s in seeds]
    pool = make_pool(args.workers) if args.workers > 1 else None
    results = pool.map(evaluate, tasks, chunksize=1) if pool else list(map(evaluate, tasks))
    print(f'{args.eval}: {args.n_seeds} seeds x {args.seconds:.0f} s')
    for k, v in enumerate(variants):
        rs = results[k * len(seeds):(k + 1) * len(seeds)]
        fit = np.array([r[0] for r in rs])
        gain = np.array([r[1] for r in rs])
        print(f'  {v:9s} fitness {fit.mean():+.3f} ± {fit.std() / math.sqrt(len(fit)):.3f}   '
              f'money {gain.mean():+.1f} ± {gain.std() / math.sqrt(len(gain)):.1f}   '
              f'clicks {np.mean([r[2]["clicks"] for r in rs]):.0f}   '
              f'bought {np.mean([r[2].get("bought", 0) for r in rs]):.1f}')


# ============================================================================
# Real game (Mac)
# ============================================================================

def evolve_real(args):
    """Fine-tune on the real game: one episode per genome, save restored before each."""
    import cma
    from game_io import FlyStopped, RealGame, check_accessibility, check_screen_permission
    check_screen_permission()
    check_accessibility()
    run_dir = RUNS_DIR / time.strftime('real-%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True)
    load_brain()
    x0 = (Genome.load(args.start) if args.start else Genome.start()).to_unit()
    es = cma.CMAEvolutionStrategy(x0, args.sigma, cma_options(args))
    env = RealGame()
    t_stop = time.time() + args.hours * 3600
    gen, best_ever = 0, None
    try:
        while time.time() < t_stop:
            pop = es.ask()
            fits = []
            for u in pop:
                fly = CursorFly(WEIGHTS['real'], WEIGHTS['flyid2i'], Genome.from_unit(np.asarray(u)), seed=gen)
                gained, stats = run_episode(env, fly, args.seconds, save=args.save)
                fits.append(-10.0 if math.isnan(gained) else signed_log(gained))
                print(f'  genome {len(fits)}/{len(pop)}: money {gained:+.1f}  clicks {stats["clicks"]}', flush=True)
            fits = np.array(fits)
            es.tell(pop, list(-fits))
            gen += 1
            i = int(np.argmax(fits))
            if best_ever is None or fits[i] > best_ever:
                best_ever = float(fits[i])
                Genome.from_unit(np.asarray(pop[i])).save(run_dir / 'best.json', fitness=best_ever, generation=gen)
            (run_dir / 'es.pkl').write_bytes(pickle.dumps(es))
            print(f'gen {gen}  best {fits.max():+.3f}  mean {fits.mean():+.3f}', flush=True)
    except FlyStopped as e:
        print(f'Stopped: {e}')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='Evolve the fly interface for Scritchy Scratchy')
    parser.add_argument('--env', choices=['sim', 'real'], default='sim')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--hours', type=float, default=1.0)
    parser.add_argument('--seconds', type=float, default=60.0, help='brain seconds per episode')
    parser.add_argument('--episodes', type=int, default=2, help='sim episodes per genome')
    parser.add_argument('--popsize', type=int, default=16)
    parser.add_argument('--sigma', type=float, default=0.2, help='CMA-ES step size, in unit coordinates')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start', help='genome json to start from (default: start_genome.json, else hand-set)')
    parser.add_argument('--resume', help='run directory to carry on')
    parser.add_argument('--save', help='(real) snapshot name to restore before each episode')
    parser.add_argument('--eval', help='genome json to evaluate instead of evolving')
    parser.add_argument('--controls', action='store_true', help='with --eval: also blind and shuffled brain')
    parser.add_argument('--n-seeds', type=int, default=10)
    args = parser.parse_args()
    if args.eval:
        eval_controls(args)
    elif args.env == 'real':
        evolve_real(args)
    else:
        evolve(args)


if __name__ == '__main__':
    main()
