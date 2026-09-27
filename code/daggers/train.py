"""
Train the fly to play Devil Daggers from recordings of a person playing.

The connectome is never changed. Evolution (CMA-ES) tunes the eyes'
genome, i.e. how strongly each screen feature drives its visual neurons.
The readout from the brain's 1,409 output neurons to keys and mouse is
fitted by ridge regression.

Fitness of a genome: the fly watches a sample of recorded clips (the
prepared features of what the player saw), and a ridge readout from its
output neurons' firing rates to what the player did is scored by 2-fold
cross-validation across clips (mean R^2 over buttons and mouse axes). Every
genome in a generation sees the same clips with the same Poisson noise. One
worker runs one genome, so --popsize defaults to the number of workers.

Usage:
    python code/daggers/train.py evolve --workers 7 --hours 1           # this machine
    python code/daggers/train.py evolve --resume data/daggers/runs/<run>
    python code/daggers/train.py export data/daggers/runs/<run>         # -> <run>/policy.json
    python code/daggers/train.py controls data/daggers/runs/<run>       # vs blind, shuffled wiring, no brain

Needs prepared recordings (dataset.py); runs anywhere with numpy, scipy,
pandas, pyarrow, numba and cma.
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

from dataset import DATA, PREPARED, WARM_S, clips, load_prepared, load_stats, split
from eyes import BANDS, CHANNELS, GENE_NAMES, SELF_MOTION, Eyes, Genome
from fly import (FPS, OUTPUTS, DaggersFly, Readout, cv_score, keep_awake, latest_run,
                 save_policy)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_brain import load_connectome  # noqa: E402

RUNS_DIR = DATA / 'runs'

# Filled in each worker (before forking where possible, so workers share the connectome)
W = {}


# ============================================================================
# Workers
# ============================================================================

def shuffled(weights, seed=0):
    """Same synapses, targets relabelled at random: a brain with no FlyWire wiring."""
    w = weights.copy()
    w.indices = np.random.default_rng(seed).permutation(w.shape[0])[w.indices].astype(w.indices.dtype)
    return w


def init_worker(all_clips, scales, controls=False):
    if 'weights' not in W:
        W['weights'], W['flyid2i'] = load_connectome()
    W['clips'], W['scales'] = all_clips, scales
    W['fly'] = DaggersFly(W['weights'], W['flyid2i'])
    if controls and 'fly_shuffled' not in W:
        W['fly_shuffled'] = DaggersFly(shuffled(W['weights']), W['flyid2i'])


def watch(fly, eyes, clip, seed, variant='normal'):
    """The fly watches one clip. Returns (readout rates, targets) for the
    scored looks, and the eyes' rates (the no-brain control's input)."""
    fly.reset(seed)
    eyes.reset()
    n = len(clip['raw'])
    X = np.zeros((n, fly.n_readout), dtype=np.float32)
    D = np.zeros((n, 2 * BANDS * len(CHANNELS)), dtype=np.float32)
    for k in range(n):
        rates = eyes.rates(clip['raw'][k], float(clip['dt'][k]))
        D[k] = np.concatenate([rates[ch].ravel() for ch in CHANNELS])
        if variant == 'blind':
            rates = {ch: np.zeros((2, BANDS)) for ch in CHANNELS}
        if variant != 'nobrain':
            X[k] = fly.look(rates)
    warm = int(WARM_S * FPS)
    return X[warm:], clip['y'][warm:], D[warm:]


def with_history(D, tau_looks=4.0):
    """The eyes' rates now and smoothed, like the fly's two readout timescales."""
    keep = math.exp(-1 / tau_looks)
    slow = np.zeros_like(D)
    acc = np.zeros(D.shape[1])
    for k in range(len(D)):
        acc = keep * acc + (1 - keep) * D[k]
        slow[k] = acc
    return np.concatenate([D, slow], axis=1)


def evaluate(task):
    """Worker: one genome over a list of clips -> (score, per-output R^2, lambda)."""
    unit, clip_ids, seeds, variant = task
    eyes = Eyes(Genome.from_unit(np.asarray(unit)), W['scales'])
    fly = W['fly_shuffled'] if variant == 'shuffled' else W['fly']
    Xs, Ys = [], []
    for c, seed in zip(clip_ids, seeds):
        X, Y, D = watch(fly, eyes, W['clips'][c], seed, variant)
        Xs.append(with_history(D) if variant == 'nobrain' else X)
        Ys.append(Y)
    score, r2, lam = cv_score(Xs, Ys)
    return score, r2.tolist(), lam.tolist()


def collect(task):
    """Worker: one genome over clips -> per clip (active columns, their rates, targets)."""
    unit, clip_ids, seeds = task
    eyes = Eyes(Genome.from_unit(np.asarray(unit)), W['scales'])
    out = []
    for c, seed in zip(clip_ids, seeds):
        X, Y, _ = watch(W['fly'], eyes, W['clips'][c], seed)
        cols = np.flatnonzero(X.any(axis=0))
        out.append((cols, X[:, cols], Y))
    return out


def make_pool(workers, all_clips, scales, controls=False):
    """Fork where the OS allows, so workers share the parent's connectome."""
    if 'fork' in mp.get_all_start_methods():
        W['weights'], W['flyid2i'] = load_connectome()
        ctx = mp.get_context('fork')
    else:
        ctx = mp.get_context('spawn')
    return ctx.Pool(workers, initializer=init_worker, initargs=(all_clips, scales, controls))


def load_data():
    """All clips, the indices of the training and held-out ones, and dataset stats."""
    stats = load_stats()
    sessions = load_prepared()
    all_clips = clips(sessions, stats['mouse_scale'])
    train_ids, val_ids = split(list(range(len(all_clips))))
    if not train_ids:
        raise SystemExit(f'No training clips: record some play and run dataset.py ({PREPARED})')
    print(f'{len(all_clips)} clips ({stats["minutes"]:.1f} min): '
          f'{len(train_ids)} train, {len(val_ids)} held out')
    return all_clips, train_ids, val_ids, stats


# ============================================================================
# Evolution
# ============================================================================

def evolve(args):
    import cma
    all_clips, train_ids, val_ids, stats = load_data()
    val_ids = val_ids[:args.clips]
    if args.resume:
        run_dir = Path(args.resume)
        es = pickle.loads((run_dir / 'es.pkl').read_bytes())
        meta = json.loads((run_dir / 'meta.json').read_text())
        gen = meta['generation']
        print(f'Resuming {run_dir} at generation {gen}')
    else:
        run_dir = RUNS_DIR / time.strftime('%Y%m%d-%H%M%S')
        run_dir.mkdir(parents=True)
        x0 = (Genome.from_dict(json.loads(Path(args.start).read_text())['genes'])
              if args.start else Genome.default()).to_unit()
        popsize = args.popsize or max(args.workers, 8)
        opts = {'bounds': [0, 1], 'popsize': popsize, 'seed': args.seed, 'verbose': -9,
                'tolflatfitness': 10 ** 9, 'tolfun': 0, 'tolfunhist': 0}
        if not args.self_motion:
            opts['fixed_variables'] = {GENE_NAMES.index(f'{ch}_gain'): 0.0 for ch in SELF_MOTION}
        es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)
        gen = 0
        meta = {'args': vars(args), 'genes': GENE_NAMES, 'outputs': OUTPUTS, 'generation': 0,
                'best_fitness': None, 'feature_scales': stats['feature_scales'],
                'mouse_scale': stats['mouse_scale']}
        with open(run_dir / 'log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(['generation', 'best', 'mean', 'best_ever', 'val_mean_genome']
                                   + [f'r2_{o}' for o in OUTPUTS] + ['seconds'])
    print(f'Run directory: {run_dir}')

    pool = make_pool(args.workers, all_clips, stats['feature_scales'])
    t_stop = time.time() + args.hours * 3600
    best_ever = meta.get('best_fitness')
    rng = np.random.default_rng(args.seed + 7919 * gen)
    try:
        while time.time() < t_stop and not es.stop():
            t0 = time.time()
            pop = es.ask()
            ids = list(rng.choice(train_ids, size=min(args.clips, len(train_ids)), replace=False))
            seeds = [args.seed * 1_000_003 + gen * 1000 + k for k in range(len(ids))]  # shared by all genomes
            tasks = [(list(u), ids, seeds, 'normal') for u in pop]
            if gen % args.val_every == 0 and val_ids:
                tasks.append((list(es.result.xfavorite), val_ids, list(range(len(val_ids))), 'normal'))
            results = pool.map(evaluate, tasks, chunksize=1)
            val_score = results.pop()[0] if len(results) > len(pop) else float('nan')
            fits = np.array([r[0] for r in results])
            es.tell(pop, list(-fits))
            gen += 1

            i = int(np.argmax(fits))
            if best_ever is None or fits[i] > best_ever:
                best_ever = float(fits[i])
                (run_dir / 'best.json').write_text(json.dumps({
                    'genes': Genome.from_unit(np.asarray(pop[i])).as_dict(), 'fitness': best_ever,
                    'generation': gen, 'r2': dict(zip(OUTPUTS, results[i][1]))}, indent=2))
            (run_dir / 'mean.json').write_text(json.dumps({
                'genes': Genome.from_unit(es.result.xfavorite).as_dict(), 'generation': gen}, indent=2))
            (run_dir / 'es.pkl').write_bytes(pickle.dumps(es))
            meta.update(generation=gen, best_fitness=best_ever)
            (run_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
            row = ([gen, round(float(fits.max()), 4), round(float(fits.mean()), 4), round(best_ever, 4),
                    round(val_score, 4)] + [round(r, 3) for r in results[i][1]]
                   + [round(time.time() - t0, 1)])
            with open(run_dir / 'log.csv', 'a', newline='') as f:
                csv.writer(f).writerow(row)
            r2 = '  '.join(f'{o} {r:+.2f}' for o, r in zip(OUTPUTS, results[i][1]))
            val_txt = f'  val(mean genome) {val_score:+.3f}' if not math.isnan(val_score) else ''
            print(f'gen {gen:4d}  best {fits.max():+.3f}  mean {fits.mean():+.3f}  '
                  f'best ever {best_ever:+.3f}{val_txt}  ({time.time() - t0:.0f}s)\n'
                  f'          R^2 of the best: {r2}', flush=True)
    finally:
        pool.close()
    print(f'Done. Next: python code/daggers/train.py export {run_dir}')


# ============================================================================
# Export and controls
# ============================================================================

def genome_of(run_dir, which='best'):
    return Genome.from_dict(json.loads((Path(run_dir) / f'{which}.json').read_text())['genes'])


def export(args):
    """Fit the readout for the run's best genome on every training clip."""
    all_clips, train_ids, val_ids, stats = load_data()
    genome = genome_of(args.run, args.genome)
    unit = list(genome.to_unit())
    pool = make_pool(args.workers, all_clips, stats['feature_scales'])
    try:
        def gather(ids):
            batches = [ids[k::args.workers] for k in range(args.workers) if ids[k::args.workers]]
            tasks = [(unit, b, [10_000 + c for c in b]) for b in batches]
            return [clip for part in pool.map(collect, tasks, chunksize=1) for clip in part]
        print('The fly watches every training clip...')
        tr = gather(train_ids)
        va = gather(val_ids) if val_ids else []
    finally:
        pool.close()

    cols = np.unique(np.concatenate([c for c, _, _ in tr]))
    pos = {c: k for k, c in enumerate(cols)}

    def dense(parts):
        Xs = []
        for c, x, _ in parts:
            full = np.zeros((len(x), len(cols)), dtype=np.float32)
            keep = [k for k, cc in enumerate(c) if cc in pos]
            full[:, [pos[c[k]] for k in keep]] = x[:, keep]
            Xs.append(full)
        return Xs, [y for _, _, y in parts]

    Xtr, Ytr = dense(tr)
    score, r2_cv, lam = cv_score(Xtr, Ytr)
    readout = Readout.fit(np.concatenate(Xtr), np.concatenate(Ytr), lam)
    readout.cols = cols[readout.cols]              # indices into the full readout vector
    report = {'train_cv_r2': score, 'lambda': dict(zip(OUTPUTS, lam.tolist())),
              'train_cv_r2_per_output': dict(zip(OUTPUTS, np.round(r2_cv, 3).tolist())),
              'readout_neurons_used': int(len(readout.cols)),
              'train_looks': int(sum(len(y) for y in Ytr))}
    if va:
        Xva, Yva = dense(va)
        X, Y = np.concatenate(Xva), np.concatenate(Yva)
        pred = ((X[:, np.searchsorted(cols, readout.cols)] - readout.mean) / readout.std) @ readout.coef + readout.bias
        ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
        r2 = np.where(ss_tot > 0, 1 - ((Y - pred) ** 2).sum(0) / np.maximum(ss_tot, 1e-12), np.nan)
        report['heldout_r2_per_output'] = dict(zip(OUTPUTS, np.round(r2, 3).tolist()))
    out = Path(args.run) / 'policy.json'
    save_policy(out, genome, stats['feature_scales'], readout, stats['mouse_scale'], report)
    print(json.dumps(report, indent=1))
    print(f'Wrote {out}. Play it: python code/daggers/play.py --policy {out}')


def controls(args):
    """The genome on held-out clips, against a blind fly, a shuffled
    connectome, and no brain at all (the readout fitted straight to the eyes)."""
    all_clips, train_ids, val_ids, stats = load_data()
    ids = (val_ids or train_ids)[:args.clips]
    unit = list(genome_of(args.run, args.genome).to_unit())
    variants = ['normal', 'blind', 'shuffled', 'nobrain']
    pool = make_pool(min(args.workers, len(variants)), all_clips, stats['feature_scales'], controls=True)
    try:
        results = pool.map(evaluate, [(unit, ids, list(range(len(ids))), v) for v in variants],
                           chunksize=1)
    finally:
        pool.close()
    print(f'{args.run} ({args.genome}): cross-validated R^2 on {len(ids)} held-out clips')
    print('  ' + ' ' * 10 + '  mean  ' + ' '.join(f'{o:>8}' for o in OUTPUTS))
    for v, (score, r2, _) in zip(variants, results):
        print(f'  {v:10s} {score:+.3f}  ' + ' '.join(f'{r:+8.3f}' for r in r2))


def main():
    parser = argparse.ArgumentParser(description='Train the fly to play Devil Daggers')
    parser.add_argument('command', choices=['evolve', 'export', 'controls'])
    parser.add_argument('run', nargs='?', help="run directory (export, controls), or 'latest'")
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--hours', type=float, default=1.0)
    parser.add_argument('--clips', type=int, default=16, help='clips per genome per generation')
    parser.add_argument('--popsize', type=int, default=None, help='default: --workers (at least 8)')
    parser.add_argument('--sigma', type=float, default=0.2, help='CMA-ES step size, in unit coordinates')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--val-every', type=int, default=5, help='generations between held-out checks')
    parser.add_argument('--start', help="genome json to start from, or 'latest' for the newest "
                                        "run's best.json (default: eyes.Genome.default)")
    parser.add_argument('--resume', help="run directory to carry on, or 'latest'")
    parser.add_argument('--self-motion', action='store_true',
                        help='let evolution use the optic-flow channels (pan, scroll); '
                             'off by default because they can make the fly spin in play')
    parser.add_argument('--genome', default='best', choices=['best', 'mean'],
                        help='export/controls: which genome of the run')
    args = parser.parse_args()
    if args.resume == 'latest':
        args.resume = str(latest_run(RUNS_DIR, 'es.pkl'))
    if args.start == 'latest':
        args.start = str(latest_run(RUNS_DIR, 'best.json') / 'best.json')
    if args.run == 'latest':
        args.run = str(latest_run(RUNS_DIR, 'best.json'))
    keep_awake()
    if args.command == 'evolve':
        evolve(args)
    elif not args.run:
        parser.error(f'{args.command} needs a run directory')
    elif args.command == 'export':
        export(args)
    else:
        controls(args)


if __name__ == '__main__':
    main()
