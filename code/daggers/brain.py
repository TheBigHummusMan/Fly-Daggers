"""
Event-driven whole-brain LIF model, fast enough to play a real-time game.

Same equations, parameters, and update order as fast_brain.FastBrain (and so
run_pytorch.TorchModel), but a neuron is only computed when it matters: when
a synaptic input arrives, when a Poisson input kicks it, or while it is
"hot", meaning its membrane might still cross threshold on its own. FastBrain
updates all 138,639 neurons every 0.1 ms step; this updates the few thousand
that anything is happening to.

Between updates a neuron follows the model's own per-step recurrence. With
u = v - vRest and no input, one step is

    u' = a u + f g,    g' = d g,    a = 1 - dt/tauMem,  f = dt/tauMem,  d = 1 - dt/tauSyn

which has the closed form, k steps later,

    g_k = d^k g,       u_k = a^k u + f g (a^k - d^k) / (a - d),

so a quiet neuron is brought up to date in one jump when it is next needed.
That changes only float rounding (this runs in float64, FastBrain in
float32). Leftover excitatory conductance can carry a neuron over threshold
after its input stops. Its free trajectory peaks at c P(u/c), with
c = f g / (a - d) and P a fixed function tabulated at start-up. While that
peak is above threshold, the neuron is stepped every step, like FastBrain
does; once it is below, the neuron can't spike without new input and goes
quiet.

Checked against FastBrain with identical Poisson input
(python code/daggers/brain.py --validate).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_brain import DT, MODEL_PARAMS  # noqa: E402

NO_HIT = np.iinfo(np.int64).max
POW_TABLE = 1 << 14              # steps of decay looked up rather than computed
QUEUE = 1 << 22                  # synaptic events in flight (~45k in a busy brain)
RHO_MAX = 32.0                   # peak table covers u/c in [0, RHO_MAX]
RHO_BINS = 4096

# Per-neuron state, one 64-byte row each, so touching a neuron costs one
# cache line. Step numbers are stored as float64, exact up to 2**53.
U, G, ACC, SINCE, REFRAC, STAMP, KICK = range(7)


def _aligned_rows(n, width=8, align=64):
    """Zeroed float64 (n, width) array whose rows start on cache-line boundaries."""
    raw = np.zeros(n * width + align // 8, dtype=np.float64)
    offset = (-raw.ctypes.data % align) // 8
    return raw[offset:offset + n * width].reshape(n, width)


def _peak_table(a_pow, d_pow):
    """Upper bounds on the free trajectory's peak, in units of c.

    With c = f g / (a - d) and rho = u / c, the peak over k >= 1 is
    c P(rho), P(rho) = max_k (rho + 1) a^k - d^k. P rises with rho, so entry
    j = P at the top of bin j bounds every rho in the bin. The last entry is
    P(0), which bounds rho <= 0; beyond the table, P(rho) <= rho + P(0).
    """
    rho = (np.arange(RHO_BINS) + 1) * (RHO_MAX / RHO_BINS)
    a, d = a_pow[1:], d_pow[1:]
    table = np.array([np.max((r + 1) * a - d) for r in rho])
    return np.append(table, np.max(a - d)) * (1 + 1e-9) + 1e-12


class EventBrain:
    """Whole-brain LIF network, advanced any number of DT steps at a time.

    Parameters
    ----------
    weights : scipy.sparse.csr_matrix
        Presynaptic-row weights from fast_brain.load_connectome().
    stim_indices : array-like
        Every neuron that may receive Poisson input (zero refractory period,
        as in FastBrain, Brian2's poi() and TorchModel's exc_indices).
    readout_indices : array-like
        Neurons whose spikes run() counts, in this order.
    seed : int, optional
        Seed for the Poisson input (numba's generator, one per process).
    """

    def __init__(self, weights, stim_indices, readout_indices=(), params=MODEL_PARAMS,
                 dt=DT, seed=None):
        p = params
        self.n = weights.shape[0]
        self.indptr = weights.indptr.astype(np.int64)
        self.indices = weights.indices.astype(np.int32)
        self.data = weights.data.astype(np.float64) * p['wScale']

        self.a = 1.0 - dt / p['tauMem']
        self.f = dt / p['tauMem']
        self.d = 1.0 - dt / p['tauSyn']
        self.theta = p['vThreshold'] - p['vRest']
        self.u_reset = p['vReset'] - p['vRest']
        self.kick = p['wScale'] * p['scalePoisson']
        # FastBrain's ring holds int(tDelay/dt) + 1 slots and is read one step
        # after it is written, so a spike reaches its targets that many + 1 steps later
        self.delay = int(p['tDelay'] / dt) + 2
        self.refrac = int(round(p['tRefrac'] / dt))
        self.ring_len = 1 << int(np.ceil(np.log2(self.delay + 1)))
        ks = np.arange(POW_TABLE)
        self.a_pow = self.a ** ks
        self.d_pow = self.d ** ks
        self.peak = _peak_table(self.a_pow, self.d_pow)
        self.p_scale = dt / 1000.0
        self.dt = dt

        self.stim = np.asarray(stim_indices, dtype=np.int64)
        self.is_stim = np.zeros(self.n, dtype=np.bool_)
        self.is_stim[self.stim] = True
        self.readout = np.asarray(readout_indices, dtype=np.int64)
        self.code = np.zeros(self.n, dtype=np.int64)     # 0 = not read out, k = readout[k - 1]
        self.code[self.readout] = np.arange(1, len(self.readout) + 1)
        self.p_hit = np.zeros(len(self.stim))
        self.seed = seed

        self.st = _aligned_rows(self.n)
        self.q_tgt = np.zeros(QUEUE, dtype=np.int32)      # synaptic events in flight (FIFO)
        self.q_w = np.zeros(QUEUE)
        self.q_pos = np.zeros(3, dtype=np.int64)          # head, tail, events dropped
        self.q_cnt = np.zeros(self.ring_len, dtype=np.int64)
        self.hot = np.zeros(self.n + 1, dtype=np.int64)   # [count, neurons...]
        self.proc = np.zeros(self.n, dtype=np.int64)
        self.spk = np.zeros(self.n, dtype=np.int64)
        self.next_hit = np.full(len(self.stim), NO_HIT, dtype=np.int64)
        self.reset()

    def reset(self):
        """Every neuron at rest, nothing in flight, t = 0."""
        self.st[:] = 0.0
        self.st[:, [REFRAC, STAMP, KICK]] = -1.0
        self.q_pos[:] = 0
        self.q_cnt[:] = 0
        self.hot[0] = 0
        self.t = 0
        self.n_spikes = 0
        if self.seed is not None:
            _seed(self.seed)
        _redraw(self.t, self.p_hit, self.next_hit)

    def set_rates(self, rates):
        """Set Poisson rates (Hz) for stim_indices, in the same order."""
        p_hit = np.minimum(np.asarray(rates, dtype=np.float64) * self.p_scale, 1.0)
        if not np.array_equal(p_hit, self.p_hit):
            self.p_hit = p_hit
            _redraw(self.t, self.p_hit, self.next_hit)

    @property
    def dropped(self):
        """Synaptic events lost to a full queue (only in runaway activity)."""
        return int(self.q_pos[2])

    def run(self, n_steps, hits=None, record=None):
        """Advance n_steps DT steps. Returns spike counts per readout neuron.

        hits = (steps, neurons) replaces the Poisson draws with explicit
        input kicks (sorted by step; used for validation). record = (steps,
        neurons, count) arrays collect every spike while they have room.
        """
        counts = np.zeros(len(self.readout) + 1, dtype=np.int64)
        if hits is None:
            hit_t = hit_i = np.zeros(0, dtype=np.int64)
            explicit = False
        else:
            hit_t, hit_i = (np.asarray(h, dtype=np.int64) for h in hits)
            explicit = True
        if record is None:
            record = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                      np.zeros(1, dtype=np.int64))
        self.n_spikes += _run(
            self.t, n_steps, self.st, self.q_tgt, self.q_w, self.q_pos, self.q_cnt, self.hot,
            self.stim, self.is_stim, self.next_hit, self.p_hit, explicit, hit_t, hit_i,
            self.indptr, self.indices, self.data, self.code, counts, self.proc, self.spk,
            record[0], record[1], record[2], self.a_pow, self.d_pow, self.peak,
            self.a, self.d, self.f, self.theta, self.u_reset, self.kick,
            self.delay, self.refrac)
        self.t += n_steps
        return counts[1:]


# ============================================================================
# Kernels
# ============================================================================

@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True, nogil=True)
def _redraw(t, p_hit, next_hit):
    """Next Poisson hit for each input neuron, from step t on. Per-step
    Bernoulli draws have geometric gaps, and they are memoryless, so redrawing
    whenever the rates change is exact."""
    for s in range(len(p_hit)):
        p = p_hit[s]
        next_hit[s] = t + np.random.geometric(p) - 1 if p > 0 else NO_HIT


@njit(cache=True, inline='always')
def _pow(table, base, k):
    return table[k] if k < len(table) else base ** k


@njit(cache=True, inline='always')
def _may_cross(u, g, theta, peak, amd, f):
    """Could the free trajectory from (u, g) still go above threshold?"""
    if g <= 0.0:
        return False                        # u only falls (or stays below 0)
    c = f * g / amd
    rho = u / c
    if rho <= 0.0:
        bound = peak[-1]
    elif rho < RHO_MAX:
        bound = peak[int(rho * (RHO_BINS / RHO_MAX))]
    else:
        bound = rho + peak[-1]
    return c * bound > theta


@njit(cache=True, inline='always')
def _touch(st, i, t, proc, m):
    """Queue neuron i for this step (once); returns the new queue length."""
    if st[i, STAMP] != t:
        st[i, STAMP] = t
        st[i, ACC] = 0.0
        proc[m] = i
        return m + 1
    return m


@njit(cache=True, nogil=True)
def _run(t0, n_steps, st, q_tgt, q_w, q_pos, q_cnt, hot,
         stim, is_stim, next_hit, p_hit, explicit, hit_t, hit_i,
         indptr, indices, data, code, counts, proc, spk,
         rec_t, rec_i, rec_n, a_pow, d_pow, peak,
         a, d, f, theta, u_reset, kick, delay, refrac):
    ring_mask = len(q_cnt) - 1
    q_mask = len(q_tgt) - 1
    amd = a - d
    total = 0
    hp = 0
    while hp < len(hit_t) and hit_t[hp] < t0:
        hp += 1
    for t in range(t0, t0 + n_steps):
        tf = float(t)
        m = 0
        # Who needs computing this step: neurons still hot from the last one...
        for q in range(hot[0]):
            m = _touch(st, hot[q + 1], tf, proc, m)
        # ...synaptic input arriving...
        slot = t & ring_mask
        head = q_pos[0]
        for e in range(q_cnt[slot]):
            k = (head + e) & q_mask
            i = q_tgt[k]
            m = _touch(st, i, tf, proc, m)
            st[i, ACC] += q_w[k]
        q_pos[0] = head + q_cnt[slot]
        q_cnt[slot] = 0
        # ...or a Poisson kick
        if explicit:
            while hp < len(hit_t) and hit_t[hp] == t:
                i = hit_i[hp]
                m = _touch(st, i, tf, proc, m)
                st[i, KICK] = tf
                hp += 1
        else:
            for s in range(len(stim)):
                if next_hit[s] == t:
                    i = stim[s]
                    m = _touch(st, i, tf, proc, m)
                    st[i, KICK] = tf
                    next_hit[s] = t + np.random.geometric(p_hit[s])

        n_hot = 0
        n_spk = 0
        for q in range(m):
            i = proc[q]
            ui = st[i, U]
            gi = st[i, G]
            k = t - np.int64(st[i, SINCE])
            if k > 0 and (ui != 0.0 or gi != 0.0):
                ak = _pow(a_pow, a, k)
                dk = _pow(d_pow, d, k)
                ui = ak * ui + f * gi * (ak - dk) / amd
                gi = dk * gi
            arriving = st[i, ACC]
            if tf <= st[i, REFRAC]:
                arriving = 0.0
            if st[i, KICK] == tf:
                ui += kick
            # FastBrain's step: membrane from the old conductance, then conductance
            u2 = a * ui + f * gi
            g2 = d * gi + arriving
            if u2 > theta:
                u2 = u_reset
                g2 = 0.0
                # Input arriving in the next refrac steps is dropped (Poisson
                # input neurons have no refractory period, as in FastBrain)
                if not is_stim[i]:
                    st[i, REFRAC] = tf + refrac
                spk[n_spk] = i
                n_spk += 1
                counts[code[i]] += 1
                r = rec_n[0]
                if r < len(rec_t):
                    rec_t[r] = t
                    rec_i[r] = i
                    rec_n[0] = r + 1
            elif abs(u2) < 1e-12 and abs(g2) < 1e-12:
                u2 = 0.0
                g2 = 0.0
            st[i, U] = u2
            st[i, G] = g2
            st[i, SINCE] = tf + 1.0
            if _may_cross(u2, g2, theta, peak, amd, f):
                n_hot += 1
                hot[n_hot] = i
        hot[0] = n_hot

        # This step's spikes reach their targets delay steps from now
        tail = q_pos[1]
        added = 0
        for q in range(n_spk):
            j = spk[q]
            lo, hi = indptr[j], indptr[j + 1]
            if tail + added + (hi - lo) - q_pos[0] > len(q_tgt):
                q_pos[2] += hi - lo
                continue
            for kk in range(lo, hi):
                k = (tail + added) & q_mask
                q_tgt[k] = indices[kk]
                q_w[k] = data[kk]
                added += 1
        q_pos[1] = tail + added
        q_cnt[(t + delay) & ring_mask] = added
        total += n_spk
    return total


# ============================================================================
# Validation and timing
# ============================================================================

def _demo_inputs(flyid2i):
    """The Fly Daggers input neurons, and a busy mix of rates for them."""
    import pandas as pd
    df = pd.read_csv(Path(__file__).resolve().parents[2] / 'data' / 'daggers_neurons.csv')
    df = df[~df.role.str.startswith('readout')]
    stim = np.array([flyid2i[f] for f in df.flywire_id], dtype=np.int64)
    rates = np.zeros(len(df))
    for role, hz in dict(object=150, loom=60, bustle=80, pan=100, scroll=40).items():
        rates[((df.role == role) & (df.side == 'left')).to_numpy()] = hz
    rates[((df.role == 'object') & (df.side == 'right')).to_numpy()] = 40
    return stim, rates


def validate(steps=3000, seed=0):
    """Same Poisson kicks into FastBrain and EventBrain; compare every spike."""
    from fast_brain import FastBrain, load_connectome
    weights, flyid2i = load_connectome()
    stim, rates = _demo_inputs(flyid2i)
    rng = np.random.default_rng(seed)
    kicks = [stim[rng.random(len(stim)) < rates * DT / 1000] for _ in range(steps)]

    ref = FastBrain(weights, stim)
    ref_spikes = set()
    for t in range(steps):
        ref_spikes.update((t, int(i)) for i in ref.step(poisson_idx=kicks[t]))

    brain = EventBrain(weights, stim)
    hit_t = np.concatenate([np.full(len(k), t) for t, k in enumerate(kicks)])
    hit_i = np.concatenate(kicks)
    rec = (np.zeros(10 ** 7, dtype=np.int64), np.zeros(10 ** 7, dtype=np.int64),
           np.zeros(1, dtype=np.int64))
    brain.run(steps, hits=(hit_t, hit_i), record=rec)
    ev_spikes = set(zip(rec[0][:rec[2][0]].tolist(), rec[1][:rec[2][0]].tolist()))

    both = len(ref_spikes & ev_spikes)
    diff = ref_spikes ^ ev_spikes
    first = min(t for t, _ in diff) if diff else None
    print(f'{steps} steps ({steps * DT:.0f} ms): FastBrain {len(ref_spikes)} spikes, '
          f'EventBrain {len(ev_spikes)}, identical {both} '
          f'({both / max(len(ref_spikes | ev_spikes), 1):.4%} of all); '
          f'first difference at step {first}')
    return ref_spikes, ev_spikes


def bench(seconds=2.0):
    """Brain speed under the busy input mix, as a multiple of real time."""
    from fast_brain import load_connectome
    weights, flyid2i = load_connectome()
    stim, rates = _demo_inputs(flyid2i)
    brain = EventBrain(weights, stim, seed=0)
    brain.set_rates(rates)
    brain.run(2000)                                   # compile and settle
    n = int(seconds * 1000 / DT)
    t = time.perf_counter()
    brain.run(n)
    wall = time.perf_counter() - t
    speed = seconds / wall
    print(f'EventBrain: {wall / n * 1e6:.1f} us/step, {speed:.2f}x real time, '
          f'{brain.n_spikes / (brain.t * DT):.0f} spikes/ms')
    return speed


def main():
    parser = argparse.ArgumentParser(description='Check or time the event-driven brain')
    parser.add_argument('--validate', action='store_true', help='compare spikes with FastBrain')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    if args.validate:
        validate(args.steps)
    bench()


if __name__ == '__main__':
    main()
