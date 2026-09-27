"""
Event-driven NumPy version of the PyTorch LIF model for closed-loop use.

Implements exactly the same equations, parameters, and update order as
the Shiu et al. PyTorch model (TorchModel; Shiu et al. LIF + alpha synapse + delay +
refractory period), but instead of a 138k x 138k sparse matmul every 0.1 ms
step, the recurrent input is gathered only from the handful of neurons that
spiked. That makes it fast enough on a laptop CPU to drive an interactive
game (see code/daggers), where the stimulus changes every few milliseconds.

Unlike TorchModel, Poisson input rates can be changed between steps, which is
what lets the game feed sensory input into the connectome in real time.
"""

from collections import deque

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
path_comp = DATA_DIR / '2025_Completeness_783.csv'     # FlyWire v783 neuron list
path_con = DATA_DIR / '2025_Connectivity_783.parquet'  # FlyWire v783 synapses

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

# ============================================================================
# Model Parameters (identical to the Shiu et al. PyTorch model)
# ============================================================================

MODEL_PARAMS = {
    'tauSyn': 5.0,        # ms
    'tDelay': 1.8,        # ms
    'v0': -52.0,          # mV
    'vReset': -52.0,      # mV
    'vRest': -52.0,       # mV
    'vThreshold': -45.0,  # mV
    'tauMem': 20.0,       # ms
    'tRefrac': 2.2,       # ms
    'scalePoisson': 250,
    'wScale': 0.275,
}

DT = 0.1  # ms


def load_connectome(comp_path=path_comp, conn_path=path_con):
    """Load the FlyWire connectome as a presynaptic-row CSR matrix.

    Returns (weights, flyid2i) where weights[pre, post] is the signed synapse
    count ('Excitatory x Connectivity') and flyid2i maps FlyWire IDs to row
    indices.
    """
    df_comp = pd.read_csv(comp_path, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    n = len(df_comp)

    table = pq.read_table(
        conn_path,
        columns=['Presynaptic_Index', 'Postsynaptic_Index',
                 'Excitatory x Connectivity'],
    )
    pre = table.column('Presynaptic_Index').to_numpy()
    post = table.column('Postsynaptic_Index').to_numpy()
    w = table.column('Excitatory x Connectivity').to_numpy().astype(np.float32)

    weights = sparse.csr_matrix((w, (pre, post)), shape=(n, n), dtype=np.float32)
    weights.sum_duplicates()
    return weights, flyid2i


class FastBrain:
    """Whole-brain LIF network stepped one DT at a time.

    Parameters
    ----------
    weights : scipy.sparse.csr_matrix
        Presynaptic-row weight matrix from load_connectome().
    stim_indices : array-like
        Every neuron that may ever receive Poisson input. Like Brian2's
        poi() and TorchModel's exc_indices, these get a zero refractory
        period.
    seed : int, optional
        Seed for the Poisson input generator.
    accel : bool, optional
        Use the numba kernels (default: when numba is installed). They give
        bit-identical spikes faster, by fusing the per-neuron updates into
        one pass and clearing only the delay-ring entries that were written.
    """

    def __init__(self, weights, stim_indices=(), params=MODEL_PARAMS, dt=DT,
                 seed=None, accel=None):
        self.accel = HAVE_NUMBA if accel is None else accel and HAVE_NUMBA
        self.n = weights.shape[0]
        self.indptr = weights.indptr
        self.indices = weights.indices
        self.data = weights.data

        p = params
        self.syn_decay = np.float32(1 - dt / p['tauSyn'])
        self.mem_factor = np.float32(dt / p['tauMem'])
        self.v0 = np.float32(p['v0'])
        self.v_rest = np.float32(p['vRest'])
        self.v_reset = np.float32(p['vReset'])
        self.v_th = np.float32(p['vThreshold'])
        self.w_scale = np.float32(p['wScale'])
        self.poisson_kick = np.float32(p['wScale'] * p['scalePoisson'])
        self.p_scale = dt / 1000.0
        self.dt = dt

        # Same integer conversions as AlphaSynapse / AlphaLIF
        self.ring_len = int(p['tDelay'] / dt) + 1
        self.refrac_steps = int(round(p['tRefrac'] / dt))

        self.stim_indices = np.asarray(stim_indices, dtype=np.int64)
        self.is_stim = np.zeros(self.n, dtype=bool)
        self.is_stim[self.stim_indices] = True
        self.rates = np.zeros(len(self.stim_indices), dtype=np.float64)

        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Return every neuron to rest and clear in-flight synaptic input."""
        self.v = np.full(self.n, self.v0, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.ring = np.zeros((self.ring_len, self.n), dtype=np.float32)
        self.spike_idx = np.empty(0, dtype=np.int64)
        self.recent = deque(maxlen=self.refrac_steps)
        self.t = 0
        self._tmp = np.empty(self.n, dtype=np.float32)
        # numba path: which ring entries each slot holds, so only those are cleared
        self._spike_buf = np.empty(self.n, dtype=np.int64)
        self._touched_buf = np.empty(self.n, dtype=np.int64)
        self._mark = np.full(self.n, -1, dtype=np.int64)
        self._slot_touched = [np.empty(0, dtype=np.int64) for _ in range(self.ring_len)]

    def set_rates(self, rates):
        """Set Poisson rates (Hz) for stim_indices, in the same order."""
        self.rates[:] = rates

    def step(self, poisson_idx=None):
        """Advance one DT. Returns indices of neurons that spiked.

        poisson_idx overrides the random Poisson draw with an explicit set of
        neurons that receive an input kick this step (used for validation).
        """
        if self.accel:
            return self._step_accel(poisson_idx)
        prev = self.spike_idx
        self.recent.append(prev)

        # --- Refractory mask (AlphaLIF.forward) --------------------------
        # A neuron is refractory for refrac_steps steps after it spikes;
        # stimulated neurons have a zero refractory period.
        arriving = self.ring[self.t % self.ring_len]
        if self.recent:
            refractory = np.concatenate(self.recent)
            refractory = refractory[~self.is_stim[refractory]]
            arriving[refractory] = 0.0

        # --- Poisson input (PoissonSpikeGenerator) -----------------------
        if poisson_idx is None:
            hits = self.rng.random(len(self.rates)) < self.rates * self.p_scale
            poisson_idx = self.stim_indices[hits]
        if len(poisson_idx):
            self.v[poisson_idx] += self.poisson_kick

        # --- Membrane update uses the pre-update conductance (LIFNeuron) -
        tmp = self._tmp
        np.subtract(self.v, self.v_rest, out=tmp)
        np.subtract(self.g, tmp, out=tmp)
        tmp *= self.mem_factor
        self.v += tmp

        # --- Conductance update (AlphaSynapse) ---------------------------
        self.g *= self.syn_decay
        self.g += arriving

        # --- Spikes and reset --------------------------------------------
        spikes = np.flatnonzero(self.v > self.v_th)
        if len(spikes):
            self.v[spikes] -= self.v[spikes] - self.v_reset
            self.g[spikes] = 0.0

        # --- Queue recurrent input from the previous step's spikes -------
        # Read-then-write on the same ring slot reproduces the torch.roll
        # delay buffer exactly.
        arriving[:] = 0.0
        for j in prev:
            lo, hi = self.indptr[j], self.indptr[j + 1]
            arriving[self.indices[lo:hi]] += self.data[lo:hi]
        if len(prev):
            arriving *= self.w_scale

        # --- Flush negligible conductances -------------------------------
        # Below 1e-20 mV a conductance cannot move v (float32 resolution near
        # -52 mV is ~4e-6 mV), so zeroing it leaves every spike unchanged. Left
        # alone, decaying conductances become denormal floats, which slow
        # every dense op several-fold on x86 CPUs.
        if self.t % 100 == 0:
            self.g[np.abs(self.g) < 1e-20] = 0.0

        self.spike_idx = spikes
        self.t += 1
        return spikes

    def _step_accel(self, poisson_idx=None):
        """step() with numba kernels; same operations in the same order."""
        prev = self.spike_idx
        self.recent.append(prev)
        slot = self.t % self.ring_len
        arriving = self.ring[slot]
        if self.recent:
            refractory = np.concatenate(self.recent)
            refractory = refractory[~self.is_stim[refractory]]
            arriving[refractory] = 0.0

        if poisson_idx is None:
            hits = self.rng.random(len(self.rates)) < self.rates * self.p_scale
            poisson_idx = self.stim_indices[hits]
        if len(poisson_idx):
            self.v[poisson_idx] += self.poisson_kick

        n = _integrate(self.v, self.g, arriving, self.v_rest, self.mem_factor,
                       self.syn_decay, self.v_th, self.v_reset, self._spike_buf)
        spikes = self._spike_buf[:n].copy()

        # Clear only what this slot held, then queue the previous step's spikes
        arriving[self._slot_touched[slot]] = 0.0
        m = _deliver(prev, self.indptr, self.indices, self.data, arriving,
                     self.w_scale, self._mark, self.t, self._touched_buf)
        self._slot_touched[slot] = self._touched_buf[:m].copy()

        if self.t % 100 == 0:
            self.g[np.abs(self.g) < 1e-20] = 0.0

        self.spike_idx = spikes
        self.t += 1
        return spikes


if HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def _integrate(v, g, arriving, v_rest, mem_factor, syn_decay, v_th, v_reset, spikes):
        """Membrane and conductance update for every neuron, then threshold
        and reset. The first loop has no branches, so it vectorizes; spikes
        are rare, so the second loop's branch is predictable."""
        for i in range(v.size):
            vi = v[i]
            gi = g[i]
            v[i] = vi + (gi - (vi - v_rest)) * mem_factor
            g[i] = gi * syn_decay + arriving[i]
        n = 0
        for i in range(v.size):
            if v[i] > v_th:
                v[i] = v[i] - (v[i] - v_reset)
                g[i] = 0.0
                spikes[n] = i
                n += 1
        return n

    @njit(cache=True, nogil=True)
    def _deliver(prev, indptr, indices, data, arriving, w_scale, mark, stamp, touched):
        """Add each spiking neuron's outgoing weights to arriving, then scale
        every touched entry once (as arriving *= w_scale would)."""
        m = 0
        for j in prev:
            for k in range(indptr[j], indptr[j + 1]):
                t = indices[k]
                if mark[t] != stamp:
                    mark[t] = stamp
                    touched[m] = t
                    m += 1
                arriving[t] += data[k]
        for q in range(m):
            arriving[touched[q]] *= w_scale
        return m
