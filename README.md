# Fly Daggers: a fruit fly's brain plays Devil Daggers

A whole-brain model of the adult fruit fly learns to play
[Devil Daggers](https://store.steampowered.com/app/422970/Devil_Daggers/) on
Windows, from recordings of a person playing. The brain is built from the
[FlyWire](https://flywire.ai/) connectome: 138,639 neurons and about 15 million
connections, as a leaky integrate-and-fire network. The wiring is never
changed. Training only tunes how the game drives the fly's eyes and how its
output neurons become keys and mouse movement.

**To run it, see [code/daggers/README.md](code/daggers/README.md).** In short,
on Windows: install 64-bit Python 3.13, then double-click `daggers\setup.bat`
once, and afterwards `record.bat`, `train.bat`, and `play.bat`.

```
record  ->  train  ->  play
```

## How it works

```
game frame ──> Retina (fixed features) ──> Eyes (15 evolved genes)
           ──> 772 visual input neurons ──> 138k-neuron connectome (unchanged)
           ──> 1,409 descending + motor neurons ──> Readout (fitted) ──> keys, mouse
```

| Game | Input neurons | |
|---|---|---|
| Bright or red things, weighted by how far off-centre they are | LC10a | 4 strips per eye |
| A sudden rise in how much of a strip is things | LC4 + LPLC2 | 4 strips per eye |
| Motion that keeps going, after removing the view's own turning | LC9 | 4 strips per eye |
| Redness (gems, some enemies) | 21 sugar GRNs | |
| The view sliding sideways or up and down | HS + H2, VS | off by default |
| **Output** | **1,409 descending + motor neurons** | **Readout: W, A, S, D, jump, both mouse buttons, mouse x and y** |

- **Eyes.** The left half of the game window is the fly's left eye. Each eye
  is cut into 4 vertical strips, and each strip drives its own quarter of a
  visual population. Those quarters were chosen from the connectome so their
  downstream pathways differ as much as possible (`make_neurons.py`). The
  FlyWire annotations don't say where each neuron looks, so whether these
  quarters match the neurons' real receptive fields is unknown. The fly sees
  only these features, never game state.
- **Self-motion.** These channels are off by default. A readout fitted to a
  player's smooth turning can learn "the view is sliding, so keep turning",
  and in closed loop that makes the fly spin.
- **Training.** CMA-ES evolves the eyes' 15 genes: how strongly each feature
  drives its neurons. A genome's fitness is how much of what the player did
  (cross-validated R²) a ridge readout can predict from the fly's output
  neurons while it watches the recordings. Each fifth clip is held out.
  `train.py controls` compares the trained fly with a blind fly, shuffled
  wiring, and no brain at all; if it doesn't beat these, the connectome isn't
  adding anything.
- **Recording.** 20 frames a second from the game window at 128×72, with the
  keys held and raw mouse movement until the next frame. It records only while
  the game has focus and a run is on. The player's hand and the HUD are masked
  out. Stretches where the capture froze (exclusive fullscreen) are dropped.
- **Playing.** The fly presses keys and moves the mouse through SendInput,
  only while the game is focused and in a run. It hands control to you when
  you touch the mouse or keys; F9 pauses it and F10 stops it. After dying, it
  presses R to start the next run and prints how long each run lasted.

### Status

On about an hour of one player's recordings, the signal is small. The eyes'
features predict about 2.5% of the variation in the player's inputs (held-out
R² 0.025), and the fly passes on part of that. In an early test game, the
trained fly walked straight off the arena: it holds W about as often as the
player did, and it can't see the edge. The next steps are more training, then
recording corrections while the fly plays.

## The brain simulator

`code/daggers/brain.py` (`EventBrain`) runs the model of
[Shiu et al. (Nature 2024)](https://www.nature.com/articles/s41586-024-07763-9)
with the same equations, parameters, and update order as `code/fast_brain.py`.
That file is a NumPy port of their PyTorch version, itself checked against the
original Brian2 model. `EventBrain` is event-driven per neuron. A neuron is
computed only when input reaches it, or while its leftover conductance could
still carry it over threshold on its own. In between, it jumps ahead with the
closed form of the model's own per-step recurrence.

- **Accuracy.** With identical Poisson input it produces the same spikes as
  `FastBrain`, until float rounding first puts a membrane on the other side
  of threshold (float64 here, float32 there), by under 0.001 mV
  (`brain.py --validate`).
- **Speed.** On a laptop CPU (Ryzen 7 7730U) under the busiest input it runs
  at about 0.85× real time, 2.2× faster than `FastBrain`. During an actual
  Devil Daggers run it ran at about 1.5× real time.

It needs only numpy, scipy, pandas, pyarrow, and numba: no GPU.

## Project structure

```
Fly-Daggers/
├── daggers/                    # double-click launchers: setup, record, train, play, check
├── code/
│   ├── fast_brain.py           # NumPy LIF model (reference for EventBrain; loads the connectome)
│   └── daggers/
│       ├── README.md           # how to run everything
│       ├── brain.py            # EventBrain: event-driven LIF model, real-time capable
│       ├── eyes.py             # Retina (fixed features) and Eyes (evolved genome)
│       ├── fly.py              # neuron groups, brain wrapper, ridge readout, policy files
│       ├── winio.py            # Windows: window capture, raw input, SendInput
│       ├── record.py           # record a person playing
│       ├── dataset.py          # recordings -> prepared features -> training clips
│       ├── train.py            # CMA-ES evolution, export, controls
│       ├── play.py             # the trained fly plays
│       ├── make_neurons.py     # builds data/daggers_neurons.csv
│       └── requirements.txt
├── data/
│   ├── 2025_Completeness_783.csv       # FlyWire v783 neuron list (3.4 MB)
│   ├── 2025_Connectivity_783.parquet   # FlyWire v783 connectivity (97 MB)
│   ├── daggers_neurons.csv             # input neurons (with strips) and readout neurons
│   ├── fly_screen_neurons.csv          # the input groups daggers_neurons.csv starts from
│   └── daggers/
│       ├── recordings/                 # your recorded play (not in git)
│       ├── prepared/                   # features for training (not in git)
│       └── runs/<run>/                 # training runs; policy.json and best.json are in git
└── tests/test_daggers.py
```

## Credits and data

- **Model:** Shiu et al., *A Drosophila computational brain model reveals
  sensorimotor processing*, Nature (2024).
- **Connectome:** FlyWire, public release v783.
- **Cell types and sides:** the FlyWire annotations of Schlegel et al. (2024),
  [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).

This project began as a benchmark of that model across simulators (Brian2,
Brian2CUDA, PyTorch, NEST GPU, GeNN, Brian2GeNN), with other closed-loop
demos (Fly Pong, Fly Screen, Fly Scratch). Those were removed to keep this
repository to Fly Daggers; they are in the git history before this cleanup.

## License

Except where otherwise noted, this project is licensed under the GNU General
Public License version 2 or any later version (`GPL-2.0-or-later`). See
[LICENSE](LICENSE).
