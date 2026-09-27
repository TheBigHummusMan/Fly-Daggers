# Emulation of the *Drosophila Fly* Brain

Whole-brain leaky integrate-and-fire model of the adult fruit fly, built from the
[FlyWire](https://flywire.ai/) connectome (~138k neurons, ~5M synapses).
Activate and silence arbitrary neurons; observe downstream spike propagation.

Based on the paper
[*A leaky integrate-and-fire computational model based on the connectome of the
entire adult Drosophila brain reveals insights into sensorimotor processing*](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1)
(Shiu et al.).

## Usage

With this computational model, one can manipulate the neural activity of a set of _Drosophila_ neurons.
The output of the model is the spike times and rates of all affected neurons.

Two types of manipulations are currently implemented:
- *Activation*:
Neurons can be activated at a fixed frequency to model optogenetic activation.
This triggers Poisson spiking in the target neurons. 
Two sets of neurons with distinct frequencies can be defined.
- *Silencing*:
In addition to activation, a different set of neurons can be silenced to model optogenetic silencing.
This sets all synaptic connections to and from those neurons to zero.

The entrypoint is [main.py](main.py), which parses CLI arguments and calls
[code/benchmark.py](code/benchmark.py) -- the central orchestrator that dispatches
to framework-specific runners:
[run_brian2_cuda.py](code/run_brian2_cuda.py),
[run_pytorch.py](code/run_pytorch.py),
[run_nestgpu.py](code/run_nestgpu.py), and
[run_genn.py](code/run_genn.py). The optional Brian2GeNN backend lives in
[run_brian2_genn.py](code/run_brian2_genn.py) and uses a separate conda
environment because Brian2GeNN 1.7.0 pins Brian2<2.6 while Brian2CUDA uses
Brian2 2.8.0.

```bash
# Run the 5 main-environment frameworks with default durations (0.1s–1000s)
# and trials (1,4,8,16,32)
python main.py

# Specific durations and trial count
python main.py --t_run 0.1 1 10 --n_run 1

# Single framework
python main.py --nestgpu --t_run 1 --n_run 1
python main.py --genn --t_run 1 --n_run 1
python main.py --brian2genn --t_run 1 --n_run 1

# Combine frameworks
python main.py --brian2-cpu --pytorch --t_run 0.1 1 --n_run 1 4 8 16 32

# Five-round Nature-paper benchmark suite
# Uses the March grid: t_run=(0.1,1,10,100), n_run=(1,4,8,16,32), 5 core backends
python main.py --paper --run-label nature_2026_07

# Add Brian2GeNN as the 6th framework from the brain-fly-brian2genn environment
python main.py --brian2genn --paper --run-label nature_2026_07
```

Results are incrementally saved to `data/benchmark-results.csv` as each
benchmark completes, with separate columns for setup time (loading, compilation)
and simulation time (the always-on cost). For repeated paper runs, the CSV keeps
the original March rows and appends new rows keyed by `run_label` and `round`;
the corresponding spike parquet path is recorded in `spike_path`.

Spike timing exports are written to parquet outside the timed simulation section
so file I/O does not contaminate `sim_time`. GeNN additionally flushes bounded
on-device spike-recording windows during long batched runs; that transfer time
is tracked as result collection rather than simulation time. A labeled paper run
writes partitioned outputs like:

```text
data/results/nature_2026_07/
├── manifest.csv
├── checksums.sha256
├── round_01/
│   ├── brian2cpp_t1.0s_n1.parquet
│   ├── brian2cuda_t1.0s_n1.parquet
│   ├── pytorch_t1.0s_n1.parquet
│   ├── nestgpu_t1.0s_n1.parquet
│   ├── genn_t1.0s_n1.parquet
│   └── brian2genn_t1.0s_n1.parquet
└── round_02/
```

The consolidated publication bundle contains 600 spike parquet files: 20 grid
points for each of six frameworks across five rounds. The `no_io/` subfolder
contains the corresponding one-round, 120-row timing dataset collected with
spike probing and output disabled; it intentionally contains no spike parquet
files.

Each spike parquet has one row per spike. The canonical timing column for new
exports is `time_ms`, with `trial`, `neuron_index`, `flywire_id`, and `exp_name`.
The legacy `t` column is kept for existing analysis scripts.

The full `nature_2026_07` spike parquet bundle is too large for regular Git
tracking, so parquet files are intentionally gitignored. The committed metadata
files are `manifest.csv` and `checksums.sha256`; the full bundle is stored in
Google Drive:

https://drive.google.com/drive/folders/1jiSfb5lNfm9gwP0YyyRz5ATIrDpBAcjs

After downloading the Drive folder into `data/results/nature_2026_07/`, verify
the bundle with:

```bash
cd data/results/nature_2026_07
sha256sum -c checksums.sha256
```

### Ground truth comparison

Brian2 (CPU) serves as the ground truth for neural accuracy: it implements the
canonical LIF model from
[Shiu et al. (Nature 2024)](https://www.nature.com/articles/s41586-024-07763-9),
which achieved 91% prediction accuracy against experimental _Drosophila_ data.
Each backend also saves per-neuron spike trains to `data/results/`, and a
comparison script measures how closely the other backends reproduce Brian2's
output:

```bash
python code/compare_ground_truth.py                  # default: t_run=1s, n_run=1
python code/compare_ground_truth.py --t_run 10 --n_run 4   # longer / averaged
python code/compare_ground_truth.py --run-label nature_2026_07 --round 1
```

This computes active-neuron overlap (Jaccard), per-neuron firing-rate
correlation, and spike-count ratios, and writes structured results to
`data/ground-truth-comparison.json`.

For all-framework pairwise comparisons, including firing-rate parity rows and
spike-time matches within a tolerance window, use:

```bash
python code/compare_spike_outputs.py \
  --run-label nature_2026_07 \
  --round 1 \
  --output-dir data/results/nature_2026_07/comparisons
```

This writes `pairwise_summary.csv`, `pairwise_summary.json`,
`parity_rates.csv`, and `missing_inputs.json`. The pairwise summary has one row
per framework pair and `t_run`/`n_run` combination.

For paper-support parity files comparing one backend against Brian2 CPU across
all five labeled rounds, use:

```bash
python code/compare_backend_to_brian2.py \
  --run-label nature_2026_07 \
  --backend brian2genn \
  --output-dir data/results/nature_2026_07/comparisons
```

This writes `<backend>_vs_brian2_rate_summary.csv/json`,
`<backend>_vs_brian2_rate_parity.csv`, and
`<backend>_vs_brian2_missing_inputs.json`. Add `--include-timing` only for
smaller targeted checks where greedy spike-time matching is scientifically
useful and computationally reasonable.

## Fly Pong

The emulated fly can play Pong. [code/fly_pong.py](code/fly_pong.py) closes
a sensorimotor loop around the whole-brain model: the ball is shown to the
fly's visual neurons, the connectome runs, and the fly's steering neurons move
its paddle. Nothing is trained or fitted. Whether the fly turns toward the
ball depends only on the connectome's wiring.

```bash
python code/fly_pong.py                           # you (↑/↓ or W/S) vs the fly
python code/fly_pong.py --autopilot               # watch the fly play the computer
python code/fly_pong.py --headless --points 10    # no window, just the score
python code/fly_pong.py --headless --blind        # control: the fly sees nothing
```

In game: `A` toggles the autopilot, `Space` pauses, `Esc` quits.

| | Neurons | Mapping |
|---|---|---|
| **Eyes** | LC10a visual projection neurons (115 left, 119 right) | Ball bearing from the fly's heading drives Poisson input (up to 200 Hz) to the LC10a on that side |
| **Brain** | All 138,639 neurons, 15M connections | Same LIF model and parameters as the benchmarks |
| **Muscles** | DNa01 + DNa02 steering descending neurons | Left − right firing rate sets paddle velocity; a left turn moves the paddle up |

The fly rides its paddle facing the opponent, so its left is up on screen.
LC10a is the pathway flies use to track small moving objects, and in the
model each side's LC10a drives the same side's DNa02. So the fly turns toward
the ball, and the paddle follows it.

The game runs in brain time: one physics tick per simulated millisecond. It
plays in slow motion at whatever speed the CPU can simulate the brain
(about 0.3× real time on a laptop CPU). The brain monitor under the field
shows the live LC10a input and DN firing rates.

Against the built-in computer opponent (`--headless --points 10 --seed 1`),
the fly won 8–2 and returned 92 of 94 balls (98%). With `--blind`, the only
returns are balls that happen to hit the still paddle: 4 of 14 (29%), and it
lost 0–10.

The brain is simulated with [code/fast_brain.py](code/fast_brain.py), an
event-driven NumPy port of the PyTorch model. It uses the same equations,
parameters, and update order, and it produces the same spikes as
`run_pytorch.TorchModel` given the same input (checked with 300 ms of
sugar-GRN stimulation: 5108 of 5108 spikes identical). It also runs 65–80×
faster on CPU and lets input rates change every step. It needs only
numpy, scipy, pandas, pyarrow, and tkinter, with no GPU or PyTorch. If numba is
installed, it fuses the per-neuron updates into compiled kernels. They produce
bit-identical spikes about 1.8× faster (checked with 5000 steps of 100 Hz and
200 Hz input to about 1,000 neurons: every spike, voltage, and conductance
identical).

Neuron IDs and sides are in `data/fly_pong_neurons.csv`. They come from the
FlyWire cell-type annotations
([Schlegel et al. 2024](https://github.com/flyconnectome/flywire_annotations)).
The left/right names for DNa01, DNa02, and P9 in `example.ipynb` are the
opposite of FlyWire's `side` labels. Fly Pong uses FlyWire's labels for both
eyes and muscles, so they are consistent with each other.

## Fly Screen

The emulated fly can watch your screen. [code/fly_screen.py](code/fly_screen.py)
turns what is on screen into input for the fly's visual neurons and runs the
whole-brain model. When the fly's descending neurons command a behavior, it
posts a text message: turning toward something, an escape takeoff, walking,
feeding, or moving its head.

```bash
pip install mss                            # screen capture
python code/fly_screen.py                  # messages in a floating window
python code/fly_screen.py --no-window      # messages in the terminal
python code/fly_screen.py --demo           # synthetic scenes, no screen capture
python code/fly_screen.py --game           # trash talk even when no game is detected
```

While you play a game, the fly trash-talks you ("Wait, what's that on the
left? Did you even see it?"). It reacts to the same neurons at the same
moments; only the wording changes. On macOS, games are detected from the
frontmost app, every 2 seconds. An app counts as a game if its Info.plist
has a games category, it is installed in a Steam library, or it is Roblox or
Minecraft. Other games, including browser games, need `--game`. Clicking the
fly's own window doesn't end game mode, and the fly announces when you start
and stop a game.

On macOS, the app you run it from needs Screen Recording permission (System
Settings → Privacy & Security → Screen Recording). Without it, the fly sees
only the desktop wallpaper.

| On screen | Input neurons | Output neurons | Reaction |
|---|---|---|---|
| Something small moving (cursor, typing) | LC10a (115 left, 119 right) | DNa02 steering | Turns toward it |
| Something big appearing suddenly | LC4 + LPLC2 (162 left, 152 right) | DNp01 Giant Fiber | Escape takeoff |
| Motion that keeps going (a video) | LC9 (87 left, 92 right) | DNp09 (P9) | Walks forward |
| The whole view sliding sideways | HSE, HSN, HSS, H2 | DNp15 | Turns its head |
| Scrolling | VS1–VS8 | DNp20 | Tilts its head |
| Saturated red, orange, or yellow | 21 sugar GRNs | MN9 | Extends its proboscis |

The left half of the screen is the fly's left eye. The screen features stand
in for the optic lobe: in this model, driving the photoreceptors (R7, R8)
directly reaches nothing downstream. The input channels were chosen by
stimulating candidate populations at 150 Hz for 300 ms and keeping those
with a clean, same-side path to one behavior's descending neurons. Sending
colors to taste neurons isn't biology; it's there for fun.

From the input neurons to the descending neurons, the connectome is
unmodified, and a message reports only what the descending neurons do. So the
messages include the network's cross-talk. For example, right-eye LC10a also
drives MN9, so a fly tracking something on its right sometimes sticks out
its proboscis. Each message lists the firing rates behind it and the input
neurons that were driven:

```text
[14:02:11] Fly: What's that on my left? Turning to look.
           (DNa02 steering L 79 · R 0 Hz   |   input: LC10a L 150 Hz)
```

The fly looks about four times a second and simulates 50 ms of brain time
per look (0.1–0.3× real time on a laptop CPU). Changes that keep recurring in
one spot, like a blinking cursor or a spinner, habituate. Looming stops
startling it after about a second, so a video that starts playing causes an
escape and then walking. The message window is masked out of what the fly
sees. With `--no-window`, the fly can see the terminal it prints to, so keep
that out of view. Neuron IDs and roles are in `data/fly_screen_neurons.csv`,
from the same FlyWire annotations as Fly Pong.

## Fly Scratch: the fly plays Scritchy Scratchy

The fly plays the scratch-card game
[Scritchy Scratchy](https://store.steampowered.com/app/3948120/Scritchy_Scratchy/)
(Steam, macOS or Windows) with the mouse cursor as its body. The connectome is
never changed. What gets trained is only the interface between the screen, the
neurons, and the mouse: 51 numbers (the `Genome`), evolved with CMA-ES.

| Game | Input neurons | Output neurons | Mouse |
|---|---|---|---|
| Contrast in two eye patches ahead of the cursor | LC10a | DNa01 + DNa02, left − right | Turn |
| Change in the eye patches (e.g. foil coming off) | LC9 | DNp09 | Walk speed; above a threshold the button is held, which scratches |
| Sudden darkening | LC4 + LPLC2 | DNp01 Giant Fiber | Jump backwards |
| Colours under the cursor | 21 sugar GRNs, 32 bitter GRNs | MN9 | Click |

Evolution chooses which colours taste sweet or bitter. The fly is given
perceptual features only, never game state. In the connectome, bitter GRNs
silence the sugar → MN9 pathway: MN9 drops from 52/84 Hz to 0 Hz when the left
bitter GRNs are driven at 150 Hz alongside the sugar GRNs. So something bitter
is never clicked, however sweet it also is.

The real game runs in real time, one copy at a time, so most evolution happens
in `ScratchSim`, a fast Python stand-in. It renders the same layout, on a
backdrop taken from a real screenshot, at the fly's own resolution. The real
game is for recording, validation, and fine-tuning.

```bash
pip install cma pillow numba pyobjc-framework-Quartz pyobjc-framework-Vision
python code/scratch/game_io.py --selftest        # finds the window, captures, OCRs, moves the cursor
python code/scratch/recorder.py --minutes 30     # record yourself playing (to calibrate the sim)
python code/scratch/evolve.py --workers 3 --hours 8          # evolve in the sim
python code/scratch/evolve.py --eval <run>/best.json --controls   # vs blind fly and shuffled wiring
python code/scratch/play.py --genome <run>/best.json         # the fly plays the real game
code/scratch/cloud.sh setup <ip>                 # the same evolution on a rented Linux box
```

- **Real game.** `game_io.py` finds the window with Quartz, captures it with
  mss, posts mouse events, reads the money counter with Apple Vision OCR, and
  snapshots or restores the save
  (`~/Library/Application Support/Lunch Money Games/Scritchy Scratchy/save.json`,
  the game's full state as JSON). Turn off Steam Cloud for the game before
  restoring saves.
- **Safety.** The fly only touches the mouse while the game is the frontmost
  app. It never clicks the title bar or the settings gear
  (`data/scratch/layout.json`). Moving the mouse yourself, or into a screen
  corner, stops it.
- **Needs.** Screen Recording and Accessibility permission for the app you
  run it from.
- **Status.** The sim's ticket rules (`sim.RULES`) are placeholders until
  they're calibrated from recordings. Upgrades, the Scratch Bot, loans, and
  prestige aren't simulated yet.

Neuron IDs and roles are in `data/scratch_neurons.csv`: the Fly Screen groups,
plus the bitter GRNs and DNa01 from the FlyWire annotations.

## Fly Daggers: the fly plays Devil Daggers

Every command, with all its options: [code/daggers/README.md](code/daggers/README.md).

The fly plays [Devil Daggers](https://store.steampowered.com/app/422970/Devil_Daggers/)
on Windows, after learning from recordings of a person playing. As in Fly
Scratch, the connectome is never changed. Two things are trained: how the game
drives the fly's visual neurons (15 numbers, the `Genome`, evolved with
CMA-ES), and a linear readout from all 1,409 descending and motor neurons,
the brain's output to the body, to the keys and mouse (ridge regression).

| Game | Input neurons | |
|---|---|---|
| Bright or red things, weighted by how far off-centre they are | LC10a | per eye |
| A sudden rise in how much of one side is things | LC4 + LPLC2 | per eye |
| Motion that keeps going, after removing the view's own turning | LC9 | per eye |
| Redness (gems, some enemies) | 21 sugar GRNs | |
| The view sliding sideways or up and down | HS + H2, VS | off by default |
| **Output** | **1,409 descending + motor neurons** | **Readout: W, A, S, D, jump, both mouse buttons, mouse x and y** |

The left half of the game window is the fly's left eye. The fly sees only
these features, never game state. The self-motion channels are off by default.
When the player turns smoothly, a readout fitted to their play can learn "the
view is sliding, so keep turning". Played in closed loop, that makes the fly
spin. With them off, the fly has to turn toward what it sees.

A genome's fitness is how much of what the player did can be read from the
fly's output neurons while it watches the player's recordings. That is the
cross-validated R² of the readout, averaged over buttons and mouse axes,
counting unpredictable outputs as 0. Training uses only the Retina's fixed
features, a few MB per hour of play; the recorded frames are needed only to
make them.

On Windows, double-click the launchers in `daggers\`: `setup.bat` once, then
`record.bat`, `train.bat`, `play.bat`. Details and every option are in
[code/daggers/README.md](code/daggers/README.md).

- **Recording.** `record.py` captures the game window 20 times a second at
  128×72 and records which buttons were held and how far the mouse moved
  until the next frame. It reads your keyboard and mouse through raw input,
  so it gets the game's own mouse counts. It records only while the game has
  focus and the cursor is hidden, so menus and the death screen are left out.
  Play windowed or borderless: in exclusive fullscreen the capture shows the
  window behind the game, and those stretches are dropped. The player's glowing hand is masked out (`eyes.MASKS`); check it
  with `--preview`.
- **Training.** Record at least an hour of play; every fifth clip is held
  out. `train.py controls <run>` compares the trained fly with three controls
  on the held-out clips: a blind fly, a connectome with shuffled wiring, and no
  brain at all (the readout fitted straight to the eyes). If the fly doesn't
  beat "no brain", the connectome is not adding anything.
- **Playing.** The fly presses keys and moves the mouse with SendInput, and
  only while the game has focus and is in play. If you move the mouse or
  press a key yourself, it lets go for 2 seconds. F9 pauses the fly and F10
  stops it. When the fly dies, it presses R to start the next run.

Neither Devil Daggers nor a recording has been run through this yet. The
pipeline has been checked end to end on synthetic gameplay (moving bright
enemies on a textured floor, and a scripted player who aims at them), not on
the real game.

The brain is simulated by `code/daggers/brain.py` (`EventBrain`), the same
model as `fast_brain.py` and in the same update order, but event-driven per
neuron. A neuron is computed only when input reaches it, or while its leftover
conductance could still carry it over threshold on its own. In between, it
jumps ahead with the closed form of the model's own per-step recurrence.
Checked against `FastBrain` with identical Poisson input: the spikes are the
same until float rounding first puts a membrane on the other side of
threshold (float64 here, float32 there), by under 0.001 mV. On a laptop CPU
(Ryzen 7 7730U), with every input channel busy, it runs at 0.85× real time,
2.2× faster than `FastBrain`. With half that input it runs at 1.4× real time.

Neuron IDs and roles are in `data/daggers_neurons.csv`, built by
`code/daggers/make_neurons.py`: the Fly Screen input groups, and every
`descending` and `motor` neuron in the FlyWire annotations. Tests:
`python -m unittest tests.test_daggers`.

## Installation

### Conda environment

The `brain-fly` conda environment provides everything needed to run the
**Brian2**, **Brian2CUDA**, **PyTorch**, **NEST GPU**, and **GeNN** backends
(including CUDA-enabled PyTorch and PyGeNN):

```bash
conda env create -f environment.yml
conda activate brain-fly
```

On Ubuntu/WSL, PyGeNN's source build also needs the system `pkg-config` binary
and libffi headers:

```bash
sudo apt-get install -y pkg-config libffi-dev
```

### GeNN

The `--genn` backend uses PyGeNN 5.4.0 with the CUDA backend. It implements
Brian2-style Poisson activation into membrane voltage, delayed sparse recurrent
synapses, GeNN batching for `n_run`, and the same parquet spike schema as the
other benchmark runners.

Large batched GeNN runs cap the on-device spike recording buffer with
`GENN_RECORDING_WINDOW_MAX_SLOTS` (default: `800000`). This preserves full spike
timing exports while avoiding CUDA out-of-memory errors for large
`n_run * t_run` combinations.

If PyGeNN was not installed when the conda environment was created, install it
inside `brain-fly` with:

```bash
export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
pip install https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip
```

### Brian2GeNN

The `--brian2genn` backend uses Brian2GeNN 1.7.0 as a Brian2 standalone device
targeting GeNN/CUDA. It is intentionally isolated from the main `brain-fly`
environment because Brian2GeNN pins Brian2<2.6, which conflicts with
Brian2CUDA's Brian2 2.8.0 requirement.

Create the environment with:

```bash
conda env create -f environment-brian2genn.yml
conda activate brain-fly-brian2genn
```

Brian2GeNN 1.7.0 expects GeNN 4.x command-line scripts such as
`genn-buildmodel.sh`. If they are not already installed, place GeNN 4.9.0 at
`~/.local/src/genn-4.9.0` or set `BRIAN2GENN_GENN_PATH`/`GENN_PATH` to your
GeNN 4.x source tree:

```bash
export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_HOME=$CUDA_PATH
export BRIAN2GENN_GENN_PATH=$HOME/.local/src/genn-4.9.0
export GENN_PATH=$BRIAN2GENN_GENN_PATH
export PATH=$GENN_PATH/bin:$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

For scientific comparability, the Brian2GeNN runner exports the same per-spike
parquet schema as the other backends and uses the same upstream Poisson drive.
Brian2GeNN cannot run this model's independent trials as a true GeNN batch in
the way the direct `--genn` backend can, so `n_run>1` is implemented as
independent build/run trials with deterministic per-trial C RNG seeds. The
`sim_time` column records GeNN executable time; `build_time` records the
Brian2GeNN code generation/compilation overhead.

### NEST GPU

NEST GPU requires a separate build from source with a custom neuron model
(`user_m1`). This is only needed if you want to use the `--nestgpu` backend.

**Prerequisites:**

- **NVIDIA CUDA Toolkit** (12.x) — follow the
  [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- **CMake** — `sudo apt install cmake` (or see
  [cmake.org](https://cmake.org/download/)).

**Steps:**

1. Clone NEST GPU:

```bash
git clone https://github.com/nest/nest-gpu
```

2. Copy the custom source files into the NEST GPU tree. You must replace `/path/to/nest-gpu` with your own local path:

```bash
cp scripts/nestgpu_source_files/src/user_m1.{h,cu}    /path/to/nest-gpu/src/
cp scripts/nestgpu_source_files/pythonlib/nestgpu.py   /path/to/nest-gpu/pythonlib/
```

   The patched `nestgpu.py` fixes weight array initialization (lines 2225-2227).

3. Build and install (set `-DCMAKE_CUDA_ARCHITECTURES` to match your GPU, e.g.
   `89` for RTX 4070):

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_INSTALL_PREFIX=$HOME/.nest-gpu-build \
      /path/to/nest-gpu
make -j$(nproc) && make install
```

For a full setup from a fresh Windows machine (WSL2 + CUDA + Miniconda), see
[scripts/setup_WSL_CUDA.sh](scripts/setup_WSL_CUDA.sh).

----

## Frameworks

| Framework | Backend | Status |
|---|---|---|
| **Brian2** | C++ standalone (multi-core CPU) | ready |
| **Brian2CUDA** | CUDA standalone (GPU) | ready |
| **PyTorch** | CUDA (GPU) | ready |
| **NEST GPU** | CUDA (GPU, custom `user_m1` neuron) | ready |
| **GeNN** | CUDA (GPU, PyGeNN 5.4.0) | ready |
| **Brian2GeNN** | Brian2GeNN 1.7.0 / GeNN CUDA | ready, separate env |

All six frameworks share the same data, model parameters, spike-output schema,
and folder structure. The five main backends run from `brain-fly` plus a
system-level NEST GPU install; Brian2GeNN runs from `brain-fly-brian2genn`
because of its Brian2 version pin.

## Quickstart

```bash
# Create the conda environment (includes CUDA-enabled PyTorch)
conda env create -f environment.yml
conda activate brain-fly

# Run a 1-second benchmark on the five main-environment backends
python main.py --t_run 1 --n_run 1 --no_log_file

# Specific backends (combinable)
python main.py --brian2-cpu                    # Brian2 CPU only
python main.py --brian2cuda-gpu               # Brian2CUDA GPU only
python main.py --pytorch                      # PyTorch only
python main.py --nestgpu                      # NEST GPU only
python main.py --genn                         # GeNN only
python main.py --brian2genn                   # Brian2GeNN only, from brain-fly-brian2genn
python main.py --pytorch --genn               # PyTorch + GeNN

# Full benchmark suite (all durations, n_run=1,4,8,16,32, five main backends)
python main.py

# Nature-paper suite: five main backends, March parameter grid, 5 rounds
python main.py --paper --run-label nature_2026_07

# Brian2GeNN Nature-paper add-on from the separate brain-fly-brian2genn env
python main.py --brian2genn --paper --run-label nature_2026_07
```

### `main.py` options

| Flag | Description |
|---|---|
| *(default)* | Run all: Brian2 (CPU) → Brian2CUDA (GPU) → PyTorch → NEST GPU → GeNN |
| `--brian2-cpu` | Brian2 C++ standalone (CPU) only |
| `--brian2cuda-gpu` | Brian2CUDA (GPU) only |
| `--pytorch` | PyTorch (GPU/CPU) only |
| `--nestgpu` | NEST GPU only |
| `--genn` | GeNN CUDA backend only |
| `--brian2genn` | Brian2GeNN backend only; use the `brain-fly-brian2genn` environment |
| `--t_run` | Simulation duration(s) in seconds, from `0.1 1 10 100 1000` (default: all), e.g. `--t_run 0.1 1 10` |
| `--n_run` | Number of independent trials, e.g. `--n_run 1 4 8 16 32` (default: all five) |
| `--experiment` | Stimulation protocol: `sugar` (21 sugar GRNs at 200 Hz, default) or `p9` (P9 forward-walking neurons at 100 Hz) |
| `--disable-spike-io` | Disable spike probing/recording and parquet output for timing-only (no-I/O) benchmarks |
| `--paper` | Run the paper suite: `t_run=[0.1,1,10,100]`, `n_run=[1,4,8,16,32]`, 5 rounds |
| `--rounds` | Repeat the full selected backend/parameter suite N times |
| `--round-start` | First round number to write, useful for resuming a labeled run |
| `--run-label` | Group repeated spike outputs under `data/results/<label>/` and append labeled CSV rows |
| `--log_file FILE` | Write log to file (default: `data/results/benchmarks.log`) |
| `--no_log_file` | Console output only |

Backend flags are combinable: `--brian2-cpu --pytorch` runs Brian2 CPU then PyTorch.

## Project structure

```
fly-brain/
├── main.py                     # Entrypoint (benchmark runner CLI)
├── environment.yml             # Conda env definition (brain-fly)
├── environment-brian2genn.yml  # Separate Brian2GeNN env definition
├── daggers/                    # Fly Daggers launchers: setup, record, train, play, check (.bat)
├── code/
│   ├── benchmark.py            # Orchestrator: config, logging, dispatcher
│   ├── run_brian2_cuda.py      # Brian2 / Brian2CUDA benchmark runner
│   ├── run_pytorch.py          # PyTorch benchmark runner (model + utils)
│   ├── run_nestgpu.py          # NEST GPU benchmark runner (subprocess per trial)
│   ├── run_genn.py             # GeNN/PyGeNN benchmark runner
│   ├── run_brian2_genn.py      # Brian2GeNN benchmark runner (separate env)
│   ├── compare_ground_truth.py # Compare backends against Brian2 (CPU) ground truth
│   ├── compare_spike_outputs.py      # Pairwise spike comparisons across all frameworks
│   ├── compare_backend_to_brian2.py  # One backend vs Brian2 (CPU) across labeled rounds
│   ├── fast_brain.py           # Event-driven NumPy port of the PyTorch model (closed-loop use)
│   ├── fly_pong.py             # Fly Pong: the emulated fly plays Pong
│   ├── fly_screen.py           # Fly Screen: the emulated fly watches your screen
│   ├── daggers/                # Fly Daggers: the fly learns to play Devil Daggers
│   │   ├── brain.py            # EventBrain: event-driven LIF model, real-time capable
│   │   ├── eyes.py             # Retina (fixed features) and Eyes (evolved genome)
│   │   ├── fly.py              # Neuron groups, brain wrapper, ridge readout, policy files
│   │   ├── winio.py            # Windows: window capture, raw input, SendInput
│   │   ├── record.py           # Record a person playing
│   │   ├── dataset.py          # Recordings -> prepared features -> training clips
│   │   ├── train.py            # CMA-ES evolution, readout export, controls
│   │   ├── play.py             # The trained fly plays
│   │   └── make_neurons.py     # Builds data/daggers_neurons.csv from FlyWire annotations
│   └── paper-phil-drosophila/  # Original paper code (not used by benchmarks)
│       ├── LICENSE             # Upstream MIT license
│       ├── model.py            # Core LIF network model (Brian2)
│       ├── utils.py            # Analysis helpers (load_exps, get_rate)
│       ├── example.ipynb       # Tutorial: activation, silencing, rate analysis
│       └── figures.ipynb       # Reproduce paper figures (uses archive 630 data)
├── data/
│   ├── 2025_Completeness_783.csv       # Neuron list (FlyWire v783)
│   ├── 2025_Connectivity_783.parquet   # Synapse connectivity (FlyWire v783)
│   ├── benchmark-results.csv           # Accumulated benchmark timings
│   ├── fly_pong_neurons.csv            # Fly Pong eye (LC10a) and steering (DNa01/02) neurons
│   ├── fly_screen_neurons.csv          # Fly Screen input and behavior neurons
│   ├── daggers_neurons.csv             # Fly Daggers inputs and 1,409 readout neurons
│   ├── ground-truth-comparison.json   # Backend accuracy vs Brian2 (CPU)
│   ├── sez_neurons.pickle              # SEZ neuron subset (for figures)
│   ├── weight_coo.pkl                  # Cached sparse weights COO (gitignored)
│   ├── weight_csr.pkl                  # Cached sparse weights CSR (gitignored)
│   ├── archive/
│   │   ├── 2023_Completeness_630.csv   # Legacy v630 data
│   │   └── 2023_Connectivity_630.parquet
│   └── results/
│       └── nature_2026_07/             # Paper dataset: manifest, checksums, no_io/ timings
└── scripts/
    ├── setup_WSL_CUDA.sh       # WSL2 + CUDA + Miniconda setup
    └── nestgpu_source_files/   # Custom user_m1 neuron + patched nestgpu.py for NEST GPU
```

## Data

The model uses FlyWire connectome data version **783** (public release).
Legacy version 630 data is kept in `data/archive/` for paper figure reproduction.

| File | Description | Size |
|---|---|---|
| `2025_Completeness_783.csv` | Neuron IDs and metadata | 3.2 MB |
| `2025_Connectivity_783.parquet` | Pre/post-synaptic indices + weights | 97 MB |
| `weight_coo.pkl` | Sparse weight matrix (COO), auto-generated by PyTorch | ~288 MB |
| `weight_csr.pkl` | Sparse weight matrix (CSR), auto-generated by PyTorch | ~289 MB |

## Architecture per framework

| | Brian2 / Brian2CUDA | PyTorch | NEST GPU |
|---|---|---|---|
| Build step | C++ / CUDA codegen + compile | None (eager mode) | None |
| Trial parallelism | Sequential (`device.run`) | Batched (`batch_size=n_run`) | Subprocess per trial (cannot reset in-process) |
| Weight format | Brian2 `Synapses` object | Sparse CSR tensor | Array-based `Connect` |
| Neuron model | Brian2 equations | Custom `nn.Module` classes | Custom CUDA kernel (`user_m1`) |
| Timestep | 0.1 ms | 0.1 ms | 0.1 ms |

## System requirements

- Linux (tested on Ubuntu 22.04 under WSL2 on Windows 11)
- NVIDIA GPU with CUDA 12.x (tested on RTX 4070)
- Miniconda / Anaconda
- NEST GPU compiled from source (for `--nestgpu` backend)
- `scripts/setup_WSL_CUDA.sh` documents the full setup from a fresh Windows machine

## License

Except where otherwise noted, this project is licensed under the GNU General
Public License version 2 or any later version
(`GPL-2.0-or-later`). See [LICENSE](LICENSE).

Third-party components retain their original notices. In particular, the
Shiu et al. Brian2 materials in `code/paper-phil-drosophila/` remain available
under their upstream [MIT License](code/paper-phil-drosophila/LICENSE), and the
adapted NEST GPU model files retain their GPL-2.0-or-later notices.
