# Fly Daggers: how to run it

Everything needed to set up, record, train, and run the fly that plays Devil
Daggers, on any Windows laptop. For how it works, see the "Fly Daggers" section
of the main [README](../../README.md).

```
setup  ->  record  ->  train  ->  play
```

## Quick start (double-click)

The launchers are in the `daggers\` folder at the top of the repository.
Double-click them in File Explorer, or run them from a terminal to add options.

| Launcher | What it does |
|---|---|
| `setup.bat` | one-time setup: finds Python, installs everything, runs the tests, times the brain |
| `record.bat` | records you playing, then prepares the recording for training |
| `train.bat` | trains for 8 hours, then exports a playable policy and runs the controls |
| `play.bat` | the fly plays, using the newest trained policy |
| `check.bat` | tests + brain speed (run it during a game to see if the laptop keeps up) |

### 1. Before the first run

1. Get the whole repository, including the `data` folder. The connectome file
   `data\2025_Connectivity_783.parquet` is about 100 MB.
2. Install **64-bit Python 3.13** from <https://www.python.org/downloads/>
   (3.10–3.12 also work). In the installer, tick **"Add python.exe to PATH"**.
3. Double-click **`daggers\setup.bat`**. It takes a few minutes the first
   time. At the end it times the brain: about **0.7× or more is fine for
   playing**, because this test uses heavier input than real play.

To play only, without recording or training, you can stop here and go to step
4: the repository already includes trained policies (`data\daggers\runs\<run>\policy.json`).

### 2. Record yourself playing

Set Devil Daggers to **borderless or windowed** mode, not exclusive fullscreen.

- **`record.bat`** records until you press F10.
- **`record.bat 30`** stops after 30 minutes of play.

| Key | Action |
|---|---|
| F9 | pause / resume |
| F10 | stop and save |

It records only while the game has focus and you are in a run; menus and the
death screen are skipped. If the window says **FROZEN CAPTURE**, press F10 and
switch the game to borderless. When you stop, it prepares the recording for
training automatically.

Record at least an hour in total. Sessions add up, so several short ones are
fine.

### 3. Train

- **`train.bat`** starts a new 8-hour run.
- **`train.bat 2`** starts a new 2-hour run.
- **`train.bat resume`** continues the newest run for 8 hours.
- **`train.bat resume 3`** continues the newest run for 3 hours.
- **`train.bat export`** only re-exports the policy and runs the controls for the newest run.

Training uses every CPU core, and the laptop stays awake while it runs. Keep it
**plugged in**. Closing the window stops training; `train.bat resume` picks it
up again.

When training ends it exports the policy and prints the **controls**: the fly
scored against a blind fly, shuffled wiring, and no brain on held-out clips. If
it doesn't beat all three, more training alone won't help.

### 4. Play

Start Devil Daggers (borderless), then:

- **`play.bat`**: the fly plays.
- **`play.bat dry`**: shows what it would do, without pressing anything.

Start a run in the game; the fly takes over after 1 second.

| Key | Action |
|---|---|
| F9 | pause / resume the fly |
| F10 | stop; every key is released |
| move the mouse / press a key | you take control for 2 seconds |

When the fly dies, it presses **R** to start the next run by itself. The
terminal prints how long each run lasted, and the average. It won't restart
if you pressed a key just before, for example Esc to pause. The status line
shows which keys the fly holds, its mouse speed, and the brain's speed. Below
about 0.7×, it reacts late.

## Full commands (for more control)

The launchers call these. Run them from the repository root in PowerShell;
`train.py` from inside `code\daggers`.

### Record and prepare

```powershell
.venv\Scripts\python code\daggers\record.py [--minutes 30] [--no-cursor-check]
.venv\Scripts\python code\daggers\eyes.py --preview data\daggers\recordings\<session>   # check masks -> preview.png
.venv\Scripts\python code\daggers\dataset.py [--force]    # --force: redo all (after changing eyes.MASKS)
```

Use `--no-cursor-check` if recording never starts, because the game doesn't
hide the cursor.

### Train

```powershell
cd code\daggers
..\..\.venv\Scripts\python train.py evolve --hours 8
..\..\.venv\Scripts\python train.py evolve --resume latest --hours 8
..\..\.venv\Scripts\python train.py export latest
..\..\.venv\Scripts\python train.py controls latest --clips 40 --workers 4
```

`train.py evolve` options:

| Option | Default | Meaning |
|---|---|---|
| `--hours H` | 1 | stop after this long |
| `--workers N` | CPU count − 1 | parallel processes |
| `--popsize N` | = workers (at least 8) | genomes per generation |
| `--clips N` | 16 | 10-second clips each genome is scored on |
| `--val-every N` | 5 | generations between held-out checks |
| `--sigma S` | 0.2 | CMA-ES step size |
| `--seed N` | 1 | random seed |
| `--start FILE` or `latest` | default genes | start from a genome (`best.json`) |
| `--resume DIR` or `latest` | | continue a stopped run |
| `--self-motion` | off | let the fly use the "view is turning" channels (can make it spin) |

`export` and `controls` take a run folder or `latest`, plus
`--genome best|mean` to choose which genes (default `best`).

Watch a run's progress (one row per generation; the number to watch is
`val_mean_genome`, the score on held-out clips):

```powershell
Get-Content data\daggers\runs\<run>\log.csv -Wait
```

### Play

```powershell
.venv\Scripts\python code\daggers\play.py [--policy latest|<path to policy.json>] [--dry-run] [--no-restart] [--no-cursor-check] [--seed N]
```

### Checks

```powershell
.venv\Scripts\python -m unittest tests.test_daggers          # unit tests (~1 s)
.venv\Scripts\python code\daggers\brain.py                   # brain speed
.venv\Scripts\python code\daggers\brain.py --validate        # spikes vs FastBrain, then speed
.venv\Scripts\python code\daggers\make_neurons.py            # rebuild data\daggers_neurons.csv (downloads FlyWire annotations)
```

## Where things are

| Path | What | In git? |
|---|---|---|
| `data\daggers\recordings\` | your recorded sessions (frames + inputs) | no (large) |
| `data\daggers\prepared\` | features extracted for training | no (rebuilt by `record.bat` / `dataset.py`) |
| `data\daggers\runs\<run>\` | `log.csv`, `best.json`, `policy.json`, ... | only `policy.json` and `best.json` |
| `data\daggers_neurons.csv` | which neurons are driven and read | yes |

To train on another laptop, copy `data\daggers\recordings\` over, then run
`record.bat` once (press F10 straight away) or `dataset.py` to prepare it.
The environment variable `FLY_DAGGERS_DATA=<folder>` moves `data\daggers\`
elsewhere.

## Troubleshooting

| Problem | Fix |
|---|---|
| `setup.bat`: "No suitable Python found" | install 64-bit Python 3.13 with "Add to PATH", reopen the window |
| status stays "waiting: game window not found" | start Devil Daggers; the window must be titled "Devil Daggers" |
| status stays "waiting: menu or cursor showing" while playing | add `--no-cursor-check` (`record.py` / `play.py`) |
| "FROZEN CAPTURE" | game settings → borderless or windowed |
| fly reacts late, brain speed well below 0.7× | plug in, close browsers and other heavy programs |
| fly walks off the map or spins | F10; it needs more / better training |
| fly doesn't restart after dying | the death screen may not show the cursor in your setup; restart yourself, or tell us |
| training stopped overnight | the laptop slept or the window closed: `train.bat resume` |
