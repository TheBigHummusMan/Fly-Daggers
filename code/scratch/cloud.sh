#!/usr/bin/env bash
# Run sim evolution on a rented Linux box (e.g. a Vultr CPU-Optimized instance).
#
# Create the instance yourself (Ubuntu 24.04, SSH key added), then from the repo root:
#   code/scratch/cloud.sh setup  <ip>              # upload code + data (~100 MB), install, time the brain
#   code/scratch/cloud.sh run    <ip> [evolve args] # start evolve.py in the background
#   code/scratch/cloud.sh status <ip>              # tail the log
#   code/scratch/cloud.sh pull   <ip>              # copy runs back to data/scratch/evolve/
#
# When done: pull, then DESTROY the instance in the Vultr dashboard.
# Stopped instances are still billed.
set -euo pipefail
cmd=${1:?usage: cloud.sh setup|run|status|pull <ip> [args]}
host=root@${2:?need the server ip}
shift 2
cd "$(dirname "$0")/../.."
REMOTE=fly

case "$cmd" in
setup)
    ssh "$host" "mkdir -p $REMOTE/code/scratch $REMOTE/data/scratch"
    rsync -az code/fast_brain.py code/benchmark.py "$host:$REMOTE/code/"
    rsync -az code/scratch/cursor_fly.py code/scratch/sim.py code/scratch/evolve.py "$host:$REMOTE/code/scratch/"
    rsync -az data/2025_Completeness_783.csv data/2025_Connectivity_783.parquet \
        data/scratch_neurons.csv "$host:$REMOTE/data/"
    rsync -az data/scratch/sprites "$host:$REMOTE/data/scratch/"
    ssh "$host" "set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq python3-venv tmux >/dev/null
        python3 -m venv $REMOTE/.venv
        $REMOTE/.venv/bin/pip install -q 'numpy<2' scipy pandas pyarrow cma pillow numba
        cd $REMOTE/code && ../.venv/bin/python - <<'EOF'
import os, time, numpy as np
from fast_brain import FastBrain, load_connectome
W, _ = load_connectome()
b = FastBrain(W, np.arange(300), seed=0); b.set_rates(np.full(300, 100.0))
for _ in range(300): b.step()
t = time.perf_counter()
for _ in range(3000): b.step()
dt = (time.perf_counter() - t) / 3000
print(f'{os.cpu_count()} CPUs; one fly runs at {1e-4 / dt:.2f}x real time ({dt * 1e6:.0f} us/step)')
EOF"
    ;;
run)
    ssh "$host" "cd $REMOTE/code/scratch && tmux new -d -s evolve \
        'OMP_NUM_THREADS=1 ../../.venv/bin/python -W ignore evolve.py --workers \$((\$(nproc) - 1)) $* 2>&1 | tee -a ../../evolve.log'"
    echo "Started. Check with: code/scratch/cloud.sh status ${host#root@}"
    ;;
status)
    ssh "$host" "tail -n 20 $REMOTE/evolve.log; uptime"
    ;;
pull)
    mkdir -p data/scratch/evolve
    rsync -az "$host:$REMOTE/data/scratch/evolve/" data/scratch/evolve/
    rsync -az "$host:$REMOTE/evolve.log" data/scratch/evolve/cloud-evolve.log
    echo "Pulled into data/scratch/evolve/. Now destroy the instance in the Vultr dashboard."
    ;;
*)
    echo "unknown command $cmd" >&2; exit 1 ;;
esac
