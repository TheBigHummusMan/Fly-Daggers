#!/usr/bin/env bash
# Train Fly Daggers on a rented Linux box (e.g. a Vultr CPU-Optimized instance).
#
# Create the instance in the Vultr dashboard yourself (Ubuntu 24.04, your SSH
# key added). Each vCPU evaluates one genome at a time, so more vCPUs = bigger
# generations. Then, from the repo root (Git Bash on Windows is fine):
#
#   code/daggers/cloud.sh setup  <ip>               # upload code, connectome, prepared recordings; install; time the brain
#   code/daggers/cloud.sh data   <ip>               # upload the prepared recordings again (after recording more)
#   code/daggers/cloud.sh run    <ip> [train args]  # start evolution in the background, e.g. --hours 8
#   code/daggers/cloud.sh status <ip>               # the log's last lines
#   code/daggers/cloud.sh export <ip> [run]         # fit the readout there for the latest (or given) run
#   code/daggers/cloud.sh pull   <ip>               # copy runs back into data/daggers/runs/
#
# When done: pull, then DESTROY the instance in the Vultr dashboard.
# Stopped instances are still billed.
set -euo pipefail
cmd=${1:?usage: cloud.sh setup|data|run|status|export|pull <ip> [args]}
host=root@${2:?need the server ip}
shift 2
cd "$(dirname "$0")/../.."
REMOTE=fly
PY=../../.venv/bin/python

upload_data() {
    [ -f data/daggers/prepared/stats.json ] || { echo "Nothing prepared: run code/daggers/dataset.py first" >&2; exit 1; }
    tar czf - data/daggers/prepared | ssh "$host" "mkdir -p $REMOTE && tar xzf - -C $REMOTE"
}

case "$cmd" in
setup)
    echo "Uploading code and connectome (~100 MB)..."
    tar czf - code/fast_brain.py code/benchmark.py \
        code/daggers/brain.py code/daggers/eyes.py code/daggers/fly.py \
        code/daggers/dataset.py code/daggers/train.py \
        data/2025_Completeness_783.csv data/2025_Connectivity_783.parquet data/daggers_neurons.csv \
        | ssh "$host" "mkdir -p $REMOTE && tar xzf - -C $REMOTE"
    upload_data
    ssh "$host" "set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq python3-venv tmux >/dev/null
        python3 -m venv $REMOTE/.venv
        $REMOTE/.venv/bin/pip install -q numpy scipy pandas pyarrow numba cma
        cd $REMOTE/code/daggers && $PY -c 'import os, brain; print(os.cpu_count(), \"vCPUs\"); brain.bench()'"
    echo "Ready. Start with: code/daggers/cloud.sh run ${host#root@} --hours 8"
    ;;
data)
    upload_data
    ;;
run)
    ssh "$host" "cd $REMOTE/code/daggers && tmux new -d -s fly \
        'OMP_NUM_THREADS=1 $PY -u train.py evolve --workers \$((\$(nproc) - 1)) $* 2>&1 | tee -a ../../train.log'"
    echo "Started. Check with: code/daggers/cloud.sh status ${host#root@}"
    ;;
status)
    ssh "$host" "tail -n 12 $REMOTE/train.log; uptime"
    ;;
export)
    # The given run, or else the newest one on the server
    run=${1:+../../data/daggers/runs/$(basename "$1")}
    ssh "$host" "cd $REMOTE/code/daggers && run=${run:-\$(ls -d ../../data/daggers/runs/*/ | tail -n 1)} && \
        OMP_NUM_THREADS=1 $PY -u train.py export \$run --workers \$((\$(nproc) - 1)) && \
        OMP_NUM_THREADS=1 $PY -u train.py controls \$run --workers 4"
    ;;
pull)
    mkdir -p data/daggers/runs
    ssh "$host" "tar czf - -C $REMOTE/data/daggers runs" | tar xzf - -C data/daggers
    ssh "$host" "cat $REMOTE/train.log" > data/daggers/runs/cloud-train.log
    echo "Pulled into data/daggers/runs/. Now destroy the instance in the Vultr dashboard."
    ;;
*)
    echo "unknown command $cmd" >&2; exit 1 ;;
esac
