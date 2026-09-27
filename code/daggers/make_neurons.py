"""
Build data/daggers_neurons.csv: the neurons Fly Daggers drives and reads.

Inputs are Fly Screen's groups (data/fly_screen_neurons.csv), chosen there
for a clean same-side path to one behavior. The readout is every descending
and motor neuron in the FlyWire annotations (Schlegel et al. 2024): the
brain's whole output to the body. Training fits how those firing rates
become keys and mouse motion; it never touches the connectome.

Bands. Each object, loom and bustle population (one cell type on one side)
is split into BANDS groups, one per strip of the screen on that eye, so
where something is can reach the brain instead of being averaged away. The
annotations don't say where each neuron looks (their positions are cell
bodies, which aren't retinotopic), so the split follows the connectome: the
neurons are ordered along the direction in which their outputs differ most
(first principal component of their normalized output weights) and cut into
equal groups. Neighbouring strips thus drive the most distinct downstream
pathways. LC outputs are known to keep some retinotopic order, but whether
these bands match where the neurons actually look is not known.

Usage:
    python code/daggers/make_neurons.py                   # downloads the annotations (~32 MB)
    python code/daggers/make_neurons.py --annotations Supplemental_file1_neuron_annotations.tsv
"""

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_brain import load_connectome  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_URL = ('https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/'
                   'supplemental_files/Supplemental_file1_neuron_annotations.tsv')
READOUT_CLASSES = ('descending', 'motor')
BANDED = ('object', 'loom', 'bustle')
BANDS = 4                        # per eye; eyes.N_BINS // 2


def output_bands(weights, idx, n_bands=BANDS):
    """Band 0..n_bands-1 for each neuron in idx, by its outputs' first principal component."""
    rows = weights[idx]
    targets = np.unique(rows.indices)
    dense = np.abs(rows[:, targets].toarray())
    dense /= np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-9)
    dense -= dense.mean(axis=0)
    _, _, vt = np.linalg.svd(dense, full_matrices=False)
    score = dense @ vt[0]
    order = np.argsort(score, kind='stable')
    band = np.empty(len(idx), dtype=int)
    band[order] = np.arange(len(idx)) * n_bands // len(idx)
    return band


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--annotations', type=Path, help='local copy of the annotation TSV')
    args = parser.parse_args()

    path = args.annotations
    if path is None:
        path = Path(tempfile.gettempdir()) / 'flywire_neuron_annotations.tsv'
        if not path.exists():
            print(f'Downloading {ANNOTATIONS_URL}')
            urllib.request.urlretrieve(ANNOTATIONS_URL, path)

    ann = pd.read_csv(path, sep='\t', usecols=['root_id', 'super_class', 'cell_type', 'side'],
                      dtype={'root_id': 'int64'})
    in_brain = set(pd.read_csv(ROOT / 'data' / '2025_Completeness_783.csv', index_col=0).index)

    inputs = pd.read_csv(ROOT / 'data' / 'fly_screen_neurons.csv')
    inputs = inputs[inputs.role.isin(['object', 'loom', 'bustle', 'pan', 'scroll', 'taste'])].copy()
    inputs['band'] = -1
    weights, flyid2i = load_connectome()
    for (role, cell_type, side), group in inputs[inputs.role.isin(BANDED)].groupby(
            ['role', 'cell_type', 'side']):
        idx = np.array([flyid2i[f] for f in group.flywire_id])
        inputs.loc[group.index, 'band'] = output_bands(weights, idx)

    readout = ann[ann.super_class.isin(READOUT_CLASSES) & ann.root_id.isin(in_brain)]
    readout = pd.DataFrame({
        'flywire_id': readout.root_id,
        'cell_type': readout.cell_type.fillna(readout.super_class),
        'side': readout.side.fillna('center'),
        'role': 'readout:' + readout.super_class,
        'band': -1,
    }).sort_values(['role', 'cell_type', 'side', 'flywire_id'])

    out = pd.concat([inputs, readout], ignore_index=True)
    dest = ROOT / 'data' / 'daggers_neurons.csv'
    out.to_csv(dest, index=False)
    print(f'{len(inputs)} input and {len(readout)} readout neurons -> {dest.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
