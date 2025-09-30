"""Configuration constants and color palettes for phosphonetworks plots."""

import os
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

DATA_DIR = os.path.join(
    'data'
)
FIGURES_DIR = os.path.abspath(
    'figures'
)

KINSUB_LABELS = {
    'literature': 'Literature',
    'phosformer': 'Phosformer',
    'kinlib': 'Kinase Library',
    'combined': 'Combined'
}

KINSUB_COLORS = {
    'literature': '#fde725',
    'phosformer': '#35b779',
    'kinlib': '#31688e',
    'combined': '#440154'
}

# DEFINE STUDY COLORS USING THE SET2 CMAP palette
studies = [
    'Tuechler et al. 2025 (PDGFRb)',
    'Chen et al. 2025 (HEK293T)',
    'Skowronek et al. 2022 (HeLa)',
    'Bortel et al. 2024 (HeLa)',
    'Lancaster et al. 2024 (HeLa)',
    'This study (HEK293T)',
    'This study (HEK293F)',
    'This study (HEK293F TR)'
]
# get colors as HTML hex codes
cdict = plt.get_cmap('tab10').colors[:len(studies)][::-1]
STUDY_COLORS = {study: to_hex(cdict[i]) for i, study in enumerate(studies)}

GT_LABELS = {
    'signor': 'SIGNOR\nPathway',
    'lun_lenient': 'Overexpression\n(Lenient)',
    'lun_moderate': 'Overexpression\n(Moderate)',
    'lun_strict': 'Overexpression\n(Strict)',
    'hek293_lenient': 'Correlation\n(Lenient)',
    'hek293_moderate': 'Correlation\n(Moderate)',
    'hek293_strict': 'Correlation\n(Strict)'
}


def set_data_dir(path):
    """Update the base directory used for cached and downloaded data."""
    print('Setting data directory to', path)
    global DATA_DIR
    DATA_DIR = os.path.abspath(path)
