import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test


def plot_psdtopo(epochs_ENC,epochs_NO_ENC,sub):
    fig=epochs_ENC.plot_psd_topomap(ch_type='eeg', normalize=True)
    eve=epochs_ENC.event_id
    fig.savefig(f'psdtopomap/subject{sub}_ENC.png')
    fig1= epochs_NO_ENC.plot_psd_topomap(ch_type='eeg', normalize=True)
    eve=epochs_NO_ENC.event_id
    fig1.savefig(f'psdtopomap/subject{sub}_NO_ENC.png')