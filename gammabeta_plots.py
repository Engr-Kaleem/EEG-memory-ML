import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test


def plot_band_psd(epochs_ENC,epochs_NO_ENC,band,subno):
    print(f'subject{subno}')      
    kwargs = dict(fmin=20, fmax=40, n_jobs=4)
    spectrum = epochs_ENC.compute_psd(
        'welch', average='median', **kwargs)

    mean_spectrum = spectrum.average()
    psds_welch_mean, freqs_mean = mean_spectrum.get_data(return_freqs=True)
    spectrum = epochs_NO_ENC.compute_psd(
        'welch', average='median', **kwargs)
    median_spectrum = spectrum.average()
    psds_welch_median, freqs_median=median_spectrum.get_data(return_freqs=True)
    # Convert power to dB scale.
    psds_welch_mean = 10 * np.log10(psds_welch_mean)
    psds_welch_median = 10 * np.log10(psds_welch_median)
    psds_welch_mean = psds_welch_mean.mean(axis=0)
    psds_welch_median = psds_welch_median.mean(axis=0)
    plt.figure()
    _, ax = plt.subplots()
    ax.plot(freqs_mean, psds_welch_mean, color='k',
                ls='-', label='ENC')
    ax.plot(freqs_median, psds_welch_median, color='k',
                ls='--', label='NO_ENC')

    ax.set(title=f'Welch {band} PSD,subject {subno} ',
    xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
    ax.legend(loc='upper right')
    plt.savefig(f'bandanaylsis/{band}_band_subject{subno}psd.png')
    
    
def plot_ERD_bands(epochs,freqs,baseline,tmin,tmax,cnorm,sub):
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                        return_itc=False, average=False, decim=2,n_jobs=4)
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
    
    df = tfr.to_data_frame(time_format=None, long_format=True)

    # Map to frequency bands:
    freq_bounds = {'_': 13,
                'beta': 35,
                'gamma': 140}
    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                        labels=list(freq_bounds)[1:])

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['beta','gamma']
    df = df[df.band.isin(freq_bands_of_interest)]
    df['band'] = df['band'].cat.remove_unused_categories()
    df['channel'] = df['channel'].cat.reorder_categories(('Fp1', 'Fpz', 'Fp2','FC1', 'FC2'),
                                                     ordered=True)

    g = sns.FacetGrid(df, row='channel', col='band', margin_titles=True)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(None, 6.5))
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc='lower center')
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    g.fig.savefig(f'ERDbandplots/subject{sub}.png')