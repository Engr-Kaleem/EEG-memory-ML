import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test



def Plot_subpsd(epochs_ENC,epochs_NO_ENC,subno):
    kwargs = dict(fmin=2, fmax=38, n_jobs=4)
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

    ax.set(title=f'Welch PSD,subject {subno} )',
    xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
    ax.legend(loc='upper right')
    plt.savefig(f'subjectpsdplots/subject{subno}.png')

def Plot_bandpsd(epochs_ENC,epochs_NO_ENC, band,subno):
    if band=='delta':
        kwargs = dict(fmin=1.0, fmax=4, n_jobs=4)
    if band=='theta':
        kwargs = dict(fmin=4, fmax=8, n_jobs=4)
    if band=='alpha':
        kwargs = dict(fmin=8, fmax=14, n_jobs=4)
    if band=='beta':
        kwargs = dict(fmin=14, fmax=38, n_jobs=4)
          

    
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

    ax.set(title=f'Welch PSD ({band},subject {subno} )',
    xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
    ax.legend(loc='upper right')
    plt.savefig(f'bandplots/{band}subject{subno}{band}.png')


def Plot_eventpsd(epochs_ENC,epochs_NO_ENC,ch_names,subno):
    kwargs = dict(fmin=2, fmax=40, n_jobs=4)
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
    

    for ch in ch_names:
       
        ch_idx = epochs_ENC.info['ch_names'].index(ch)
        _, ax = plt.subplots()
        ax.plot(freqs_mean, psds_welch_mean[ch_idx, :], color='k',
                ls='-', label='ENC')
        ax.plot(freqs_median, psds_welch_median[ ch_idx, :], color='k',
                ls='--', label='NO_ENC')

        ax.set(title=f'Welch PSD ({ch},subject {subno} )',
            xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
        ax.legend(loc='upper right')
        plt.savefig(f'PSDplots/{ch}subject{subno}.png')
        


def plot_ERD(epochs,freqs,baseline,tmin,tmax,kwargs,cnorm,sub):
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                        return_itc=False, average=False, decim=2,n_jobs=4)
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
    for event in epochs.event_id:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, 11, figsize=(20, 6),
                                gridspec_kw={"width_ratios": [10, 10,10, 10, 10, 10, 10,10,10,10,1]})
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            # positive clusters
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
                                colorbar=False, show=False, mask=mask,
                                mask_style="mask")

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({event}subject{sub})")
        fig.savefig(f'ERDplot/{event}subject{sub}.png')
    

def plot_ERD_bands(epochs,freqs,baseline,tmin,tmax,cnorm,sub):
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                        return_itc=False, average=False, decim=2,n_jobs=2)
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
    df = tfr.to_data_frame(time_format=None, long_format=True)

    # Map to frequency bands:
    freq_bounds = {'_': 0,
                'delta': 3,
                'theta': 7,
                'alpha': 13,
                'beta': 35,
                'gamma': 140}
    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                        labels=list(freq_bounds)[1:])

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['delta', 'theta', 'alpha', 'beta']
    df = df[df.band.isin(freq_bands_of_interest)]
    df['band'] = df['band'].cat.remove_unused_categories()
    df['channel'] = df['channel'].cat.reorder_categories(('Fp1', 'Fpz','Fp2',  'T7', 'Cz',  'T8',  'P7' , 'Pz', 'P8' ),
                                                     ordered=True)

    g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(None, 3))
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc='lower center')
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    g.fig.savefig(f'ERDbandplots/subject{sub}.png')
    
 

def plot_ERD_stats(epochs,freqs,baseline,tmin,tmax,cnorm,sub):
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                        return_itc=False, average=False, decim=2,n_jobs=4)
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
    df = tfr.to_data_frame(time_format=None, long_format=True)
      # Map to frequency bands:
    freq_bounds = {'_': 0,
                'delta': 3,
                'theta': 7,
                'alpha': 13,
                'beta': 35,
                'gamma': 140}
    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                        labels=list(freq_bounds)[1:])

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['delta', 'theta', 'alpha', 'beta']
    df = df[df.band.isin(freq_bands_of_interest)]
    df['band'] = df['band'].cat.remove_unused_categories()
    df['channel'] = df['channel'].cat.reorder_categories(('Fp1', 'Fpz', 'Fp2','FC1', 'FC2'),
                                                     ordered=True)
    df_mean = (df.query('time > 1')
             .groupby(['condition', 'epoch', 'band', 'channel'])[['value']]
             .mean()
             .reset_index())

    g = sns.FacetGrid(df_mean, col='condition', col_order=['EV_ENC', 'EV_NO_ENC'],
                    margin_titles=True)
    g = (g.map(sns.violinplot, 'channel', 'value', 'band', n_boot=10,
            palette='deep', order=['Fp1', 'Fpz', 'Fp2','FC1', 'FC2'],
            hue_order=freq_bands_of_interest,
            linewidth=0.5).add_legend(ncol=4, loc='lower center'))
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, **axline_kw)
    g.set_axis_labels("", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    g.fig.savefig(f'ERDstatplots/subject{sub}.png')
    
    
    
