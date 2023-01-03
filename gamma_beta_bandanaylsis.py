import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.time_frequency import tfr_multitaper
from gammabeta_plots import plot_band_psd

#load  epcohs 
sub=range(1,15)
picks=['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4',
 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 
'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7'
, 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
i=4

# code fore plotting ERD of beta and gamma band 
epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
epochs_ENC=epochs['EV_ENC']
freqs = np.arange(13, 45)  # frequencies from 2-35Hz
vmin, vmax = -1, 6.5  # set min and max ERDS values in plot
baseline = (-0.25, 0)  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
tmin,tmax=-1,2.0
tfr = tfr_multitaper(epochs_ENC, freqs=freqs, n_cycles=freqs, use_fft=True,
                        return_itc=False, average=True, decim=2,n_jobs=4)
tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
df = tfr.to_data_frame(time_format=None, long_format=True)
print(df.head)


# Map to frequency bands:
freq_bounds = {'_': 13,
            'beta': 35,
            'gamma': 45}
df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                    labels=list(freq_bounds)[1:])

# Filter to retain only relevant frequency bands:
freq_bands_of_interest = ['beta','gamma']
df = df[df.band.isin(freq_bands_of_interest)]
df['band'] = df['band'].cat.remove_unused_categories()


g = sns.FacetGrid(df, row='ch_type', col='band', margin_titles=True)
g.map(sns.lineplot, 'time', 'value', n_boot=10)
axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
g.map(plt.axhline, y=0, **axline_kw)
g.map(plt.axvline, x=0, **axline_kw)
g.set(ylim=(None, 6.5))
g.set_axis_labels("Time (s)", "ERDS (%)")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.add_legend(ncol=2, loc='lower center')
g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
g.fig.savefig(f'ERDbandplots/subject{sub}.png')




# enable following code to pllot beta and  gamma band PSd of each subject

""" for i in sub:
    print(f'subject{i}')
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    gamma_epochs=epochs.filter(29,45.0,n_jobs=4)
    beta_epochs=epochs.filter(13.0,29.0,n_jobs=4)
    plot_band_psd(gamma_epochs['EV_ENC'],gamma_epochs['EV_NO_ENC'],'gamma',i) 
    plot_band_psd(beta_epochs['EV_ENC'],beta_epochs['EV_NO_ENC'],'beta',i)  """