
# import libraraires
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import pandas as pd
import seaborn as sns
import mne
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from Data_plot import Plot_eventpsd, plot_ERD, Plot_bandpsd,Plot_subpsd,plot_ERD_bands
from Data_plot import plot_ERD_stats
from topo_plots import plot_psdtopo
#load  epcohs 

picks=['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4',
 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 
'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7'
, 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
s=range(1,3)
subdata=[]

# Deal with NaN values when the model cannot detect peaks in any given range
def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""
    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))
    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')
    
def plot_psd_fooof(enc_data,non_enc_data,subno):
  
  bands =['theta','alpha','beta','gamma']
  freq_range = [3, 45]
  # Calculate power spectra across the the continuous data by MNE
  maplist=[]
  mappsd1=[]
  mappsd2=[]
  
   
  kwargs = dict(fmin=3, fmax=8, n_jobs=4)
  
  
  

  spectra1 = enc_data.compute_psd('welch', average='median', **kwargs)
  spectra1=spectra1.average()
  psds1, freqs1 = spectra1.get_data(return_freqs=True)
  spectra2 = non_enc_data.compute_psd('welch', average='median', **kwargs)
  spectra2=spectra2.average()
  psds2, freqs2 = spectra2.get_data(return_freqs=True)
  spectra_diff_t=psds1-psds2
  mappsd1.append(psds1)
  mappsd2.append(psds2)
  
  kwargs = dict(fmin=8, fmax=12, n_jobs=4)
  spectra1 = enc_data.compute_psd('welch', average='median', **kwargs)
  spectra1=spectra1.average()
  psds1, freqs1 = spectra1.get_data(return_freqs=True)
  spectra2 = non_enc_data.compute_psd('welch', average='median', **kwargs)
  spectra2=spectra2.average()
  psds2, freqs2 = spectra2.get_data(return_freqs=True)
  spectra_diff_a=psds1-psds2
  mappsd1.append(psds1)
  mappsd2.append(psds2)
  
  kwargs = dict(fmin=12, fmax=30, n_jobs=4)
  spectra1 = enc_data.compute_psd('welch', average='median', **kwargs)
  spectra1=spectra1.average()
  psds1, freqs1 = spectra1.get_data(return_freqs=True)
  spectra2 = non_enc_data.compute_psd('welch', average='median', **kwargs)
  spectra2=spectra2.average()
  psds2, freqs2 = spectra2.get_data(return_freqs=True)
  spectra_diff_b=psds1-psds2
  mappsd1.append(psds1)
  mappsd2.append(psds2)
  
  kwargs = dict(fmin=30, fmax=45, n_jobs=4)
  spectra1 = enc_data.compute_psd('welch', average='median', **kwargs)
  spectra1=spectra1.average()
  psds1, freqs1 = spectra1.get_data(return_freqs=True)
  spectra2 = non_enc_data.compute_psd('welch', average='median', **kwargs)
  spectra2=spectra2.average()
  psds2, freqs2 = spectra2.get_data(return_freqs=True)
  spectra_diff_g=psds1-psds2
  mappsd1.append(psds1)
  mappsd2.append(psds2)
  
  maplist.append(spectra_diff_t)
  maplist.append(spectra_diff_a)
  maplist.append(spectra_diff_b)
  maplist.append(spectra_diff_g)
  
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  for ind in range(0,4):
      band_power=maplist[ind]
      mne.viz.plot_topomap(band_power.mean(axis = 1),enc_data.info,names=picks,axes=axes[ind],show=False);
      axes[ind].set_title(bands[ind] + ' power', {'fontsize' : 16}) 
  fig.savefig(f'topmapdiff/subject{subno}.png')
  
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  for ind in range(0,4):
      band_power=mappsd1[ind]
      mne.viz.plot_topomap(band_power.mean(axis = 1),enc_data.info,names=picks, axes=axes[ind],show=False);
      axes[ind].set_title(bands[ind] + ' power enc', {'fontsize' : 16}) 
  fig.savefig(f'topmapdiff/subject{subno}_enc.png')
  
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  for ind in range(0,4):
      band_power=mappsd2[ind]
      mne.viz.plot_topomap(band_power.mean(axis = 1),enc_data.info,names=picks, axes=axes[ind],show=False);
      axes[ind].set_title(bands[ind] + ' power non enc', {'fontsize' : 16}) 
  fig.savefig(f'topmapdiff/subject{subno}non_ENC.png')
  
  """ fg.fit(freqs1, spectra_diff, freq_range)
  # Plot the topographies across different frequency bands
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  for ind, (label, band_def) in enumerate(bands):
      # Extract the power peaks across channels for the current band
      band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])
      # Create a topomap for the current oscillation band
      mne.viz.plot_topomap(band_power, enc_data.info, cmap=Colormap.viridis, axes=axes[ind],show=False);
      axes[ind].set_title(label + ' power', {'fontsize' : 16}) """
   
s=range(1,15)
subdata=[]
      
for i in s:
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    epochs_ENC=(epochs['EV_ENC'])
    epochs_NO_ENC=(epochs['EV_NO_ENC'])
    plot_psd_fooof(epochs_ENC,epochs_NO_ENC,i)


