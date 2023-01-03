
# import libraraires
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from Data_plot import Plot_eventpsd, plot_ERD, Plot_bandpsd,Plot_subpsd,plot_ERD_bands
from Data_plot import plot_ERD_stats
from topo_plots import plot_psdtopo
#load  epcohs 
s=range(1,14)
subdata=[]

""" 
picks=['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4',
 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 
'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7'
, 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
 """


""" for i in range(1,len(s)):
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    epochs.filter(1.0,38.0,picks)
    Plot_subpsd(epochs['EV_ENC'],epochs['EV_NO_ENC'],s[i])  """

""" for i in range(1,len(s)):
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    epochs.filter(1.0,38.0,picks)
    plot_psdtopo(epochs['EV_ENC'],epochs['EV_NO_ENC'],s[i])   """

""" ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4',
 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 
'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7'
, 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'] """





""" for i in range(1,len(s)):
    bands = ['delta', 'theta', 'alpha', 'beta']
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    for band in bands:         
        #epochs.filter(4.0,38.0,picks)
        Plot_bandpsd(epochs['EV_ENC'],epochs['EV_NO_ENC'],band,s[i])  """
        

""" for i in range(1,len(s)):
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    epochs.filter(1.0,38.0)
    Plot_eventpsd(epochs['EV_ENC'],epochs['EV_NO_ENC'],picks,s[i]) 
 """ 
picks=['Fp1', 'Fp2', 'FC5',  'FC6', 'T7', 'Cz',  'T8',  'P7' , 'Pz', 'P8' ]
 
''' for i in range(1,len(s)):
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    freqs = np.arange(2, 36)  # frequencies from 2-35Hz
    vmin, vmax = -1, 6.5  # set min and max ERDS values in plot
    baseline = (-0.25, 0)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    tmin,tmax=-1,2.99


    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                buffer_size=None, out_type='mask')  # for cluster test

    plot_ERD(epochs.pick_channels(picks),freqs,baseline,tmin,tmax,kwargs,cnorm,i) '''

picks=['Fp1', 'Fpz','Fp2',  'T7', 'Cz',  'T8',  'P7' , 'Pz', 'P8' ]
for i in range(1,len(s)):
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    epochs.filter(2.0,45.0)
    freqs = np.arange(2, 36)  # frequencies from 2-35Hz
    vmin, vmax = -1, 3  # set min and max ERDS values in plot
    baseline = (-0.25, 0)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    tmin,tmax=-0.25,2.99
    plot_ERD_bands(epochs.pick_channels(picks),freqs,baseline,tmin,tmax,cnorm,i)



""" picks=['Fp1', 'Fpz', 'Fp2','FC1', 'FC2']  
for i in range(1,len(s)):
    epochs = mne.read_epochs('data/clean_s'+str(i)+'_erp_epochs.fif', preload=True)
    freqs = np.arange(2, 36)  # frequencies from 2-35Hz
    vmin, vmax = -1, 6.5  # set min and max ERDS values in plot
    baseline = (-0.25, 0)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    tmin,tmax=-1,2.99
    plot_ERD_stats(epochs.pick_channels(picks),freqs,baseline,tmin,tmax,cnorm,i)

 """
''' picks=[ 'Fp2','CP5', 'CP1']
# check the events
print(epochs)
print(epochs.event_id)
print(epochs.ch_names)
event_dict = {'EV_ENC': 1 ,'EV_NO_ENC': 2}
##epochs.event_id = event_dict;
epochs.filter(2.0,38.0,picks)

epochs_ENC=(epochs['EV_ENC']).pick_channels(picks)
epochs_NO_ENC=(epochs['EV_NO_ENC']).pick_channels(picks)

#Plot_eventpsd(epochs_ENC,epochs_NO_ENC,epochs.ch_names[0:5])  

f




 '''