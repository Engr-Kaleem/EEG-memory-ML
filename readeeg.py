
# import libraraires
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from Data_plot import Plot_eventpsd, plot_ERD

#load  epcohs 
epochs = mne.read_epochs('clean_s1_erp_epochs.fif', preload=True)

""" ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4',
 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 
'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7'
, 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'] """

picks=[ 'Fp2','CP5', 'CP1']
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

freqs = np.arange(2, 36)  # frequencies from 2-35Hz
vmin, vmax = -1, 2.5  # set min and max ERDS values in plot
baseline = (-0.25, 0)  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
tmin,tmax=-0.25,2.99


kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

plot_ERD(epochs[:200].pick_channels(picks),freqs,baseline,tmin,tmax,kwargs,cnorm)






