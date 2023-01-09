
import mne;
import matplotlib.pyplot  as mp
import numpy as np








picks=['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4',
 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 
'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7'
, 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
info = mne.create_info(32, sfreq=100)
info = mne.create_info(picks, ch_types=30*['eeg'], sfreq=100);
info.set_montage('standard_1020');
epochs = mne.read_epochs('data/clean_s1_erp_epochs.fif', preload=True)
epochs.plot_sensors(show_names=True)
mp.show()
