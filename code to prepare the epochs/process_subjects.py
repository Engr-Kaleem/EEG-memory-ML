import mne
import numpy as np
from glob import glob
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
import collections
import warnings
import sklearn
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 2000)

sys.path.append('../scripts/')
from data import BioSemiData
import constants as cst

warnings.filterwarnings("ignore")
mne.set_log_level('CRITICAL')

# CONSTANTS
dcu_channels = cst.DCU_channels
OUTPUT_DIR = cst.path_to_epochs

RESAMPLING_HZ_TARGET = 100
RAW_LP_HZ = 30
RAW_HP_HZ = 0.1

# for extract epochs for viz
EPOCHS_TMIN = -.5
EPOCHS_TMAX = 1

# this is the longer window used for wavelets, output, ...
EPOCHS_TMIN_LONG = -1
EPOCHS_TMAX_LONG = 3

# we use these times in epochs to do trial rejection
EPOCH_FILT_TIME_BEGIN = -.25
EPOCH_FILT_TIME_END = 1

FILT_D_HP = 1
FILT_D_LP = 20

# Events constants
EV_ENC = 1  # Remembered clip
EV_NO_ENC = 2  # Not remembered clip
EVENT_IDS_ALL = dict(EV_ENC=EV_ENC, EV_NO_ENC=EV_NO_ENC)

reject = {'4': dict(eeg=7e-5), '5': dict(eeg=9e-5), '6': dict(eeg=8e-5), '7': dict(eeg=7e-5), '8': dict(eeg=7e-5),
          '9': dict(eeg=7e-5), '10': dict(eeg=7e-5), '11': dict(eeg=7e-5), '12': dict(eeg=7e-5), '13': dict(eeg=7e-5),
          '14': dict(eeg=7e-5), '15': dict(eeg=7e-5), '16': dict(eeg=7e-5), '17': dict(eeg=7e-5), '18': dict(eeg=7e-5),
          '19': dict(eeg=6e-5), '20': dict(eeg=7e-5), '21': dict(eeg=7e-5), '22': dict(eeg=8e-5), '23': dict(eeg=9e-5),
          '24': dict(eeg=7e-5), '25': dict(eeg=9e-5), '26': dict(eeg=7e-5), '28': dict(eeg=11e-5), '29': dict(eeg=7e-5),
          '30': dict(eeg=7e-5), '31': dict(eeg=7e-5), '32': dict(eeg=7e-5), '33': dict(eeg=7e-5), '34': dict(eeg=7e-5),
          '35': dict(eeg=9e-5), '36': dict(eeg=7e-5), '37': dict(eeg=8e-5), '38': dict(eeg=7e-5), '39': dict(eeg=7e-5),
          '41': dict(eeg=7e-5), '42': dict(eeg=7e-5), '43': dict(eeg=7e-5), '44': dict(eeg=7e-5), '45': dict(eeg=7e-5)
          }

bad_ics = {'4': [0, 4], '5': [1, 4], '6': [], '7': [], '8': [], '9': [], '10': [8, 0], '11': [], '12': [],
           '13': [0, 1, 4],
           '14': [],
           '15': [0, 1, 5], '16': [0, 1], '17': [9, 8, 7, 6, 4, 0, 1, 2], '18': [], '19': [0, 2, 3, 4, 8, 9],
           '20': [], '21': [0, 6], '22': [0, 1, 2], '23': [0, 5], '24': [], '25': [0],
           '26': [], '28': [], '29': [], '30': [0, 2], '31': [5, 7], '32': [], '33': [], '34': [], '35': [],
           '36': [0, 2, 3],
           '37': [0, 2], '38': [], '39': [], '41': [0, 3], '42': [], '43': [0, 1, 3, 4], '44': [], '45': []
           }  # empty ones haven't been processed


def main(fname):
    """
    Load raw.fif file given as parameter

    :param fname: name of raw.fif file
    """
    # Load FIF file
    raw = mne.io.Raw(fname, preload=True)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    return raw


def ica_no_eog_epochs(data):
    """
    Fit ICA for EOG correction
    :param data: MNE Epochs
    :return: Fitted ICA estimator
    """
    print("Starting ICA...")
    n_components = 32
    method = 'fastica'
    decim = 3
    random_state = 23
    # ica on filtered data above 1 Hz
    picks = mne.pick_types(data.info, eeg=True, eog=False, stim=False)
    ica_fitted = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state)
    ica_filt_d = data.copy().resample(RESAMPLING_HZ_TARGET, npad='auto').filter(1.5, 20.,
                                                                                h_trans_bandwidth='auto',
                                                                                filter_length='auto',
                                                                                phase='zero',
                                                                                picks=picks)
    ica_fitted.fit(ica_filt_d, decim=decim, picks=picks)
    return ica_fitted


def logs_and_events(sub_id):
    """
    Load events for subject
    :param sub_id: subject id
    :return: logfile - pandas DataFrame
    """
    logfile = pd.read_csv('%s%d_log.csv' % (cst.path_to_logs, sub_id))
    return logfile


def recode_events(logfile, events):
    """
    Convert the logfile info into MNE-compatible events for labelling the epochs (remembered / not remembered)
    :param logfile: pandas dataframe
    :param events: original events extracted from EEG
    :return: updated events in MNE-compatible format
    """
    synthetic_events = []  # MNE-compatible data structure for events we will make
    print(len(logfile), len(events))
    for i, row in logfile.iterrows():
        # print(i, row)
        t2 = events[i][0]  # video
        if row['tp'] == 1:
            synthetic_events.append([t2, 0, EV_ENC])
        elif row['fn'] == 1:
            synthetic_events.append([t2, 0, EV_NO_ENC])
    print(len(synthetic_events), len(logfile), len(events))
    print(set(np.asarray(synthetic_events)[:, -1]))
    return synthetic_events

# Visually inspected in check_trigger_numbers.py -- these subjects might have to be excluded
# 7, 10, 23, 24 - too much noise in general?
# 18, 34 movements?
# 25? 28?
# 41 moves a lot


if __name__ == '__main__':
    for S_ID in range(4, 46):
        if S_ID in cst.EXCLUDED_PARTICIPANTS:
            continue
        fname = '%srawArray_s%d.raw.fif' % (cst.path_to_processed_data, S_ID)
        print(f"Subject {S_ID}: {fname}")
        if not os.path.exists(fname):
            print("No raw.fif file for subject")
            continue
        elif os.path.exists(os.path.join(OUTPUT_DIR + 's{}_erp_epochs_ica.fif'.format(S_ID))):
            print(f"Data already processed for subject {S_ID}")
            continue
        eeg_data = main(fname)
        eeg_data, _ = mne.io.set_eeg_reference(eeg_data, ref_channels='average', ch_type='eeg')
        eeg_data = eeg_data.filter(RAW_HP_HZ, RAW_LP_HZ, h_trans_bandwidth='auto', l_trans_bandwidth='auto',
                                   filter_length='auto', phase='zero')
        eeg_data.plot(n_channels=33)
        plt.show()
        if len(eeg_data.info['bads']) > 0:
            eeg_data.interpolate_bads(reset_bads=True)
        picks = mne.pick_types(eeg_data.info, eeg=True, eog=False, stim=False)
        ica_fitted = ica_no_eog_epochs(eeg_data)
        FILT_D = eeg_data.copy().filter(FILT_D_HP, FILT_D_LP, h_trans_bandwidth='auto', filter_length='auto',
                                        phase='zero', picks=picks)
        # Epoch rejection
        events_from_eeg = mne.find_events(eeg_data, stim_channel='photodiode', output='onset')
        events_from_eeg = events_from_eeg[np.isin(events_from_eeg[:, 2], [128])]  # 128 is start of clip
        print("Number of events in EEG: %d" % events_from_eeg.shape[0])
        good_epochs_find = mne.Epochs(FILT_D, events_from_eeg, event_id=128, tmin=EPOCH_FILT_TIME_BEGIN,
                                      tmax=EPOCH_FILT_TIME_END, proj=True,
                                      picks=picks, baseline=(-.25, 0), preload=True, reject_by_annotation=False,
                                      reject=reject[str(S_ID)])
        GOOD_EPOCH_INDEXES = good_epochs_find.selection
        print('Remaining %d out of %d' % (len(GOOD_EPOCH_INDEXES), len(events_from_eeg)))
        del good_epochs_find, FILT_D
        
        epochs_all = mne.Epochs(eeg_data, events_from_eeg, event_id=128, tmin=EPOCHS_TMIN, tmax=EPOCHS_TMAX,
                                baseline=(-.25, 0), picks=picks, preload=True, proj=True).resample(RESAMPLING_HZ_TARGET,
                                                                                                   npad='auto')
        if len(bad_ics[str(S_ID)]) == 0:  # plot them and choose manually; update list at the top of this script so they're removed in the next if statement
            ica_fitted.plot_properties(epochs_all[GOOD_EPOCH_INDEXES], picks=range(10))
            plt.show()
        # ICA correction, but first we need to know which ICs to remove
        if len(bad_ics[str(S_ID)]) > 0:
            epochs_ica = ica_fitted.apply(epochs_all.copy(), exclude=bad_ics[str(S_ID)])
            epochs_ica[GOOD_EPOCH_INDEXES].plot()
            mne.viz.plot_epochs_image(epochs_ica[GOOD_EPOCH_INDEXES], picks=['Fp1', 'Fz', 'O1', 'Cz', 'Pz'], sigma=3,
                                      colorbar=True, order=None)
            epochs_ica[GOOD_EPOCH_INDEXES].average().plot_joint(show=False, times=[.1, .16, .2, .3, .4, .5, .6, .7, .8])
            plt.show()
        
        # Load logs
        logfile = logs_and_events(S_ID)
        synthetic_events = recode_events(logfile, events_from_eeg)
        
        # Re-compute epochs and prepare for ML
        
        # what are the good epochs?
        print('Remaining epochs: %d/%d' % (len(GOOD_EPOCH_INDEXES), len(synthetic_events)))
        epochs_all = mne.Epochs(eeg_data, synthetic_events, event_id=EVENT_IDS_ALL,
                                tmin=EPOCHS_TMIN, tmax=EPOCHS_TMAX, baseline=(-.25, 0),
                                picks=picks, preload=True, proj=True).resample(RESAMPLING_HZ_TARGET, npad='auto')
        epochs_all_long = mne.Epochs(eeg_data, synthetic_events, event_id=EVENT_IDS_ALL,
                                     tmin=EPOCHS_TMIN_LONG, tmax=EPOCHS_TMAX_LONG, baseline=(-.25, 0),
                                     picks=picks, preload=True, proj=True).resample(RESAMPLING_HZ_TARGET, npad='auto')
        
        print(len(epochs_all), len(epochs_all_long))
        epochs_ica = ica_fitted.apply(epochs_all.copy(), exclude=bad_ics[str(S_ID)])
        epochs_ica_long = ica_fitted.apply(epochs_all_long.copy(), exclude=bad_ics[str(S_ID)])
        
        CH_VIZ_PICKS = mne.pick_channels(epochs_ica.info['ch_names'], include=['Fp1', 'Fz', 'O1', 'Cz', 'Pz'])
        '''
        ## Successfully encoded video
        epochs_ica[GOOD_EPOCH_INDEXES]["EV_ENC"].average().plot_joint(show=False, times=[.1,.16,.2,.3,.4,.5,.6,.7,.8])
        ## Not encoded video
        epochs_ica[GOOD_EPOCH_INDEXES]["EV_NO_ENC"].average().plot_joint(show=False, times=[.1,.16,.2,.3,.4,.5,.6,.7,.8])
        ## Difference of remembered vs not remembered
        avg1 = epochs_ica[GOOD_EPOCH_INDEXES]["EV_ENC"].copy().average()
        avg2 = epochs_ica[GOOD_EPOCH_INDEXES]["EV_NO_ENC"].copy().average()
        mne.combine_evoked([avg1, -avg2], weights='equal').plot_joint(show=False, times=[.1, .2, .25, .3, .46,.5,.6])
        plt.show()
        '''
        
        # Metadata and save outputs
        ### all ica filt epochs
        metadata = pd.DataFrame({'IS_GOOD': [False] * len(epochs_ica), 'SUB_ID': S_ID})
        metadata['IS_GOOD'][GOOD_EPOCH_INDEXES] = True
        epochs_ica.metadata = metadata
        epochs_ica.save(os.path.join(OUTPUT_DIR + 's{}_erp_epochs_ica.fif'.format(S_ID)), overwrite=True)
        ### all epochs
        epochs_all.metadata = metadata
        epochs_all.save(os.path.join(OUTPUT_DIR + 's{}_erp_epochs_all.fif'.format(S_ID)), overwrite=True)
        ### all ica filt epochs_long
        epochs_ica_long.metadata = metadata
        epochs_ica_long.save(os.path.join(OUTPUT_DIR + 's{}_erp_epochs_ica_long.fif'.format(S_ID)), overwrite=True)
        ### all epochs_long
        epochs_all_long.metadata = metadata
        epochs_all_long.save(os.path.join(OUTPUT_DIR + 's{}_erp_epochs_all_long.fif'.format(S_ID)), overwrite=True)
        print(metadata)
        metadata.to_csv('%s/metadata_s%d.csv' % (OUTPUT_DIR, S_ID))
