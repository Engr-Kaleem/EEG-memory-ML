import numpy as np
import pandas as pd
import os
import mne.io
import warnings

warnings.filterwarnings("ignore")
mne.set_log_level('CRITICAL')


INPUT_DIR = "./data/mne_epochs/"

SUB_IDS = [2, 4, 9, 10, 13, 16, 19, 30, 31, 36, 37, 41]

# vid trigs - EV - Event for video
EV_ENC = 1  # clip remembered
EV_NO_ENC = 2  # clip not remembered

montage = mne.channels.make_standard_montage('standard_1020')

for sub_id in SUB_IDS:
    epochs = mne.read_epochs(os.path.join(INPUT_DIR, 'clean_s{}_erp_epochs.fif'.format(sub_id)))
    epochs.drop_channels(['T7', 'T8'])
    metadata = pd.read_csv("%sclean_final_s%d.csv" % (INPUT_DIR, sub_id))
    print(len(epochs), len(metadata))
    assert len(epochs) == len(metadata)
    print(metadata.columns)
    # ['filename', 'response', 'responseTime', 'target', 'tp', 'fp', 'tn', 'fn', 'epoch_number', 'IS_GOOD', 'SUB_ID',
    # 'LABEL', 'IS_TEST']

    # ERPs
    erps = epochs.copy()
    picks = mne.pick_channels(erps.ch_names, include=erps.ch_names)
    erps = epochs.copy().filter(l_freq=0, h_freq=15).apply_baseline((-.25, 0)).resample(30).\
                         get_data(picks=picks, tmin=0, tmax=1)
    X = erps.reshape(erps.shape[0], -1)  # one row per epoch
    smps_per_channel = erps.shape[-1]

    column_names = []
    for ch in epochs.ch_names:
        column_names += ['%s_%d' % (ch, t) for t in range(smps_per_channel)]

    subject_data = pd.DataFrame(X, columns=column_names)
    subject_data['subject'] = sub_id
    subject_data['filename'] = metadata['filename']
    subject_data['LABEL'] = metadata['LABEL']
    subject_data['IS_TEST'] = metadata['IS_TEST']
    subject_data.loc[subject_data['IS_TEST'] == True, 'LABEL'] = np.nan
    subject_data.drop(['IS_TEST'], axis=1, inplace=True)
    if sub_id == SUB_IDS[0]:  # first subject
        erps_all_subjects = subject_data
    else:
        erps_all_subjects = pd.concat([erps_all_subjects, subject_data], ignore_index=True)

    erps_all_subjects.to_csv('./data/EEG_task/ERP_data.csv', index=False)

    ## ERSPs
    freqs = np.arange(2, 30, 1)
    n_cycles = np.log(freqs + 1) ** 2.1
    downsample_factor = 20

    ersps = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles, average=False, return_itc=False, picks=picks)
    baseline_method = 'zscore'  # logratio ratio mean percent zscore zlogratio
    pwr_baselined = ersps.copy().apply_baseline(baseline=(-.25, 0), mode=baseline_method).data[:, :, :, 100::downsample_factor]
    X = pwr_baselined.reshape(pwr_baselined.shape[0], -1)

    column_names = []
    for ch in ersps.ch_names:
        for f in freqs:
            column_names += ['%s_%d_%d' % (ch, f, t) for t in range(pwr_baselined.shape[-1])]
    subject_data = pd.DataFrame(X, columns=column_names)
    subject_data['subject'] = sub_id
    subject_data['filename'] = metadata['filename']
    subject_data['LABEL'] = metadata['LABEL']
    subject_data['IS_TEST'] = metadata['IS_TEST']
    subject_data.loc[subject_data['IS_TEST'] == True, 'LABEL'] = np.nan
    subject_data.drop(['IS_TEST'], axis=1, inplace=True)
    if sub_id == SUB_IDS[0]:
        ersps_all_subjects = subject_data
    else:
        ersps_all_subjects = pd.concat([ersps_all_subjects, subject_data], ignore_index=True)

    ersps_all_subjects.to_csv('./data/EEG_task/ERSP_data.csv', index=False)
