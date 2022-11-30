import numpy as np
import pandas as pd
import os
import mne.io
import warnings

warnings.filterwarnings("ignore")
mne.set_log_level('CRITICAL')


INPUT_DIR = "./data/mne_epochs/"

SUB_DCU = [1, 2, 3, 40]
SUB_IDS = [2, 4, 9, 10, 13, 15, 16, 17, 19, 30, 31, 36, 37, 41]
# not enough samples for subject 15, 17

# vid trigs - EV - Event for video
EV_ENC = 1  # clip remembered
EV_NO_ENC = 2  # clip not remembered

montage = mne.channels.make_standard_montage('standard_1020')

for sub_id in SUB_IDS:
    if not os.path.exists(os.path.join(INPUT_DIR + 'clean_s{}_erp_epochs.fif'.format(sub_id))):
        epochs = mne.read_epochs(os.path.join(INPUT_DIR, 's{}_erp_epochs_ica_long.fif'.format(sub_id)))
        kept = len(epochs["IS_GOOD"]) / len(epochs)
        assert kept >= 0.6, 'Not enough epochs for this subject: %.2f' % kept
        if sub_id in SUB_DCU:
            epochs.drop_channels(['TP9', 'TP10'])
        else:
            epochs.drop_channels(['TP7', 'TP8'])
        epochs.set_montage(montage)
        epochs['IS_GOOD'].save(os.path.join(INPUT_DIR + 'clean_s{}_erp_epochs.fif'.format(sub_id)), overwrite=True)
    epochs = mne.read_epochs(os.path.join(INPUT_DIR, 'clean_s{}_erp_epochs.fif'.format(sub_id)))
    print(len(epochs), len(epochs.metadata))
    # We just need the metadata really
    metadata = epochs.metadata.copy()
    metadata['LABEL'] = epochs.events[:, -1]
    metadata['IS_TEST'] = False  # True if epoch belongs to test set, otherwise (i.e., training set) False
    # Check that there are enough positive (EV_ENC) samples
    pos = epochs['EV_ENC'].selection
    neg = epochs['EV_NO_ENC'].selection
    if len(pos) < 40:
        print("Not enough positive samples for subject %d" % sub_id)
        continue
    if sub_id in [10, 36, 2]:  # these are left for the cross-subject task
        metadata['IS_TEST'] = True
    else:
        # class EV_ENC
        pos_test_ids = np.random.choice(pos, 20, replace=False)
        metadata['IS_TEST'][pos_test_ids] = True
        # class EV_NO_ENC
        neg_test_ids = np.random.choice(neg, 80, replace=False)
        metadata['IS_TEST'][neg_test_ids] = True
    print(metadata.head())
    metadata.to_csv('%s/clean_metadata_s%d.csv' % (INPUT_DIR, sub_id))
