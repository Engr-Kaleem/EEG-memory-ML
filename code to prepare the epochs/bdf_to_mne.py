import mne
import numpy as np
from glob import glob
from data import BioSemiData
from align_event_channels import start_of_clip_triggers
import os
import constants as cst
import matplotlib.pyplot as plt

bad_channels = cst.bad_channels  
dcu_channels = cst.DCU_channels
conversion_dict = cst.sname_to_sid


def main(fname, sid):
    """
    Given a subject, convert the bdf eeg_data, together with the information from the logs
    of that participant's experimental session into a single MNE-readable file

    :param fname: name of bdf file
    :param sid: subject id
    """
    # Load BDF file
    load_channels = dcu_channels + ['EXG4', 'Status']  # we don't load external electrodes
    bdf = BioSemiData(filename=fname, channels=load_channels)
    bdf.load(num_sec=-1)
    print("Sampling frequency", bdf.fs)  # added
    print("Shape of EEG array:", bdf.shape())  # added
    print("List of channels:", len(bdf.channels), bdf.channels)  # 64 EEG channels + 8 EXG channels
    print("Number of seconds in file:", bdf.duration())
    bdf.data /= 1e6
    # Save photodiode data
    photodiode = bdf.statusch
    trg_channel = np.zeros((bdf.shape()[-1]))
    print("Extracting events...")
    bdf.channels[bdf.channels.index('EXG4')] = 'photodiode'
    print('Updated channel list:', bdf.channels)  # checking that there's no EXG4 anymore and there's a channel called 'photodiode'
    bdf.data[bdf.channels.index('photodiode')] = start_of_clip_triggers(photodiode, bdf.fs, False)

    # MNE
    info = mne.create_info(ch_names=bdf.channels,
                           ch_types=['eeg'] * len(dcu_channels) + ['stim'],
                           sfreq=bdf.fs
                           )
    print(bdf.data.shape)
    print(bdf.data[:2, :5])  # check data are actual numbers
    raw = mne.io.RawArray(bdf.data, info)
    raw.set_montage('biosemi64')
    if str(sid) in bad_channels.keys():  # bad channels for this participant are interpolated
        bad_ = []
        for ch in bad_channels[str(sid)]:
            if ch in raw.ch_names:
                bad_.append(ch)
        if len(bad_) > 0:
            print('Bad channels:', bad_)
            raw.info['bads'] = bad_
            raw.interpolate_bads(reset_bads=True)
    print(raw)
    if raw.info['sfreq'] > 2048:
        raw.resample(sfreq=2048.)
    # Save it
    #raw.plot(n_channels=32)
    raw.save(fname='%srawArray_s%d.raw.fif' % (cst.path_to_processed_data, sid), overwrite=True)
    #plt.show()


if __name__ == '__main__':
    fnames = sorted(glob("%s*.bdf" % cst.path_to_data))
    for f, fname in enumerate(fnames):
        sname = fname.split('\\')[-1].split('.')[0]
        print(f"Subject {f + 1}/{len(fnames)}: {fname} ({sname})")
        if os.path.exists('%srawArray_s%d.raw.fif' % (cst.path_to_processed_data, conversion_dict[sname])):
            print("Subject already processed.")
            if not sname == 'quebec':
                continue
        main(fname, conversion_dict[sname])

