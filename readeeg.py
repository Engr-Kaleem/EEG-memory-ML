import numpy as np
import matplotlib.pyplot as plt
import mne

epochs = mne.read_epochs('clean_s5_erp_epochs.fif', preload=False)
#print(epochs)

#print(epochs.events)
event_dict = {'EV_ENC': 1 ,'EV_NO_ENC': 2}
epochs.event_id = event_dict;


#epochs[3].plot(n_epochs=1);
#plt.show()


#print(epochs.events)

epochs_ENC=epochs['EV_ENC']
epochs_NO_ENC=epochs['EV_NO_ENC']

print(epochs.ch_names)

kwargs = dict(fmin=2, fmax=40, n_jobs=None)
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

# We will only plot the PSD for a single sensor in the first epoch.
ch_name = 'F4'
ch_idx = epochs.info['ch_names'].index(ch_name)
epo_idx = 0

_, ax = plt.subplots()
ax.plot(freqs_mean, psds_welch_mean[ch_idx, :], color='k',
        ls='-', label='ENC')
ax.plot(freqs_median, psds_welch_median[ ch_idx, :], color='k',
        ls='--', label='NO_ENC')

ax.set(title=f'Welch PSD ({ch_name}, Epoch {epo_idx})',
       xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
ax.legend(loc='upper right')



print(epochs_ENC.event_id)
print(epochs_NO_ENC.event_id)
epochs_NO_ENC.plot_psd(fmin=5., fmax=40., average=True)
plt.figure;
epochs_ENC.plot_psd(fmin=5., fmax=40., average=True)
plt.show()