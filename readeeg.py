import numpy as np
import matplotlib.pyplot as plt
import mne

epochs = mne.read_epochs('clean_s1_erp_epochs.fif', preload=False)
print(epochs)

event_dict = {'EV_ENC': 1 ,'EV_NO_ENC': 2}
epochs.event_id = event_dict;


epochs[3].plot(n_epochs=1);
plt.show()