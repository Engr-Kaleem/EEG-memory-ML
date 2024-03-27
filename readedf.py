import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import mne
import numpy as np
import matplotlib.pyplot as plt

# Define raw globally
raw = None
org_raw=None

def upload_file():
    global raw
    global org_raw
    filename = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf")])
    if filename:
        try:
            raw = mne.io.read_raw_edf(filename,preload=True)
            org_raw=raw.copy()
            print(raw)
            print(raw.info)
            update_channel_combobox()
            sfreq_label.config(text=f"Sampling Frequency: {raw.info['sfreq']} Hz")
            return raw
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading the file: {e}")
            raw = None

def update_channel_combobox():
    global raw
    if raw:
        try:
            channels = raw.info['ch_names']
            channel_combobox['values'] = channels
            if channels:
                channel_combobox.current(0)  # Select the first channel by default
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading channel names: {e}")

def plot_eeg_channel():
    global raw
    if raw is None:
        messagebox.showerror("Error", "Please upload an EDF file first.")
        return
    
    try:
        selected_channel = channel_combobox.get()
        channel_index = raw.ch_names.index(selected_channel)
        raw.compute_psd(fmax=80).plot(picks=[channel_index], exclude="bads")
        raw.plot(picks=selected_channel)
        data, times = raw[raw.ch_names.index(selected_channel), :]
        
        plt.figure(figsize=(10, 4))
        plt.plot(times, data[0], color='b')
        plt.title(f"EEG Channel: {selected_channel}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def plot_all_channels():
    global raw
    if raw is None:
        messagebox.showerror("Error", "Please upload an EDF file first.")
        return
    
    try:
        n_channels = len(raw.ch_names)
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols

        plt.figure(figsize=(15, 4 * n_rows))
        for i, ch_name in enumerate(raw.ch_names):
            data, times = raw[ch_name, :]
            plt.subplot(n_rows, n_cols, i + 1)
            plt.plot(times, data[0], color='b')
            plt.title(f"EEG Channel: {ch_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
        plt.tight_layout()
        plt.show()
        raw.plot(duration=5)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def apply_ica():
    global raw
    if raw is None:
        messagebox.showerror("Error", "Please upload an EDF file first.")
        return
    
    try:
        ica = mne.preprocessing.ICA(n_components=len(raw.ch_names), random_state=42)
        ica.fit(raw)
        ica.plot_components()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def plot_fft():
    global raw
    if raw is None:
        messagebox.showerror("Error", "Please upload an EDF file first.")
        return
    
    try:
        n_channels = len(raw.ch_names)
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols

        plt.figure(figsize=(15, 4 * n_rows))
        for i, ch_name in enumerate(raw.ch_names):
            data, times = raw[ch_name, :]
            fft_vals = np.abs(np.fft.fft(data[0]))
            freqs = np.fft.fftfreq(len(data[0]), 1/raw.info['sfreq'])
            
            plt.subplot(n_rows, n_cols, i + 1)
            plt.plot(freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2], color='b')
            plt.title(f"FFT of EEG Channel: {ch_name}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def remove_dc_and_filter():
    global raw
    if raw is None:
        messagebox.showerror("Error", "Please upload an EDF file first.")
        return
    
    # Remove DC offset and apply bandpass filter
    try:
        raw.notch_filter(60)
        raw.filter(1, 80, fir_design='firwin')
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def plot_montage():
    global raw
    if raw is None:
        messagebox.showerror("Error", "Please upload an EDF file first.")
        return
    
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        remove_dc_and_filter()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        raw.plot_sensors(ch_type='eeg', show_names=True, axes=ax)
        plt.title('EEG Montage')
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def restore_original_raw():
    global raw
    global org_raw
    if org_raw is not None:
        raw = org_raw.copy()
        messagebox.showinfo("Info", "Original data has been restored successfully.")
    else:
        messagebox.showerror("Error", "No original data available.")

# Create the main window
root = tk.Tk()
root.title("EEG Channel Plotter")

# Create widgets
edf_file_label = tk.Label(root, text="EDF File:")
edf_file_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
edf_file_entry = tk.Entry(root, width=50)
edf_file_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
upload_button = tk.Button(root, text="Upload", command=upload_file)
upload_button.grid(row=0, column=3, padx=5, pady=5)

sfreq_label = tk.Label(root, text="Sampling Frequency: - Hz")
sfreq_label.grid(row=0, column=4, padx=5, pady=5, sticky="e")

channel_label = tk.Label(root, text="Select EEG Channel:")
channel_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

channel_combobox = ttk.Combobox(root, state='readonly')
channel_combobox.grid(row=1, column=1, padx=5, pady=5)

update_channel_combobox()  # Initially populate the channel combobox

plot_button = tk.Button(root, text="Plot EEG Channel", command=plot_eeg_channel)
plot_button.grid(row=1, column=2, padx=5, pady=5)

# Add button to restore original raw
restore_button = tk.Button(root, text="Restore Original Data", command=restore_original_raw)
restore_button.grid(row=2, column=5, padx=5, pady=5)
plot_all_button = tk.Button(root, text="Plot All EEG Channels", command=plot_all_channels)
plot_all_button.grid(row=1, column=3, padx=5, pady=5)

ica_button = tk.Button(root, text="Apply ICA", command=apply_ica)
ica_button.grid(row=1, column=4, padx=5, pady=5)

fft_button = tk.Button(root, text="Plot FFT of Each Channel", command=plot_fft)
fft_button.grid(row=2, column=2, padx=5, pady=5)

montage_button = tk.Button(root, text="Show Montage", command=plot_montage)
montage_button.grid(row=2, column=3, padx=5, pady=5)

remove_dc_filter_button = tk.Button(root, text="Remove DC & Filter", command=remove_dc_and_filter)
remove_dc_filter_button.grid(row=2, column=4, padx=5, pady=5)

# Run the application
root.mainloop()
