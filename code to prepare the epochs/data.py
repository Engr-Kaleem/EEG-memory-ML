import numpy as np
import scipy as sci


def find_events(statusch):
    """Adapted from BCI-Mouse erpbase.py """
    TM = 0x0200  # TM (timing) bit set  # 512
    DV = 0x0100  # DV (data valid) bit set  # 256
    CTL = 128  # first possible stimulus
    # we make the assumption that:
    # - every chunk of data sent is followed by a TM mark lasting 1 or 2 samples
    # - all TM marks related to different chunks are separated (not sent in a row)
    # - every chunk of data starts with a byte >= 128
    # - the only byte in a chunk >= 128 is the first one
    # get rid of all the non-interesting samples
    ev_i = (statusch & (TM | DV)).nonzero()[0]  # find indices of elements having any of TM or DV set (the [0] takes the first and only dimension)
    ev = statusch[ev_i]  # ev contains a subset of statusch, the elements at indices ev_i
    # find the time marks
    # find elements with data valid bit set
    dv_i = (ev & DV).astype(bool)  # elements of ev having DV bit set (bool)
    dv = ev[dv_i].astype(sci.uint8)  # get those elements
    dv_i = ev_i[dv_i]  # real indices (in the recordings) of elements having DV bit set
    tm_ana = dv_i[(dv & CTL).astype(bool)]
    # find the start and end of each chunk in dv-indices
    sc = (dv & CTL).nonzero()[0]  # start of chunk
    ec = sci.append(sc[1:], len(dv))  # it is the first sample after the end of a chunk
    # create the output
    ev = {}
    ev['smp'] = tm_ana #tm_i
    ev['code'] = dv[sc]
    ev['data'] = [dv[(sc[i] + 1):ec[i]] for i in range(len(sc))]
    assert len(ev['smp']) == len(ev['code']), "Something is wrong!!"
    return ev


def _find_events(statusch):  # Jacobo's version
    """Adapted from BCI-Mouse erpbase.py """
    TM = 0x0200  # TM (timing) bit set
    DV = 0x0100  # DV (data valid) bit set
    CTL = 128  # first possible stimulus
    ev_i = (statusch & DV).nonzero()[0]  # find indices of elements having any of TM or DV set (the [0] takes the first and only dimension)
    ev = statusch[ev_i]  # ev contains a subset of statusch, the elements at indices ev_i
    dv = ev.astype(np.uint8)  # get those elements
    smp = (dv & CTL).astype(bool)
    smp = ev_i[smp]
    sc = (dv & CTL).nonzero()[0]  # start of chunk
    ec = np.append(sc[1:], len(dv))  # it is the first sample after the end of a chunk
    # create the output
    ev = {}
    ev['smp'] = smp
    ev['code'] = dv[sc]
    ev['data'] = [dv[(sc[i] + 1):ec[i]] for i in range(len(sc))]
    assert len(ev['smp']) == len(ev['code']), "Something is wrong!!"
    return ev


class BioSemiData:
    def __init__(self, filename=None, channels=None):
        self.file = filename
        self.channels = [] if (channels is None or channels=='Status') else channels[:-1]  # the last one is the status channel
        self.data = None
        self.filter_params = []
        self.fs = 0
        self.statusch = None
        self.whitened = False
        self.events = None
        self.labels = None
        self.ref = None
        self.verbose = True

    def load(self, num_sec=-1):
        print(self.file)
        import pyedflib as edf
        fin = edf.EdfReader(self.file)
        missing_channels = False
        for lab in self.channels:
            if lab not in fin.getSignalLabels():
                print("Channel %s not loaded (doesn't exist in data!" % lab)
                missing_channels = True
        if missing_channels:
            exit()
        num_sig = fin.signals_in_file
        smp_per_datarecord = fin.samples_in_datarecord(1)
        self.fs = fin.getSampleFrequency(0)
        length = int(fin.getNSamples()[0]) if num_sec < 0 else int(num_sec * self.fs)
        if len(self.channels) > 0:
            self.data = np.zeros((len(self.channels), length), dtype=np.float64)
        else:  # Load all data
            self.data = np.zeros((num_sig - 1, length))
            self.channels = fin.getSignalLabels()[:-1]
        buf = np.zeros((smp_per_datarecord), dtype='int32')
        self.statusch = np.zeros((0), dtype='int32')
        for record in range(int(length/self.fs)):
            fin.read_digital_signal(num_sig - 1, record * smp_per_datarecord, smp_per_datarecord, buf)
            self.statusch = np.hstack((self.statusch, buf))
        print("Loaded status channel")
        for i, lab in enumerate(fin.getSignalLabels()[:-1]):
            if lab in self.channels:
                chan_id = self.channels.index(lab)
                # Read signal
                if num_sec < 0:
                    self.data[chan_id, :] = fin.readSignal(i)
                else:
                    fin.readsignal(i, 0, length, self.data[chan_id, :])
                # Subtract mean (to get rid of the offset)  # TODO: this could be parallelised?
                self.data[chan_id, :] -= np.mean(self.data[chan_id, :])
        for lab in self.channels:
            if lab not in fin.getSignalLabels():
                print("%s not loaded (doesn't exist in data!" % lab)

    def load_statusch(self, num_sec=-1):
        print(self.file)
        import pyedflib as edf
        fin = edf.EdfReader(self.file)
        num_sig = fin.signals_in_file
        smp_per_datarecord = fin.samples_in_datarecord(1)
        self.fs = fin.getSampleFrequency(0)
        length = int(fin.getNSamples()[0]) if num_sec < 0 else int(num_sec * self.fs)
        buf = np.zeros((smp_per_datarecord), dtype='int32')
        self.statusch = np.zeros((0), dtype='int32')
        for record in range(int(length/self.fs)):
            fin.read_digital_signal(num_sig - 1, record * smp_per_datarecord, smp_per_datarecord, buf)
            self.statusch = np.hstack((self.statusch, buf))
        print("Loaded status channel")
        self.statusch = np.asarray(self.statusch)
        return self

    def shape(self):
        return self.data.shape
    
    def duration(self):
        try:
            return self.data.shape[-1] / self.fs
        except:
            return len(self.statusch) / self.fs

    def reference(self, ref_ch='all'):
        if type(ref_ch) == list:
            idx = []
            for ch in ref_ch:
                idx.append(self.channels.index(ch))
            if len(ref_ch) == 1:
                for ch in range(self.data.shape[0]):
                    self.data[ch] -= self.data[idx[0], :]
            else:
                self.data -= np.mean(self.data[np.asarray(idx), :], axis=0)
            self.ref = ref_ch
            return idx
        elif ref_ch == 'all':
            # Common Average Reference (CAR)
            self.data -= np.mean(self.data, axis=0)
            self.ref = 'CAR'
        else: # Assume one channel used as reference
            self.data -= self.data[self.channels.index(ref_ch), :]
            return self.channels.index(ref_ch)

    def get_events(self, statusch=None, save=True):
        if statusch is None:
            statusch = self.statusch
        if save:
            self.events = find_events(statusch)
            self.labels = [i.tostring().decode("utf-8") for i in self.events['data']]
        else:
            events = find_events(statusch)
            labels = [i.tostring().decode("utf-8") for i in events['data']]
            return events, labels

    def get_trials(self, code):
        # TODO: make it compatible with inputting multiple codes
        returnable = {'smp': [], 'labels': [], 'code': []}
        these = np.asarray(self.events.code) == code
        #print(these)
        returnable['smp'] = self.events['smp'][these]
        returnable['labels'] = np.asarray(self.labels)[these]
        returnable['code'] = self.events['code'][these]
        return returnable

    def get_trials_str(self, str):
        """
        Looks for a given string in the labels of the events and extracts the code and sample number of those events
        :param str: Search string
        :return: Dictionary of samples, labels and codes for the events
        """
        returnable = {'smp': [], 'labels': [], 'code': []}
        idx = []
        for i in range(len(self.labels)):
            #if str+'_' in self.labels[i].lower():
            if str in self.labels[i].lower():
                idx.append(i)
        if len(idx) == 0:
            print("No trials found")
            return []
        these = np.asarray(idx)
        # print(these)
        returnable['smp'] = self.events['smp'][these]
        returnable['labels'] = np.asarray(self.labels)[these]
        returnable['code'] = self.events['code'][these]
        return returnable

    def extract_stimuli(self, type):
        """
        Extracts the stimuli of a given type
        :param type: type of stimuli we want to extract ('alternate'/'touch'/'press')
        :return: a list of 2D arrays (each array is a trial, in channels x samples form) containing the data from those trials
        """
        if not self.events:
            self.get_events()
        tr = self.get_trials_str(type)
        print("Number of trials:", int(len(tr['labels'])/2))
        #print(tr['labels'])#, tr['code'])
        # Obtain start and end samples
        starts = []
        ends = []
        for i in range(len(tr['labels'])):
            if "start" in tr['labels'][i]:
                starts.append(tr['smp'][i])
            elif "end" in tr['labels'][i]:
                ends.append(tr['smp'][i])
        assert len(starts) == len(ends)
        trials = [self.data[:,st:end] for st, end in zip(starts, ends)]
        time_st = [self.statusch[st:end] for st, end in zip(starts, ends)]  # new status channel
        return trials, time_st
