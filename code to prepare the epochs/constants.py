
# Paths
path_to_data = './data/bdfs/'
path_to_dcu_data = './data/dcu_data/'
path_to_processed_data = "./data/mne_raw/"
path_to_epochs = "./data/mne_epochs/"
path_to_annotations = "./data/annotations/"
path_to_logs = "./data/logs/"


bad_channels = {'5': ['T7', 'T8'], '6': ['POz'], '36': ['POz'], '44': ['POz'], '3': ['T8', 'TP9'],
                '6': ['P7', 'TP7']}

# DCU channels:
DCU_channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'TP7', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'TP8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
gdf_channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'TP9', 'T7', 'C3', 'Cz',
                'C4', 'T8', 'TP10', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2',
                'photodiode']

#### Note: DCU has TP9/10, in BioSemi these don't exist, so we use TP7/8 instead.

sname_to_sid = {'charlie': 4, 'echo': 5, 'foxtrot': 6, 'golf': 7, 'hotel': 8, 'india': 9, 'juliett': 10, 'kilo': 11,
                'lima': 12, 'mike': 13, 'november': 14, 'oscar': 15, 'papa': 16, 'quebec': 17, 'romeo': 18,
                'sierra': 19, 'tango': 20, 'uniform': 21, 'victor': 22, 'whiskey': 23, 'xray': 24, 'yankee': 25,
                'zulu': 26, 'beta': 28,  #'epsilon': 29, # epsilon is excluded because we don't have the triggers in the bdf
                'omicron': 30, 'lambda': 31, 'theta': 32, 'sigma': 33, 'zeta': 34,
                'kappa': 35, 'omega': 36, 'mercury': 37, 'venus': 38, 'earth': 39, 'jupiter': 41, 'saturn': 42,
                'uranus': 43, 'neptune': 44, 'pluto': 45,
                'alpha': 1, 'bravo': 2, 'gamma': 3, 'mars': 40  # these are from DCU
                }

video_watched = {'charlie': 4, 'echo': 5, 'foxtrot': 6, 'golf': 7, 'hotel': 8, 'india': 9, 'juliett': 10, 'kilo': 11,
                'lima': 12, 'mike': 1, 'november': 2, 'oscar': 3, 'papa': 4, 'quebec': 5, 'romeo': 6,
                'sierra': 7, 'tango': 8, 'uniform': 9, 'victor': 10, 'whiskey': 11, 'xray': 12, 'yankee': 1,
                'zulu': 2, 'beta': 4,  #'epsilon': 5, # epsilon is excluded because we don't have the triggers in the bdf
                'omicron': 6, 'lambda': 7, 'theta': 8, 'sigma': 9, 'zeta': 10,
                'kappa': 11, 'omega': 12, 'mercury': 1, 'venus': 2, 'earth': 3, 'jupiter': 5, 'saturn': 6,
                'uranus': 7, 'neptune': 8, 'pluto': 9, 'alpha': 1, 'bravo': 2, 'gamma': 3, 'mars': 4}


EXCLUDED_PARTICIPANTS = [3, 40,    # based on signal quality
                         29,  # ?
                         5, 6, 21, 22, 23, 28, 32,  # too many epochs rejected
                         42,  # didn't do online annotation
                         20, 7, 14, 39, 44, 26, 8, 11, 38, 18, 34, 33, 45, 24, 12  # FP>30% in online annotation
                         ]
