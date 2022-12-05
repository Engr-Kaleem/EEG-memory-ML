import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import mne
from MLEngine import FilterBank
from FBCSP import FBCSP
from MLEngine import MLEngine
import CSP

epochs = mne.read_epochs('data/clean_s10_erp_epochs.fif', preload=True)
print(epochs['EV_ENC'].shape,epochs[EV_NO_ENC].shape)
eeg_data = epochs.get_data()
m_filters = 2
fbank = FilterBank(fs=100.0)
fbank_coeff = fbank.get_filter_coeff()
filtered_data = fbank.filter_data(eeg_data)
labels = epochs.events[:, -1]
labels[labels==1]=0
labels[labels==2]=1
print(epochs)
print(f'unique labels{labels[1:30]}')
print('shape of data{filtered_data.shape{} and labels{len(labels)}}')
fbcsp = FBCSP(m_filters)
fbcsp.fit(filtered_data, labels)
features_mat = fbcsp.transform(filter_data, class_idx=0)
""" #Use SMOTE to oversample the minority class
oversample = SMOTE()
over_X, over_y = oversample.fit_resample(X, y)
over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y)
#Build SMOTE SRF model
SMOTE_SRF = RandomForestClassifier(n_estimators=150, random_state=0)
#Create Stratified K-fold cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scoring = ('f1', 'recall', 'precision')
#Evaluate SMOTE SRF model
scores = cross_validate(SMOTE_SRF, over_X, over_y, scoring=scoring, cv=cv)
#Get average evaluation metrics
print('Mean f1: %.3f' % mean(scores['test_f1']))
print('Mean recall: %.3f' % mean(scores['test_recall']))
print('Mean precision: %.3f' % mean(scores['test_precision']))

#Randomly spilt dataset to test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)
#SMOTE SRF prediction result
y_pred = SMOTE_SRF.predict(X_test)
#Create confusion matrix
fig = plot_confusion_matrix(SMOTE_SRF, X_test, y_test, display_labels=['Will Not Buy', 'Will Buy'], cmap='Greens')
plt.title('SMOTE + Standard Random Forest Confusion Matrix')
plt.show() """