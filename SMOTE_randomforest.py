import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import mne
from MLEngine import FilterBank
from FBCSP import FBCSP
import pandas as pd
import numpy as np
from metrics_plot import plot_metrics
from sklearn.metrics import f1_score,precision_score,recall_score
''' training_accuracy = []
testing_accuracy = []
f1=[]
prec=[]
recal=[] '''
subs=14
clf='SMOTE+TREE'
All_meterics=np.zeros([6, subs])

selected='seendata'

for sub in range(1,subs+1,1):
    epochs = mne.read_epochs('data/clean_s'+str(sub)+'_erp_epochs.fif', preload=True) 
    
    #epochs = mne.read_epochs('data/clean_s10_erp_epochs.fif', preload=True)
    eeg_data = epochs.get_data()
    # epochs = mne.read_epochs('data/clean_s10_erp_epochs.fif', preload=True)
    print(epochs['EV_ENC'],epochs['EV_NO_ENC'])
  
    m_filters = 2
    fbank = FilterBank(fs=100.0)
    fbank_coeff = fbank.get_filter_coeff()
    filtered_data = fbank.filter_data(eeg_data)
    labels = epochs.events[:, -1]
    labels[labels==1]=0
    labels[labels==2]=1
    ''' print(epochs)
    print(f'unique labels{labels[1:30]}')
    print(f'shape of data{filtered_data.shape} and labels{len(labels)}') '''
    
    fbcsp = FBCSP(m_filters)
    fbcsp.fit(filtered_data, labels)
    features_mat = fbcsp.transform(filtered_data, class_idx=0)
    #print(f'feature shape after transform{features_mat.shape}')
    #Use SMOTE to oversample the minority class
    oversample = SMOTE()
    over_X, over_y = oversample.fit_resample(features_mat,labels)
    over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y)
    #Build SMOTE SRF model
    SMOTE_SRF = RandomForestClassifier(n_estimators=150, random_state=0)
    #Create Stratified K-fold cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scoring = ('accuracy','f1', 'recall', 'precision')
    #Evaluate SMOTE SRF model
    scores = cross_validate(SMOTE_SRF, over_X, over_y, scoring=scoring, cv=cv,return_train_score=True)
    #Get average evaluation metrics
    df=pd.DataFrame(scores)
    All_meterics[0,sub-1]=sub
    All_meterics[1,sub-1]= mean(scores['train_accuracy'])
    All_meterics[2,sub-1]= mean(scores['test_accuracy'])
        
    #Randomly spilt dataset to test and train set
    X_train, X_test, y_train, y_test = train_test_split(features_mat, labels, test_size=0.1, stratify=labels)
    print(f'tarin={X_train.shape},test={X_test.shape},tr_label{y_train.shape},tst_lable{y_test.shape}')
    #Train SMOTE SRF
    SMOTE_SRF.fit(over_X_train, over_y_train)
    #SMOTE SRF prediction result
    y_pred = SMOTE_SRF.predict(X_test)
    f1_test = f1_score(y_test, y_pred)
    print('The f1 score for the testing data:', f1_test)    
    recal_test = recall_score(y_test, y_pred)
    print('The recall score for the testing data:', recal_test)  
    precision_test = precision_score(y_test, y_pred)
    print('The precison score for the testing data:', precision_test)
    All_meterics[3,sub-1]= f1_test
    All_meterics[4,sub-1] = recal_test
    All_meterics[5,sub-1] = precision_test
    #Create confusion matrix
    #fig = confusion_matrix(SMOTE_SRF, X_test, y_test, display_labels=['EV_ENC', 'EV_NO,ENC'], cmap='Greens')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['EV_ENC', 'EV_NO,ENC'])
    cmd.plot()
    plt.savefig(f'confmatSMOTE/subj{sub}clf{clf}')
    
plot_metrics(All_meterics,clf,selected)