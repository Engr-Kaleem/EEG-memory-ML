import MLEngine
from MLEngine import FilterBank
from Classifier import Classifier
from FBCSP import FBCSP
from MLEngine import MLEngine
import CSP
import mne
from sklearn.svm import SVR
import numpy as np

# Read epochs

epochs = mne.read_epochs('clean_s4_erp_epochs.fif', preload=True)
eeg_data = epochs.get_data()

# applying filter bank
fbank = FilterBank(fs=100.0)
fbank_coeff = fbank.get_filter_coeff()
filtered_data = fbank.filter_data(eeg_data)
labels = epochs.events[:, -1]
labels[labels==2]=0
print(np.unique(labels))
training_accuracy = []
testing_accuracy = []
kfold = 10
ntimes = 10
m_filters = 2
for k in range(ntimes):
    '''for N times x K fold CV'''
    CV = MLEngine()
    train_indices, test_indices = CV.cross_validate_Ntimes_Kfold(y_labels=labels, ifold=k)
    for i in range(kfold):
        train_idx = train_indices.get(i)
        test_idx = test_indices.get(i)
        print(f'Times {str(k)}, Fold {str(i)}\n')
        y_train, y_test = CV.split_ydata(labels, train_idx, test_idx)
        x_train_fb, x_test_fb = CV.split_xdata(filtered_data, train_idx, test_idx)

        y_classes_unique = np.unique(y_train)
        print(y_classes_unique)
        n_classes = len(np.unique(y_train))


        fbcsp = FBCSP(m_filters)
        fbcsp.fit(x_train_fb, y_train)
        y_train_predicted = np.zeros((y_train.shape[0], n_classes), dtype=float)
        y_test_predicted = np.zeros((y_test.shape[0], n_classes), dtype=float)

        for j in range(n_classes):
            cls_of_interest = y_classes_unique[j]
            print(f'cls_of_interest={cls_of_interest}')
            select_class_labels = lambda cls, labels: [0 if y == cls else 1 for y in labels]

            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            y_test_cls = np.asarray(select_class_labels(cls_of_interest, y_test))

            x_features_train = fbcsp.transform(x_train_fb, class_idx=cls_of_interest)
            x_features_test = fbcsp.transform(x_test_fb, class_idx=cls_of_interest)

            classifier_type = SVR(gamma='auto')
            classifier = Classifier(classifier_type)
            y_train_predicted[:, j] = classifier.fit(x_features_train, np.asarray(y_train_cls, dtype=float))
            y_test_predicted[:, j] = classifier.predict(x_features_test)
            
        print(y_train_predicted.shape,y_test_predicted.shape)
        y_train_predicted_multi = MLEngine.get_multi_class_label(y_train_predicted,0)
        y_test_predicted_multi = m_filters.get_multi_class_label(y_test_predicted,0)

        tr_acc = np.sum(y_train_predicted_multi == y_train, dtype=np.float) / len(y_train)
        te_acc = np.sum(y_test_predicted_multi == y_test, dtype=np.float) / len(y_test)

        print(f'Training Accuracy = {str(tr_acc)}\n')
        print(f'Testing Accuracy = {str(te_acc)}\n \n')

        training_accuracy.append(tr_acc)
        testing_accuracy.append(te_acc)

mean_training_accuracy = np.mean(np.asarray(training_accuracy))
mean_testing_accuracy = np.mean(np.asarray(testing_accuracy))

print('*' * 10, '\n')
print(f'Mean Training Accuracy = {str(mean_training_accuracy)}\n')
print(f'Mean Testing Accuracy = {str(mean_testing_accuracy)}')
print('*' * 10, '\n')
