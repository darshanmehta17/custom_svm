from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.multiclass import OneVsOneClassifier
from models.svm import CustomSVM


# Get the training and testing data for the Vowel dataset from ESL
train_dataset_url = 'https://web.stanford.edu/~hastie/ElemStatLearn//datasets/vowel.train'
test_dataset_url = 'https://web.stanford.edu/~hastie/ElemStatLearn//datasets/vowel.test'

train_data = pd.read_csv(train_dataset_url)
y_train = train_data.y.values
X_train = train_data.drop(columns=['row.names', 'y']).values

test_data = pd.read_csv(test_dataset_url)
y_test = test_data.y.values.reshape(-1, 1)
X_test = test_data.drop(columns=['row.names', 'y']).values

# Standardize the X values for both the train and the test data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the CustomSVM classifier with RBF kernel with sigma=1 and
# regularization coefficient=0.01
clf = CustomSVM(kernel='rbf', sigma=1, lambda_=0.01)

# Since the dataset contains multiple classes, we initialize the 
# OneVsOneClassifier and use it to wrap the CustomSVM
ovo_clf = OneVsOneClassifier(clf, n_jobs=-1)  # -1 instructs it to use all CPUs

# Fit the model on the training data and time the process
print('Training the models...')
start = time()
ovo_clf.fit(X_train_scaled, y_train)
end = time()
print('Training completed.')

# Calculate the accuracy of the model on the training and the testing data
train_score = ovo_clf.score(X_train_scaled, y_train)
test_score = ovo_clf.score(X_test_scaled, y_test)

# Print the performance metrics of the model
print('\nPerformance metrics:')
print('Time taken to fit the model: {0:.4f}s'.format(end - start))
print('Accuracy on the training data: {0:.4f}'.format(train_score))
print('Accuracy on the testing data: {0:.4f}'.format(test_score))
