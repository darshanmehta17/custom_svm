from time import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.svm import CustomSVM

# Get the digits dataset from scikit-learn for 2 classes
X, y = load_digits(n_class=2, return_X_y=True)
y = y.reshape(-1, 1)

random_state = 0  # set random seed to allow reproducing results

# Split the dataset into train and test in the ratio 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=random_state)

# Standardize the X values for both the train and the test data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the CustomSVM classifier with default parameters
clf_custom = CustomSVM()

# Initialize the scikit-learn SVC with default parameters
clf_sklearn = SVC()

# Fit the custom model on the training data and time the process
print('Training the custom model...')
start_custom = time()
clf_custom.fit(X_train_scaled, y_train)
end_custom = time()
print('Training completed.')

# Fit the scikit-learn model on the training data and time the process
print('Training the scikit-learn model...')
start_sklearn = time()
clf_sklearn.fit(X_train_scaled, y_train.reshape(-1, ))
end_sklearn = time()
print('Training completed.')

# Calculate the accuracy of the custom model on the training and the testing data
train_score_custom = clf_custom.score(X_train_scaled, y_train)
test_score_custom = clf_custom.score(X_test_scaled, y_test)

# Calculate the accuracy of the scikit-learn model on the training and the testing data
train_score_sklearn = clf_sklearn.score(X_train_scaled, y_train)
test_score_sklearn = clf_sklearn.score(X_test_scaled, y_test)

# Print the performance metrics of the custom model
print('\nPerformance metrics of custom model:')
print('Time taken to fit the model: {0:.4f}s'.format(end_custom - start_custom))
print('Accuracy on the training data: {0:.4f}'.format(train_score_custom))
print('Accuracy on the testing data: {0:.4f}'.format(test_score_custom))

# Print the performance metrics of the scikit-learn model
print('\nPerformance metrics of scikit-learn model:')
print('Time taken to fit the model: {0:.4f}s'.format(end_sklearn - start_sklearn))
print('Accuracy on the training data: {0:.4f}'.format(train_score_sklearn))
print('Accuracy on the testing data: {0:.4f}'.format(test_score_sklearn))
