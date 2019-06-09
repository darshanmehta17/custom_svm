from time import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.svm import CustomSVM
from utils import plot_misclassification_error


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
clf = CustomSVM()

# Fit the model on the training data and time the process
print('Training the model...')
start = time()
clf.fit(X_train_scaled, y_train)
end = time()
print('Training completed.')

# Calculate the accuracy of the model on the training and the testing data
train_score = clf.score(X_train_scaled, y_train)
test_score = clf.score(X_test_scaled, y_test)

# Print the performance metrics of the model
print('\nPerformance metrics:')
print('Time taken to fit the model: {0:.4f}s'.format(end - start))
print('Accuracy on the training data: {0:.4f}'.format(train_score))
print('Accuracy on the testing data: {0:.4f}'.format(test_score))

# Plot the misclassification error and save the image to a file
output_file = './misclassification_error_digits_custom_svm.png'
print('Saving misclassification plot to ' + output_file)
plot_misclassification_error(clf, X_train_scaled, y_train, output_file)
