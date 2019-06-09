import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


def make_dataset(n_samples=200, n_features=20, n_classes=2, random_state=None):
    """
    Generate a random dataset for classification with n_classes classes. Uses 
    scikit-learn under the hood. Has n_features // 2 informative features.
    
    Parameters:
    -----------
    n_samples: Specifies the number of datapoints in the generated dataset.
    Default is 200.
    
    n_features: Specifies the number of features in the generated dataset.
    Default is 20.
    
    n_classes: Specifies the number of classes (labels) in the generated
    dataset. Default is 2.
    
    random_state: Specifies the seed for random number generated for the
    creation of the dataset.
    
    Returns:
    --------
    X: Generated dataset samples of shape (n_samples, n_features).
    
    y: Class labels of the datapoints in X. Has shape (n_samples, 1).
    
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, n_redundant=0, 
                               n_informative=n_features // 2, 
                               random_state=random_state)
    
    return X, y.reshape(-1, 1)


def plot_misclassification_error(clf, X, y, output_file):
    """
    Plots the misclassification error of the classifier on the data provided 
    against the iterations. The passed classifier must have a param_log_ 
    attribute and a score function. The plot is saved to output_file passed.
    
    Parameters:
    -----------
    clf: Instance of the classifier for which the misclassification error needs
    to be plotted.
    
    X: Data vectors of shape (n_samples, n_features) for which misclassification
    error needs to be computed.
    
    y: True class labels for the samples in X. Has shape (n_samples, 1).
    
    output_file: Specifies path to the file where the plot must be saved.
    
    """
    # Calculate the misclassification error after each iteration on the data
    misclassification_log = []
    for iter_, weight in clf.param_log_:
        score = 1 - clf.score(X, y, weight)
        misclassification_log.append((iter_, score))
    
    plt.rcParams['figure.figsize'] = [15, 10]  # set the plot size
    plt.plot(*tuple(zip(*misclassification_log)))  # plot the data
    plt.title('Misclassification error on data vs. Iteration #')
    plt.xlabel('Iteration #')
    plt.ylabel('Misclassification Error')
    plt.savefig(output_file)  # save the plot
