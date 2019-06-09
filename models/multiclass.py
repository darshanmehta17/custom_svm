from copy import copy
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mode
from tqdm import tqdm


class OneVsOneClassifier(object):
    """
    A wrapper class to perform one vs. one style multiclass classification using
    any classifier of choice. It creates one classifier instance for each pair
    of classes in the dataset. The classifier being used inside the wrapper must
    have a fit and a predict function to interface with the wrapper.
    
    Parameters:
    -----------
    clf: Object instance of the classifier to be used inside the wrapper.
    
    n_jobs: The number of parallel jobs to use for the computation. None means
    1 and -1 means using all processors. Default is 1.
    
    """
    def __init__(self, clf, n_jobs=1):
        self._clf = clf
        self._n_jobs = n_jobs
        
    
    def fit(self, X, y):
        """
        Fits all the underlying classifiers. Passes each of them a subset of the
        dataset which corresponds to the class pairs they are associated with.
        
        Parameters:
        -----------
        X: Training vectors of shape (n_samples, n_features).
        
        y: Multiclass target values of shape (n_samples, ).
        
        Returns:
        --------
        self: Instance of the current object.
        
        """
        def _get_class_specific_data(X, y, class_pair):
            valid_indices = (y == class_pair[0]) + (y == class_pair[1])
            
            X_target = X[valid_indices]
            y_target = y[valid_indices].reshape(-1, 1)
            
            return X_target, y_target

    
        def _fit_subset_classifier(clf, X, y):
            clf.fit(X, y)
            return clf
        
        classes = np.sort(np.unique(y)).ravel()  # get a list of unique classes
        class_pairs = list(combinations(classes, 2))  # generate all possible pairs of classes
        
        # Train a classifier for each pair of classes
        self._clfs = Parallel(n_jobs=self._n_jobs)(delayed(_fit_subset_classifier)
                                                   (copy(self._clf), 
                                                    *_get_class_specific_data(X, y, class_pair))
                                                    for class_pair in tqdm(class_pairs))
        
        return self
    
    
    def predict(self, X):
        """
        Estimates the best class label for each data point in X. The best label
        is picked via a majority vote for a certain class by most of the 
        classifiers. 
        
        Parameters:
        -----------
        X: Data vectors of shape (n_samples, n_features) for which 
        classification needs to be performed.
        
        Returns:
        --------
        predictions: Predicted multiclass targets for each sample in X. Has
        shape (n_samples, 1).
        
        """
        predictions_all = [clf.predict(X) for clf in self._clfs]
        predictions = mode(predictions_all, axis=0).mode.reshape(-1, 1)
        return predictions
    
    
    def score(self, X, y):
        """
        Calculates the mean accuracy of the predictions of the model on the 
        given test data as compared to the given true labels. Generates class
        predictions for each datapoint in X using the predict function and 
        compares them against the true class labels present in y.
        
        Parameters:
        -----------
        X: Data vectors of shape (n_samples, n_features) for which 
        accuracy needs to be computed.
        
        y: True class labels for the samples in X. Has shape (n_samples, 1).
        
        Returns:
        --------
        score: Mean accuracy of the model on the given test data.
        
        """
        y_pred = self.predict(X)
        score = np.mean(y_pred == y.reshape(-1, 1))
        return score
