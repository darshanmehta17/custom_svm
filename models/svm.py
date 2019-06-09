import numpy as np


class CustomSVM(object):
    """
    Custom implementation of SVM for binary classification with support for 
        Gaussian RBF kernel, Polynomial kernel and Linear kernel. Uses Fast Gradient
        Descent algorithm to minimize smoothed hinge loss. 
    
    Parameters:
    -----------
    kernel: Specifies the kernel type to be used in the algorithm. It must be
    one of 'rbf', 'polynomial, or 'linear'. Kernel hyperparameters are passed
    via the kwagrs parameter. For more infomation of the hyperparameters, refer
    to the kwargs section and the _get_kernel function. It is used to compute
    the similarity between data points. Default is 'rbf'.
    
    h: Specifies the smoothness coefficient of the smoothed hinged loss. Default
    is 0.5.
    
    lambda_: Specifies the L2 regularization coefficient. Default is 0.1.
    
    eps: Specifies the tolerance value for the stopping criterion. Default is 
    0.001.
    
    bt_alpha: Specifies the sufficient decrease factor for backtracking line
    search. Default is 0.5.
    
    bt_eta: Specifies the factor of decrease of step size at each step in the
    backtracking line search. Default is 0.8.
    
    max_iter: Specifies the maximum number of iterations the algorithm would run
    for. The algorithm would be forced to terminate even if the convergence 
    criterion has not been reached yet. Default is 1000.
    
    init_weights: Specifies the weight vector of shape (n, 1) with which the 
    model must start. Default is set to zero vector of the size (n, 1).
    
    init_weights_fast: Specifies the initial additional weight vector of shape
    (n, 1) which would be used by the Fast Gradient Descent algorithm. Default 
    is set to zero vector of the size (n, 1).
    
    step_size_init: Specifies the initial step size used by the Fast Gradient
    Descent algorithm. Default is calculated by the _get_init_step_size using 
    the smoothness constant. 
    
    kwargs: Specifies any additional parameters required by the kernel. The 
    Gaussian RBF kernel requires providing a parameter 'sigma' and Polynomial 
    kernel requires parameters 'power' and 'bias'.
    
    Attributes:
    -----------
    params_: The final weight values of the model which minimize the smoothed
    hinge loss.
    
    objective_log_: List of objective values calculated at each iteration of
    the Fast Gradient Descent algorithm.
    
    param_log_: List of updated weights after each iteration of the Fast Gradient
    Descent algorithm.
    
    """
    def __init__(self, kernel='rbf', h=0.5, lambda_=0.1, eps=1e-3, bt_alpha=0.5, 
                 bt_eta=0.8, max_iter=1000, init_weights=None, 
                 init_weights_fast=None, step_size_init=None, **kwargs):
        self._h = h
        self._lambda_ = lambda_
        self._eps = eps
        self._bt_alpha = bt_alpha
        self._bt_eta = bt_eta
        self._max_iter = max_iter
        self._kernel = self._get_kernel(kernel, **kwargs)
        self._beta0 = init_weights
        self._theta0 = init_weights_fast
        self._step_size_init = step_size_init
    
        
    def _get_kernel(self, kernel='rbf', **kwargs):
        """
        Returns a callable to the kernel function of choice created using the 
        hyperparameters. This kernel function can be used to measure the 
        similarity between two data points or multiple data points.
        
        Parameters:
        -----------
        kernel: Specifies the kernel type to be used in the algorithm. It must 
        be one of 'rbf', 'polynomial, or 'linear'. Default is 'rbf'.
        
        kwargs: Specifies any additional parameters required by the kernel. The 
        Gaussian RBF kernel requires providing a parameter 'sigma' and 
        Polynomial kernel requires parameters 'power' and 'bias'.
        
        Returns:
        --------
        kernel_function: A callable to the prepared kernel function of choice.
        
        """
        
        def _linear_kernel():
            """
            Prepares a linear kernel as a dot product of two vectors. Performs
            the same operation on two matrices vector-wise.
            
            Returns:
            --------
            kernel_function: A callable to the linear kernel function.
            
            """
            kernel_function = lambda X, y: (X @ y.T)
            return kernel_function


        def _polynomial_kernel(power=2, bias=1):
            """
            Prepares a polynomial kernel using the provided bias and power.
            
            Parameters:
            -----------
            power: Specifies the degree to which the kernel is raised. Default is 2.
            
            bias: Specifies the bias term which is added to each element. 
            Default is 1.
            
            Returns:
            --------
            kernel_function: A callable to the polynomial kernel function.
            
            """
            kernel_function = lambda X, y: ((X @ y.T) + bias) ** power
            return kernel_function


        def _rbf_kernel(sigma=0.5):
            """
            Prepares a Gaussian RBF kernel using the provided sigma.
            
            Parameters:
            -----------
            sigma: Specifies the sigma value used in the RBF kernel. Default is 0.5.
            
            Returns:
            --------
            kernel_function: A callable to the Gaussian RBF kernel function.
            
            """
            gamma = -1 / (2 * sigma ** 2)
            kernel_function = lambda X, y: np.exp(gamma * np.square(X[:, np.newaxis] - y).sum(axis=2))
            return kernel_function
        
        kernel_mapping = {
            'rbf': _rbf_kernel,
            'polynomial': _polynomial_kernel,
            'linear': _linear_kernel
        }
        
        if kernel not in kernel_mapping.keys():
            raise ValueError("The provided kernel option %s is invalid. Please"
                             " choose from the following kernels: %s"
                             % (kernel, list(kernel_mapping.keys())))
            
        return kernel_mapping[kernel](**kwargs)
        
        
    def _compute_objective(self, K, y, beta):
        """
        Calculates the loss or the objective value of the smoothed hinge loss.
        
        Parameters:
        -----------
        K: The kernel gram matrix of shape (n, n).
        
        y: Modified ground truth labels of shape (n, 1) consisting of values {-1, 1}.
        
        beta: Specifies the weights / parameters of the model. Shape is (n, 1).
        
        Returns:
        --------
        loss: The objective loss value for the provided weights.
        
        """
        K_beta = K @ beta
        reg_val = (self._lambda_ * (beta.T @ K_beta)).ravel()[0]
        condition_exp = 1 - np.multiply(y, K_beta)
        
        mid_val = np.square(self._h + condition_exp) / (4 * self._h)
        final_cost = (mid_val * (np.abs(condition_exp) <= self._h)) + (condition_exp * (condition_exp > self._h))
        loss = np.mean(final_cost) + reg_val
        return loss
    
    
    def _computegrad(self, K, y, beta):
        """
        Computes the gradient of the objective function with respect to the
        specified weights.
        
        Parameters:
        -----------
        K: The kernel gram matrix of shape (n, n).
        
        y: Modified ground truth labels of shape (n, 1) consisting of values {-1, 1}.
        
        beta: Specifies the weights / parameters of the model. Shape is (n, 1).
        
        Returns:
        --------
        gradient: The gradient vector of shape (n, 1) for the provided weights.
        
        """
        n = beta.shape[0]
        K_beta = K @ beta
        
        reg_val = (2 * self._lambda_ * (K_beta)).reshape(-1, 1)
        condition_exp = 1 - np.multiply(y, K_beta).reshape(-1, 1)
        
        delta_condition = -1 * (y * K)
        mid_val = np.multiply((condition_exp + self._h), delta_condition) / (2 * self._h)
        
        final_betas = ((mid_val.T @ (np.abs(condition_exp) <= self._h)) + \
                        (delta_condition.T @ (condition_exp > self._h))).reshape(-1, 1) / n
        gradient = final_betas + reg_val
        return gradient
    
    
    def _computegram(self, X, Z):
        """
        Computes the kernel gram matrix which is a measure of similarity between
        each point in X and each point in Z.
        
        Parameters:
        -----------
        X: A matrix of shape (n, -1).
        
        Z: Another matrix of shape (n, -1).
        
        Returns:
        --------
        gram: The similarity matrix of shape (n, n).
        
        """
        gram = self._kernel(X, Z)
        return gram
    
    
    def _get_init_step_size(self, X):
        """
        Estimates an initial step size which would help in making a good 
        gradient descent step using the smoothness constant.
        
        Parameters:
        -----------
        X: The kernel gram matrix of size (n, n).
        
        Returns:
        --------
        step_size: Value of the initial step size estimate.
        
        """
        n = X.shape[1]
        
        mat = (X @ X.T) / n
        
        L = np.linalg.eigvalsh(mat)[-1] + self._lambda_
        step_size = 1 / L
        
        return step_size

    
    def _backtracking(self, X, y, beta, step_size_prev):
        """
        Uses backtracking line search algorithm to estimate the best step size.
        Scales down the last best step size by a factor of _bt_eta until the
        minimum move condition is satisfied.
        
        Parameters:
        -----------
        X: The kernel gram matrix of size (n, n).
        
        y: Modified ground truth labels of shape (n, 1) consisting of values {-1, 1}.
        
        beta: Specifies the weights / parameters of the model. Shape is (n, 1).
        
        step_size_prev: Specifies the last picked value of step size.
        
        Returns:
        --------
        step_size: Value of the step size.
        
        """
        step_size = step_size_prev
        
        gradient = self._computegrad(X, y, beta)
        objective_prior = self._compute_objective(X, y, beta)
        gradient_norm = np.sum(np.square(gradient))
            
        for _ in range(self._max_iter):
            beta_posterior = beta - step_size * gradient
            objective_posterior = self._compute_objective(X, y, beta_posterior)

            minimum_move = objective_prior - self._bt_alpha * step_size * gradient_norm

            if objective_posterior <= minimum_move:
                return step_size
            else:
                step_size *= self._bt_eta

        print("WARNING: Could not find a good step size in %d iterations. Might"
              " affect the convergence of the algorithm." % (self._max_iter))
            
        return step_size
    
    
    def _fast(self, X, y, beta0, theta0, step_size_init):
        """
        Updates the parameters (weights) of the model using the iterative Fast
        Gradient Descent algorithm. Minimizes the smoothed hinge loss function
        until the norm of the gradient falls below the tolerance level (eps).
        Will take a hard stop after _max_iter iterations have been completed
        despite not having satisfied the convergence criterion.
        
        Parameters:
        -----------
        X: The kernel gram matrix of size (n, n).
        
        y: Modified ground truth labels of shape (n, 1) consisting of values {-1, 1}.
        
        beta0: Specifies the weight vector of shape (n, 1) with which the 
        model must start.
        
        theta0: Specifies the initial additional weight vector of shape
        (n, 1) which would be used by the Fast Gradient Descent algorithm. 
        
        step_size_init: Specifies the initial step size used by the Fast 
        Gradient Descent algorithm.     

        Returns:
        --------
        beta: The final weight values of the model which minimize the smoothed
        hinge loss.
        
        objective_log: List of objective values calculated at each iteration of
        the Fast Gradient Descent algorithm.
        
        betas: List of updated weights after each iteration of the Fast Gradient
        Descent algorithm.
        
        """
        beta = beta0
        theta = theta0
        step_size = step_size_init
        
        objective_log = [(0, self._compute_objective(X, y, beta))]
        betas = [(0, beta)]
        
        for iter in range(self._max_iter):
            grad_beta = self._computegrad(X, y, beta)

            if np.linalg.norm(grad_beta) <= self._eps:
                return beta, objective_log, betas
            
            step_size = self._backtracking(X, y, beta, step_size)
            grad_theta = self._computegrad(X, y, theta)
            
            beta_new = theta - step_size * grad_theta
            theta = beta_new + (iter / (iter + 3)) * (beta_new - beta)
            beta = beta_new
            objective_log.append((iter + 1, self._compute_objective(X, y, beta)))
            betas.append((iter + 1, beta))
        
        print("WARNING: Fast Grad could not converge in %d iterations." 
              % (self._max_iter))
            
        return beta, objective_log, betas

    
    def fit(self, X, y):
        """
        Fits the SVM model according to the given training data.
        
        Parameters:
        -----------
        X: Training vectors of shape (n_samples, n_features).
        
        y: Target values of shape (n_samples, 1). Must contain only 2 classes.
        
        Returns:
        --------
        self: Instance of the current object.
        
        """
        self._classes = np.unique(y)
        if len(self._classes) > 2:
            raise ValueError("More than 2 classes were found in the labels "
                             "provided. Please use the one vs. one classifier "
                             "to wrap this SVM class for a multiclass "
                             "classification scenario.")
        y = np.where(y == self._classes[0], 1, -1)
        
        if self._beta0 is None:
            self._beta0 = np.zeros((X.shape[0], 1))

        if self._theta0 is None:
            self._theta0 = np.zeros((X.shape[0], 1))
            
        self._X_train = X
        K = self._computegram(X, X)  # compute the kernel gram matrix

        if self._step_size_init is None:
            self._step_size_init = self._get_init_step_size(K)
        
        beta, objective_log, betas = self._fast(K, y, self._beta0, self._theta0,
                                                self._step_size_init)

        self.params_ = beta
        self.objective_log_ = objective_log
        self.param_log_ = betas
        
        return self
        
    
    def predict(self, X, weights=None):
        """
        Performs classification on the samples in X.
        
        Parameters:
        -----------
        X: Data vectors of shape (n_samples, n_features) for which 
        classification needs to be performed.
        
        weights: Weights of the model to use for predicting the class labels.
        Has shape (n_samples, 1). It is usually an entry from the class
        attribute param_log_. Default value is the class attribute params_.
        
        Returns:
        --------
        y_pred: Class labels for the samples in X. Has shape (n_samples, 1).
        
        """
        if weights is None:
            weights = self.params_
        gram = self._computegram(self._X_train, X)
        
        y_pred_raw = np.sign(np.sum(weights * gram, axis=0)).reshape(-1, 1)
        y_pred = np.where(y_pred_raw == 1, self._classes[0], self._classes[1])
        return y_pred
    
    
    def score(self, X, y, weights=None):
        """
        Calculates the mean accuracy of the predictions of the model on the 
        given test data as compared to the given true labels.
        
        Parameters:
        -----------
        X: Data vectors of shape (n_samples, n_features) for which 
        accuracy needs to be computed.
        
        y: True class labels for the samples in X. Has shape (n_samples, 1).
        
        weights: Weights of the model to use for predicting the class labels.
        Has shape (n_samples, 1). It is usually an entry from the class
        attribute param_log_. Default value is the class attribute params_.
        
        Returns:
        --------
        score: Mean accuracy of the model on the given test data.
        
        """
        y_pred = self.predict(X, weights)
        score = np.mean(y_pred == y)
        return score
