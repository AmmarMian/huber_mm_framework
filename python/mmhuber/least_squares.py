import numpy as np
from numpy.linalg import norm, pinv

def estimate_beta(X, y):
    """ Least-square estimate of regression parameters.
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Outputs in the model.
    X : array-like, shape (n_samples, n_features)
        Inputs in the model.

    Returns
    -------
    beta : array-like, shape (n_samples,)
        Regression coefficients estimate.
    
    """

    #return np.linalg.inv(X.T@X)@X.T@y
    return pinv(X)@y

def estimate_sigma(X, y, beta):
    """ Least-square estimate of scale parameter.
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Outputs in the model.
    X : array-like, shape (n_samples, n_features)
        Inputs in the model.
    beta : array-like, shape (n_samples,)
        Regression coefficients estimate.

    Returns
    -------
    sigma : float
        Scale estimate.
    
    """

    n_samples, n_features = X.shape
    # sigma_square = 0
    # for i in range(n_samples):
    #     sigma_square += (y[i] - (X[i,:].reshape((n_features,1))).T @ beta.reshape((n_features,1)))**2
    sigma_square = norm(y.reshape((n_samples,1)) - X@beta.reshape((n_features,1)), ord=2)
    return float(np.sqrt(1/(n_samples-n_features)) * sigma_square)
    

def estimate_beta_sigma(X, y):
    """ Least-square estimate of regression and scale parameters.
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Outputs in the model.
    X : array-like, shape (n_samples, n_features)
        Inputs in the model.

    Returns
    -------
    beta : array-like, shape (n_samples,)
        Regression coefficients estimate.
    sigma : float
        Scale estimate.
    
    """
    beta = estimate_beta(X, y)
    sigma = estimate_sigma(X, y, beta)
    return (beta, sigma)

