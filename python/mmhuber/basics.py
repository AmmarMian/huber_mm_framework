import numpy as np
from scipy.stats import chi2

def loss_function(x, c):
    """Huber's loss function
    
    Parameters
    ----------
    x : array_like
        Input array.
    c : float
        User-defined threshold for Huber's loss.
    
    Returns
    -------
    array_like
        Output array
    """
    return .5*((np.abs(x)<=c)*np.abs(x)**2 + \
            (np.abs(x)>c)*(2*c*np.abs(x)-c**2))


def loss_derivative_function(x, c):
    """Huber's loss function derivative
    
    Parameters
    ----------
    x : array_like
        Input array.
    c : float
        User-defined threshold for Huber's loss.
    
    Returns
    -------
    array_like
        Output array
    """

    return (np.abs(x)<=c)*x +  (np.abs(x)>c)*(c*np.sign(x))


def lfd_pdf(x, c):
    """Huber's Least Favourable distribution pdf
    
    Parameters
    ----------
    x : array_like
        Input array.
    c : float
        User-defined threshold for Huber's loss.
    
    Returns
    -------
    array_like
        Output array
    """

    return np.exp(-loss_function(x,c))


def criterion(y, X, beta, sigma, c, alpha=.5):
    """Compute the Huber's criterion for given regression vector and scale estimates,
       according to the linear model:
       y = X*beta + v
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Outputs in the model.
    X : array-like, shape (n_samples, n_features)
        Inputs in the model.
    beta : array-like, shape (n_features,)
        Regression coefficients.
    sigma : float
        Scale value.
    c : float
        User-defined threshold for Huber's loss.
    alpha : float, optional
        Scaling factor, by default .5 (LS-loss)

    Returns
    -------
    float
        The value of the criterion  
    """

    # Asserting that we have consistent inputs
    # assert (X.ndim==2) and (y.ndim==1), "Input or Output dimensions not valid"
    # assert (X.shape[1]==len(beta)), "Dimensions of function inputs are not consistent"
    # assert (X.shape[0]==len(y)), "Dimensions of function inputs are not consistent"

    n_samples, n_features = X.shape
    # L = len(y)*alpha*sigma
    # for i in range(len(y)):
    #     L += loss_function((y[i]-X[i,:].T.reshape((1, n_features))@beta.reshape((n_features, 1)))/sigma, c)*sigma
    r = y.reshape((n_samples,1)) - X@beta
    L = sigma*np.sum(loss_function(r/sigma,c))/(n_samples-n_features) + alpha*sigma
    return float(L)


def pseudo_residual(y, X, beta, sigma, c):
    """Compute the Huber's pseudo-residual for given regression vector and scale estimates,
       according to the linear model:
       y = X*beta + v
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Outputs in the model.
    X : array-like, shape (n_features, n_samples)
        Inputs in the model.
    beta : array-like, shape (n_samples,)
        Regression coefficients.
    sigma : float
        Scale value.
    c : float
        User-defined threshold for Huber's loss.

    Returns
    -------
    array_like, shape (n_samples,)
        Output residual
    """

    # Asserting that we have consistent inputs
    assert (X.ndim==2) and (y.ndim==1), "Input or Output dimensions not valid"
    assert (len(y)==X.shape[1]==len(beta)), "Dimensions of function inputs are not consistent"

    return loss_derivative_function((y - X@beta)/sigma, c)*sigma


def chi_c(x, c):
    """ Handy function for the M-estimating equations.
    
    Parameters
    ----------
    x : array_like
        Input array.
    c : float
        User-defined threshold for Huber's loss.
    
    Returns
    -------
    array_like
        Output array
    """

    return .5*((np.abs(x)<=c)*np.abs(x)**2 + (np.abs(x)>c)*c*c)


def compute_alpha(c):
    """Compute the value of alpha constant used in Huber's criterion.
    
    Parameters
    ----------
    c : float
        User-defined threshold for Huber's loss.
    
    Returns
    -------
    float
        Value of alpha constant used in Huber's criterion.
    """

    return (c**2/2) * (1-chi2.cdf(c**2, df=1)) + .5*chi2.cdf(c**2,df=3) 


def M_estimating_equations(y, X, beta, sigma, c):
    
    # Useful values
    n_samples, n_features = X.shape
    alpha = compute_alpha(c)    
    temp1 = (y.reshape((n_samples,1)) - X@beta.reshape((n_features,1))) / sigma
    temp2 = loss_derivative_function(temp1, c)

    # First equation on beta
    eq_beta = (X.T @ temp2) / (n_samples*sigma)

    # Second equation on sigma
    eq_sigma = (temp2**2).sum()/(n_samples-n_features) - 2*alpha

    return (float(eq_beta), eq_sigma)

