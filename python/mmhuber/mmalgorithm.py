import warnings
import numpy as np
from numpy.linalg import inv, norm, pinv
from .basics import loss_function, loss_derivative_function, criterion, compute_alpha
from .least_squares import estimate_beta, estimate_sigma
from tqdm import tqdm

def _w_function(x, c):
    return 1.0 * (x<=c) + (c*(1.0/x))*(x>c)


def _scalar_w(a,b,w):
    return (a*b*w).sum()


def _norm_w(x,w):
    return _scalar_w(x,x,w)


def hubreg(y, X, c, beta_0=None, sigma_0=None, epsilon=1e-5, mu='optimal', lbda='optimal', 
           mu_0=0.0, lbda_0=0.0, n_iter=100, check_decreasing=False, return_all=False, pbar=False):
    """Robust linear regression with scale by minimizing Huber's criterion
       with a Majorization-Minimization algorithm.
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Outputs in the model.
    X : array-like, shape (n_samples, n_features)
        Inputs in the model.
    c : float
        User-defined threshold for Huber's loss.
    beta_0 : either string 'LS' or array-like, shape (n_features,), optional
        First guess of regression coefficients.
        If 'LS', will use the Least-squares solution.
        By default, will use a zero vector.
    sigma_0 : either string 'LS' or float, optional
        First guess of scale value.
        If 'LS', will use the Least-squares solution.
        By default, will use median statistic.
    epsilon : float, optional
        Convergence tolerance value, by default 1e-5.
    mu : either string 'optimal' or float, optional
        Learning rate for beta. If 'optimal', the learning rate will be updated.
        1 results in the standard MM algorithm without stepsize.
        By default 'optimal'.
    lbda : either string 'optimal' or float, optional
        Learning rate for sigma. If 'optimal', the learning rate will be updated.
        1 results in the standard MM algorithm without stepsize.
        By default 'optimal'.
    mu_0: float, optional
        Initial value for stepsize in 'optimal' setup, by default 0.0.
    lbda_0: float, optional
        Initial value for stepsize in 'optimal' setup, by default 0.0.
    n_iter: int, optional
        Number of iterations max, by default 100.
    check_decreasing: bool, optional
        Sanity check to see that the criterion is decreasing and stop when not, by default False.
    return_all: bool, optional
        Return all the iterates and value of criterion (if check_decreasing=True) or not, by default False.
    p_bar: bool, optional
        Display or not a progressbar, by default False.

    
    Returns
    -------
    beta : array-like, shape (n_samples,)
        Final guess of regression coefficients. Only when return_all=False.
    sigma : float
        Final guess of scale value. Only when return_all=False.
    estimates_list: list
        List of all the estimates iterates in te form of a tuple (beta, sigma). Only when return_all=True.
    L: list
        List of all the value of the criterion at each iteration. Only when return_all=True and check_decreasing=True.
    """

    # Managing input if 1-D
    if isinstance(beta_0, (int, float, complex)):
        beta_0 = np.array(beta_0).reshape((1,1))

    # Initialization
    n_samples, n_features = X.shape
    X_plus = pinv(X)
    alpha = compute_alpha(c)

    # Managing initial guesses
    beta, sigma = (beta_0, sigma_0)
    if beta_0 is None:
        beta = np.zeros((n_features,1))
    elif isinstance(beta_0, str) and beta_0 == 'LS':
        beta = np.array(estimate_beta(X, y))
    if sigma_0 is None:
        sigma = 1.4826*np.median(abs(y))
    elif isinstance(sigma_0, str) and sigma_0 == 'LS':
        sigma = estimate_sigma(X, y, beta)
 
    
    # Defining variables used in the algorithm
    beta = beta.reshape((n_features, 1))
    mu_algorithm,  lbda_algorithm = mu, lbda
    if mu == 'optimal':
        mu_algorithm = mu_0
    if lbda == 'optimal':
        lbda_algorithm = lbda_0
    if check_decreasing:
        L = [criterion(y, X, beta, sigma, c, alpha)]  # Contains the values of the criterion at each iteration
    estimates_list = [] # Contains the values of the estimates at each iteration

    # Updates
    constant_value = (1/np.sqrt((n_samples-n_features)*2*alpha))
    n = 0
    convergence_criterion = False
    if pbar:
        bar = tqdm(total=n_iter)
    while (n<n_iter) and not convergence_criterion:
        
        # Step 1: Update residual
        r = y.reshape((n_samples,1)) - X@beta
        
        # Step 2: Update tau
        tau =  constant_value * norm(loss_derivative_function(r/sigma, c), ord=2)

        # Step 3: Update stepsize for sigma if needed
        if lbda == 'optimal':
            temp = loss_derivative_function(r/(sigma*tau**(lbda_algorithm)), c) * constant_value
            lbda_temp = lbda_algorithm + np.log(norm(temp))/np.log(tau)
            lbda_algorithm = np.max([0.01, np.min([lbda_temp, 1.99])])
        
        # Step 4: Update the scale estimate
        sigma_new = sigma*tau**(lbda_algorithm)

        # Step 5: Update delta
        temp = loss_derivative_function(r/sigma_new, c)*sigma_new
        delta = X_plus @ temp.reshape((n_samples, 1))

        # Step 6: Update the stepsize for beta if needed
        if mu == 'optimal':
            z = X@delta
            w = _w_function(np.abs(r-mu_algorithm*z)/sigma_new,c)
            mu_temp = _scalar_w(r, z, w) / _norm_w(z,w)
            mu_algorithm = np.max([0.01, np.min([mu_temp, 1.99])])

        # Step 7: Update the regression estimate
        beta_new = beta + mu_algorithm*delta

        # Step 8: Managing convergence
        if check_decreasing:
            L.append( criterion(y, X, beta_new, sigma_new, c, alpha) )
            
            if L[n+1]>L[n]:
                print('Huber criterion has increased at this iteration. Stopping here.')
                beta_new, sigma_new = beta, sigma
                convergence_criterion = True

        beta, sigma = (beta_new, sigma_new)
        estimates_list.append((np.squeeze(beta), sigma))
        convergence_criterion = convergence_criterion or \
                            ( (norm(mu_algorithm*delta)/norm(beta) < epsilon) and \
                                (np.abs(tau**(lbda_algorithm)-1) < epsilon) )
        n += 1

        if pbar:
            bar.update(1)


    if pbar:
        bar.close()
    if return_all:
        if check_decreasing:
            return (estimates_list, L)
        else:
            return estimates_list
    else:
        return (np.squeeze(beta), sigma)


def hubniht(y, X, c, K, support_0=None, beta_0=None, sigma_0=None, epsilon=1e-5, mu='optimal', lbda='optimal', 
           mu_0=0.0, lbda_0=0.0, n_iter=100, check_decreasing=False, return_all=False, pbar=False):
    """Normalized Iterative Hard Thresholding (SNIHT) algorihtm of real-valued 
       signals using Huber's criterion for joint estimation of regression and
       scale.
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Outputs in the model.
    X : array-like, shape (n_samples, n_features)
        Inputs in the model (dictionary used).
    c : float
        User-defined threshold for Huber's loss.
    K : int
        Number of non-zero elements in beta.
    support_0 : list of int, optional
        Initial guess of the support, by default None.
    beta_0 : either string 'LS' or array-like, shape (n_features,), optional
        First guess of regression coefficients.
        If 'LS', will use the Least-squares solution.
        By default, will use a zero vector.
    sigma_0 : either string 'LS' or float, optional
        First guess of scale value.
        If 'LS', will use the Least-squares solution.
        By default, will use median statistic.
    epsilon : float, optional
        Convergence tolerance value, by default 1e-5.
    mu : either string 'optimal' or float, optional
        Learning rate for beta. If 'optimal', the learning rate will be updated.
        1 results in the standard MM algorithm without stepsize.
        By default 'optimal'.
    lbda : either string 'optimal' or float, optional
        Learning rate for sigma. If 'optimal', the learning rate will be updated.
        1 results in the standard MM algorithm without stepsize.
        By default 'optimal'.
    mu_0: float, optional
        Initial value for stepsize in 'optimal' setup, by default 0.0.
    lbda_0: float, optional
        Initial value for stepsize in 'optimal' setup, by default 0.0.
    n_iter: int, optional
        Number of iterations max, by default 100.
    check_decreasing: bool, optional
        Sanity check to see that the criterion is decreasing and stop when not, by default False.
    return_all: bool, optional
        Return all the iterates and value of criterion (if check_decreasing=True) or not, by default False.
    p_bar: bool, optional
        Display or not a progressbar, by default False.

    
    Returns
    -------
    beta : array-like, shape (n_samples,)
        Final guess of regression coefficients. Only when return_all=False.
    sigma : float
        Final guess of scale value. Only when return_all=False.
    support : list of int
        Final guess of the support. Only when return_all=False.
    estimates_list: list
        List of all the estimates iterates in te form of a tuple (beta, sigma, support). Only when return_all=True.
    L: list
        List of all the value of the criterion at each iteration. Only when return_all=True and check_decreasing=True.
    """

    # Managing input if 1-D
    if isinstance(beta_0, (int, float, complex)):
        beta_0 = np.array(beta_0).reshape((1,1))

    # Initialization
    n_samples, n_features = X.shape
    X_plus = pinv(X)
    alpha = compute_alpha(c)

    # Managing initial guesses
    beta, sigma, support = (beta_0, sigma_0, support_0)
    if beta_0 is None:
        beta = np.zeros((n_features,1))
    elif isinstance(beta_0, str) and beta_0 == 'LS':
        beta = np.array(estimate_beta(X, y))
    if sigma_0 is None:
        sigma = 1.4826*np.median(abs(np.squeeze(y)))
    elif isinstance(sigma_0, str) and sigma_0 == 'LS':
        sigma = estimate_sigma(X, y, beta)
    if support_0 is None:
        y_psi = loss_derivative_function(y/sigma, c)*sigma # winsorized observations
        delta = X.T @ y_psi # Correlations 
        indexes = np.argsort(np.squeeze(delta))
        support = indexes[-1:-K-1:-1]
 
    
    # Defining variables used in the algorithm
    beta = beta.reshape((n_features, 1))
    r = y.reshape((n_samples,1)) - X@beta
    mu_algorithm,  lbda_algorithm = mu, lbda
    if mu == 'optimal':
        mu_algorithm = mu_0
    if lbda == 'optimal':
        lbda_algorithm = lbda_0
    if check_decreasing:
        L_temp = sigma*np.sum(loss_function(r/sigma,c))/(n_samples-K) + alpha*sigma
        L = [float(L_temp)]  # Contains the values of the criterion at each iteration
    estimates_list = [] # Contains the values of the estimates at each iteration

    # Updates
    constant_value = (1/np.sqrt((n_samples-K)*2*alpha))
    n = 0
    convergence_criterion = False
    if pbar:
        bar = tqdm(total=n_iter)
    while (n<n_iter) and not convergence_criterion:
       

        # Step 2: Update tau
        tau =  constant_value * norm(loss_derivative_function(r/sigma, c), ord=2)
        # Step 3: Update stepsize for sigma if needed
        if lbda == 'optimal':
            temp = loss_derivative_function(r/(sigma*tau**(lbda_algorithm)), c) * constant_value
            lbda_temp = lbda_algorithm + np.log(norm(temp))/np.log(tau)
            lbda_algorithm = np.max([0.01, np.min([lbda_temp, 1.99])])
        
        # Step 4: Update the scale estimate
        sigma_new = sigma*tau**(lbda_algorithm)
        
    
        # Step 5: Update delta
        temp = loss_derivative_function(r/sigma_new, c)*sigma_new
        delta = X_plus @ temp.reshape((n_samples, 1))
        
        # Step 6: Update the stepsize for beta if needed
        if mu == 'optimal':
            z = X[:, support]@delta[support]
            w = _w_function(np.abs(r-mu_algorithm*z)/sigma_new,c)
            mu_temp = _scalar_w(r, z, w) / _norm_w(z,w)
            mu_algorithm = np.max([0.01, np.min([mu_temp, 1.99])])

        # Step 7: Update the regression estimate
        beta_temp = beta + mu_algorithm*delta
        indexes = np.argsort(np.squeeze(beta_temp))
        support_new = indexes[-1:-K-1:-1]
        beta_new  = np.zeros((n_features,1))
        beta_new[support_new] = beta_temp[support_new]

        if np.any(np.isnan(beta_new)):
            print(f'Error, there are nan in the new regression estimate at iteration {n+1}.')

        # Update residual
        r = y.reshape((n_samples,1)) - X@beta_new

        # Step 8: Managing convergence
        if check_decreasing:
            L_temp = sigma*np.sum(loss_function(r/sigma,c))/(n_samples-K) + alpha*sigma
            L.append( float(L_temp) )
            
            if L[n+1]>L[n]:
                print(f'Huber criterion has increased at iteration {n+1}. Stopping here.')
                beta_new, sigma_new = beta, sigma
                convergence_criterion = True

        beta, sigma, support_new = (beta_new, sigma_new, support_new)
        estimates_list.append((np.squeeze(beta), sigma, support_new))
        convergence_criterion = convergence_criterion or \
                            ( (norm(mu_algorithm*delta)/norm(beta) < epsilon) and \
                                (np.abs(tau**(lbda_algorithm)-1) < epsilon) )
        n += 1

        if pbar:
            bar.update(1)


    if pbar:
        bar.close()
    if return_all:
        if check_decreasing:
            return (estimates_list, L)
        else:
            return estimates_list
    else:
        return (np.squeeze(beta), sigma, support)

