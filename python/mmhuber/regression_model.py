import warnings
import numpy as np
from sklearn import linear_model, datasets
from .tools.itsample import sample, normalize
from .basics import lfd_pdf


def generate_samples(n_samples, n_features, c=1.345, sigma=1, noise='lfd', random_state=None):

    # Generate linear model without noise
    X, y, beta = datasets.make_regression(n_samples=n_samples, n_features=n_features,
                            n_informative=n_features, coef=True, random_state=random_state)
    
    # Generate noise according to the model chosen
    np.random.seed(random_state)
    if noise == 'Gaussian':
        y += sigma * np.random.randn(n_samples)

    elif noise == 'lfd':
        pdf = lambda x: (1/sigma)*lfd_pdf(x/sigma, c)
        pdf = normalize(pdf)
        y += np.array(sample(pdf, n_samples))

    elif noise == 'none':
        pass

    else:
        warnings.warn(f'Choice of noise {noise} not recognized. No noise was added.')

    return (y, X, beta, sigma)

