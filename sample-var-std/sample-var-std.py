import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x, dtype=float)
    x_mean = x.mean()
    sqr = 0.0
    for i in x:
        sqr += (i-x_mean)**2
    sqr /= x.size-1
    return {
        "variance": float(sqr),
        "standard_deviation": float(np.sqrt(sqr))
    }