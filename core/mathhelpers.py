
import math
import numpy as np

### Log Space ###

# safe log mathstyle
def logp(p: float):
    if p <= 0:
        return float("-inf")
    return math.log(p)

# safe log numpystyle
def logp_np(arr: np.ndarray):
    with np.errstate(divide="ignore"):
        # lets numpy output neginf
        return np.log(arr)

# helper to sum more than two logs at once
def logsumexp(log_values, axis=None):
    log_values = np.asarray(log_values, dtype=float)
    return np.logaddexp.reduce(log_values, axis=axis)

### Normalize numpy arrays ###

# normalize a 1D vector 
def normalize_vector(vec: np.ndarray) -> np.ndarray:
    total = vec.sum()

    if total > 0:
        return vec / total

    # vec's total is under 0?...
    return vec.copy()

# normalize an array row by row
def normalize_rows(mat: np.ndarray) -> np.ndarray:

    # prepare shape of output
    out = np.zeros_like(mat, dtype=float)

    # get how many rows there are
    n_rows, _ = mat.shape

    for i in range(n_rows):
        out[i] = normalize_vector(mat[i])

    return out