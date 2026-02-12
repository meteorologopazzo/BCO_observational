import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussian_filter_nan(arr, sigma, **kwargs):
    """Gaussian filter that ignores NaNs."""
    nan_mask = np.isnan(arr)
    arr_filled = np.where(nan_mask, 0, arr)
    weights = (~nan_mask).astype(float)
    
    filtered = gaussian_filter1d(arr_filled, sigma=sigma, **kwargs)
    norm = gaussian_filter1d(weights, sigma=sigma, **kwargs)
    
    # Avoid divide-by-zero where all values were NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        filtered /= norm
    filtered[norm == 0] = np.nan
    return filtered


def esat(t, p):
    """
    usage: es = qsat(t,p)
    Returns saturation vapor pressure es (mb) given t(K) and p(mb).

    After Buck, 1981: J.Appl.Meteor., 20, 1527-1532

    Returns ndarray float for any numeric object input.
    """
    t2 = t-273.15
    return 6.1121 * np.exp(17.502 * t2 / (240.97 + t2)) * (1.0007 + p * 3.46e-6)

def mr_from_rh(t,p,rh, epsilon=0.622):
    
    '''usage:
       t in K
       p in hPa or mb
       rh in fraction between 0 and 1
       returns: q in kg/kg'''

    e_star = esat(t,p)
    e = rh*e_star
    return epsilon*e/p


## this function computes Pearson's correlation coefficient
## and provides and estimate of the corresponding p-value
## between variables in two different xarrays.
## Variables must have a common dimension along which to compute correlation
def xr_pearsonr(x, y, dim):
    import xarray as xr
    from scipy.stats import t
    r = xr.corr(x, y, dim=dim)
    n = xr.ufuncs.isfinite(x * y).sum(dim=dim)
    tstat = r * np.sqrt((n - 2) / (1 - r**2))
    p = xr.apply_ufunc(
        lambda tt, df: 2 * (1 - t.cdf(np.abs(tt), df=df)),
        tstat,
        n - 2,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return r, p