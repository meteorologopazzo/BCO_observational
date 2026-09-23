import numpy as np
import xarray as xr


def fit2d_withErrs(x, y, u):
    """
    function for multilinear regression of u over x and y
    reference model :  phi = phi0 + dxU + dyU + res
    x, y = distances (m) from the circle centre
    u    = variable to regress
    
    xarray input shape: (..., altitude, sonde_id) — sonde_id is moved to the
    last axis by xr.apply_ufunc's input_core_dims, so all axis=-1 reductions
    here operate over sonde_id.
    computations are fully implemented in numpy

    to be called in fit2d_xr_loc()
    """

    # Convert everything to numpy
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.array(u, copy=True)

    # Design matrix
    a = np.stack(
        [np.ones_like(x), x, y],
        axis=-1
    )

    # Invalid observations
    invalid = (
        np.isnan(u)
        | np.isnan(x)
        | np.isnan(y)
    )

    under_constraint = np.sum(~invalid, axis=-1) < 6

    # Remove invalid observations from fit
    u[invalid] = 0
    a[invalid] = 0

    # Pseudoinverse
    a_inv = np.linalg.pinv(a)

    # Regression coefficients
    intercept, dudx, dudy = np.einsum(
        "...rm,...m->r...",
        a_inv,
        u
    )

    # Covariance matrix factor
    # BE CAREFUL WITH THE INDICES AND DIMENSIONS HERE
    cov_beta = np.einsum("...ir,...jr->...ij", a_inv, a_inv)

    # Fitted values / residuals
    u_fit = (
        intercept[..., None]
        + dudx[..., None] * x
        + dudy[..., None] * y
    )

    u_err = u - u_fit

    # Don't let invalid points contribute to RSS
    u_err[invalid] = 0

    # Residual sum of squares
    u_sq_sum = np.sum(
        u_err**2,
        axis=-1
    )

    # Number of valid observations
    u_n = np.sum(
        ~invalid,
        axis=-1
    )

    # implement check :
    # the number of degrees of freedom should be more than 3
    dof = u_n - 3
    safe_dof = np.where(under_constraint, 1, dof)

    # Residual variance
    sigma2 = np.where(
        under_constraint,
        np.nan,
        u_sq_sum / safe_dof
    )

    # Standard errors
    se_beta = np.sqrt(
        sigma2[..., None]
        * np.diagonal(
            cov_beta,
            axis1=-2,
            axis2=-1
        )
    )

    se_intercept = se_beta[..., 0]
    se_dudx = se_beta[..., 1] 
    se_dudy = se_beta[..., 2] 

    # Enforce minimum number of observations
    intercept[under_constraint] = np.nan
    dudx[under_constraint] = np.nan
    dudy[under_constraint] = np.nan

    se_intercept[under_constraint] = np.nan
    se_dudx[under_constraint] = np.nan
    se_dudy[under_constraint] = np.nan

    return (
        intercept,
        dudx,
        dudy,
        se_intercept,
        se_dudx,
        se_dudy,
    )



def fit2d_xr_loc(x, y, u):
    """
    this function vectorizes fit2d_withErrs over an Xarray datarray
    """

    return xr.apply_ufunc(
        fit2d_withErrs,
        x,
        y,
        u,
        input_core_dims=[
            ["sonde_id"],
            ["sonde_id"],
            ["sonde_id"],
        ],
        output_core_dims=[
            [],
            [],
            [],
            [],
            [],
            [],
        ],
        vectorize=False,
    )




def fit_circle_loc(g, u_name):

    """ 
    this function allows to map the fit2d_xr_loc() on an xarray dataset, 
    selecting the particular variable with name u_name (str)

    since regressions should be carried out for every circle at each single altitude, 
    the xarray 
    """

    (
        intercept,
        dudx,
        dudy,
        se_intercept,
        se_dudx,
        se_dudy,
    ) = fit2d_xr_loc(
        g.x,
        g.y,
        g[u_name],
    )

    return xr.Dataset({
        f"{u_name}_mean": intercept,
        f"d_{u_name}_dx": dudx,
        f"d_{u_name}_dy": dudy,
        f"se_d{u_name}dx": se_dudx,
        f"se_d{u_name}dy": se_dudy,
        f"se_{u_name}_intercept": se_intercept,
    })
