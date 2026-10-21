import numpy as np
import xarray as xr

import sys
sys.path.append("/Users/alessandrostorer/Desktop/PhD/1st_yy/work/codes/BCO_observational/") # functions_collector
from functions_collector import great_circle, great_circle_haversine


#####  computation of local derivatives along a single circle  #####
def _unwrap_nan(a):
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    if ok.sum() > 1:
        out[ok] = np.unwrap(a[ok])
    return out

def unwrap_along_sonde(g):
    return xr.apply_ufunc(
        _unwrap_nan, g,
        input_core_dims=[["sonde_id"]],
        output_core_dims=[["sonde_id"]],
        vectorize=True,
        keep_attrs=True,
    )

def compute_circle_arcs(prepared_ds):
    r = np.sqrt(prepared_ds.x**2 + prepared_ds.y**2)
    theta = np.arctan2(prepared_ds.y, prepared_ds.x)

    theta_unwrapped = theta.groupby("circle").map(unwrap_along_sonde).transpose("sonde_id", "altitude")

    print(f"DEBUG, Nan in theta: {int(np.isnan(theta).sum())} vs Nans in wrapped {int(np.isnan(theta_unwrapped).sum())}" )

    s = (r * theta_unwrapped).transpose("sonde_id", "altitude")
    prepared_ds_ret = prepared_ds.copy()
    prepared_ds_ret = prepared_ds_ret.assign_coords(arc=(("sonde_id", "altitude"),s.values), 
                                            theta_unwrapped=(("sonde_id", "altitude"), theta_unwrapped.values))

    return prepared_ds_ret



def local_sonde_diff(grouped_var):

    '''
    this function computes local gradients in variable "var"
    by computing the delta between consecutive sondes over the circumference arc flown by HALO .
    This is a function that should be mapped starting from a grouped dataset:

    usage example : dqp_darc = beach3_clean.groupby("circle").map(local_sonde_diff)

    Input : 
    group == xarray dataset or datarray grouped by "circle" (see functions to prepare datasets); dims (sonde_id, altitude)
    the grouped dataset must have already "arc" as a coordinate, from compute_circle_arcs()

    var == str containing the name of the variable to differntiate
    
    Output : zonal-meridional components of the arc derivatives
    dvar_x, dvar_y  , these are DataArrays
    '''

    variable = grouped_var.transpose("sonde_id", "altitude")
    var_dims, var_coords = variable.dims, variable.coords
    variable = variable.values
    arc      = grouped_var["arc"].transpose("sonde_id", "altitude").values

    result = np.full(variable.shape, np.nan)
    for j in range(variable.shape[1]):
        ok = np.isfinite(variable[:, j]) & np.isfinite(arc[:, j])
        if ok.sum() >= 2:
            result[ok, j] = np.gradient(variable[ok, j], arc[ok, j], edge_order=2)

    # variable = grouped_var.values
    # arc = grouped_var["arc"].values

    # n_sondes, n_altitudes = variable.shape

    # result = np.full_like(variable, np.nan, dtype=np.result_type(variable, float))

    # # Interior points: central difference
    # result[1:-1, :] = (
    #     (variable[2:, :] - variable[:-2, :]) / (arc[2:, :] - arc[:-2, :])
    # )

    # # First sonde: forward difference
    # result[0, :] = (
    #     (variable[1, :] - variable[0, :])
    #     / (arc[1, :] - arc[0, :])
    # )

    # # Last sonde: backward difference
    # result[-1, :] = (
    #     (variable[-1, :] - variable[-2, :])
    #     / (arc[-1, :] - arc[-2, :])
    # )

    dvar_darc = xr.DataArray(
        result,
        dims=var_dims,
        coords=var_coords
    )

    cos_t = np.cos(dvar_darc.theta_unwrapped)
    sin_t = np.sin(dvar_darc.theta_unwrapped)

    dvar_x = -sin_t * dvar_darc
    dvar_y =  cos_t * dvar_darc


    return xr.Dataset({"dvar_x": dvar_x, "dvar_y": dvar_y})



##### FUNCTIONS TO PREPARE THE DATASETs 

## Eurec4a
def sign(x, y):
    '''
    the zonal and meridional distances from the centres of circles should be both positive and negative
    input : 
    x == lon or lat of sondes
    y == lon or lat of circles
    '''
    return xr.where(x >= y, 1.0, -1.0).where(x.notnull() & y.notnull())



def prepare_joanne34(joanne3_path, joanne4_path):

    joanne4  = xr.open_dataset(joanne4_path)  # level 4
    joanne3 = xr.open_dataset(joanne3_path)  # level 3

    halo3 = joanne3.platform_id == "HALO"
    halo4 = joanne4.platform_id == "HALO"
    joanne3 = joanne3.where(halo3, drop=True).sortby("launch_time").rename({"alt": "altitude"})
    joanne4 = joanne4.where(halo4, drop=True).sortby("circle_time").rename({"alt": "altitude"})

    ## compute sondes_per_circle
    sondes4 = joanne4.sonde_id
    sondes_per_circle = joanne4.sonde_id.shape[1] - (sondes4 == '').sum(dim="sounding")

    ## select valid sondes
    sonde_sel = sondes4.values.flatten()
    sonde_sel = sonde_sel[sonde_sel != '']
    joanne3 = joanne3.sel(sonde_id=sonde_sel)


    ## assign labels to define which circle each sonde belongs to
    circle_labels = np.repeat(
        joanne4.circle.values,
        sondes_per_circle.values
    )

    circle_lon_rep = np.repeat(
        joanne4.circle_lon.values,
        sondes_per_circle.values
    )

    circle_lat_rep = np.repeat(
        joanne4.circle_lat.values,
        sondes_per_circle.values
    )

    joanne3_clean = joanne3.assign_coords(
        circle=("sonde_id", circle_labels), 
        circle_lat_rep=("sonde_id", circle_lat_rep), 
        circle_lon_rep=("sonde_id", circle_lon_rep)
    )

    ## compute zonal - meridional distances from circle centres with great_circle
    x = sign(joanne3_clean.lon, joanne3_clean.circle_lon_rep)*great_circle_haversine(joanne3_clean.lon, joanne3_clean.lat, joanne3_clean.circle_lon_rep, joanne3_clean.lat)
    y = sign(joanne3_clean.lat, joanne3_clean.circle_lat_rep)*great_circle_haversine(joanne3_clean.lon, joanne3_clean.lat, joanne3_clean.lon, joanne3_clean.circle_lat_rep)

    joanne3_clean['x'], joanne3_clean['y'] = x, y
    joanne3_clean['x'].attrs.update({"long name": "zonal signed great circle distance of single sondes from centre of circle", "unit":"km"})
    joanne3_clean['y'].attrs.update({"long name": "meridional signed great circle distance of single sondes from centre of circle", "unit":"km"})


    return joanne3_clean.sortby("launch_time"), joanne4.sortby("circle_time")



### Orcestra
def prepare_beach34_new():
    beach4  = xr.open_dataset("ipfs://bafybeibgeeqs5uhmbqy4hz4v3pihrfisiklcetisxkf63d2r473szaprwi", engine="zarr")  # level 4
    beach3 = xr.open_dataset("ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q", engine="zarr")  # level 3

    b4 = beach4.swap_dims({"sonde": "sonde_id"})
    b4 = b4.assign_coords(circle=np.arange(b4.sizes["circle"]))
    b3 = beach3.swap_dims({"sonde": "sonde_id"})

    # 1. sonde -> circle labels from the ORIGINAL beach4 (before any subsetting)
    n = b4.sondes_per_circle.values
    assert n.sum() == b4.sizes["sonde_id"]

    sid = b4.sonde_id.values
    lab_all = xr.DataArray(np.repeat(b4.circle.values, n),    dims="sonde_id", coords={"sonde_id": sid})
    lon_all = xr.DataArray(np.repeat(b4.circle_lon.values, n), dims="sonde_id", coords={"sonde_id": sid})
    lat_all = xr.DataArray(np.repeat(b4.circle_lat.values, n), dims="sonde_id", coords={"sonde_id": sid})

    # 2. keep sondes present in both datasets
    common = np.intersect1d(b3.sonde_id, b4.sonde_id)
    b3 = b3.sel(sonde_id=common)
    b4 = b4.sel(sonde_id=common)
    lab = lab_all.sel(sonde_id=common).values

    # 5. keep only circles with >= 6 of the sondes that are actually left
    circ, cnt = np.unique(lab, return_counts=True)
    good, good_cnt = circ[cnt >= 6], cnt[cnt >= 6]
    keep = np.isin(lab, good)

    b3 = b3.isel(sonde_id=keep)
    b4 = b4.isel(sonde_id=keep).sel(circle=good)          # drop the discarded circles too

    # 3. same sondes, same order in both
    b3 = b3.sortby("launch_time")
    b4 = b4.sel(sonde_id=b3.sonde_id)

    # 4. update circle-level counts and attach labels
    b4["sondes_per_circle"] = ("circle", good_cnt)

    sel_ids = b3.sonde_id.values
    b3 = b3.assign_coords(
        circle=("sonde_id", lab_all.sel(sonde_id=sel_ids).values),
        circle_lon_rep=("sonde_id", lon_all.sel(sonde_id=sel_ids).values),
        circle_lat_rep=("sonde_id", lat_all.sel(sonde_id=sel_ids).values),
    )
    # beach4 already has a `circle` dimension, so use a different name for the per-sonde label
    b4 = b4.assign_coords(sonde_circle=("sonde_id", b3.circle.values))

    # 2. keep only sondes at least 75km far from the centre
    x = sign(b3.lon, b3.circle_lon_rep)*great_circle_haversine(b3.lon, b3.lat, b3.circle_lon_rep, b3.lat)
    y = sign(b3.lat, b3.circle_lat_rep)*great_circle_haversine(b3.lon, b3.lat, b3.lon, b3.circle_lat_rep)

    b3['x'], b3['y'] = x, y
    b3['x'].attrs.update({"long name": "zonal signed great circle distance of single sondes from centre of circle", "unit":"km"})
    b3['y'].attrs.update({"long name": "meridional signed great circle distance of single sondes from centre of circle", "unit":"km"})

    dist = np.sqrt(b3.x**2 + b3.y**2)
    keep_sonde = dist.min("altitude") > 75. 
    b3 = b3.isel(sonde_id=keep_sonde.values)

    return b3, b4



def prepare_beach34_NEW(min_distance, min_sondes_perCircle):
    beach4  = xr.open_dataset("ipfs://bafybeibgeeqs5uhmbqy4hz4v3pihrfisiklcetisxkf63d2r473szaprwi", engine="zarr")  # level 4
    beach3 = xr.open_dataset("ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q", engine="zarr")  # level 3

    b4 = beach4.swap_dims({"sonde": "sonde_id"})
    b4 = b4.assign_coords(circle=np.arange(b4.sizes["circle"]))
    b3 = beach3.swap_dims({"sonde": "sonde_id"})

    # 1. sonde -> circle labels from the ORIGINAL beach4 (before any subsetting)
    n = b4.sondes_per_circle.values
    assert n.sum() == b4.sizes["sonde_id"]
    sid = b4.sonde_id.values
    lab_all = xr.DataArray(np.repeat(b4.circle.values, n),     dims="sonde_id", coords={"sonde_id": sid})
    lon_all = xr.DataArray(np.repeat(b4.circle_lon.values, n), dims="sonde_id", coords={"sonde_id": sid})
    lat_all = xr.DataArray(np.repeat(b4.circle_lat.values, n), dims="sonde_id", coords={"sonde_id": sid})

    # 2. keep sondes present in both datasets (before labelling)
    common = np.intersect1d(b3.sonde_id, b4.sonde_id)
    b3 = b3.sel(sonde_id=common)
    b4 = b4.sel(sonde_id=common)

    b3 = b3.assign_coords(
        circle=("sonde_id", lab_all.sel(sonde_id=common).values),
        circle_lon_rep=("sonde_id", lon_all.sel(sonde_id=common).values),
        circle_lat_rep=("sonde_id", lat_all.sel(sonde_id=common).values),
    )

    # 3. x, y offsets from circle centre, then distance cut on whole sondes
    x = sign(b3.lon, b3.circle_lon_rep) * great_circle_haversine(b3.lon, b3.lat, b3.circle_lon_rep, b3.lat)
    y = sign(b3.lat, b3.circle_lat_rep) * great_circle_haversine(b3.lon, b3.lat, b3.lon, b3.circle_lat_rep)
    b3["x"], b3["y"] = x, y
    b3["x"].attrs.update({"long_name": "zonal signed great circle distance of single sondes from centre of circle", "units": "km"})
    b3["y"].attrs.update({"long_name": "meridional signed great circle distance of single sondes from centre of circle", "units": "km"})

    dist = np.sqrt(b3.x**2 + b3.y**2)
    keep_dist = (dist.min("altitude") > min_distance).values
    b3 = b3.isel(sonde_id=keep_dist)
    b4 = b4.isel(sonde_id=keep_dist)

    # 4. keep only circles with enough sondes left
    lab = b3.circle.values
    circ, cnt = np.unique(lab, return_counts=True)
    ok = cnt >= min_sondes_perCircle
    good, good_cnt = circ[ok], cnt[ok]
    keep = np.isin(lab, good)
    b3 = b3.isel(sonde_id=keep)
    b4 = b4.isel(sonde_id=keep).sel(circle=good)

    # 5. same sondes, same order in both; update circle-level info
    b3 = b3.sortby("launch_time")
    b4 = b4.sel(sonde_id=b3.sonde_id.values)
    b4["sondes_per_circle"] = ("circle", good_cnt)
    b4 = b4.assign_coords(sonde_circle=("sonde_id", b3.circle.values))

    # 6. consistency checks
    # assert np.array_equal(b3.sonde_id.values, b4.sonde_id.values)
    # assert set(np.unique(b3.circle.values)) == set(b4.circle.values)
    # assert (b3.groupby("circle").count()["sonde_id"].values == b4.sondes_per_circle.values).all()

    return b3, b4





##### COMPUTATION OF MESOSCALE AREA-AVERAGED REGRESSIONS  #####
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

    out_res = xr.Dataset({
        f"{u_name}_mean": intercept,
        f"d_{u_name}_dx": dudx,
        f"d_{u_name}_dy": dudy,
        f"se_d{u_name}dx": se_dudx,
        f"se_d{u_name}dy": se_dudy,
        f"se_{u_name}_mean": se_intercept,
    })

    return out_res

#####  END OF COMPUTATION OF MESOSCALE AREA-AVERAGED QUANTITIES  #### 

