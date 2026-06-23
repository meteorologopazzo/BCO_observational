import numpy as np
import xarray as xr
import metpy

from scipy.optimize import root_scalar

from metpy.units import units

import sys
sys.path.append("/work/mh1498/m301248/TCO_data/COARE-algorithm/Python/COARE3.5/") 
import meteo


## what is the reference layer to compute mixed layer AVERAGED quantities?
mixed_avg_levels   = slice(200, 500)

ref_bl_hgt = 500.      # m
ref_bl_density = 1.    # kg / m3

## what height is entrained air from?
entrainment_levels = slice(700, 1000)    # m

## what height does convective downdrafts come from?
downdraft_levels_eu = slice(750, 1000)
downdraft_levels_or = slice(1000, 1500)


Rd = 287.
cp = 1004. # j / kg K
Lv = 2.5e6 # J/kg
g = 9.81   # m/s2
  
 


def theta_to_T(theta, p):
    """
    input:
    - p (hPa) pressure
    - theta (K) pot temperature
    """
    return theta * (p / 1000.)**(Rd/cp)

def T_to_theta(T, p):
    """
    input:
    - p (hPa) pressure
    - T (K) temperature
    """
    return T * (p / 1000.)**(-Rd/cp)

'''
def rad_terms(sw_in, lw_in, sst):
    return (eps/(1-eps))*lw_in +eps*oc_abs*stefan*(sst)**4 + ((eta/(1-eta)) + eta*(1-oc_alb))*sw_in

def R_net(sw_in, lw_in, sst, theta_bl):
    return rad_terms(sw_in, lw_in, sst) - ((2 - eps)/(1 - eps))*eps*stefan*(theta_bl)**4'''


## compute terms in radiative fluxes, following YK22 
rad_params = {
    "eps": 0.295,
    "oc_abs": 0.97,
    "stefan": 5.67*1e-8,
    "eta": 0.02,
    "oc_alb": 0.055
}
def rad_terms(sw_in, lw_in, sst):
    return (rad_params["eps"]/(1-rad_params["eps"]))*lw_in +rad_params["eps"]*rad_params["oc_abs"]*rad_params["stefan"]*(sst)**4 + ((rad_params["eta"]/(1-rad_params["eta"])) + rad_params["eta"]*(1-rad_params["oc_alb"]))*sw_in

def rad_SB_term(theta_bl):
    return ((2 - rad_params["eps"])/(1 - rad_params["eps"])) * rad_params["eps"] * rad_params["stefan"] * (theta_bl)**4

def compute_RNet(sw_in, lw_in, sst, theta_bl):
    return rad_terms(sw_in, lw_in, sst) - rad_SB_term(theta_bl)

################################################################################################

## TO USE IN SOLVER find_theta_for_pair
## h_{cd} - s_{cd} - Lv*q_sat(s_{cd}/cp) = 0
## where I find theta as cp*\theta = s_cd
def temp_from_h(theta, p, h_cd, Lv, cp, sat_frac):
    # input:
    # - p : pressure in hPa
    # - h_cd : moist static energy at convective downdraft level, J/kg
    # - Lv : latent heat of vaporization = 2.5e6 J/kg
    # - cp : specific heat dry air = 1004.67 J/kg K
    # - sat_frac : fraction of saturation, to test when different from 1.

    es = 6.1121 * np.exp(17.502 * ( theta -273.15) / (240.97 + ( theta -273.15)))     # TRY HERE TO USE theta*(p/1000.)**(R/cp) - returns physically consistent values, but mass fluxes become negative :/
    denominator = p - 0.378 * es * (1.0007 + p * 3.46e-6)
    
    return h_cd/cp - theta - sat_frac*(Lv / cp)*0.622*es*(1.0007 + p * 3.46e-6) / denominator

def find_theta_for_pair(p, h, sat_frac):
    
    f = lambda t : temp_from_h(t, p=p, h_cd=h, Lv=Lv, cp=cp, sat_frac=sat_frac)

    try:
        sol = root_scalar(f, bracket=[270,350], method='brentq')
        return sol.root if sol.converged else np.nan
    except:
        return np.nan
    
# Vectorize the function
vectorized_theta_root = np.vectorize(find_theta_for_pair)


## this function aims at extracting the thermodynamical properties 
## from entrainment heights and downdrafts, from a single profile at a time
def properties_from_profile(profile, mixed_avg_levels, entrainment_levels, downdraft_levels, CD_sat_frac, vert_dim):

    h_cd = (cp*profile["ta"] + Lv*profile["q"] + g*profile[vert_dim]).sel({vert_dim:downdraft_levels})
    p_cd = profile["p"].sel({vert_dim:downdraft_levels}).values/100.  ## hPa

    cd_levels = profile[vert_dim].sel({vert_dim:downdraft_levels}).values

    thetaD = (vectorized_theta_root( p_cd , h_cd, sat_frac=CD_sat_frac))
    thetaD = xr.DataArray(thetaD, dims=[vert_dim], coords={vert_dim:cd_levels, "p_cd": (vert_dim, p_cd)})
    thetaD = thetaD.rename({vert_dim:"height_cd"})

    qD = CD_sat_frac*(meteo.qsea((thetaD)-273.15 , p_cd)/1e3)
    qD = xr.DataArray(qD, dims=[vert_dim], coords={vert_dim:cd_levels, "p_cd": (vert_dim, p_cd)})
    qD = qD.rename({vert_dim:"height_cd"})



    theta_out = profile.theta.sel({vert_dim:entrainment_levels})  #.rename({vert_dim:"height_e"})
    theta_bl  = profile.theta.sel({vert_dim:mixed_avg_levels}).mean(dim = vert_dim) #.rename({vert_dim:"height_e"})

    q_out = profile.q.sel({vert_dim:entrainment_levels})  #.rename({vert_dim:"height_e"})
    q_bl  = profile.q.sel({vert_dim:mixed_avg_levels}).mean(dim = vert_dim) #.rename({vert_dim:"height_e"})


    return xr.Dataset(
        dict(
            thetaD=thetaD,
            qD=qD,
            theta_out=theta_out,
            theta_bl=theta_bl,
            q_out=q_out,
            q_bl=q_bl
        )
        )


## input the properties_from_profile() results
## build the matrix with due coefficients
## to solve for the entrainment and CD mass fluxes
def solve_qTh_equations(profile_props, shf, lhf, RNet, Fa_q, Fa_th):

    # size = (Elevs, 2, 2)
    A = np.zeros((2, 2))
    A[0, 0] = (profile_props["theta_out"].mean() - profile_props["theta_bl"]).values
    A[0, 1] = (profile_props["thetaD"].weighted((profile_props["thetaD"].p_cd).diff(dim="height_cd", n=1)).mean() - profile_props["theta_bl"]).values
    A[1, 0] = (profile_props["q_out"].mean() - profile_props["q_bl"]).values
    A[1, 1] = (profile_props["qD"].weighted((profile_props["thetaD"].p_cd).diff(dim="height_cd", n=1)).mean() - profile_props["q_bl"]).values

    # shape (Elevs, 2)
    b = np.zeros((2))
    b[0] = - shf / cp - (RNet) / cp - ref_bl_density*ref_bl_hgt*Fa_th
    b[1] = - lhf / Lv - ref_bl_density*ref_bl_hgt*Fa_q

    # results shape (Elevs, 2)
    x = np.linalg.solve(A, b)

    me = xr.DataArray(x[0], attrs={"units":"kg / m2 s"})
    mD = xr.DataArray(x[1], attrs={"units":"kg / m2 s"})

    return xr.Dataset(
        dict(
            me = me, 
            mD = mD
        )
    )

################################################################################################



######  mass fluxes computation for generic profile, following YK22  #######
def compute_BL_height(profile):
    ''' to implement with criterion on MSE '''
    return 500.