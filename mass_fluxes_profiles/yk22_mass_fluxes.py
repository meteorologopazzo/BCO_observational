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
ref_p_bl = metpy.calc.height_to_pressure_std(ref_bl_hgt*units.m)
ref_p_bl = ref_p_bl.to(units.hPa).magnitude

## what height is entrained air from?
entrainment_levels = slice(700, 1000)    # m

## what height does convective downdrafts come from?
downdraft_levels_eu = slice(750, 1000)
downdraft_levels_or = slice(1000, 1500)


Rd = 287.
cp = 1004. # j / kg K
Lv = 2.5e6 # J/kg
g = 9.81   # m/s2

## compute terms in radiative balance which do not depend on \theta_BL
def rad_terms(sw_in, lw_in, sst):
    eps = 0.14
    oc_abs = 0.97        ## ocean absorptivity == emissivity
    stefan = 5.67*1e-8
    eta = 0.02
    oc_alb = 0.055
    
    return (eps/(1-eps))*lw_in +eps*oc_abs*stefan*(sst)**4 + ((eta//(1-eta)) + eta*(1-oc_alb))*sw_in

def rad_SB_term(theta_bl):
    eps = 0.14
    stefan = 5.67*1e-8
    return ((2 - eps)/(1 - eps)) * eps * stefan * theta_bl**4

################################################################################################

## TO USE IN SOLVER find_theta_for_pair
## h_{cd} - s_{cd} - Lv*q_sat(s_{cd}/cp) = 0
## where I find theta as cp*\theta = s_cd
def temp_from_h(theta, p, h_cd, Lv, cp):
    # input:
    # - p : pressure in hPa
    # - h_cd : moist static energy at convective downdraft level, J/kg
    # - Lv : latent heat of vaporization = 2.5e6 J/kg
    # - cp : specific heat dry air = 1004.67 J/kg K

    es = 6.1121 * np.exp(17.502 * (theta-273.15) / (240.97 + (theta-273.15)))
    denominator = p - 0.378 * es * (1.0007 + p * 3.46e-6)
    
    return h_cd/cp - theta - (Lv / cp) * 0.622 * es * (1.0007 + p * 3.46e-6) / denominator

def find_theta_for_pair(p, h):
    # def f(t):
    #     return temp_from_h(t, p=p, h_cd=h, Lv=Lv, cp=cp)
    
    f = lambda t : temp_from_h(t, p=p, h_cd=h, Lv=Lv, cp=cp)

    try:
        sol = root_scalar(f, bracket=[280,300], method='brentq')
        return sol.root if sol.converged else np.nan
    except:
        return np.nan
    
# Vectorize the function
vectorized_theta_root = np.vectorize(find_theta_for_pair)

################################################################################################


ingr_eu = xr.open_dataset("/work/mh1498/m301248/TCO_data/fluxes_data/EUREC4A_IngrFlux.nc")
sst_eu = ingr_eu.sst.sel(cell=[17,24,31,18,25,32, 19,26,33]).mean(dim=["cell", "time"])

fluxes_eu = xr.open_dataset("/work/mh1498/m301248/TCO_data/fluxes_data/sfcFluxes_EUREC4A.nc")
lhf_era5_eu = fluxes_eu.lhf.sel(cell=[17,24,31,18,25,32, 19,26,33]).mean(dim=["cell", "time"])
shf_era5_eu = fluxes_eu.shf.sel(cell=[17,24,31,18,25,32, 19,26,33]).mean(dim=["cell", "time"])



def mass_fluxes_mean_profile(input_ingredients, sfc_fluxes, radiosonde_ds, entrainment_levels, downdraft_levels):

    lhf, shf = sfc_fluxes[0], sfc_fluxes[1]

    non_thBL_terms_eu = rad_terms(input_ingredients.sw_global.sel(cell=0).mean(dim="time").values,
                                input_ingredients.lw_diff.sel(cell=0).mean(dim="time").values, 
                                input_ingredients.sst.sel(cell=[17,24,31,18,25,32, 19,26,33]).mean(dim=["cell", "time"]).values)


    h_cd = (cp*radiosonde_ds["ta"] + Lv*radiosonde_ds["mr"] + g*radiosonde_ds.height).mean(dim="launch_time").sel(height=downdraft_levels)

    p_cd = radiosonde_ds["p"].mean(dim="launch_time").sel(height=downdraft_levels).values

    ## find \theta_{cd}
    thetaD = vectorized_theta_root(p_cd / 100., h_cd)

    ## compute q_sat(s_{cd}/cp = \theta_{cd}) at a pressure level representative of BL depth ~ 500m
    qSat_out = meteo.qsea(  thetaD-273.15 , ref_p_bl  )/1e3    # kg / kg   # p_cd_eu/100



    ## set all remaining thermodynamics
    theta_out = theta.sel(height=entrainment_levels).values
    q_out     = q.sel(height=entrainment_levels).values

    theta_bl = theta.sel(height=mixed_avg_levels).mean()
    q_bl     = q.sel(height=mixed_avg_levels).mean()

    ## missing member of radiative cooling, dependent on \theta_BL
    thBL_term = rad_SB_term(theta_bl)





    ## matrix inversion and computation of mass fluxes
    A_eu = np.zeros((theta_out.size, 2, 2))
    A_eu[:, 0, 0] = theta_out - theta_bl
    A_eu[:, 0, 1] = np.nanmean(thetaD) - theta_bl
    A_eu[:, 1, 0] = q_out - q_bl
    A_eu[:, 1, 1] = np.nanmean(qSat_out) - q_bl

    # b_eu: shape (N, 2)
    b_eu = np.zeros((len(theta_out), 2))
    b_eu[:, 0] = - shf / cp - (non_thBL_terms_eu - thBL_term) / cp
    b_eu[:, 1] = - lhf / Lv

    # Step 3: Solve batch system: A_eu @ x_eu = b_eu
    # Use np.linalg.solve for batch matrix inversion (fastest)
    x_eu = np.linalg.solve(A_eu, b_eu)  # Shape: (N, 2)

    ## x[0] == entrainment mass flux
    ## x[1] == convective downdraft mass flux
    return x



