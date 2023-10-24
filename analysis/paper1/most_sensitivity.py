# In this notebook we run TurbPy and examine the sensivity of TurbPy's MOST solutions to:
# 1. the saturation vapor pressure curve used to calculate surface water vapor pressure from surface temperature measurement
# 2. choice of surface temperature measusrement (including estimating surface temperature as 2/3-meter surface temperature)
#
# See `sat_vapor_pressure_curve.ipynb`, the analysis which preceded this, and `create_turbulence_dataset.ipynb`, where we copied the code for running TurbPy.

import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from metpy.calc import saturation_vapor_pressure
from metpy.units import units
import metpy.calc
from metpy.units import units
import pint_xarray
import turbpy


# # Prerequisite files
# * use script `analysis/sail/lidar_wind_profile_synoptic_wind_coherence.py` to download synoptic wind data
# * use `cumulative_sublimation.ipynb` to create daily dataset
# * use `analysis/sos/create_turbulence_dataset.ipynb` to create SoS tidy dataset and the (disdrometer) precip data

# Inputs
start_date = '20221130'
end_date = '20230517'
met_gothic = 'gucmetM1.b1'
tsi_sail_code = 'guctsiskycoverM1.b1'
data_dir = '/data2/elilouis/sublimationofsnow'
username = os.getenv("ARM_USERNAME")
token = os.getenv("ARM_TOKEN")

# # Open Data 
tidy_df_30Min = pd.read_parquet(f"../sos/tidy_df_30Min_{start_date}_{end_date}_noplanar_fit.parquet")
tidy_df_5Min = pd.read_parquet(f"../sos/tidy_df_{start_date}_{end_date}_noplanar_fit.parquet")
    
tidy_df_5Min['time'] = pd.to_datetime(tidy_df_5Min['time'])
tidy_df_30Min['time'] = pd.to_datetime(tidy_df_30Min['time'])

# returns in Pascals
def e_sat_metpy(temp_in_c):
    millibars = 6.112*np.exp(17.67*temp_in_c / (243.5 + temp_in_c))
    return millibars*100
def e_sat_alduchov(temp_in_c):
    millibars = 6.1168*np.exp(22.587*temp_in_c / (273.86 + temp_in_c))
    return millibars*100

# EXTRACT VARIABLES
VARIABLES = [
    ## Input Variables for Turbpy
    'Tsurf_c',
    'Tsurf_d',
    'Tsurf_rad_d',
    'P_10m_c',
    ## Input variables for calculating dewpoint temperature
    'T_3m_c',
    'RH_3m_c',
    ## Input Variables
    'spd_3m_c',
    'P_10m_c',
    # 'Tpot_3m_c',
    'airdensity_3m_c',
    'mixingratio_3m_c',
    # 'T_3m_c',
    ## Measurement Variables
    'w_h2o__3m_c',
    'u*_3m_c',
    'Ri_3m_c',
    'SnowDepth_d',
]

# CREATE WIDE DATAFRAME
variables_df = tidy_df_30Min[tidy_df_30Min.variable.isin(VARIABLES)].pivot_table(
    values = 'value',
    index = 'time',
    columns='variable'
).reset_index()

# Calculate surface temperature estimated as dewpoint temperature
variables_df['vaporpressure_3m_c'] = variables_df.apply(
    lambda row: metpy.calc.vapor_pressure( 
        metpy.calc.add_height_to_pressure(
            row['P_10m_c']*units("millibars"), -7*units("m")
        ),  
    row['mixingratio_3m_c']
    ).to(units("pascals")
).magnitude, axis=1)

variables_df['Td_3m_c'] = variables_df['vaporpressure_3m_c'].apply(lambda x: metpy.calc.dewpoint(x*units("Pa")).magnitude)

# MAKE CONVERSIONS
# convert from ˚C to K
variables_df['Td_3m_c'] = (variables_df['Td_3m_c'].values * units("celsius")).to("kelvin").magnitude
variables_df['Tsurf_c'] = (variables_df['Tsurf_c'].values * units("celsius")).to("kelvin").magnitude
variables_df['Tsurf_d'] = (variables_df['Tsurf_d'].values * units("celsius")).to("kelvin").magnitude
variables_df['Tsurf_rad_d'] = (variables_df['Tsurf_rad_d'].values * units("celsius")).to("kelvin").magnitude
variables_df['T_3m_c'] = (variables_df['T_3m_c'].values * units("celsius")).to("kelvin").magnitude

# Calculate specific humidity at 3m
variables_df['specifichumidity_3m_c'] = metpy.calc.specific_humidity_from_mixing_ratio(
    xr.DataArray(variables_df['mixingratio_3m_c'])*units('g/g')
).pint.to('g/kg').values

# Create measurement height variables
height = 3

snowDepth = variables_df['SnowDepth_d']
airTemp = variables_df['T_3m_c']
windspd = variables_df['spd_3m_c']
airPressure = (variables_df['P_10m_c'].values * units.millibar).to(units.pascal).magnitude
# the turbpy.vapPress function requires specific humidity in units of g/g
specific_humidity = xr.DataArray(variables_df['specifichumidity_3m_c'])*units('g/kg').to('g/g').magnitude
airVaporPress = turbpy.vapPress(
    specific_humidity,
    airPressure
)











def run_turbpy(inputs):
    z0, scheme_name, surface_temp_variable, e_sat_curve_func_name = inputs

    e_sat_curve_func = e_sat_curve_options[e_sat_curve_func_name]

    sfcTemp = variables_df[surface_temp_variable]
    sfcVaporPress = e_sat_curve_func(sfcTemp - 273.15)

    model_run_name = f"{str(scheme_name)} {str(surface_temp_variable)} {str(e_sat_curve_func.__name__)} {str(z0)}"

    stability_correction[model_run_name] = np.zeros_like(sfcTemp)
    conductance_sensible[model_run_name] = np.zeros_like(sfcTemp)
    conductance_latent[model_run_name] = np.zeros_like(sfcTemp)
    sensible_heat[model_run_name] = np.zeros_like(sfcTemp)
    latent_heat[model_run_name] = np.zeros_like(sfcTemp)
    zeta[model_run_name] = np.zeros_like(sfcTemp)

    for n, (tair, vpair, tsfc, vpsfc, u, airP, snDep) in enumerate(zip(
        airTemp, airVaporPress, sfcTemp, sfcVaporPress, windspd, airPressure, snowDepth
    )):
        if any(np.isnan([tair, vpair, tsfc, vpsfc, u, airP])):
            stability_correction[model_run_name][n] = np.nan
            conductance_sensible[model_run_name][n] = np.nan
            conductance_latent[model_run_name][n] = np.nan
            sensible_heat[model_run_name][n] = np.nan
            latent_heat[model_run_name][n] = np.nan
            zeta[model_run_name][n] = np.nan
        else:
            (
                conductance_sensible[model_run_name][n], 
                conductance_latent[model_run_name][n], 
                sensible_heat[model_run_name][n],
                latent_heat[model_run_name][n],
                stab_output,
                p_test
            ) = turbpy.turbFluxes(tair, airP,
                                                    vpair, u, tsfc,
                                                     vpsfc, snDep,
                                                    height, param_dict=scheme_dict[scheme_name],
                                                    z0Ground=z0, groundSnowFraction=1)
            # Get the Zeta value from the stability parameters dictionary
            if scheme_dict[scheme_name]['stability_method'] != 'monin_obukhov':
                stability_correction[model_run_name][n] = stab_output['stabilityCorrection']
                # SHOULD I JUST BE ASSIGNING NAN HERE?
                zeta[model_run_name][n] = np.nan
            else:
                stability_correction[model_run_name][n] = np.nan
                zeta[model_run_name][n] = stab_output['zeta']

    return model_run_name, latent_heat, sensible_heat, zeta


if __name__ == '__main__':

    SNOW_SURFACE_ROUGHNESS_VALUES = [0.0001, 0.0005, 0.001, 0.005]

    scheme_dict = {
        ################################################
        ###### BULK AERODYNAMIC METHODS
        ################################################
        "Standard": {
                    "stability_method": "standard"
        },
        "Louis b = 12": {
                    "stability_method": "louis",
                    "stability_params": {
                        "louis": 24.0
                    }
        },
        ################################################
        ###### MOST METHODS USING YANG LENGTHS
        ################################################
        # I added this one to the Turbpy Code base to match my own solution
        'MO Marks Dozier': {
                    'stability_method': 'monin_obukhov',
                    'monin_obukhov': {
                        'gradient_function': 'marks_dozier',
                        'roughness_function': 'yang_08'
                    },
                    'stability_params': {
                        'marks_dozier': 5.2
                    }
        },
        'MO Holtslag de Bruin': {
                    'stability_method': 'monin_obukhov',
                    'monin_obukhov': {
                        'gradient_function': 'holtslag_debruin',
                        'roughness_function': 'yang_08'
                    }
        },
        'MO Webb NoahMP': {
                    'stability_method': 'monin_obukhov',
                    'monin_obukhov': {
                        'gradient_function': 'webb_noahmp',
                        'roughness_function': 'yang_08'
                    },
        },
        "MO Beljaars Holtslag": {
                    "monin_obukhov": {
                        "gradient_function": "beljaar_holtslag",
                        'roughness_function': 'yang_08'
                    },
                    "stability_method": "monin_obukhov"
        },
        "MO Cheng Brutsaert": {
                    "monin_obukhov": {
                        "gradient_function": "cheng_brutsaert",
                        'roughness_function': 'yang_08'
                    },
                    "stability_method": "monin_obukhov"
        },
        ################################################
        ###### THESE ARE ALL USING ANDREAS LENGTHS
        ################################################
        'MO Marks Dozier andreas lengths': {
                    'stability_method': 'monin_obukhov',
                    'monin_obukhov': {
                        'gradient_function': 'marks_dozier',
                        'roughness_function': 'andreas'
                    },
                    'stability_params': {
                        'marks_dozier': 5.2
                    }
        },
        'MO Holtslag de Bruin andreas lengths': {
                    'stability_method': 'monin_obukhov',
                    'monin_obukhov': {
                        'gradient_function': 'holtslag_debruin',
                        'roughness_function': 'andreas'
                    }
        },
        'MO Webb NoahMP andreas lengths': {
                    'stability_method': 'monin_obukhov',
                    'monin_obukhov': {
                        'gradient_function': 'webb_noahmp',
                        'roughness_function': 'andreas'
                    },
        },
        "MO Beljaars Holtslag andreas lengths": {
                    "monin_obukhov": {
                        "gradient_function": "beljaar_holtslag",
                        'roughness_function': 'andreas'
                    },
                    "stability_method": "monin_obukhov"
        },
        "MO Cheng Brutsaert andreas lengths": {
                    "monin_obukhov": {
                        "gradient_function": "cheng_brutsaert",
                        'roughness_function': 'andreas'
                    },
                    "stability_method": "monin_obukhov"
        },
    }

    surface_temp_options = [
        'Tsurf_c',
        'Tsurf_d',
        'Tsurf_uw',
        'Tsurf_ue',
        'Tsurf_rad_d',
        'Td_3m_c'
    ]

    e_sat_curve_options = {
        'e_sat_metpy': e_sat_metpy,
        'e_sat_alduchov': e_sat_alduchov
    }

    config_list = []
    for z0 in SNOW_SURFACE_ROUGHNESS_VALUES:
        for scheme_name, scheme in scheme_dict.items():
            for surface_temp_variable in surface_temp_options:
                for e_sat_curve_name in e_sat_curve_options.keys():
                    config_list.append([z0,scheme_name,surface_temp_variable, e_sat_curve_name])

    # Initialzie dictionaries for containing output
    stability_correction = {}
    conductance_sensible = {}
    conductance_latent = {}
    sensible_heat = {}
    latent_heat = {}
    zeta = {}

    print("Running the models")
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm

    config_list_tqdm = tqdm(config_list)

    processed_results =  Parallel(n_jobs = 64)(
        delayed(run_turbpy)(config) for config in config_list_tqdm
    )

    df = pd.DataFrame()
    for result in processed_results:
        model_run_name, latent_heat, sensible_heat, zeta = result
        new_df = pd.DataFrame({
                    'time': variables_df.time.values,
                    'config': np.full(len(latent_heat[model_run_name]), model_run_name),
                    'latent heat flux': latent_heat[model_run_name],
                    'sensible heat flux': sensible_heat[model_run_name],
                    'zeta': zeta[model_run_name]
                })
        # convert from W/m^2 to g/m^2/s
        new_df['latent heat flux'] = - new_df['latent heat flux']/2838
        new_df['sensible heat flux'] = - new_df['sensible heat flux']
        # convert from W/m^2 to ˚C*m/s
        new_df[f'sensible heat flux'] = (new_df[f'sensible heat flux']/(variables_df['airdensity_3m_c']*0.718*1000))
        df = pd.concat([df, new_df])

    df.to_parquet("model_results.parquet")