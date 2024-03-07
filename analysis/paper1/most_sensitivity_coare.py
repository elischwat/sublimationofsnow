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
from metpy.units import units
import metpy.calc
from metpy.units import units
import pint_xarray

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import sys
sys.path.append('../splash/')
import functions_library

# # Prerequisite files
# * use script `analysis/sail/lidar_wind_profile_synoptic_wind_coherence.py` to download synoptic wind data
# * use `cumulative_sublimation.ipynb` to create daily dataset
# * use `analysis/sos/create_turbulence_dataset.ipynb` to create SoS tidy dataset and the (disdrometer) precip data

# paseed to Parallel as the n_jobs parameter
PARALLELISM = 4

# Inputs
start_date = '20221130'
end_date = '20230509'

# # Open Data
tidy_df = pd.read_parquet(f"tidy_df_{start_date}_{end_date}_noplanar_fit_clean.parquet")
tidy_df['time'] = pd.to_datetime(tidy_df['time'])
[v for v in tidy_df.variable.unique() if 'z0' in v]

# returns in Pascals
def e_sat_metpy(temp_in_c):
    millibars = 6.112*np.exp(17.67*temp_in_c / (243.5 + temp_in_c))
    return millibars*100
def e_sat_alduchov(temp_in_c):
    millibars = 6.1168*np.exp(22.587*temp_in_c / (273.86 + temp_in_c))
    return millibars*100


z0_variable_names = [
    'z0_andreas',
    'z0_andreas_weekly',
    # 'z0_windprofile_weekly',
]
z0_values_constant = [
    0.00001,
    0.00005,    
    0.0001, 
    0.0005, 
    0.001,  
    0.005,  
]

# EXTRACT VARIABLES
VARIABLES = [
    ## Input Variables for Turbpy
    'Tsurf_c',
    'Tsurf_d',
    # 'Tsurf_uw',
    # 'Tsurf_ue',
    'Tsurf_rad_d',
    'Tsnow_1_0m_d',
    'Tsnow_0_4m_d',
    'P_10m_c',
    ## Input variables
    'T_3m_c',           'T_5m_c',           'T_10m_c',           'T_15m_c',           'T_20m_c',           
    'RH_3m_c',          'RH_5m_c',          'RH_10m_c',          'RH_15m_c',          'RH_20m_c',          
    'spd_3m_c',         'spd_5m_c',         'spd_10m_c',         'spd_15m_c',         'spd_20m_c',         
    'mixingratio_3m_c', 'mixingratio_5m_c', 'mixingratio_10m_c', 'mixingratio_15m_c', 'mixingratio_20m_c', 
    ## Measurement Variables
    'SnowDepth_c',
] + z0_variable_names


# these are the names of the variables that the coare algorithm function in functions_library outputs
output_var_names = [
    'hsb', 'hlb', 'tau', 'zo', 'zot', 'zoq', 'L', 'usr', 'tsr', 'qsr', 'dter', 'dqer', \
        'hl_webb', 'Cd', 'Ch', 'Ce', 'Cdn_10', 'Chn_10', 'Cen_10', 'rr', 'rt', 'rq', 
]

latent_heat_of_vaporization = 2838 * units("J/g")
INVERSION_HEIGHT = 600

# CREATE WIDE DATAFRAME
sos_inputs_df = tidy_df[tidy_df.variable.isin(VARIABLES)].pivot_table(
    values = 'value',
    index = 'time',
    columns='variable'
).reset_index()


def run_coare(inputs):
    z0_var_name, surface_temp_variable, e_sat_curve_func_name, meas_height = inputs

    t_var =             f'T_{meas_height}m_c'
    rh_var =            f'RH_{meas_height}m_c'
    spd_var =           f'spd_{meas_height}m_c'
    mixingratio_var =   f'mixingratio_{meas_height}m_c'
    
    e_sat_curve_func = e_sat_curve_options[e_sat_curve_func_name]
    model_run_name = f"{str(surface_temp_variable)} {str(e_sat_curve_func.__name__)} {z0_var_name} {meas_height}m"
    
    results_list = []

    for time, row in sos_inputs_df.iterrows():
        bulk_inputs = [
            row[spd_var],
            row[surface_temp_variable],
            row[t_var],
            row[mixingratio_var],
            INVERSION_HEIGHT,
            row['P_10m_c'],
            meas_height - row['SnowDepth_c'],
            meas_height - row['SnowDepth_c'],
            meas_height - row['SnowDepth_c'],
            row[rh_var],
            1 # vwc "volumetric water content" doesn't matter if snow_flag = 1
        ]
        if any(pd.isnull(np.array(bulk_inputs))):
            print(f"Failed on timestamp: {time}")
            nan_filler = np.full(len(output_var_names), np.nan)
            results_list.append(nan_filler)
        else:
            if z0_var_name in z0_values_constant:
                # runs with constant z0 values
                bulk_outputs = functions_library.cor_ice_A10(
                    bulk_inputs, 
                    le_flag=1,
                    snow_flag=1,
                    sta='asfs30', 
                    snow_z0=z0_var_name
                )
                results_list.append(bulk_outputs)
            elif z0_var_name in z0_variable_names:
                bulk_outputs = functions_library.cor_ice_A10(
                    bulk_inputs, 
                    le_flag=1,
                    snow_flag=1,
                    sta='asfs30', 
                    snow_z0=row[z0_var_name]
                )
                results_list.append(bulk_outputs)
            else:
                raise ValueError(f"z0_var_name provided invalid: {z0_var_name}")

    results_df = pd.DataFrame(results_list)
    results_df.columns = output_var_names
    results_df['time'] = sos_inputs_df['time']
    results_df = results_df.set_index('time')
    results_df['hlb_gperm2s'] = results_df['hlb'] / latent_heat_of_vaporization
    # add config name to df
    results_df.insert(loc=0, column='config', value=np.full(results_df['hlb_gperm2s'].shape, model_run_name))
    return results_df

if __name__ == '__main__':
    SNOW_SURFACE_ROUGHNESS_VALUES = z0_variable_names + z0_values_constant

    surface_temp_options = [
        'Tsurf_c',
        'Tsurf_d',
        # 'Tsurf_uw',
        # 'Tsurf_ue',
        'Tsurf_rad_d',
        # Including these two for an analysis, see flux_divergence.ipynb
        'Tsnow_1_0m_d',
        'Tsnow_0_4m_d',
    ]

    meas_heights = [
        3,
        5,
        10,
        15,
        20
    ]

    e_sat_curve_options = {
        # 'e_sat_metpy': e_sat_metpy,
        'e_sat_alduchov': e_sat_alduchov
    }

    config_list = []
    for z0 in SNOW_SURFACE_ROUGHNESS_VALUES:
        for surface_temp_variable in surface_temp_options:
            for e_sat_curve_name in e_sat_curve_options.keys():
                for h in meas_heights:
                    config_list.append([z0,surface_temp_variable, e_sat_curve_name, h])  

    print("Running the models")

    config_list_tqdm = tqdm(config_list)

    processed_results =  Parallel(n_jobs = PARALLELISM)(
        delayed(run_coare)(config) for config in config_list_tqdm
    )
    
    combined_results = pd.concat(processed_results)
    
    combined_results.to_parquet("coare_model_results.parquet")