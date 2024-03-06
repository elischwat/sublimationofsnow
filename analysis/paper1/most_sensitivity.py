# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: sublimationofsnow
#     language: python
#     name: python3
# ---

# %% [markdown]
# In this notebook we run TurbPy and examine the sensivity of TurbPy's MOST solutions to:
# 1. the saturation vapor pressure curve used to calculate surface water vapor pressure from surface temperature measurement
# 2. choice of surface temperature measusrement (including estimating surface temperature as 2/3-meter surface temperature)
#
# See `sat_vapor_pressure_curve.ipynb`, the analysis which preceded this, and `create_turbulence_dataset.ipynb`, where we copied the code for running TurbPy.

# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from metpy.units import units
import metpy.calc
from metpy.units import units
import pint_xarray
import turbpy
from dask.distributed import Client
import dask.dataframe as dd
from datetime import datetime


# %% [markdown]
# # Prerequisite files
# * use script `analysis/sail/lidar_wind_profile_synoptic_wind_coherence.py` to download synoptic wind data
# * use `cumulative_sublimation.ipynb` to create daily dataset
# * use `analysis/sos/create_turbulence_dataset.ipynb` to create SoS tidy dataset and the (disdrometer) precip data

# %%
# paseed to Parallel as the n_jobs parameter
PARALLELISM = 4

# %%
# Inputs
start_date = '20221130'
end_date = '20230509'

# %%
# # Open Data                     
tidy_df = pd.read_parquet(f"tidy_df_{start_date}_{end_date}_noplanar_fit_clean.parquet")
tidy_df['time'] = pd.to_datetime(tidy_df['time'])

# %%
# returns in Pascals
def e_sat_metpy(temp_in_c):
    millibars = 6.112*np.exp(17.67*temp_in_c / (243.5 + temp_in_c))
    return millibars*100
def e_sat_alduchov(temp_in_c):
    millibars = 6.1168*np.exp(22.587*temp_in_c / (273.86 + temp_in_c))
    return millibars*100

# %%
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

# %%
# EXTRACT VARIABLES
VARIABLES = [
    ## Input Variables for Turbpy
    'Tsurf_c',
    'Tsurf_d',
    # 'Tsurf_uw',
    # 'Tsurf_ue',
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
    'SnowDepth_c',
] + z0_variable_names

# %%
# CREATE WIDE DATAFRAME
variables_df = tidy_df[tidy_df.variable.isin(VARIABLES)].pivot_table(
    values = 'value',
    index = 'time',
    columns='variable'
).reset_index()

# %%
# MAKE CONVERSIONS
# convert from ˚C to K
variables_df['Tsurf_c'] = (variables_df['Tsurf_c'].values * units("celsius")).to("kelvin").magnitude
variables_df['Tsurf_d'] = (variables_df['Tsurf_d'].values * units("celsius")).to("kelvin").magnitude
# variables_df['Tsurf_uw'] = (variables_df['Tsurf_uw'].values * units("celsius")).to("kelvin").magnitude
# variables_df['Tsurf_ue'] = (variables_df['Tsurf_ue'].values * units("celsius")).to("kelvin").magnitude
variables_df['Tsurf_rad_d'] = (variables_df['Tsurf_rad_d'].values * units("celsius")).to("kelvin").magnitude
variables_df['T_3m_c'] = (variables_df['T_3m_c'].values * units("celsius")).to("kelvin").magnitude

# %%
# Calculate specific humidity at 3m
variables_df['specifichumidity_3m_c'] = metpy.calc.specific_humidity_from_mixing_ratio(
    xr.DataArray(variables_df['mixingratio_3m_c'])*units('g/g')
).pint.to('g/kg').values

# %%
# Create measurement height variables
HEIGHT = 3

# %%
timestamps = variables_df.time.values
snowDepth = variables_df['SnowDepth_c']
airTemp = variables_df['T_3m_c']
windspd = variables_df['spd_3m_c']
airPressure = (variables_df['P_10m_c'].values * units.millibar).to(units.pascal).magnitude
# the turbpy.vapPress function requires specific humidity in units of g/g
specific_humidity = xr.DataArray(variables_df['specifichumidity_3m_c'])*units('g/kg').to('g/g').magnitude
airVaporPress = turbpy.vapPress(
    specific_humidity,
    airPressure
)

# %% [markdown]
# # Organize all inputs into a dataset

# %%

SNOW_SURFACE_ROUGHNESS_VALUES = z0_variable_names + z0_values_constant
scheme_dict = {
    ################################################
    ###### BULK AERODYNAMIC METHODS
    ################################################
    "Standard": {
                "stability_method": "standard"
    },
    # "Louis b = 12": {
    #             "stability_method": "louis",
    #             "stability_params": {
    #                 "louis": 24.0
    #             }
    # },
    ################################################
    ###### MOST METHODS USING YANG LENGTHS
    ################################################
    # I added this one to the Turbpy Code base to match my own solution
    # 'MO Marks Dozier': {
    #             'stability_method': 'monin_obukhov',
    #             'monin_obukhov': {
    #                 'gradient_function': 'marks_dozier',
    #                 'roughness_function': 'yang_08'
    #             },
    #             'stability_params': {
    #                 'marks_dozier': 5.2
    #             }
    # },
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
                'stability_method': 'monin_obukhov'
    },
    "MO Cheng Brutsaert": {
                "monin_obukhov": {
                    "gradient_function": "cheng_brutsaert",
                    'roughness_function': 'yang_08'
                },
                'stability_method': 'monin_obukhov'
    },
    ################################################
    ###### THESE ARE ALL USING ANDREAS LENGTHS
    ################################################
    # 'MO Marks Dozier andreas lengths': {
    #             'stability_method': 'monin_obukhov',
    #             'monin_obukhov': {
    #                 'gradient_function': 'marks_dozier',
    #                 'roughness_function': 'andreas'
    #             },
    #             'stability_params': {
    #                 'marks_dozier': 5.2
    #             }
    # },
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
                'stability_method': 'monin_obukhov'
    },
    "MO Cheng Brutsaert andreas lengths": {
                "monin_obukhov": {
                    "gradient_function": "cheng_brutsaert",
                    'roughness_function': 'andreas'
                },
                'stability_method': 'monin_obukhov'
    },
}

surface_temp_options = [
    'Tsurf_c',
    'Tsurf_d',
    # 'Tsurf_uw',
    # 'Tsurf_ue',
    'Tsurf_rad_d'
]

e_sat_curve_options = {
    # 'e_sat_metpy': e_sat_metpy,
    'e_sat_alduchov': e_sat_alduchov
}

config_list = []
for z0 in SNOW_SURFACE_ROUGHNESS_VALUES:
    for scheme_name, scheme in scheme_dict.items():
        for surface_temp_variable in surface_temp_options:
            for e_sat_curve_name in e_sat_curve_options.keys():
                config_list.append([z0,scheme_name,surface_temp_variable, e_sat_curve_name])

# %%
config_df = pd.DataFrame(config_list).rename(columns={
    0: 'z0',
    1: 'scheme_name',
    2: 'surface_measurement',
    3: 'e_sat_curve'
})
config_df

# %%
run_df_list = []
for i, row in config_df.iterrows():
    e_sat_curve_func = e_sat_curve_options[row['e_sat_curve']]
    sfcTemp = variables_df[row['surface_measurement']]
    sfcVaporPress = e_sat_curve_func(sfcTemp - 273.15)
    if row['z0'] in z0_values_constant:
        z0_values_local = np.full(sfcTemp.shape, row['z0'])
    elif row['z0'] in z0_variable_names:
        z0_values_local = variables_df[row['z0']]
    else:
        raise ValueError(f"z0_var_name provided invalid: {row['z0']}")
    model_run_name = f"{row['scheme_name']} {row['surface_measurement']} {row['e_sat_curve']} {str(row['z0'])}"
    new_inputs_df = pd.DataFrame({
        'scheme_name'       : np.full(len(airTemp), row['scheme_name']),
        'ts_variable'       : np.full(len(airTemp), row['surface_measurement']),
        'e_sat_curv'        : np.full(len(airTemp), row['e_sat_curve']),
        'z0'                : np.full(len(airTemp), row['z0']),
        'time'              : timestamps,
        'airTemp'           : airTemp,
        'airVaporPress'     : airVaporPress,
        'sfcTemp'           : sfcTemp,
        'sfcVaporPress'     : sfcVaporPress,
        'windspd'           : windspd,
        'airPressure'       : airPressure,
        'snowDepth'         : snowDepth,
        'z0_values_local'   : z0_values_local,
    })
    run_df_list.append(new_inputs_df)

# %%
run_df = pd.concat(run_df_list)


# %%
def run_turbpy_for_row(row):
    model_run_name = f"{str(row['scheme_name'])} {str(row['ts_variable'])} {str(row['e_sat_curv'])} {str(row['z0'])}"
    if any(np.isnan(np.array([
        row['airTemp'], 
        row['airPressure'],
        row['airVaporPress'],
        row['windspd'],
        row['sfcTemp'],
        row['sfcVaporPress'],
        row['snowDepth'],
    ]))):
        return (
            model_run_name,
            row['time'],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan 
        )
    else:
        try:
            return (
                model_run_name,
                row['time'],
            ) + turbpy.turbFluxes(
                    row['airTemp'], 
                    row['airPressure'],
                    row['airVaporPress'],
                    row['windspd'],
                    row['sfcTemp'],
                    row['sfcVaporPress'],
                    row['snowDepth'],
                    HEIGHT,
                    param_dict=scheme_dict[row['scheme_name']],
                    z0Ground=row['z0_values_local'],
                    groundSnowFraction=1
                )
        except UnboundLocalError:
            return (
                model_run_name,
                row['time'],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan 
            )


# %%
if __name__ == '__main__':
    print("Starting dask client...")
    client = Client()
    print('Access client gere:')
    print(client.dashboard_link)

    # %%
    ddf = dd.from_pandas(run_df, npartitions=8)
    q = ddf.apply(run_turbpy_for_row, axis=1, meta=(None, 'int64'))

    # %%
    print("Starting computation...")
    start_time = datetime.now()
    results = q.compute()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    

        # %%
    results_df = pd.DataFrame(results.reset_index(drop=True))
    results_df = pd.DataFrame({'results':results_df[0].apply(lambda tup: np.array(tup))})
    results_df = pd.DataFrame(
        results_df['results'].to_list(),
        columns = [
            'config', 
            'time', 
            'conductanceSensible',
            'conductanceLatent',
            'senHeatGround',
            'latHeatGround',
            'stabilityCorrectionParameters',
            'param_dict',
        ]
    )

    # %%
    final_results_df = results_df.rename(columns={
        'latHeatGround': 'latent heat flux',
        'senHeatGround': 'sensible heat flux',
        'conductanceLatent': 'latent heat conductance',
        'conductanceSensible': 'sensible heat conductance',
    })
    final_results_df

    # %%
    final_results_df.to_parquet(
        "model_results.parquet"
    )
