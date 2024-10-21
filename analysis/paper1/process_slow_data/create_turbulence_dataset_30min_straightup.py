import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
from sublimpy import utils, tidy
import xarray as xr
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse


# Constants
sos_data_dir = '/Users/elischwat/Development/data/sublimationofsnow/sosqc/sos_qc_geo_tiltcor_v20240307/'
PLANAR_FIT = True
start_date = '20221101'
end_date = '20230619'
DATE_FORMAT_STR = '%Y%m%d'
# Threshold number of 20hz samples in a 5 minute average determining if the value
# is replaced with NaN
datelist = pd.date_range(
    dt.datetime.strptime(start_date, DATE_FORMAT_STR),
    dt.datetime.strptime(end_date, DATE_FORMAT_STR),
    freq='d'
).strftime(DATE_FORMAT_STR).tolist()

def create_tidy_dataset(
        planar_fitted_dir,
        filtering_str,
        FILTER_SNOWFALL,
        snowfall_mask_file,
        snowfall_mask_str,
        PERCENTAGE_DIAG,
        output_dir
):
    # +
    # Separate out the eddy covariance measurement variable names because they are very repetitive
    ec_measurement_suffixes = [
        '1m_ue',    '2m_ue',    '3m_ue',    '10m_ue', 
        '1m_d',     '2m_d',     '3m_d',     '10m_d',
        '1m_uw',    '2m_uw',    '2_5m_uw',  '3m_uw',    '10m_uw', 
        '1m_c',     '2m_c',     '3m_c',     '5m_c',     '10m_c',    '15m_c',    '20m_c'
    ]

    sonic_measurement_prefixes = [
        'u_', 'v_', 'w_', 'tc_', 'spd_', 'dir_', 
        'u_u__', 'v_v__', 'w_w__', 'tc_tc__', 
        'u_w__', 'v_w__', 'u_v__', 
        'u_tc__', 'v_tc__', 'w_tc__', 
        'u_u_u__', 'v_v_v__', 'w_w_w__', 
        'tc_tc_tc__', 
    ]
    irga_measurement_prefixes = [
        'h2o_', 'h2o_h2o__', 'h2o_h2o_h2o__', 'co2_', 'co2_co2__', 'co2_co2_co2__', 
    ]
    sonic_plus_irga_measurement_prefixes = [
        'u_h2o__', 'v_h2o__', 'w_h2o__', 'u_co2__', 'v_co2__', 'w_co2__', 
    ]
    ec_measurement_prefixes = sonic_measurement_prefixes + irga_measurement_prefixes + sonic_plus_irga_measurement_prefixes

    ec_variable_names = [
        (prefix + suffix) for prefix in ec_measurement_prefixes for suffix in ec_measurement_suffixes
    ]

    counts_vars = ['counts_' + suffix for suffix in ec_measurement_suffixes]
    counts_1_vars = ['counts_' + suffix + '_1' for suffix in ec_measurement_suffixes]
    counts_2_vars = ['counts_' + suffix + '_2' for suffix in ec_measurement_suffixes]
    irgadiag_vars = ['irgadiag_' + suffix for suffix in ec_measurement_suffixes]
    ldiag_vars = ['ldiag_' + suffix for suffix in ec_measurement_suffixes]

    diagnostic_variable_names = counts_vars + counts_1_vars + counts_2_vars + irgadiag_vars + ldiag_vars

    VARIABLE_NAMES = ec_variable_names + diagnostic_variable_names + [
        # Temperature & Relative Humidity Array 
        'T_1m_c', 'T_2m_c', 'T_3m_c', 'T_4m_c', 'T_5m_c', 'T_6m_c', 'T_7m_c', 'T_8m_c', 'T_9m_c', 'T_10m_c',
        'T_11m_c', 'T_12m_c', 'T_13m_c', 'T_14m_c', 'T_15m_c', 'T_16m_c', 'T_17m_c', 'T_18m_c', 'T_19m_c', 'T_20m_c',

        'RH_1m_c', 'RH_2m_c', 'RH_3m_c', 'RH_4m_c', 'RH_5m_c', 'RH_6m_c', 'RH_7m_c', 'RH_8m_c', 'RH_9m_c', 'RH_10m_c',
        'RH_11m_c','RH_12m_c','RH_13m_c','RH_14m_c','RH_15m_c','RH_16m_c','RH_17m_c','RH_18m_c','RH_19m_c','RH_20m_c',

        # Pressure Sensors
        'P_20m_c',
        'P_10m_c', 'P_10m_d', 'P_10m_uw', 'P_10m_ue',

        # Blowing snow/FlowCapt Sensors
        'SF_avg_1m_ue', 'SF_avg_2m_ue',

        # Apogee sensors
        "Vtherm_c", "Vtherm_d", "Vtherm_ue", "Vtherm_uw", 
        "Vpile_c", "Vpile_d", "Vpile_ue", "Vpile_uw",
        "IDir_c", "IDir_d", "IDir_ue", "IDir_uw",

        # Snow-level temperature arrays (towers D and UW)
        'Tsnow_0_4m_d', 'Tsnow_0_5m_d', 'Tsnow_0_6m_d', 'Tsnow_0_7m_d', 'Tsnow_0_8m_d', 'Tsnow_0_9m_d', 'Tsnow_1_0m_d', 'Tsnow_1_1m_d', 'Tsnow_1_2m_d', 'Tsnow_1_3m_d', 'Tsnow_1_4m_d', 'Tsnow_1_5m_d',
        'Tsnow_0_4m_uw', 'Tsnow_0_5m_uw', 'Tsnow_0_6m_uw', 'Tsnow_0_7m_uw', 'Tsnow_0_8m_uw', 'Tsnow_0_9m_uw', 'Tsnow_1_0m_uw', 'Tsnow_1_1m_uw', 'Tsnow_1_2m_uw', 'Tsnow_1_3m_uw', 'Tsnow_1_4m_uw', 'Tsnow_1_5m_uw',
        
        # Downward/Upward Facing Longwave Radiometers
        'Rpile_out_9m_d','Tcase_out_9m_d',    
        'Rpile_in_9m_d', 'Tcase_in_9m_d',
        'Tcase_uw', 'Rpile_in_uw', 'Rpile_out_uw',
        
        # Upward facing shortwave radiometer (tower D) - for measuring incoming solar radiation!
        'Rsw_in_9m_d', 'Rsw_out_9m_d',

        # Snow Pillow SWE
        'SWE_p1_c', 'SWE_p2_c', 'SWE_p3_c', 'SWE_p4_c',

        # Soil Moisture
        'Qsoil_d',

        # Ground heat flux
        'Gsoil_d',
    ]

    # # Open and concatenate daily SoS datasets
    # all_file_paths = [
    #     os.path.join(
    #         sos_data_dir,
    #         f'isfs_sos_qc_geo_tiltcor_5min_{date}.nc'
    #     ) for date in datelist
    # ]
    # datasets = []
    # for file in all_file_paths:
    #     ds = xr.open_dataset(file)
    #     # this ensures we don't access variables that aren't in this dataset, which would throw an error
    #     ds_new = ds[set(ds.data_vars).intersection(VARIABLE_NAMES)]
    #     datasets.append(ds_new)
    # sos_ds = xr.concat(datasets, dim='time')
    # Ensure time index is evenly spaced by filling in any missing timestamps
    # sos_ds = utils.fill_missing_timestamps(sos_ds)
    # # Save intermediate data product to save time rerunning in the future
    # sos_ds.to_netcdf("sos_ds_temp_storage_30min_straightup.cdf")

    sos_ds = xr.open_dataset("sos_ds_temp_storage_30min_straightup.cdf")

    # # Replace flag variables with my calculated aggregated flags
    # (sonic anemometer and irgason flags)

    flag_counts_df = pd.read_parquet("/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_flagcounts").loc[start_date: end_date]
    assert all(sos_ds.time == flag_counts_df.index)

    # +
    irga_vars = [
        'irgadiag_10m_c',	 'irgadiag_10m_d',	 'irgadiag_10m_ue',	 'irgadiag_10m_uw',	 'irgadiag_15m_c',	 'irgadiag_1m_c',	 'irgadiag_1m_d',	 'irgadiag_1m_ue',	 
        'irgadiag_1m_uw',	 'irgadiag_20m_c',	 'irgadiag_2m_c',	 'irgadiag_3m_c',	 'irgadiag_3m_d',	 'irgadiag_3m_ue',	 'irgadiag_3m_uw',	 'irgadiag_5m_c',	
    ]

    ldiag_vars = [
        'ldiag_10m_c', 'ldiag_10m_d', 'ldiag_10m_ue', 'ldiag_10m_uw', 'ldiag_15m_c', 'ldiag_1m_c', 'ldiag_1m_d', 'ldiag_1m_ue', 
        'ldiag_1m_uw', 'ldiag_20m_c', 'ldiag_2m_c', 'ldiag_3m_c', 'ldiag_3m_d', 'ldiag_3m_ue', 'ldiag_3m_uw', 'ldiag_5m_c',
    ]

    for var in irga_vars + ldiag_vars:
        sos_ds[var] = flag_counts_df[var]

    # # Resample dataset

    # ## Define dictionary defining the resampling function to use for each variable
    #
    # Covariances are resampled according to the rules of **Reynold averaging** (https://www.eol.ucar.edu/content/combining-short-term-moments-longer-time-periods).
    #
    # Meteorological and turbulence measurements (other than covariances) are resampled using the **mean**.
    #
    # EC count variables are **summed**.

    data_vars_processing_dict = {
        'reynolds_average': [
            'u_u__1m_uw',    'v_v__1m_uw',    'w_w__1m_uw',    'u_w__1m_uw',    'v_w__1m_uw',  'u_tc__1m_uw',  'v_tc__1m_uw',   'u_co2__1m_uw', 'u_h2o__1m_uw',  'v_co2__1m_uw', 'v_h2o__1m_uw',   'w_tc__1m_uw',   'w_co2__1m_uw', 'w_h2o__1m_uw',
            'u_u__2m_uw',    'v_v__2m_uw',    'w_w__2m_uw',    'u_w__2m_uw',    'v_w__2m_uw',  'u_tc__2m_uw',  'v_tc__2m_uw',   'u_co2__2m_uw', 'u_h2o__2m_uw',  'v_co2__2m_uw', 'v_h2o__2m_uw',   'w_tc__2m_uw',   'w_co2__2m_uw', 'w_h2o__2m_uw',
            'u_u__2_5m_uw', 'v_v__2_5m_uw',   'w_w__2_5m_uw',  'u_w__2_5m_uw',  'v_w__2_5m_uw','u_tc__2_5m_uw','v_tc__2_5m_uw', 'u_co2__2_5m_uw', 'u_h2o__2_5m_uw','v_h2o__2_5m_uw', 'w_tc__2_5m_uw', 'w_co2__2_5m_uw', 'w_h2o__2_5m_uw',
            'u_u__3m_uw',    'v_v__3m_uw',    'w_w__3m_uw',    'u_w__3m_uw',    'v_w__3m_uw',  'u_tc__3m_uw',  'v_tc__3m_uw',   'u_co2__3m_uw', 'u_h2o__3m_uw',  'v_co2__3m_uw', 'v_h2o__3m_uw',   'w_tc__3m_uw',   'w_co2__3m_uw', 'w_h2o__3m_uw',
            'u_u__10m_uw',   'v_v__10m_uw',   'w_w__10m_uw',   'u_w__10m_uw',   'v_w__10m_uw', 'u_tc__10m_uw', 'v_tc__10m_uw',  'u_co2__10m_uw', 'u_h2o__10m_uw', 'v_co2__10m_uw', 'v_h2o__10m_uw',  'w_tc__10m_uw',  'w_co2__10m_uw', 'w_h2o__10m_uw',
            'u_u__1m_ue',    'v_v__1m_ue',    'w_w__1m_ue',    'u_w__1m_ue',    'v_w__1m_ue',  'u_tc__1m_ue',  'v_tc__1m_ue',   'u_co2__1m_ue', 'u_h2o__1m_ue',  'v_co2__1m_ue', 'v_h2o__1m_ue',   'w_tc__1m_ue',   'w_co2__1m_ue', 'w_h2o__1m_ue',
            'u_u__2m_ue',    'v_v__2m_ue',    'w_w__2m_ue',    'u_w__2m_ue',    'v_w__2m_ue',  'u_tc__2m_ue',  'v_tc__2m_ue',   'u_co2__2m_ue', 'u_h2o__2m_ue',  'v_co2__2m_ue', 'v_h2o__2m_ue',   'w_tc__2m_ue',   'w_co2__2m_ue', 'w_h2o__2m_ue',
            'u_u__3m_ue',    'v_v__3m_ue',    'w_w__3m_ue',    'u_w__3m_ue',    'v_w__3m_ue',  'u_tc__3m_ue',  'v_tc__3m_ue',   'u_co2__3m_ue', 'u_h2o__3m_ue',  'v_co2__3m_ue', 'v_h2o__3m_ue',   'w_tc__3m_ue',   'w_co2__3m_ue', 'w_h2o__3m_ue',
            'u_u__10m_ue',   'v_v__10m_ue',   'w_w__10m_ue',   'u_w__10m_ue',   'v_w__10m_ue', 'u_tc__10m_ue', 'v_tc__10m_ue',  'u_co2__10m_ue', 'u_h2o__10m_ue', 'v_co2__10m_ue', 'v_h2o__10m_ue',  'w_tc__10m_ue',  'w_co2__10m_ue', 'w_h2o__10m_ue',
            'u_u__1m_d',     'v_v__1m_d',     'w_w__1m_d',     'u_w__1m_d',     'v_w__1m_d',   'u_tc__1m_d',   'v_tc__1m_d',    'u_co2__1m_d', 'u_h2o__1m_d',   'v_co2__1m_d', 'v_h2o__1m_d',    'w_tc__1m_d',    'w_co2__1m_d', 'w_h2o__1m_d',
            'u_u__2m_d',     'v_v__2m_d',     'w_w__2m_d',     'u_w__2m_d',     'v_w__2m_d',   'u_tc__2m_d',   'v_tc__2m_d',    'u_co2__2m_d', 'u_h2o__2m_d',   'v_co2__2m_d', 'v_h2o__2m_d',    'w_tc__2m_d',    'w_co2__2m_d', 'w_h2o__2m_d',
            'u_u__3m_d',     'v_v__3m_d',     'w_w__3m_d',     'u_w__3m_d',     'v_w__3m_d',   'u_tc__3m_d',   'v_tc__3m_d',    'u_co2__3m_d', 'u_h2o__3m_d',   'v_co2__3m_d', 'v_h2o__3m_d',    'w_tc__3m_d',    'w_co2__3m_d', 'w_h2o__3m_d',
            'u_u__10m_d',    'v_v__10m_d',    'w_w__10m_d',    'u_w__10m_d',    'v_w__10m_d',  'u_tc__10m_d',  'v_tc__10m_d',   'u_co2__10m_d', 'u_h2o__10m_d',  'v_co2__10m_d', 'v_h2o__10m_d',   'w_tc__10m_d',   'w_co2__10m_d', 'w_h2o__10m_d',
            'u_u__1m_c',     'v_v__1m_c',     'w_w__1m_c',     'u_w__1m_c',     'v_w__1m_c',   'u_tc__1m_c',   'v_tc__1m_c',    'u_co2__1m_c', 'u_h2o__1m_c',   'v_co2__1m_c', 'v_h2o__1m_c',    'w_tc__1m_c',    'w_co2__1m_c', 'w_h2o__1m_c',
            'u_u__2m_c',     'v_v__2m_c',     'w_w__2m_c',     'u_w__2m_c',     'v_w__2m_c',   'u_tc__2m_c',   'v_tc__2m_c',    'u_co2__2m_c', 'u_h2o__2m_c',   'v_co2__2m_c', 'v_h2o__2m_c',    'w_tc__2m_c',    'w_co2__2m_c', 'w_h2o__2m_c',
            'u_u__3m_c',     'v_v__3m_c',     'w_w__3m_c',     'u_w__3m_c',     'v_w__3m_c',   'u_tc__3m_c',   'v_tc__3m_c',    'u_co2__3m_c', 'u_h2o__3m_c',   'v_co2__3m_c', 'v_h2o__3m_c',    'w_tc__3m_c',    'w_co2__3m_c', 'w_h2o__3m_c',
            'u_u__5m_c',     'v_v__5m_c',     'w_w__5m_c',     'u_w__5m_c',     'v_w__5m_c',   'u_tc__5m_c',   'v_tc__5m_c',    'u_co2__5m_c', 'u_h2o__5m_c',   'v_co2__5m_c', 'v_h2o__5m_c',    'w_tc__5m_c',    'w_co2__5m_c', 'w_h2o__5m_c',
            'u_u__10m_c',    'v_v__10m_c',    'w_w__10m_c',    'u_w__10m_c',    'v_w__10m_c',  'u_tc__10m_c',  'v_tc__10m_c',   'u_co2__10m_c', 'u_h2o__10m_c',  'v_co2__10m_c', 'v_h2o__10m_c',   'w_tc__10m_c',   'w_co2__10m_c', 'w_h2o__10m_c',
            'u_u__15m_c',    'v_v__15m_c',    'w_w__15m_c',    'u_w__15m_c',    'v_w__15m_c',  'u_tc__15m_c',  'v_tc__15m_c',   'u_co2__15m_c', 'u_h2o__15m_c',  'v_co2__15m_c', 'v_h2o__15m_c',   'w_tc__15m_c',   'w_co2__15m_c', 'w_h2o__15m_c',
            'u_u__20m_c',    'v_v__20m_c',    'w_w__20m_c',    'u_w__20m_c',    'v_w__20m_c',  'u_tc__20m_c',  'v_tc__20m_c',   'u_co2__20m_c', 'u_h2o__20m_c',  'v_co2__20m_c', 'v_h2o__20m_c',   'w_tc__20m_c',   'w_co2__20m_c', 'w_h2o__20m_c',
        ],
        'average': [
            # Sonic anemometer data
            'co2_1m_uw', 'h2o_1m_uw' ,       'tc_1m_uw',     'spd_1m_uw',    'u_1m_uw',  'v_1m_uw',   'w_1m_uw',  
            'co2_3m_uw', 'h2o_3m_uw' ,       'tc_3m_uw',     'spd_3m_uw',    'u_3m_uw',  'v_3m_uw',   'w_3m_uw',  
            'co2_10m_uw', 'h2o_10m_uw' ,      'tc_10m_uw',    'spd_10m_uw',   'u_10m_uw', 'v_10m_uw',  'w_10m_uw',  
            'co2_1m_ue', 'h2o_1m_ue' ,       'tc_1m_ue',     'spd_1m_ue',    'u_1m_ue',  'v_1m_ue',   'w_1m_ue',  
            'co2_3m_ue', 'h2o_3m_ue' ,       'tc_3m_ue',     'spd_3m_ue',    'u_3m_ue',  'v_3m_ue',   'w_3m_ue',  
            'co2_10m_ue', 'h2o_10m_ue' ,      'tc_10m_ue',    'spd_10m_ue',   'u_10m_ue', 'v_10m_ue',  'w_10m_ue',  
            'co2_1m_d', 'h2o_1m_d' ,        'tc_1m_d',      'spd_1m_d',     'u_1m_d',   'v_1m_d',    'w_1m_d',  
            'co2_3m_d', 'h2o_3m_d' ,        'tc_3m_d',      'spd_3m_d',     'u_3m_d',   'v_3m_d',    'w_3m_d',  
            'co2_10m_d', 'h2o_10m_d' ,       'tc_10m_d',     'spd_10m_d',    'u_10m_d',  'v_10m_d',   'w_10m_d',  
            'co2_1m_c', 'h2o_1m_c' ,        'tc_1m_c',      'spd_1m_c',     'u_1m_c',   'v_1m_c',    'w_1m_c',  
            'co2_2m_c', 'h2o_2m_c' ,        'tc_2m_c',      'spd_2m_c',     'u_2m_c',   'v_2m_c',    'w_2m_c',  
            'co2_3m_c', 'h2o_3m_c' ,        'tc_3m_c',      'spd_3m_c',     'u_3m_c',   'v_3m_c',    'w_3m_c',  
            'co2_5m_c', 'h2o_5m_c' ,        'tc_5m_c',      'spd_5m_c',     'u_5m_c',   'v_5m_c',    'w_5m_c',  
            'co2_10m_c', 'h2o_10m_c' ,       'tc_10m_c',     'spd_10m_c',    'u_10m_c',  'v_10m_c',   'w_10m_c',  
            'co2_15m_c', 'h2o_15m_c' ,       'tc_15m_c',     'spd_15m_c',    'u_15m_c',  'v_15m_c',   'w_15m_c',  
            'co2_20m_c', 'h2o_20m_c' ,       'tc_20m_c',     'spd_20m_c',    'u_20m_c',  'v_20m_c',   'w_20m_c',  

            # Temperature & Relative Humidity Array 
            'T_1m_c', 'T_2m_c', 'T_3m_c', 'T_4m_c', 'T_5m_c', 'T_6m_c', 'T_7m_c', 'T_8m_c', 'T_9m_c', 'T_10m_c',
            'T_11m_c', 'T_12m_c', 'T_13m_c', 'T_14m_c', 'T_15m_c', 'T_16m_c', 'T_17m_c', 'T_18m_c', 'T_19m_c', 'T_20m_c',

            'RH_1m_c', 'RH_2m_c', 'RH_3m_c', 'RH_4m_c', 'RH_5m_c', 'RH_6m_c', 'RH_7m_c', 'RH_8m_c', 'RH_9m_c', 'RH_10m_c',
            'RH_11m_c','RH_12m_c','RH_13m_c','RH_14m_c','RH_15m_c','RH_16m_c','RH_17m_c','RH_18m_c','RH_19m_c','RH_20m_c',

            # Pressure Sensors
            'P_20m_c',
            'P_10m_c', 'P_10m_d', 'P_10m_uw', 'P_10m_ue',

            # Blowing snow/FlowCapt Sensors
            'SF_avg_1m_ue', 'SF_avg_2m_ue',

            # Apogee sensors
            "Vtherm_c", "Vtherm_d", "Vtherm_ue", "Vtherm_uw", 
            "Vpile_c", "Vpile_d", "Vpile_ue", "Vpile_uw",
            "IDir_c", "IDir_d", "IDir_ue", "IDir_uw",

            # Snow-level temperature arrays (towers D and UW)
            'Tsnow_0_4m_d', 'Tsnow_0_5m_d', 'Tsnow_0_6m_d', 'Tsnow_0_7m_d', 'Tsnow_0_8m_d', 'Tsnow_0_9m_d', 'Tsnow_1_0m_d', 'Tsnow_1_1m_d', 'Tsnow_1_2m_d', 'Tsnow_1_3m_d', 'Tsnow_1_4m_d', 'Tsnow_1_5m_d',
            'Tsnow_0_4m_uw', 'Tsnow_0_5m_uw', 'Tsnow_0_6m_uw', 'Tsnow_0_7m_uw', 'Tsnow_0_8m_uw', 'Tsnow_0_9m_uw', 'Tsnow_1_0m_uw', 'Tsnow_1_1m_uw', 'Tsnow_1_2m_uw', 'Tsnow_1_3m_uw', 'Tsnow_1_4m_uw', 'Tsnow_1_5m_uw',
            
            # Downward Facing Longwave Radiometer (tower D) - for measuring snow surface temperature
            'Rpile_out_9m_d',
            'Tcase_out_9m_d',    
            # Upward Facing Longwave Radiometer (tower D)
            'Rpile_in_9m_d',
            'Tcase_in_9m_d',
            # Downward Facing Longwave Radiometer (tower UW) - for measuring snow surface temperature
            'Tcase_uw', 'Rpile_in_uw', 'Rpile_out_uw',
            
            # Upward facing shortwave radiometer (tower D) - for measuring incoming solar radiation!
            'Rsw_in_9m_d',
            'Rsw_out_9m_d',

            # Snow Pillow SWE
            'SWE_p1_c', 'SWE_p2_c', 'SWE_p3_c', 'SWE_p4_c',

            # Soil Moisture
            'Qsoil_d',

            # Soil Moisture
            'Gsoil_d',
        ],
        'median' : [
            'dir_1m_uw',    
            'dir_3m_uw',    
            'dir_10m_uw',   
            'dir_1m_ue',    
            'dir_3m_ue',    
            'dir_10m_ue',   
            'dir_1m_d',     
            'dir_3m_d',     
            'dir_10m_d',    
            'dir_1m_c',     
            'dir_2m_c',     
            'dir_3m_c',     
            'dir_5m_c',     
            'dir_10m_c',    
            'dir_15m_c',    
            'dir_20m_c',    
        ],
        'sum' : [
            # Counts of UNflagged instantaneous (20hz) eddy covariance measurements
            'counts_1m_c',    'counts_1m_c_1',    'counts_1m_c_2',    
            'counts_2m_c',    'counts_2m_c_1',    'counts_2m_c_2',    
            'counts_3m_c',    'counts_3m_c_1',    'counts_3m_c_2',    
            'counts_5m_c',    'counts_5m_c_1',    'counts_5m_c_2',    
            'counts_10m_c',   'counts_10m_c_1',   'counts_10m_c_2',   
            'counts_15m_c',   'counts_15m_c_1',   'counts_15m_c_2',   
            'counts_20m_c',   'counts_20m_c_1',   'counts_20m_c_2',   
            'counts_1m_uw',   'counts_1m_uw_1',   'counts_1m_uw_2',   
            'counts_3m_uw',   'counts_3m_uw_1',   'counts_3m_uw_2',   
            'counts_10m_uw',  'counts_10m_uw_1',  'counts_10m_uw_2',  
            'counts_1m_ue',   'counts_1m_ue_1',   'counts_1m_ue_2',   
            'counts_3m_ue',   'counts_3m_ue_1',   'counts_3m_ue_2',   
            'counts_10m_ue',  'counts_10m_ue_1',  'counts_10m_ue_2',  
            'counts_1m_d',    'counts_1m_d_1',    'counts_1m_d_2',    
            'counts_3m_d',    'counts_3m_d_1',    'counts_3m_d_2',    
            'counts_10m_d',   'counts_10m_d_1',   'counts_10m_d_2',   

            # Counts of FLAGGED 20hz measurements 
            'irgadiag_1m_c',    'ldiag_1m_c',
            'irgadiag_2m_c',    'ldiag_2m_c',
            'irgadiag_3m_c',    'ldiag_3m_c',
            'irgadiag_5m_c',    'ldiag_5m_c',
            'irgadiag_10m_c',   'ldiag_10m_c',
            'irgadiag_15m_c',   'ldiag_15m_c',
            'irgadiag_20m_c',   'ldiag_20m_c',
            'irgadiag_1m_uw',   'ldiag_1m_uw',
            'irgadiag_3m_uw',   'ldiag_3m_uw',
            'irgadiag_10m_uw',  'ldiag_10m_uw',
            'irgadiag_1m_ue',   'ldiag_1m_ue',
            'irgadiag_3m_ue',   'ldiag_3m_ue',
            'irgadiag_10m_ue',  'ldiag_10m_ue',
            'irgadiag_1m_d',    'ldiag_1m_d',
            'irgadiag_3m_d',    'ldiag_3m_d',
            'irgadiag_10m_d',   'ldiag_10m_d',
        ]
    }


    # ## Define function for resampling covariances

    # +
    def separate_covariance_variable_name(cov_name):
        """Get the names of the two mean variables associated with a covariance variable. Built to use
        with SOS datasets. For example, one might provide `w_h2o__3m_c` and this function will return
        `w_3m_c` and `h2o_3m_c`.

        Args:
            cov_name (str): name of variable that you want to separate into the two names of the 
            asssociated mean variables.

        Returns:
            var1, var 2 (str, str): two strings with the names of the two mean variables
        """
        [first_parts, second_part] = cov_name.split('__')
        [var1, var2] = first_parts.split('_')
        [var1, var2] = [
            var1 +'_' + second_part,
            var2 +'_' + second_part,
        ]
        return var1, var2

    def resample_moment(df, cov, mean1, mean2, new_frequency, n_in_new_re_length, skipna=True):
        """Combines moments into longer time periods, using reynolds averaging. Built to use with SOS 
        datasets. Resampling covariances which have been calculated for a specific Reynolds
        averaging length (e.g. the SOS datasets are averaged to 5minutes), you need both the mean
        values and covariance. For example, the variable `w_h2o__3m_c` is associated with mean values
        `w_3m_c` and `h2o_3m_c`. To reasmple `w_h2o__3m_c` to another averaging length, we need the three
        variables.

        Args:
            df (pd.Dataframe): Dataframe containing the three columns required for calculations (contains)
                        the names supplied as parameters `cov`, `mean1`, and `mean2`.
            cov (str): Name of covariance variable to resample using Reynolds averaing
            mean1 (str): Name of one of the two mean variables associated with `cov`
            mean2 (str): Name of the other mean variable associated with `cov`
            new_frequency (str): String interpretable by pandas/xarray that describes the reynolds length you 
                are resampling to. EG: '60Min'
            n_in_new_re_length (_type_): Number of 5 minute intervals that fit in the new_frequency. E.G. for
                new_frequency='60Min', you would provide 12.
            skipna (bool, optional): Whether to skip NaNs when calculating the new variables. Providing True
                will allow more moments to be calculated, but those moments may be inaccurate/non-representative.
                Providing False will result in more missing data.

        Returns:
            pd.DataFrame: Dataframe with resampled data.
        """
        return pd.DataFrame({
                cov: df.groupby(pd.Grouper(freq=new_frequency)).apply(
                    lambda row: 
                        (1/n_in_new_re_length)*(row[cov] + row[mean1]*row[mean2]).sum(skipna=skipna)
                        - (
                            (1/n_in_new_re_length)*row[mean1].sum(skipna=skipna)
                            * (1/n_in_new_re_length)*row[mean2].sum(skipna=skipna)
                        )
                )
            })

    def resample(ds, new_frequency, n_in_new_re_length, skipna=True):
        """Resample SOS xarray datasets, applying the proper aggregation function
        for different variables. Some are resampled by taking the mean, some by 
        summing, and others by Reynolds averaging. 
        """
        # Resample data vars that need to be averaged (plain old averaging)
        # Use built in xarray functionality
        resampled_averages = ds[
            data_vars_processing_dict['average']
        ].to_dataframe().resample(new_frequency).mean().to_xarray()

        resampled_medians = ds[
            data_vars_processing_dict['median']
        ].to_dataframe().resample(new_frequency).median().to_xarray()
        
        # Resample data vars that need to be summed
        # Use built in xarray functionality
        resampled_sums = ds[
            data_vars_processing_dict['sum']
        ].to_dataframe().resample(new_frequency).sum().to_xarray()
        
        # Resample data vars that need to be summed using the rules of Reynolds Averaging
        # Use our custom function defined above
        resampled_reynolds_averages_list = []
        def split_covariance_name_and_resample(name):
            mean_var1, mean_var2 = separate_covariance_variable_name(name)
            resampled = resample_moment(
                ds[[mean_var1, mean_var2, name]].to_dataframe(), 
                name, 
                mean_var1, 
                mean_var2, 
                new_frequency, 
                n_in_new_re_length, 
                skipna=skipna
            )
            return resampled.to_xarray()
        resampled_reynolds_averages_list =  Parallel(n_jobs = 8)(
            delayed(split_covariance_name_and_resample)(name) 
            for name in tqdm(data_vars_processing_dict['reynolds_average'])
        )
        
        new_ds = xr.merge(
            [
                resampled_sums, 
                resampled_medians,
                resampled_averages
            ] + resampled_reynolds_averages_list
        )

        ## Copy attributes from the original dataset
        new_ds.attrs = ds.attrs
        for var in new_ds:
            new_ds[var].attrs = ds[var].attrs
        return new_ds



    # ## Resample variables

    sos_ds30min = resample(sos_ds, '30Min', 6, skipna=True)

    sos_ds5min = sos_ds
    sos_ds = sos_ds30min

    # # Replace fluxes with planar fitted fluxes
    planar_fitted_data_df = pd.read_parquet(planar_fitted_dir)


    if PLANAR_FIT:
        planar_fitted_data_df = planar_fitted_data_df[[c for c in planar_fitted_data_df.columns if c.endswith('_fit')]]
        planar_fitted_data_df.columns = [c.replace('_fit', '') for c in planar_fitted_data_df.columns]
        planar_fitted_data_df = planar_fitted_data_df.loc[ start_date : end_date ]
        planar_fitted_data_df.index = planar_fitted_data_df.index - dt.timedelta(minutes=15)
        planar_fitted_ds = planar_fitted_data_df.to_xarray()
        sos_ds = sos_ds.assign(planar_fitted_ds)

    # # Convert to local time and isolate to study period
    sos_ds = utils.modify_xarray_timezone(sos_ds, 'UTC', 'US/Mountain')
    sos_ds = sos_ds.reset_coords(
        'time (UTC)', drop=True
    ).reset_coords(
        'time (US/Mountain)', drop=True
    ) 
    sos_ds = sos_ds.to_dataframe().sort_index().loc['20221130': '20230508'].to_xarray()

    # # Copy fluxes into variables named "raw" so that we keep the raw fluxes around
    important_ec_variables = [
        'w_h2o__3m_ue', 'w_h2o__10m_ue', 
        'w_h2o__3m_d', 'w_h2o__10m_d',
        'w_h2o__3m_uw', 'w_h2o__10m_uw', 
        'w_h2o__2m_c', 'w_h2o__3m_c', 'w_h2o__5m_c', 'w_h2o__10m_c', 'w_h2o__15m_c', 'w_h2o__20m_c'
    ]
    for var in important_ec_variables:
        sos_ds[
            var + '_raw'
        ] = sos_ds[var]


    # # Remove instrument-flagged data

    # ## Set bad Irga measurements to NaN
    #
    # The NCAR report recommends all Irga-related measurements be set to NaN when irgadiag is non-zero.  They did this for some but not all of the data.
    var_ls = []
    old_nan_count_badirga_ls = []
    new_nan_count_badirga_ls = []
    old_mean_ls = []
    new_mean_ls = []
    old_median_ls = []
    new_median_ls = []
    for suffix in ec_measurement_suffixes:
        h2o_flux_var = 'w_h2o__' + suffix
        irgadiag_var = 'irgadiag_' + suffix

        if irgadiag_var in sos_ds.variables and h2o_flux_var in sos_ds.variables:
            old_nan_count_badirga = (np.isnan(sos_ds[h2o_flux_var])).sum().item()
            old_mean = sos_ds[h2o_flux_var].mean().item()
            old_median = sos_ds[h2o_flux_var].median().item()

            sos_ds[h2o_flux_var] = sos_ds[h2o_flux_var].where(sos_ds[irgadiag_var] <= PERCENTAGE_DIAG)
        
            new_nan_count_badirga = (np.isnan(sos_ds[h2o_flux_var])).sum().item()
            new_mean = sos_ds[h2o_flux_var].mean().item()
            new_median = sos_ds[h2o_flux_var].median().item()
            var_ls.append(h2o_flux_var)
            old_nan_count_badirga_ls.append(old_nan_count_badirga)
            new_nan_count_badirga_ls.append(new_nan_count_badirga)
            old_mean_ls.append(old_mean)
            new_mean_ls.append(new_mean)
            old_median_ls.append(old_median)
            new_median_ls.append(new_median)
        else:
            var_ls.append(h2o_flux_var)
            old_nan_count_badirga_ls.append(np.nan)
            new_nan_count_badirga_ls.append(np.nan)
            old_mean_ls.append(np.nan)
            new_mean_ls.append(np.nan)
            old_median_ls.append(np.nan)
            new_median_ls.append(np.nan)

    # ## Set bad Sonic measurements to Nan

    var_ls = []
    old_nan_count_badsonic_ls = []
    new_nan_count_badsonic_ls = []
    old_mean_ls = []
    new_mean_ls = []
    for suffix in ec_measurement_suffixes:
        w_var = 'w_' + suffix
        h2o_flux_var = 'w_h2o__' + suffix
        sonicdiag_var = 'ldiag_' + suffix

        if h2o_flux_var in sos_ds.variables and sonicdiag_var in sos_ds.variables:
            old_nan_count_badsonic = (np.isnan(sos_ds[h2o_flux_var])).sum().item()
            old_mean = sos_ds[h2o_flux_var].mean().item()
            
            sos_ds[h2o_flux_var] = sos_ds[h2o_flux_var].where(sos_ds[sonicdiag_var] <= PERCENTAGE_DIAG)

            new_nan_count_badsonic = (np.isnan(sos_ds[h2o_flux_var])).sum().item()
            new_mean = sos_ds[h2o_flux_var].mean().item()
            var_ls.append(h2o_flux_var)
            old_nan_count_badsonic_ls.append(old_nan_count_badsonic)
            new_nan_count_badsonic_ls.append(new_nan_count_badsonic)
            old_mean_ls.append(old_mean)
            new_mean_ls.append(new_mean)
        else:
            var_ls.append(h2o_flux_var)
            old_nan_count_badsonic_ls.append(np.nan)
            new_nan_count_badsonic_ls.append(np.nan)
            old_mean_ls.append(np.nan)
            new_mean_ls.append(np.nan)

    # # Plausibility limits
    PLAUSIBILITY_LIMIT = 0.2
    var_ls = []
    old_nan_count_plausibilitylimit_ls = []
    new_nan_count_plausibilitylimit_ls = []
    old_mean_ls = []
    new_mean_ls = []
    for suffix in ec_measurement_suffixes:
        w_var = 'w_' + suffix
        h2o_flux_var = 'w_h2o__' + suffix
        sonicdiag_var = 'ldiag_' + suffix

        if h2o_flux_var in sos_ds.variables and sonicdiag_var in sos_ds.variables:
            old_nan_count_plausibilitylimit = (np.isnan(sos_ds[h2o_flux_var])).sum().item()
            old_mean = sos_ds[h2o_flux_var].mean().item()
            
            sos_ds[h2o_flux_var] = sos_ds[h2o_flux_var].where(np.abs(sos_ds[h2o_flux_var]) < PLAUSIBILITY_LIMIT)
        
            new_nan_count_plausibilitylimit = (np.isnan(sos_ds[h2o_flux_var])).sum().item()
            new_mean = sos_ds[h2o_flux_var].mean().item()
            var_ls.append(h2o_flux_var)
            old_nan_count_plausibilitylimit_ls.append(old_nan_count_plausibilitylimit)
            new_nan_count_plausibilitylimit_ls.append(new_nan_count_plausibilitylimit)
            old_mean_ls.append(old_mean)
            new_mean_ls.append(new_mean)
        else:
            var_ls.append(h2o_flux_var)
            old_nan_count_plausibilitylimit_ls.append(np.nan)
            new_nan_count_plausibilitylimit_ls.append(np.nan)
            old_mean_ls.append(np.nan)
            new_mean_ls.append(np.nan)

    # # Analyze cleaning steps
    nan_counts_df = pd.DataFrame({
        'variable':                         var_ls,
        'n':                                len(sos_ds.time),
        'original nan count':               old_nan_count_badirga_ls, 
        'nans after bad irga removed':      new_nan_count_badirga_ls, 
        'nans after bad sonic removed':     new_nan_count_badsonic_ls, 
        'nans after plausibility limit':    new_nan_count_plausibilitylimit_ls
    })
    limited_nan_counts_df = nan_counts_df[ 
        (~nan_counts_df.variable.str.contains('__1m_'))
        
        &
        (~nan_counts_df.variable.str.contains('__2_5m_'))
    ]
    limited_nan_counts_df.dropna()
    limited_nan_counts_df['Valid measurements'] = limited_nan_counts_df['n'] - limited_nan_counts_df['original nan count']
    limited_nan_counts_df['Data removed by EC150 flag'] = limited_nan_counts_df['nans after bad irga removed'] - limited_nan_counts_df['original nan count']
    limited_nan_counts_df['Data removed by CSAT3 flag'] = limited_nan_counts_df['nans after bad sonic removed'] - limited_nan_counts_df['nans after bad irga removed']
    limited_nan_counts_df['Data removed by plausibility limit'] = limited_nan_counts_df['nans after plausibility limit'] - limited_nan_counts_df['nans after bad sonic removed']
    limited_nan_counts_df[[
        'variable',
        'n',
        'Valid measurements',
        'Data removed by EC150 flag',
        'Data removed by CSAT3 flag',
        'Data removed by plausibility limit'
    ]].dropna()
            

    # # Get Tidy Dataset
    tidy_df = tidy.get_tidy_dataset(sos_ds, list(sos_ds.data_vars))

    # # Apply mean diurnal cycle gap filling for latent heat fluxes
    for lhflux_variable in [
        'w_h2o__2m_c',
        'w_h2o__3m_c',
        'w_h2o__5m_c',
        'w_h2o__10m_c',
        'w_h2o__15m_c',
        'w_h2o__20m_c',
        'w_h2o__3m_ue',
        'w_h2o__10m_ue',
        'w_h2o__3m_uw',
        'w_h2o__10m_uw',
        'w_h2o__3m_d',
        'w_h2o__10m_d',
    ]:
        subset = tidy_df[tidy_df.variable == lhflux_variable].set_index('time')
        for i,row in subset.iterrows():
            if np.isnan(row['value']):
                start_window = i - dt.timedelta(days=3, hours=12)
                end_window = i + dt.timedelta(days=3, hours=12)
                src = subset.loc[start_window: end_window].reset_index()
                means = pd.DataFrame(
                    src.groupby([src.time.dt.hour, src.time.dt.minute])['value'].mean()
                )
                subset.loc[i, 'value'] = means.loc[i.hour, i.minute].value
        new_values = subset['value'].values
        measurement = subset['measurement'].values[0]
        height = subset['height'].values[0]
        tower = subset['tower'].values[0]
        # Add new values for variable
        tidy_df = tidy.tidy_df_add_variable(
            tidy_df,
            new_values,
            lhflux_variable + '_gapfill',
            measurement,
            height,
            tower
        )
        tidy_df.query("variable == 'w_h2o__3m_c'").set_index('time')['value'].isna().sum()



    # # Save dataset
    output_file_name = None
    if PLANAR_FIT:    
        if 'oneplane' in planar_fitted_dir:
            if FILTER_SNOWFALL:
                output_file_name = f'tidy_df_{start_date}_{end_date}_planar_fit_STRAIGHTUP_{filtering_str}_flags{PERCENTAGE_DIAG}_snowfallfiltered{snowfall_mask_str}.parquet'
            else:
                output_file_name = f'tidy_df_{start_date}_{end_date}_planar_fit_STRAIGHTUP_{filtering_str}_flags{PERCENTAGE_DIAG}.parquet'
        else:
            if FILTER_SNOWFALL:
                output_file_name = f'tidy_df_{start_date}_{end_date}_planar_fit_multiplane_STRAIGHTUP_{filtering_str}_flags{PERCENTAGE_DIAG}_snowfallfiltered{snowfall_mask_str}.parquet'
            else:
                output_file_name = f'tidy_df_{start_date}_{end_date}_planar_fit_multiplane_STRAIGHTUP_{filtering_str}_flags{PERCENTAGE_DIAG}.parquet'
    else:
        if FILTER_SNOWFALL:
            output_file_name = f'tidy_df_{start_date}_{end_date}_noplanar_fit_STRAIGHTUP_{filtering_str}_flags{PERCENTAGE_DIAG}_snowfallfiltered{snowfall_mask_str}.parquet', 
        else:
            output_file_name = f'tidy_df_{start_date}_{end_date}_noplanar_fit_STRAIGHTUP_{filtering_str}_flags{PERCENTAGE_DIAG}.parquet', 

    tidy_df = utils.modify_df_timezone(tidy_df, 'US/Mountain', 'UTC')
    tidy_df.to_parquet(os.path.join(output_dir, output_file_name), index=False)
    nan_counts_df_output_file_name = output_file_name.replace("tidy_df_", "nan_cnt_")
    limited_nan_counts_df.to_parquet(os.path.join(output_dir, nan_counts_df_output_file_name))
    
    return output_file_name, nan_counts_df_output_file_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--planar-fitted-dir",      type=str,       required=True)
    parser.add_argument("--filtering-str",          type=str,       required=True)
    parser.add_argument("--filter-snowfall",        type=bool,      required=True)
    parser.add_argument("--snowfall-mask-file",     type=str,       required=True)
    parser.add_argument("--snowfall-mask-str",      type=str,       required=True)
    parser.add_argument("--percentage-diag",        type=int,       required=True)
    parser.add_argument("--output-dir",             type=str,       required=True)
    args = parser.parse_args()  
    output_file_name, nan_counts_df_output_file_name = create_tidy_dataset(
        args.planar_fitted_dir,
        args.filtering_str,
        args.filter_snowfall,
        args.snowfall_mask_file,
        args.snowfall_mask_str,
        args.percentage_diag,
        args.output_dir
    )
    print("---")
    print("Generated file:")
    print(output_file_name)
    print("Generated nan counts file:")
    print(nan_counts_df_output_file_name)
    print("---")