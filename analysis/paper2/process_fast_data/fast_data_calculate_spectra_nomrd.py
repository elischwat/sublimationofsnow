"""
    Calculates MRD from 20hz high rate data for all dates of high rate data
    (between 20221101 - 20230619), calculating spectra over a given subset of hours
    specified by user inputs. 
    Note that the "start_hour" and "end_hour" input parameters are in UTC time.
    Operations are a parallelized to increase processing speed.

    Examples running the script
    #######################################
    python analysis/paper2/process_fast_data/fast_data_calculate_spectra_nomrd.py \
        -f \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230409_21.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230409_22.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230409_23.nc' \
        -o "/Users/elischwat/Development/data/sublimationofsnow/mrd/NOmrds/" \
        -s 1000 \
        -p 16

    python analysis/paper2/process_fast_data/fast_data_calculate_spectra_nomrd.py \
        -f \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230418_12.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230418_13.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230418_14.nc' \
        -o "/Users/elischwat/Development/data/sublimationofsnow/mrd/NOmrds/" \
        -s 1000 \
        -p 16
        
    python analysis/paper2/process_fast_data/fast_data_calculate_spectra_nomrd.py \
        -f \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230418_21.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230418_22.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230418_23.nc' \
        -o "/Users/elischwat/Development/data/sublimationofsnow/mrd/NOmrds/" \
        -s 1000 \
        -p 16

    python analysis/paper2/process_fast_data/fast_data_calculate_spectra_nomrd.py \
        -f \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20221221_21.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20221221_22.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20221221_23.nc' \
        -o "/Users/elischwat/Development/data/sublimationofsnow/mrd/NOmrds/" \
        -s 1000 \
        -p 16


    python analysis/paper2/process_fast_data/fast_data_calculate_spectra_nomrd.py \
        -f \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230304_21.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230304_22.nc' \
            '/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast_old/isfs_sos_qc_geo_tiltcor_hr_20230304_23.nc' \
        -o "/Users/elischwat/Development/data/sublimationofsnow/mrd/NOmrds/" \
        -s 1000 \
        -p 16
"""
import numpy as np
import xarray as xr

import datetime as dt
import pandas as pd
import argparse

from sublimpy import utils
import glob
import pytz
from scipy.signal import welch, csd
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import math
from datetime import datetime, timedelta

# First and second rotations that comprise the double rotation,
# as described here: http://link.springer.com/10.1023/A:1018966204465
def apply_first_rotation(df, u_var, v_var, w_var):
    mean_u = df[u_var].mean()
    mean_v = df[v_var].mean()
    theta = np.arctan2(mean_v, mean_u)
    adj_u = df[u_var]*np.cos(theta) + df[v_var]*np.sin(theta)
    adj_v = -df[u_var]*np.sin(theta) + df[v_var]*np.cos(theta)
    df[u_var] = adj_u
    df[v_var] = adj_v
    return df
def apply_second_rotation(df, u_var, v_var, w_var):
    mean_u = df[u_var].mean()
    mean_w = df[w_var].mean()
    phi = np.arctan2(mean_w, mean_u)
    adj_u = df[u_var]*np.cos(phi) + df[w_var]*np.sin(phi)
    adj_w = - df[u_var]*np.sin(phi) + df[w_var]*np.cos(phi)
    df[u_var] = adj_u
    df[w_var] = adj_w
    return df


def fast_data_files_to_dataframe(file_list, rotation = 'double'):
    assert rotation in ['single', 'double', 'none']
    ## Open files, prep timestamps, combine into dataset
    index_vars = ['base_time']
    value_vars = [
        'u_2m_c',	'v_2m_c',	'w_2m_c',	'h2o_2m_c',     'tc_2m_c',
        'u_3m_c',	'v_3m_c',	'w_3m_c',	'h2o_3m_c',     'tc_3m_c',
        'u_5m_c',	'v_5m_c',	'w_5m_c',	'h2o_5m_c',     'tc_5m_c',
        'u_10m_c',	'v_10m_c',	'w_10m_c',	'h2o_10m_c',    'tc_10m_c',
        'u_15m_c',	'v_15m_c',	'w_15m_c',	'h2o_15m_c',    'tc_15m_c',
        'u_20m_c',	'v_20m_c',	'w_20m_c',	'h2o_20m_c',    'tc_20m_c',
        'u_3m_uw',	'v_3m_uw',	'w_3m_uw',	'h2o_3m_uw',    'tc_3m_uw',
        'u_10m_uw',	'v_10m_uw',	'w_10m_uw',	'h2o_10m_uw',   'tc_10m_uw',
        'u_3m_ue',	'v_3m_ue',	'w_3m_ue',	'h2o_3m_ue',    'tc_3m_ue',
        'u_10m_ue',	'v_10m_ue',	'w_10m_ue',	'h2o_10m_ue',   'tc_10m_ue',
        'u_3m_d',	'v_3m_d',	'w_3m_d',	'h2o_3m_d',     'tc_3m_d',
        'u_10m_d',	'v_10m_d',	'w_10m_d',	'h2o_10m_d',    'tc_10m_d',
    ]
    suffixes = [ '2m_c',  '3m_c',  '5m_c',  '10m_c',  '15m_c',  '20m_c',  '3m_uw',  '10m_uw',  '3m_ue',  '10m_ue',  '3m_d',  '10m_d']
    VARIABLES = index_vars + value_vars
    fast_ds = xr.open_mfdataset( file_list, concat_dim="time",  combine="nested",  data_vars=VARIABLES)
    fast_df = fast_ds[VARIABLES].to_dataframe()

    ## Create timestamp
    fast_df = fast_df.reset_index()
    fast_df['time'] = fast_df.apply(lambda row: dt.datetime(
            year = row['time'].year, month = row['time'].month, day = row['time'].day,
            hour = row['base_time'].hour, minute = row['time'].minute, second = row['time'].second, 
            microsecond = int(row['sample'] * (1e6/20))
        ),
        axis = 1
    )
    
    # Interpolate all the variables 
    fast_df = fast_df.set_index('time')[value_vars].interpolate().reset_index()

    if rotation != 'none':
        for suffix in suffixes:
            # then rotation is single or double, and we at least need to do the first rotation
            fast_df = apply_first_rotation(fast_df, f'u_{suffix}', f'v_{suffix}', f'w_{suffix}')
            print("mean u, v, w after first rotation ", fast_df[[f'u_{suffix}', f'v_{suffix}', f'w_{suffix}']].mean())    
                    
        if rotation == 'double':
            for suffix in suffixes:
                # then rotation is single or double, and we at least need to do the first rotation
                fast_df = apply_second_rotation(fast_df, f'u_{suffix}', f'v_{suffix}', f'w_{suffix}')
                print("mean u, v, w after second rotation ", fast_df[[f'u_{suffix}', f'v_{suffix}', f'w_{suffix}']].mean())    
    
    return fast_df

##########
def newmrd(data_a, data_b, M, Mx):
    D = np.zeros(M - Mx)
    Dstd = np.copy(D)
    data_a2 = np.copy(data_a)
    data_b2 = np.copy(data_b)
    for ims in range(M - Mx + 1):
        ms = M - ims
        l = 2 ** ms
        nw = round((2 ** M) / l)
        wmeans_a = np.zeros(nw)
        wmeans_b = np.copy(wmeans_a)
        for i in range(nw):
            k = round(i * l)
            wmeans_a[i] = np.mean(data_a2[k:(i+1)*l])
            wmeans_b[i] = np.mean(data_b2[k:(i+1)*l])
            data_a2[k:(i+1)*l] -= wmeans_a[i]
            data_b2[k:(i+1)*l] -= wmeans_b[i]
        if nw > 1:
            D[ms] = np.mean(wmeans_a * wmeans_b)
            Dstd[ms] = np.std(wmeans_a * wmeans_b, ddof=0)
    return D, Dstd

def calculate_mrd_for_df(df, VAR1, VAR2, shift, parallelism):
    M = int(np.floor(np.log2(len(df))))
    print(f"Got data of length {len(df)}. using M = {M}")
    timestep = (
        df['time'].iloc[1] - df['time'].iloc[0]
    ).total_seconds() * 1000
    print(f"Timestep of fast data is: {timestep} ms")
    mrd_x = np.array([
        dt.timedelta(milliseconds=2**i * timestep).total_seconds() 
        for i in range(1, M+1)
    ])

    def send_subset_to_mrd_algorithm(i):
        i_start = i * shift
        i_end = i * shift + 2**M - 1
        this_df = df.loc[ i_start : i_end]
        result = newmrd(
            this_df[VAR1],
            this_df[VAR2],
            M, 
            0
        )
        return pd.DataFrame({
            'tau':       mrd_x,
            'Co':        result[0],
            'std':       result[1],
            'iteration': i,
            'start_time':this_df.time.min(), 
            'end_time':  this_df.time.max(),
        })
    
    n_iterations = math.floor((len(df) - 2**M) / shift)
    print(f"Dataset of length {len(df)}, with M = {M}, permits {n_iterations} iterations.")
    iterations_tqdm = tqdm(range(n_iterations))

    processed_results = Parallel(n_jobs = parallelism)(
        delayed(send_subset_to_mrd_algorithm)(i) for i in iterations_tqdm
    )

    return pd.concat(processed_results)

def process_args(input_file_list, shift, parallelism, output_path):
    fast_df = fast_data_files_to_dataframe(input_file_list)
    print(f"Number of valid rows pre-forward-filling: {len(fast_df.dropna())}")
    fast_df = fast_df.ffill()
    print(f"Number of valid rows post-forward-filling: {len(fast_df.dropna())}")

    suffixes = [
        '3m_c', 
        '5m_c', 
        '10m_c', 
        '15m_c',
        '20m_c', 
        '3m_uw', '10m_uw', 
        '3m_ue', '10m_ue', 
        '3m_d',  '10m_d'
    ]
    variable_prefixes_for_spectra_calculation = [
        ('u_', 'u_'),
        ('v_', 'v_'),
        ('w_', 'w_'),
        ('u_', 'w_'),
        ('v_', 'w_'),
        ('u_', 'v_'),
        ('w_', 'tc_'),
        ('w_', 'h2o_'),
    ]
    spectra_results_list = []
    for prefix1, prefix2 in variable_prefixes_for_spectra_calculation:
        for suffix in suffixes:
            spectra_result = calculate_mrd_for_df(fast_df, prefix1 + suffix, prefix2 + suffix, shift = shift, parallelism = parallelism)
            spectra_result['covariance'] = prefix1 + prefix2
            spectra_result['loc'] = suffix
            spectra_results_list.append(spectra_result)
    output_file = os.path.join(
            output_path, 
            f"{fast_df.time.min()}_{fast_df.time.max()}.parquet"
        )
    print(f"Saving to output_file: {output_file}")
    pd.concat(spectra_results_list).to_parquet(output_file)

if __name__ == '__main__':
    ####################################
    # THIS VERSION READS FROM TERMINAL
    ####################################

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-f",
    #     "--input-file-list",
    #     nargs="*",
    #     required=True,
    #     help="List of file paths to process, must be contiguous in time"
    # )
    # parser.add_argument(
    #     "-o",
    #     "--output-path",
    #     type=str,
    #     required=True,
    #     help="Output directory for files with calculated spectra."
    # )
    # parser.add_argument(
    #     "-s",
    #     "--shift",
    #     type=int,
    #     required=True,
    #     help="Shift for overlapping data segments on which MRDs are performed."
    # )
    # parser.add_argument(
    #     "-p",
    #     "--parallelism",
    #     type=int,
    #     required=True,
    #     help="Number of cores to use for parallel processing."
    # )
    # args = parser.parse_args()    
    # process_args(args.input_file_list, args.shift, args.parallelism, args.output_path)
    

    ####################################
    # THIS VERSION RUNS FROM THE SCRIPT
    ####################################

    # Define the start and end dates
    start_date = datetime(2023, 3, 1)
    end_date = datetime(2023, 5, 1)

    # Generate the list of date strings
    date_list = []
    current_date = start_date
    while current_date < end_date:
        date_list.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    for date in date_list:
        print(f"Processing date: {date}")

        input_file_list = [
            f'/storage/elilouis/sublimationofsnow/sosqc_fast/isfs_sos_qc_geo_tiltcor_hr_{date}_21.nc'
            f'/storage/elilouis/sublimationofsnow/sosqc_fast/isfs_sos_qc_geo_tiltcor_hr_{date}_22.nc'
            f'/storage/elilouis/sublimationofsnow/sosqc_fast/isfs_sos_qc_geo_tiltcor_hr_{date}_23.nc'
        ]
        shift = 2000
        parallelism = 20
        output_path = "/storage/elilouis/sublimationofsnow/mrd/NOmrds/"
        process_args(input_file_list, shift, parallelism, output_path)