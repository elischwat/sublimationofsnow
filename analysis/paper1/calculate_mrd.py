"""
    Calculates MRD from 20hz high rate data for all dates of high rate data
    (between 20221101 - 20230619), calculating spectra over a given subset of hours
    specified by user inputs. 
    Note that the "start_hour" and "end_hour" input parameters are in UTC time.
    Operations are a parallelized to increase processing speed.

    Examples running the script
    #######################################
    # Example 1, 0900-1700 local time ("daytime")
    python analysis/paper1/calculate_mrd.py -s 16    -e 0   -o "/Users/elischwat/Development/data/sublimationofsnow/mrd/0900_1700" -i "/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast/" -p 8
    

    # Example 2, 1900-0500 ("nighttime")
    python analysis/paper1/calculate_mrd.py -s 2   -e 12   -o "/Users/elischwat/Development/data/sublimationofsnow/mrd/1900_0500" -i "/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast/" -p 8
    
    
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

START_DATE = '20221101'
END_DATE = '20230619'

DATE_FORMAT = "%Y%m%d"
datelist = [
    timestamp.date().strftime(DATE_FORMAT)
    for timestamp in pd.date_range(
        START_DATE, 
        END_DATE
    ).tolist()
]

index_vars = ['base_time']
value_vars = [
        'u_2m_c',	'v_2m_c',	'w_2m_c',	'h2o_2m_c', 'tc_2m_c',
        'u_3m_c',	'v_3m_c',	'w_3m_c',	'h2o_3m_c', 'tc_3m_c',
        'u_5m_c',	'v_5m_c',	'w_5m_c',	'h2o_5m_c', 'tc_5m_c',
        'u_10m_c',	'v_10m_c',	'w_10m_c',	'h2o_10m_c', 'tc_10m_c',
        'u_15m_c',	'v_15m_c',	'w_15m_c',	'h2o_15m_c', 'tc_15m_c',
        'u_20m_c',	'v_20m_c',	'w_20m_c',	'h2o_20m_c', 'tc_20m_c',

        'u_3m_uw',	'v_3m_uw',	'w_3m_uw',	'h2o_3m_uw', 'tc_3m_uw',
        'u_10m_uw',	'v_10m_uw',	'w_10m_uw',	'h2o_10m_uw', 'tc_10m_uw',

        'u_3m_ue',	'v_3m_ue',	'w_3m_ue',	'h2o_3m_ue', 'tc_3m_ue',
        'u_10m_ue',	'v_10m_ue',	'w_10m_ue',	'h2o_10m_ue', 'tc_10m_ue',

        'u_3m_d',	'v_3m_d',	'w_3m_d',	'h2o_3m_d', 'tc_3m_d',
        'u_10m_d',	'v_10m_d',	'w_10m_d',	'h2o_10m_d', 'tc_10m_d',
    ]
VARIABLES = index_vars + value_vars

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

def process_date(date, file_list, start_i, end_i, output_dir): 
    print(f"Processing date: {date}")

    # WHAT IF start_i > end_i
    if start_i > end_i:
        # this means that we need to get files from multiple dates  
        date_obj = dt.datetime.strptime(date, DATE_FORMAT)
        next_date_obj = date_obj + dt.timedelta(days=1)
        next_date = next_date_obj.strftime(DATE_FORMAT)
        local_file_list = [ f for f in file_list if f"_{date}" in f or f"_{next_date}" in f]
        local_file_list = sorted(local_file_list)[start_i: end_i + 24]
    else:
        # identify files we want from this one date
        local_file_list = [ f for f in file_list if f"_{date}" in f]
        local_file_list = sorted(local_file_list)[start_i:end_i]

    # open files and convert to DF
    ds = xr.open_mfdataset(
        local_file_list, concat_dim="time", 
        combine="nested", 
        data_vars=VARIABLES
    )
    df = ds[VARIABLES].to_dataframe()

    M = int(np.floor(np.log2(len(df))))
    print(f"Got data of length {len(df)}. using M = {M}")

    df = df.head(2**M)
    # create timestamp 
    df = df.reset_index()
    df['time'] = df.apply(lambda row: dt.datetime(
            year = row['time'].year,
            month = row['time'].month,
            day = row['time'].day,
            hour = row['base_time'].hour,
            minute = row['time'].minute,
            second = row['time'].second,
            microsecond = int(row['sample'] * (1e6/20))
        ),
        axis = 1
    )
    df = utils.modify_df_timezone(df, pytz.UTC, "US/Mountain")

    # interpolate nans
    for var in value_vars:
        df[var] = df[var].interpolate()

    towers = ['c', 'd', 'uw', 'ue']

    # calculate sensible heat cospectra
    local_df_list = []
    for tower in towers:
        if tower == 'c':
            heights = [2, 3, 5, 10, 15, 20]
        else:
            heights = [3, 10]
        for height in heights:
            var1 = f'w_{height}m_{tower}'
            var2 = f'tc_{height}m_{tower}'
            result = newmrd(
                df[var1], 
                df[var2],
                M, 
                0
            )
            timestep = (
                df['time'].iloc[1] - df['time'].iloc[0]
            ).total_seconds() * 1000
            mrd_x = np.array([
                dt.timedelta(milliseconds=2**i * timestep).total_seconds() 
                for i in range(1, M+1)
            ])
            result_df = pd.DataFrame({
                'tau':      mrd_x,
                'Co':       result[0],
                'std':      result[1]
            })
            result_df['var1'] = var1
            result_df['var2'] = var2
            result_df['height'] = height
            result_df['tower'] = tower
            local_df_list.append(result_df)
    sensheat_copower_spectrum = pd.concat(local_df_list)

    # Calculate latent heat cospectra
    local_df_list = []
    for tower in towers:
        if tower == 'c':
            heights = [2, 3, 5, 10, 15, 20]
        else:
            heights = [3, 10]
        for height in heights:
            var1 = f'w_{height}m_{tower}'
            var2 = f'h2o_{height}m_{tower}'
            result = newmrd(
                df[var1], 
                df[var2],
                M, 
                0
            )
            timestep = (
                df['time'].iloc[1] - df['time'].iloc[0]
            ).total_seconds() * 1000
            mrd_x = np.array([
                dt.timedelta(milliseconds=2**i * timestep).total_seconds() 
                for i in range(1, M+1)
            ])
            result_df = pd.DataFrame({
                'tau':      mrd_x,
                'Co':       result[0],
                'std':      result[1]
            })
            result_df['var1'] = var1
            result_df['var2'] = var2
            result_df['height'] = height
            result_df['tower'] = tower
            local_df_list.append(result_df)
    latheat_copower_spectrum = pd.concat(local_df_list)

    dir_sensible_heat = os.path.join(output_dir, 'sensible_heat')
    if not os.path.exists(dir_sensible_heat):
        os.makedirs(dir_sensible_heat)
    sensheat_copower_spectrum_fn = os.path.join(dir_sensible_heat, date + '.parquet')
    sensheat_copower_spectrum = sensheat_copower_spectrum.assign(date = date)
    sensheat_copower_spectrum.to_parquet(sensheat_copower_spectrum_fn)

    dir_latent_heat = os.path.join(output_dir, 'latent_heat')
    if not os.path.exists(dir_latent_heat):
        os.makedirs(dir_latent_heat)
    latheat_copower_spectrum_fn = os.path.join(dir_latent_heat, date + '.parquet')
    latheat_copower_spectrum = latheat_copower_spectrum.assign(date = date)
    latheat_copower_spectrum.to_parquet(latheat_copower_spectrum_fn)

    return (
        sensheat_copower_spectrum_fn,
        latheat_copower_spectrum_fn
    )




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Directory containing fast sos datasets (netcdf files containing 20hz measurements)."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output directory for files with calculated spectra."
    )
    parser.add_argument(
        "-s",
        "--start-hour",
        type=int,
        required=True,
        help="Starting hour for calculating spectra. UTC time."
    )
    parser.add_argument(
        "-e",
        "--end-hour",
        type=int,
        required=True,
        help="Ending hour for calculating spectra. UTC time."
    )
    parser.add_argument(
        "-p",
        "--parallelism",
        type=int,
        required=True,
        help="Number of cores to use for parallel processing."
    )
    args = parser.parse_args()    

    

    file_list = glob.glob(os.path.join(args.input_path, "*.nc"))
    dates_tqdm = tqdm(datelist)
    start_i, end_i, output_dir = (args.start_hour, args.end_hour, args.output_path)
    print(f"N total fast files: {len(file_list)}")
    print(f"Beginning processing (parallelism = {args.parallelism})")
    processed_results =  Parallel(n_jobs = args.parallelism)(
        delayed(process_date)(date, file_list, start_i, end_i, output_dir) for date in dates_tqdm
    )