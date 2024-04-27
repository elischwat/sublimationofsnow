# ---

# ---
import numpy as np
import os
import xarray as xr
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
alt.data_transformers.enable('json')
from sublimpy import utils, extrautils
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

PARALLELISM = 16
DATA_DIR = "/storage/elilouis/"

# Save to data path
OUTPUT_PATH = f"{DATA_DIR}sublimationofsnow/planar_fit_processed_30min/"

# # Open fast data
file_list = sorted(glob.glob(f"{DATA_DIR}sublimationofsnow/planar_fit/*.nc"))
file_list = [f for f in file_list if '202210' not in f]

# Open planar fit fit data
monthly_file = f"{DATA_DIR}sublimationofsnow/monthly_planar_fits.csv"
weekly_file = f"{DATA_DIR}sublimationofsnow/weekly_planar_fits.csv"
fits_df = pd.read_csv(monthly_file, delim_whitespace=True)
weeklyfits_df = pd.read_csv(weekly_file, delim_whitespace=True)
fits_df['height'] = fits_df['height'].str.replace('_', '.').astype('float')
weeklyfits_df['start_date'] = pd.to_datetime(weeklyfits_df['start_date'], format='%Y%m%d')
weeklyfits_df['end_date'] = pd.to_datetime(weeklyfits_df['end_date'], format='%Y%m%d')
fits_df['W_f'] = fits_df.apply(
    lambda row: [row['W_f_1'], row['W_f_2'], row['W_f_3']],
    axis=1
).drop(columns=['W_f_1', 'W_f_2', 'W_f_3'])
weeklyfits_df['W_f'] = weeklyfits_df.apply(
    lambda row: [row['W_f_1'], row['W_f_2'], row['W_f_3']],
    axis=1
).drop(columns=['W_f_1', 'W_f_2', 'W_f_3'])

def process_files(file_list, output_file):

    ds = xr.open_mfdataset(file_list, concat_dim="time", combine="nested")

    ds = ds[[
        'base_time',
        'u_3m_c',	'v_3m_c',	'w_3m_c',	'h2o_3m_c',		'tc_3m_c',
        'u_3m_d',	'v_3m_d',	'w_3m_d',	'h2o_3m_d',		'tc_3m_d',
        'u_3m_ue',	'v_3m_ue',	'w_3m_ue',	'h2o_3m_ue',	'tc_3m_ue',
        'u_3m_uw',	'v_3m_uw',	'w_3m_uw',	'h2o_3m_uw',	'tc_3m_uw',
        'u_5m_c',	'v_5m_c',	'w_5m_c',	'h2o_5m_c',		'tc_5m_c',
        'u_10m_c',	'v_10m_c',	'w_10m_c',	'h2o_10m_c',	'tc_10m_c',
        'u_10m_d',	'v_10m_d',	'w_10m_d',	'h2o_10m_d',	'tc_10m_d',
        'u_10m_ue',	'v_10m_ue',	'w_10m_ue',	'h2o_10m_ue',	'tc_10m_ue',
        'u_10m_uw',	'v_10m_uw',	'w_10m_uw',	'h2o_10m_uw',	'tc_10m_uw',
        'u_15m_c',	'v_15m_c',	'w_15m_c',	'h2o_15m_c',	'tc_15m_c',
        'u_20m_c',	'v_20m_c',	'w_20m_c',	'h2o_20m_c',	'tc_20m_c',
    ]]

    # # Create timestamp
    # To use the datam, its necessary to combine 3 columns of data from the dataset to get the full timestamp. This is demonstrated below. The 'time' column actually only incudes the second and minute information. For all datapoints, the hour according to the 'time' column is 1.  The 'base_time' column indicates the hour of the day. The 'sample' column indicates the 20hz sample number. 
    df = ds.to_dataframe().reset_index()

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
    ds = df.set_index('time').to_xarray()

    # Define function to do Reynolds Averaging
    def create_re_avg_ds(
            ds, 
            re_avg_period_size, 
            var1,
            var2,
            covariance_name
    ):
        coarse_ds = ds.coarsen(time=re_avg_period_size).mean()
        coarse_ds = coarse_ds.assign_coords(time = coarse_ds.time.dt.round('1s'))
        coarse_ds = coarse_ds.reindex_like(ds, method='nearest')
        ds[f"{var1}_mean"] = coarse_ds[f"{var1}"]
        ds[f"{var1}_fluc"] = ds[f"{var1}"] - ds[f"{var1}_mean"]
        ds[f"{var2}_mean"] = coarse_ds[f"{var2}"]
        ds[f"{var2}_fluc"] = ds[f"{var2}"] - ds[f"{var2}_mean"]
        ds[covariance_name] = ds[f"{var2}_fluc"] * ds[f"{var1}_fluc"]
        ds = ds.coarsen(time = re_avg_period_size).mean()
        ds = ds.assign_coords(time = ds.time.dt.round('1s'))
        return ds


    # # Iterate over variables, apply planar fit to fast data, and calculate covariance fluxes
    MONTH = ds.time.dt.month.values[0]

    df_list = []
    for tower in ['c', 'uw', 'ue', 'd']:
        if tower == 'c':
            heights = [3,5,10,15,20]
        else:
            heights = [3,10]
        
        for height in heights:
            fitting_params = fits_df.set_index(['month', 'height', 'tower']).loc[
                MONTH,
                height,
                tower
            ]
            u, v, w = extrautils.apply_planar_fit(
                ds[f'u_{height}m_{tower}'].values.flatten(),
                ds[f'v_{height}m_{tower}'].values.flatten(),
                ds[f'w_{height}m_{tower}'].values.flatten(),
                fitting_params['a'], 
                fitting_params['W_f'],
            )
            ds[f'u_{height}m_{tower}_fit'] = ('time', u)
            ds[f'v_{height}m_{tower}_fit'] = ('time', v)
            ds[f'w_{height}m_{tower}_fit'] = ('time', w)
            
            ds_plain =  create_re_avg_ds(
                ds, 
                5*60*20,
                var1 = f'w_{height}m_{tower}', 
                var2= f'h2o_{height}m_{tower}', 
                covariance_name = f'w_h2o__{height}m_{tower}'
            )
            ds_fit =    create_re_avg_ds(
                ds, 
                5*60*20,
                var1 = f'w_{height}m_{tower}_fit', 
                var2= f'h2o_{height}m_{tower}', 
                covariance_name = f'w_h2o__{height}m_{tower}_fit'
            )
            plain_vars = [
                f'u_{height}m_{tower}',
                f'v_{height}m_{tower}',
                f'w_{height}m_{tower}',
                f'w_h2o__{height}m_{tower}'
            ]
            fit_vars = [
                f'u_{height}m_{tower}_fit',
                f'v_{height}m_{tower}_fit',
                f'w_{height}m_{tower}_fit',
                f'w_h2o__{height}m_{tower}_fit'
            ]
            merged_df = ds_plain[plain_vars].to_dataframe()[plain_vars].join(
                ds_fit[fit_vars].to_dataframe()[fit_vars]
            )
            df_list.append(merged_df)
    
    combined_df = df_list[0].join(df_list[1:])
    combined_df.to_parquet(output_file)
    return output_file


if __name__ == '__main__':
    n_days = int(len(file_list)/24)

    print(len(file_list))

    def print_and_process(i):
        start_i = i*24
        end_i = (i+1)*24
        output_file = os.path.join(OUTPUT_PATH, file_list[start_i].split('/')[-1][27:-6]) + '.parquet'
        print(f"Processing files from {file_list[start_i].split('/')[-1]} through {file_list[end_i].split('/')[-1]} and saving to: {output_file}")
        process_files(
            file_list[start_i: end_i], 
            output_file
        )

    # fast
    saved_files_list =  Parallel(n_jobs = PARALLELISM)(
        delayed(print_and_process)(i)
        for i in tqdm(list(range(0, n_days)))
    )
    print("Finished processing")