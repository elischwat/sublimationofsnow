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

"""
Initialize parameters, file inputs
"""
# base path for a number of different directories this script needs
DATA_DIR = "/Users/elischwat/Development/data/"
# path to directory where daily files are stored
OUTPUT_PATH = f"{DATA_DIR}sublimationofsnow/planar_fit_processed_30min/"
# n cores utilized by application
PARALLELISM = 8
# Reynolds averaging length, in units (1/20) seconds
SAMPLES_PER_AVERAGING_LENGTH = 30*60*20

# # Open fast data
file_list = sorted(glob.glob(f"{DATA_DIR}sublimationofsnow/sosqc_fast/*.nc"))
file_list = [f for f in file_list if '202210' not in f]

# Open planar fit data
monthly_file = f"{DATA_DIR}sublimationofsnow/monthly_planar_fits.csv"
weekly_file = f"{DATA_DIR}sublimationofsnow/weekly_planar_fits.csv"
fits_df = pd.read_csv(monthly_file, delim_whitespace=True)
weeklyfits_df = pd.read_csv(weekly_file, delim_whitespace=True)

# Transform planar fit data
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

# Set some convenience variables
always_there_vars = [
    'base_time',
    'u_2m_c',	'v_2m_c',	'w_2m_c',	'h2o_2m_c',		'tc_2m_c',
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
]
c_1m_vars = ['u_1m_c',	'v_1m_c',	'w_1m_c',	'h2o_1m_c',		'tc_1m_c']
d_1m_vars = ['u_1m_d',	'v_1m_d',	'w_1m_d',	'h2o_1m_d',		'tc_1m_d']
ue_1m_vars = ['u_1m_ue',	'v_1m_ue',	'w_1m_ue',	'h2o_1m_ue',	'tc_1m_ue']
uw_1m_vars = ['u_1m_uw',	'v_1m_uw',	'w_1m_uw',	'h2o_1m_uw',	'tc_1m_uw']

"""
Processes a list of files with 20Hz EC data, and produces a planar-fitted 
and reynolds averaged dataset, which is saved to a single file.
"""
def process_files(file_list, output_file):
    # print(f"Processing n files: {len(file_list)}")
    # Open planar fit data
    ds = xr.open_mfdataset(file_list, concat_dim="time", combine="nested")
    # Build up a list of vars to ensure we only access vars that are actually in the dataset
    all_vars_in_ds = always_there_vars
    if 'u_1m_c' in ds:
        all_vars_in_ds = all_vars_in_ds + c_1m_vars
    if 'u_1m_d' in ds:
        all_vars_in_ds = all_vars_in_ds + d_1m_vars
    if 'u_1m_ue' in ds:
        all_vars_in_ds = all_vars_in_ds + ue_1m_vars
    if 'u_1m_uw' in ds:
        all_vars_in_ds = all_vars_in_ds + uw_1m_vars

    ds = ds[all_vars_in_ds]

    # Create timestamp
    # To use the datam, its necessary to combine 3 columns of data from the dataset to get the full timestamp. 
    # This is demonstrated below. The 'time' column actually only incudes the second and minute information. For all datapoints, the hour according to the 'time' column is 1.  
    # The 'base_time' column indicates the hour of the day. The 'sample' column indicates the 20hz sample number. 
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
    
    # Iterate over variables, apply planar fit to fast data, and calculate covariance fluxes
    assert len(pd.Series(ds.time.dt.month).unique()) == 1
    MONTH = ds.time.dt.month.values[0]

    df_list = []
    for tower in ['c', 'uw', 'ue', 'd']:
        if tower == 'c':
            heights = [1,2,3,5,10,15,20]
        else:
            heights = [1,3,10]
        
        for height in heights:
            # only operate on this height/tower if those measurements are in this day's ds
            if f"u_{height}m_{tower}" in ds:
                # Retrieve the planar fit parameters for this month, height, tower
                fitting_params = fits_df.set_index(['month', 'height', 'tower']).loc[
                    MONTH,
                    height,
                    tower
                ]
                
                # Calculate the planar fitted 20Hz u, v, w values
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

                # Define function to do Reynolds Averaging
                def create_re_avg_ds(ds, var1,  var2, covariance_name):
                    coarse_ds = ds.coarsen(time=SAMPLES_PER_AVERAGING_LENGTH).mean(skipna=True)
                    coarse_ds = coarse_ds.assign_coords(time = coarse_ds.time.dt.round('1s'))
                    coarse_ds = coarse_ds.reindex_like(ds, method='nearest')
                    ds[f"{var1}_mean"] = coarse_ds[f"{var1}"]
                    ds[f"{var1}_fluc"] = ds[f"{var1}"] - ds[f"{var1}_mean"]
                    ds[f"{var2}_mean"] = coarse_ds[f"{var2}"]
                    ds[f"{var2}_fluc"] = ds[f"{var2}"] - ds[f"{var2}_mean"]
                    ds[covariance_name] = ds[f"{var2}_fluc"] * ds[f"{var1}_fluc"]
                    ds = ds.coarsen(time = SAMPLES_PER_AVERAGING_LENGTH).mean()
                    ds = ds.assign_coords(time = ds.time.dt.round('1s'))
                    return ds.to_dataframe()

                # Calculate un-fitted reynolds averaged variables
                ds_plain_w =    create_re_avg_ds(ds, f'w_{height}m_{tower}', f'h2o_{height}m_{tower}',  f'w_h2o__{height}m_{tower}')[[
                    f'u_{height}m_{tower}',
                    f'v_{height}m_{tower}',
                    f'w_{height}m_{tower}',
                    f'w_h2o__{height}m_{tower}'
                ]]
                ds_plain_u =    create_re_avg_ds(ds, f'u_{height}m_{tower}', f'h2o_{height}m_{tower}', f'u_h2o__{height}m_{tower}' )[f'u_h2o__{height}m_{tower}']
                ds_plain_v =    create_re_avg_ds(ds, f'v_{height}m_{tower}', f'h2o_{height}m_{tower}', f'v_h2o__{height}m_{tower}' )[f'v_h2o__{height}m_{tower}']
                ds_plain_w_w =  create_re_avg_ds(ds, f'w_{height}m_{tower}', f'w_{height}m_{tower}',   f'w_w__{height}m_{tower}'   )[f'w_w__{height}m_{tower}']
                ds_plain_u_u =  create_re_avg_ds(ds, f'u_{height}m_{tower}', f'u_{height}m_{tower}',   f'u_u__{height}m_{tower}'   )[f'u_u__{height}m_{tower}']
                ds_plain_v_v =  create_re_avg_ds(ds, f'v_{height}m_{tower}', f'v_{height}m_{tower}',   f'v_v__{height}m_{tower}'   )[f'v_v__{height}m_{tower}']
                ds_plain_u_w =  create_re_avg_ds(ds, f'u_{height}m_{tower}', f'w_{height}m_{tower}',   f'u_w__{height}m_{tower}'   )[f'u_w__{height}m_{tower}']
                ds_plain_v_w =  create_re_avg_ds(ds, f'v_{height}m_{tower}', f'w_{height}m_{tower}',   f'v_w__{height}m_{tower}'   )[f'v_w__{height}m_{tower}']

                # Calculate fitted reynolds averaged variables
                ds_fit_w =    create_re_avg_ds(ds,  f'w_{height}m_{tower}_fit', f'h2o_{height}m_{tower}', f'w_h2o__{height}m_{tower}_fit')[[
                    f'u_{height}m_{tower}_fit',
                    f'v_{height}m_{tower}_fit',
                    f'w_{height}m_{tower}_fit',
                    f'w_h2o__{height}m_{tower}_fit'
                ]]
                ds_fit_u =    create_re_avg_ds(ds, f'u_{height}m_{tower}_fit', f'h2o_{height}m_{tower}',       f'u_h2o__{height}m_{tower}_fit')[f'u_h2o__{height}m_{tower}_fit']
                ds_fit_v =    create_re_avg_ds(ds, f'v_{height}m_{tower}_fit', f'h2o_{height}m_{tower}',       f'v_h2o__{height}m_{tower}_fit')[f'v_h2o__{height}m_{tower}_fit']
                ds_fit_w_w =  create_re_avg_ds(ds, f'w_{height}m_{tower}_fit', f'w_{height}m_{tower}_fit',     f'w_w__{height}m_{tower}_fit')[f'w_w__{height}m_{tower}_fit']
                ds_fit_u_u =  create_re_avg_ds(ds, f'u_{height}m_{tower}_fit', f'u_{height}m_{tower}_fit',     f'u_u__{height}m_{tower}_fit')[f'u_u__{height}m_{tower}_fit']
                ds_fit_v_v =  create_re_avg_ds(ds, f'v_{height}m_{tower}_fit', f'v_{height}m_{tower}_fit',     f'v_v__{height}m_{tower}_fit')[f'v_v__{height}m_{tower}_fit']
                ds_fit_u_w =  create_re_avg_ds(ds, f'u_{height}m_{tower}_fit', f'w_{height}m_{tower}_fit',     f'u_w__{height}m_{tower}_fit')[f'u_w__{height}m_{tower}_fit']
                ds_fit_v_w =  create_re_avg_ds(ds, f'v_{height}m_{tower}_fit', f'w_{height}m_{tower}_fit',     f'v_w__{height}m_{tower}_fit')[f'v_w__{height}m_{tower}_fit']

                # Combine different variables into one dataset
                df_plain = ds_plain_w.join(ds_plain_u).join(ds_plain_v).join(ds_plain_w_w).join(ds_plain_u_u).join(ds_plain_v_v).join(ds_plain_u_w).join(ds_plain_v_w) 
                df_fit = ds_fit_w.join(ds_fit_u).join(ds_fit_v).join(ds_fit_w_w).join(ds_fit_u_u).join(ds_fit_v_v).join(ds_fit_u_w).join(ds_fit_v_w)          
                
                # Isolate the variables we want - this should be unnecessary
                plain_vars = [
                    f'u_{height}m_{tower}',         f'v_{height}m_{tower}',         f'w_{height}m_{tower}',
                    f'w_h2o__{height}m_{tower}',    f'u_h2o__{height}m_{tower}',    f'v_h2o__{height}m_{tower}',
                    f'w_w__{height}m_{tower}',      f'u_u__{height}m_{tower}',      f'v_v__{height}m_{tower}', f'u_w__{height}m_{tower}', f'v_w__{height}m_{tower}',
                ]
                fit_vars = [
                    f'u_{height}m_{tower}_fit',         f'v_{height}m_{tower}_fit',         f'w_{height}m_{tower}_fit',
                    f'w_h2o__{height}m_{tower}_fit',    f'u_h2o__{height}m_{tower}_fit',    f'v_h2o__{height}m_{tower}_fit',
                    f'w_w__{height}m_{tower}_fit',      f'u_u__{height}m_{tower}_fit',      f'v_v__{height}m_{tower}_fit', f'u_w__{height}m_{tower}_fit', f'v_w__{height}m_{tower}_fit',
                ]

                # Combined variables and add to the list of generated dataframes
                merged_df = df_plain[plain_vars].join(
                    df_fit[fit_vars]
                )
                df_list.append(merged_df)
    
    # Combine datasets from 
    combined_df = df_list[0].join(df_list[1:])
    combined_df.to_parquet(output_file)
    return output_file


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run application
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    n_files = len(file_list)
    print(f"n files: {n_files}")

    def print_and_process(i):
        output_file = os.path.join(OUTPUT_PATH, file_list[i].split('/')[-1][27:]) + '.parquet'
        # print(output_file)
        # print(f"Processing file: \n\t{file_list[i]} and saving to: \n\t{output_file}")
        process_files(
            [file_list[i]],
            output_file
        )
    # slow
    # saved_files_list = [print_and_process(i) for i in tqdm(list(range(0, n_days)))]
        
    # fast
    saved_files_list =  Parallel(n_jobs = PARALLELISM)(
        delayed(print_and_process)(i)
        for i in tqdm(list(range(0, n_files)))
    )
    print("Finished processing")