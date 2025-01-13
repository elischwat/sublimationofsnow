"""
Why are planar fit parameters calculated in another notebook (process_slow_data/planar_fit_monthly.py)?
Let's just put that in here and allow a configuration for this script on if we want monthly/weekly/etc

No... we can't do that. Calculating monthly planar fits requires opening up all the data. 
We can open up all that fast data at once.
"""
import os
import xarray as xr
import numpy as np
import datetime as dt
import pandas as pd
from sublimpy import extrautils
import glob
from tqdm import tqdm
import traceback
import concurrent.futures

"""
Initialize parameters, file inputs
"""
# base path for a number of different directories this script needs
DATA_DIR = "/storage/elilouis/"
# path to directory where daily files are stored
OUTPUT_PATH = f"{DATA_DIR}sublimationofsnow/double_rotated_30min_despiked_q7/"
DESPIKE = True
FILTERING_q = 7
# n cores utilized by application
PARALLELISM = 20
# Reynolds averaging length, in units (1/20) seconds
SAMPLES_PER_AVERAGING_LENGTH = 30*60*20

# # Open fast data
file_list = sorted(glob.glob(f"{DATA_DIR}sublimationofsnow/sosqc_fast/*.nc"))
file_list = [f for f in file_list if '202210' not in f]


# Set some convenience variables
always_there_vars = [
    'base_time',
    'u_2m_c',	'v_2m_c',	'w_2m_c',	'h2o_2m_c',		'tc_2m_c',      'irgadiag_2m_c', 'ldiag_2m_c',
    'u_3m_c',	'v_3m_c',	'w_3m_c',	'h2o_3m_c',		'tc_3m_c',      'irgadiag_3m_c', 'ldiag_3m_c',
    'u_3m_d',	'v_3m_d',	'w_3m_d',	'h2o_3m_d',		'tc_3m_d',      'irgadiag_3m_d', 'ldiag_3m_d',
    'u_3m_ue',	'v_3m_ue',	'w_3m_ue',	'h2o_3m_ue',	'tc_3m_ue',     'irgadiag_3m_ue', 'ldiag_3m_ue',
    'u_3m_uw',	'v_3m_uw',	'w_3m_uw',	'h2o_3m_uw',	'tc_3m_uw',     'irgadiag_3m_uw', 'ldiag_3m_uw',
    'u_5m_c',	'v_5m_c',	'w_5m_c',	'h2o_5m_c',		'tc_5m_c',      'irgadiag_5m_c', 'ldiag_5m_c',
    'u_10m_c',	'v_10m_c',	'w_10m_c',	'h2o_10m_c',	'tc_10m_c',     'irgadiag_10m_c', 'ldiag_10m_c',
    'u_10m_d',	'v_10m_d',	'w_10m_d',	'h2o_10m_d',	'tc_10m_d',     'irgadiag_10m_d', 'ldiag_10m_d',
    'u_10m_ue',	'v_10m_ue',	'w_10m_ue',	'h2o_10m_ue',	'tc_10m_ue',    'irgadiag_10m_ue', 'ldiag_10m_ue',
    'u_10m_uw',	'v_10m_uw',	'w_10m_uw',	'h2o_10m_uw',	'tc_10m_uw',    'irgadiag_10m_uw', 'ldiag_10m_uw',
    'u_15m_c',	'v_15m_c',	'w_15m_c',	'h2o_15m_c',	'tc_15m_c',     'irgadiag_15m_c', 'ldiag_15m_c',
    'u_20m_c',	'v_20m_c',	'w_20m_c',	'h2o_20m_c',	'tc_20m_c',     'irgadiag_20m_c', 'ldiag_20m_c',
]
c_1m_vars = ['u_1m_c',	'v_1m_c',	'w_1m_c',	'h2o_1m_c',		'tc_1m_c', 'irgadiag_1m_c', 'ldiag_1m_c']
d_1m_vars = ['u_1m_d',	'v_1m_d',	'w_1m_d',	'h2o_1m_d',		'tc_1m_d', 'irgadiag_1m_d', 'ldiag_1m_d']
ue_1m_vars = ['u_1m_ue',	'v_1m_ue',	'w_1m_ue',	'h2o_1m_ue',	'tc_1m_ue', 'irgadiag_1m_ue', 'ldiag_1m_ue']
uw_1m_vars = ['u_1m_uw',	'v_1m_uw',	'w_1m_uw',	'h2o_1m_uw',	'tc_1m_uw', 'irgadiag_1m_uw', 'ldiag_1m_uw']


def subset_variables(ds):
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
    return ds

def create_timestamp(ds):
    # To use the datum, its necessary to combine 3 columns of data from the dataset to get the full timestamp. 
    # This is demonstrated below. The 'time' column actually only incudes the second and minute information. For all datapoints, the hour according to the 'time' column is 1.  
    # The 'base_time' column indicates the hour of the day. The 'sample' column indicates the 20hz sample number. 
    time_alt = pd.to_datetime(pd.DataFrame({
        'year':         np.repeat(ds.time.dt.year, 20),
        'month':        np.repeat(ds.time.dt.month, 20),
        'day':          np.repeat(ds.time.dt.day, 20),
        'hour' :        np.tile(ds.base_time.dt.hour, 60*60*20),
        'minute' :      np.repeat(ds.time.dt.minute, 20),
        'second' :      np.repeat(ds.time.dt.second, 20),
        'microsecond' : (np.tile(ds.sample, 60*60) * (1e6/20)).astype(int),
    }))
    ds = ds.rename({'time': 'time_incomplete'})
    ds = ds.assign_coords(time_incomplete=ds.time_incomplete)
    ds = ds.stack(time = ('time_incomplete', 'sample'))
    ds = ds.drop_vars(['time', 'time_incomplete', 'sample'])
    ds = ds.assign_coords(time = time_alt)
    return ds

def despike(ds, height, tower):
    # Despiking using multiples of medians
    def block_median(timeseries, window):
        return timeseries.groupby(pd.Grouper(freq=window)).transform('median')
    def filter_spike(timeseries, q = FILTERING_q, window='30min'):
        mad = block_median(np.abs(timeseries - block_median(timeseries, window=window)), window=window)
        upper_bound = block_median(timeseries, window=window) + q*mad / 0.6745
        lower_bound = block_median(timeseries, window=window) - q*mad / 0.6745
        is_valid = (timeseries > lower_bound) & (timeseries < upper_bound)
        return timeseries.where(is_valid)

    this_df = ds.to_dataframe()

    this_df = pd.DataFrame(
            filter_spike(
                this_df[f'h2o_{height}m_{tower}'].where(this_df[f'irgadiag_{height}m_{tower}'] == 0)
            )
        ).join(
            filter_spike(
                this_df[f'w_{height}m_{tower}'].where(this_df[f'ldiag_{height}m_{tower}'] == 0)
            )
        )
    ds = ds.update(
        this_df[[f'h2o_{height}m_{tower}', f'w_{height}m_{tower}']].to_xarray()
    )
    return ds

def wind_direction(u, v):
    # From: https://www.eol.ucar.edu/content/wind-direction-quick-reference
    dir = 270 - np.rad2deg(np.arctan2(v,u))
    if dir > 360:
        dir = dir - 360
    return dir

def create_re_avg_ds(ds, var1,  var2, covariance_name):
    # Function to do Reynolds Averaging
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

def process_hourly_file(input_file, output_file):
    # open files, filter by variables, checking if the 1m variables are in the dataset
    ds = xr.open_dataset(input_file)
    ds = subset_variables(ds)
    ds = create_timestamp(ds)

    assert len(pd.Series(ds.time.dt.month).unique()) == 1
    MONTH = ds.time.dt.month.values[0]
    
    # Iterate over all height-towers, calculating the corrected variables
    df_list = []
    for height, tower in [
        (1,  'c'),  
        (2,  'c'),  
        (3,  'c'), 
        (5,  'c'), 
        (10, 'c'), (15, 'c'), (20, 'c'),
        (1,  'uw'), 
        (3,  'uw'), (10, 'uw'),
        (1,  'ue'), 
        (3,  'ue'), (10, 'ue'),
        (1,  'd'),  
        (3,  'd'),  (10, 'd')
    ]:  
        # assign convenience variables for the height-tower being operated on
        U_VAR, V_VAR, W_VAR = (f"u_{height}m_{tower}", f"v_{height}m_{tower}", f"w_{height}m_{tower}")

        # Make sure that height-tower is in this dataset (this time period may not have those measurements)
        if U_VAR in ds:
            if DESPIKE:
                ds = despike(ds, height, tower)
            # isolate the variables we want to operate on
            local_df = ds[[U_VAR, V_VAR, W_VAR]].to_dataframe()
            # calculate the 30-minute averaged angles for double rotation
            local_df['theta'] = local_df.groupby(pd.Grouper(freq='30min')).transform('mean').apply(
                lambda row: np.arctan2(row[V_VAR], row[U_VAR]), 
                axis=1
            )
            local_df[U_VAR + '_rotated1'] = local_df[U_VAR]*np.cos(local_df['theta']) + local_df[V_VAR]*np.sin(local_df['theta'])
            local_df[V_VAR + '_rotated1'] = -local_df[U_VAR]*np.sin(local_df['theta']) + local_df[V_VAR]*np.cos(local_df['theta'])
            local_df[W_VAR + '_rotated1'] = local_df[W_VAR]

            local_df['phi'] = local_df.groupby(pd.Grouper(freq='30min')).transform('mean').apply(
                lambda row: np.arctan2(row[W_VAR + '_rotated1'], row[U_VAR + '_rotated1']), 
                axis=1
            )
            local_df[U_VAR + '_rotated2'] = local_df[U_VAR + '_rotated1']*np.cos(local_df['phi']) + local_df[W_VAR + '_rotated1']*np.sin(local_df['phi'])
            local_df[V_VAR + '_rotated2'] = local_df[V_VAR + '_rotated1']
            local_df[W_VAR + '_rotated2'] = -local_df[U_VAR + '_rotated1']*np.sin(local_df['phi']) + local_df[W_VAR + '_rotated1']*np.cos(local_df['phi'])

            # add the fitted <u,v,w> values to the original xarray dataset
            ds[f'u_{height}m_{tower}_fit'] = ('time', local_df[U_VAR + '_rotated2'])
            ds[f'v_{height}m_{tower}_fit'] = ('time', local_df[V_VAR + '_rotated2'])
            ds[f'w_{height}m_{tower}_fit'] = ('time', local_df[W_VAR + '_rotated2'])

            # Calculate un-fitted Reynolds averaged variables
            ds_plain_w_h2o =create_re_avg_ds(ds, f'w_{height}m_{tower}', f'h2o_{height}m_{tower}',  f'w_h2o__{height}m_{tower}')[[
                f'u_{height}m_{tower}',
                f'v_{height}m_{tower}',
                f'w_{height}m_{tower}',
                f'w_h2o__{height}m_{tower}'
            ]]
            ds_plain_u_h2o =create_re_avg_ds(ds, f'u_{height}m_{tower}', f'h2o_{height}m_{tower}', f'u_h2o__{height}m_{tower}' )[f'u_h2o__{height}m_{tower}']
            ds_plain_v_h2o =create_re_avg_ds(ds, f'v_{height}m_{tower}', f'h2o_{height}m_{tower}', f'v_h2o__{height}m_{tower}' )[f'v_h2o__{height}m_{tower}']
            ds_plain_w_w =  create_re_avg_ds(ds, f'w_{height}m_{tower}', f'w_{height}m_{tower}',   f'w_w__{height}m_{tower}'   )[f'w_w__{height}m_{tower}']
            ds_plain_u_u =  create_re_avg_ds(ds, f'u_{height}m_{tower}', f'u_{height}m_{tower}',   f'u_u__{height}m_{tower}'   )[f'u_u__{height}m_{tower}']
            ds_plain_v_v =  create_re_avg_ds(ds, f'v_{height}m_{tower}', f'v_{height}m_{tower}',   f'v_v__{height}m_{tower}'   )[f'v_v__{height}m_{tower}']
            ds_plain_u_w =  create_re_avg_ds(ds, f'u_{height}m_{tower}', f'w_{height}m_{tower}',   f'u_w__{height}m_{tower}'   )[f'u_w__{height}m_{tower}']
            ds_plain_v_w =  create_re_avg_ds(ds, f'v_{height}m_{tower}', f'w_{height}m_{tower}',   f'v_w__{height}m_{tower}'   )[f'v_w__{height}m_{tower}']
            ds_plain_u_tc = create_re_avg_ds(ds, f'u_{height}m_{tower}', f'tc_{height}m_{tower}',  f'u_tc__{height}m_{tower}'  )[f'u_tc__{height}m_{tower}']
            ds_plain_v_tc = create_re_avg_ds(ds, f'v_{height}m_{tower}', f'tc_{height}m_{tower}',  f'v_tc__{height}m_{tower}'  )[f'v_tc__{height}m_{tower}']
            ds_plain_w_tc = create_re_avg_ds(ds, f'w_{height}m_{tower}', f'tc_{height}m_{tower}',  f'w_tc__{height}m_{tower}'  )[f'w_tc__{height}m_{tower}']

            # Calculate fitted Reynolds averaged variables
            ds_fit_w_h2o =create_re_avg_ds(ds,  f'w_{height}m_{tower}_fit', f'h2o_{height}m_{tower}', f'w_h2o__{height}m_{tower}_fit')[[
                f'u_{height}m_{tower}_fit',
                f'v_{height}m_{tower}_fit',
                f'w_{height}m_{tower}_fit',
                f'w_h2o__{height}m_{tower}_fit'
            ]]
            ds_fit_u_h2o =create_re_avg_ds(ds, f'u_{height}m_{tower}_fit', f'h2o_{height}m_{tower}',   f'u_h2o__{height}m_{tower}_fit')[f'u_h2o__{height}m_{tower}_fit']
            ds_fit_v_h2o =create_re_avg_ds(ds, f'v_{height}m_{tower}_fit', f'h2o_{height}m_{tower}',   f'v_h2o__{height}m_{tower}_fit')[f'v_h2o__{height}m_{tower}_fit']
            ds_fit_w_w =  create_re_avg_ds(ds, f'w_{height}m_{tower}_fit', f'w_{height}m_{tower}_fit', f'w_w__{height}m_{tower}_fit'  )[f'w_w__{height}m_{tower}_fit']
            ds_fit_u_u =  create_re_avg_ds(ds, f'u_{height}m_{tower}_fit', f'u_{height}m_{tower}_fit', f'u_u__{height}m_{tower}_fit'  )[f'u_u__{height}m_{tower}_fit']
            ds_fit_v_v =  create_re_avg_ds(ds, f'v_{height}m_{tower}_fit', f'v_{height}m_{tower}_fit', f'v_v__{height}m_{tower}_fit'  )[f'v_v__{height}m_{tower}_fit']
            ds_fit_u_w =  create_re_avg_ds(ds, f'u_{height}m_{tower}_fit', f'w_{height}m_{tower}_fit', f'u_w__{height}m_{tower}_fit'  )[f'u_w__{height}m_{tower}_fit']
            ds_fit_v_w =  create_re_avg_ds(ds, f'v_{height}m_{tower}_fit', f'w_{height}m_{tower}_fit', f'v_w__{height}m_{tower}_fit'  )[f'v_w__{height}m_{tower}_fit']
            ds_fit_u_tc = create_re_avg_ds(ds, f'u_{height}m_{tower}_fit', f'tc_{height}m_{tower}',    f'u_tc__{height}m_{tower}_fit' )[f'u_tc__{height}m_{tower}_fit']
            ds_fit_v_tc = create_re_avg_ds(ds, f'v_{height}m_{tower}_fit', f'tc_{height}m_{tower}',    f'v_tc__{height}m_{tower}_fit' )[f'v_tc__{height}m_{tower}_fit']
            ds_fit_w_tc = create_re_avg_ds(ds, f'w_{height}m_{tower}_fit', f'tc_{height}m_{tower}',    f'w_tc__{height}m_{tower}_fit' )[f'w_tc__{height}m_{tower}_fit']

            df_plain = ds_plain_w_h2o.join(ds_plain_u_h2o).join(ds_plain_v_h2o).join(ds_plain_w_w).join(ds_plain_u_u).join(
                ds_plain_v_v).join(ds_plain_u_w).join(ds_plain_v_w).join(ds_plain_u_tc).join(ds_plain_v_tc).join(ds_plain_w_tc)

            df_fit = ds_fit_w_h2o.join(ds_fit_u_h2o).join(ds_fit_v_h2o).join(ds_fit_w_w).join(ds_fit_u_u).join(
                ds_fit_v_v).join(ds_fit_u_w).join(ds_fit_v_w).join(ds_fit_u_tc).join(ds_fit_v_tc).join(ds_fit_w_tc)

            merged_df = df_plain.join(df_fit)
            df_list.append(merged_df)

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
        input_file = file_list[i]
        output_file = os.path.join(OUTPUT_PATH, input_file.split('/')[-1][27:]) + '.parquet'
        try:
            process_hourly_file(input_file, output_file)
        except Exception as exc:
            print('Exception caught while processing file {} : {}'.format(input_file, exc))
            traceback.print_exc()
            
    # slow
    # saved_files_list = [print_and_process(i) for i in tqdm(list(range(0, n_files)))]
        
    # fast
    with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLELISM) as executor:
        saved_files_list = list(tqdm(executor.map(print_and_process, range(0, n_files)), total=n_files))
    print("Finished processing")