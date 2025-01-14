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
OUTPUT_PATH = f"{DATA_DIR}sublimationofsnow/planar_fit_10sector_processed_30min_despiked_q7/"
DESPIKE = True
FILTERING_q = 7
SECTORS = True
# n cores utilized by application
PARALLELISM = 20
# Reynolds averaging length, in units (1/20) seconds
SAMPLES_PER_AVERAGING_LENGTH = 30*60*20

# # Open fast data
file_list = sorted(glob.glob(f"{DATA_DIR}sublimationofsnow/sosqc_fast/*.nc"))
file_list = [f for f in file_list if '202210' not in f]
wind_dir_bins = np.arange(0, 396, 36)
    
# Open planar fit data
if SECTORS:
    monthly_file = f"{DATA_DIR}sublimationofsnow/monthly_planar_fits_10sectors.csv"
else:
    monthly_file = f"{DATA_DIR}sublimationofsnow/monthly_planar_fits.csv"
fits_df = pd.read_csv(monthly_file, delim_whitespace=True)

# Transform planar fit data
fits_df['height'] = fits_df['height'].str.replace('_', '.').astype('float')
fits_df['W_f'] = fits_df.apply(
    lambda row: [row['W_f_1'], row['W_f_2'], row['W_f_3']],
    axis=1
).drop(columns=['W_f_1', 'W_f_2', 'W_f_3'])
if SECTORS:
    fits_df = fits_df.set_index(['month', 'height', 'tower', 'bin_low', 'bin_high'])
else:
    fits_df = fits_df.set_index(['month', 'height', 'tower'])
fits_df = fits_df.sort_index()


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
        # and make sure that there is a fit available for this time period and height-tower.
        # Note that, if we are doing sector planar fitting, we need to remove some extra index values for this check
        if SECTORS: 
            index_drop_level = [3,4]
        else:
            index_drop_level = []
        if U_VAR in ds and (MONTH, height, tower) in fits_df.index.droplevel(index_drop_level):
            if DESPIKE:
                ds = despike(ds, height, tower)
            # isolate the variables we want to operate on
            local_df = ds[[U_VAR, V_VAR, W_VAR]].to_dataframe()
            # Only process data if there are non-nan values
            if len(local_df) > 0:
                if SECTORS:
                    # calculate the 30-minute averaged wind direction
                    local_df['wind_direction_block_mean'] = local_df.groupby(pd.Grouper(freq='30min')).transform('mean').apply(
                        lambda row: wind_direction(row[U_VAR], row[V_VAR]), axis=1
                    )
                    # group the wind directions into discrete bins, we use the lower bound to identify each bin
                    local_df['wind_direction_block_mean_bin_low'] = pd.cut(
                        local_df['wind_direction_block_mean'],
                        wind_dir_bins,
                        labels = wind_dir_bins[:-1] 
                    )
                    # merge the planar fit parameters into the data, so each row gets the appropriate parameters
                    # WE drop nans here because if U/V/W are nan, then wind_direction_block_mean and wind_direction_block_mean_bin_low
                    # are nan, and then we try to merge using nan as a key, which fails
                    local_df = local_df.dropna()

                    if len(local_df) > 0:
                        local_df = local_df.reset_index().merge(
                            fits_df.loc[MONTH, height, tower][['a', 'W_f']],
                            left_on = 'wind_direction_block_mean_bin_low',
                            right_on = 'bin_low'
                        ).set_index('time')
                        # group by the wind speed bin and apply the planar fit for each subset of data
                        result = local_df.groupby('wind_direction_block_mean_bin_low').apply(
                            lambda df: (
                                df.index, 
                                extrautils.apply_planar_fit(df[U_VAR], df[V_VAR], df[W_VAR], df['a'].values[0], df['W_f'].values[0])
                            )
                        )
                        # Wrangle the results from the groupby-apply
                        new_values_df = pd.DataFrame()
                        for key, results in result:
                            new_values_df = pd.concat([
                                new_values_df,
                                pd.DataFrame({
                                    'time': key,
                                    'u':    results[0],
                                    'v':    results[1],
                                    'w':    results[2],
                                }).set_index('time')
                            ])
                        new_values_df = new_values_df.sort_index()
                else: 
                    new_u, new_v, new_w = extrautils.apply_planar_fit(
                        local_df[U_VAR], 
                        local_df[V_VAR], 
                        local_df[W_VAR], 
                        fits_df.loc[MONTH, height, tower]['a'], 
                        fits_df.loc[MONTH, height, tower]['W_f'], 
                    )
                    new_values_df = pd.DataFrame({
                        'time': local_df.index,
                        'u':    new_u,
                        'v':    new_v,
                        'w':    new_w,
                    }).set_index('time')
                
                if 'new_values_df' in locals():
                    # add the fitted <u,v,w> values to the original xarray dataset
                    ds[f'u_{height}m_{tower}_fit'] =    new_values_df['u'].to_xarray()
                    ds[f'v_{height}m_{tower}_fit'] =    new_values_df['v'].to_xarray()
                    ds[f'w_{height}m_{tower}_fit'] =    new_values_df['w'].to_xarray()

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

def output_file_from_input_file(input_file):
    return os.path.join(OUTPUT_PATH, input_file.split('/')[-1][27:]) + '.parquet'
if __name__ == '__main__':
    n_files = len(file_list)
    print(f"n files: {n_files}")

    output_files = [output_file_from_input_file(f) for f in file_list]
    file_list = [
        inf for inf, outf in zip(file_list, output_files) 
        if not os.path.exists(outf)
    ]
    n_files = len(file_list)
    print(f"processing subset of files: {n_files}")
    

    def print_and_process(i):
        input_file = file_list[i]
        output_file = output_file_from_input_file(input_file)
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