import numpy as np
import xarray as xr

import datetime as dt
import pandas as pd

import matplotlib.pyplot as plt

import altair as alt
alt.data_transformers.enable('json')

from sublimpy import utils
import glob
import pytz
from scipy.signal import welch, csd
from scipy.stats import chi2
from joblib import Parallel, delayed
from tqdm import tqdm
import os

PARALLELISM = 8
START_DATE = '20221101'
END_DATE = '20230619'

# start_i = 16 # this corresponds to hour 0900
# end_i = 24 # this corresponds to hour 1700
# output_dir = "/Users/elischwat/Development/data/sublimationofsnow/spectra/0900_1700"

start_i = 2 # this corresponds to hour 1900 of the night before
end_i = 12 # this corresponds to hour 0500
output_dir = "/Users/elischwat/Development/data/sublimationofsnow/spectra/1900_0500"

file_path_fast_data = "/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast/"
file_list = glob.glob(os.path.join(file_path_fast_data, "*.nc"))


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

def process_date(date): 
    # identify files we want from this date
    print(f"Processing date: {date}")
    local_file_list = [ f for f in file_list if f"_{date}" in f]
    local_file_list = sorted(local_file_list)[start_i:end_i]

    # open files and convert to DF
    ds = xr.open_mfdataset(
        local_file_list, concat_dim="time", 
        combine="nested", 
        data_vars=VARIABLES
    )
    df = ds[VARIABLES].to_dataframe()

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

    # Calculate covatiance spectra of uu, vv, ww
    spectrum_ls = []
    for tower in towers:
        if tower == 'c':
            heights = [2, 3, 5, 10, 15, 20]
        else:
            heights = [3, 10]
        for height in heights:
            for var in ['u', 'v', 'w']:
                spectrum = pd.DataFrame(dict(zip(
                    ['frequency', 'power spectrum'],
                    list(welch(
                            df[f"{var}_{height}m_c"],
                            fs=20, #Hz
                            window='hann', #'hann' is the default,
                            nperseg=72000 # one hour window
                    ))
                )))
                spectrum = spectrum.assign(height=height).assign(tower=tower)
                spectrum = spectrum.assign(variance = f"{var}'{var}'")
                spectrum_ls.append(spectrum)
    variance_spectrum_df = pd.concat(spectrum_ls)
        
    # Calculate momentum spectra - sqrt[(u'w')^2 + (v'w')^2]
    local_df_list = []
    
    for tower in towers:
        if tower == 'c':
            heights = [2, 3, 5, 10, 15, 20]
        else:
            heights = [3, 10]
        for height in heights:
            local_df = pd.DataFrame(dict(zip(
                ['frequency', 'power spectrum'],
                list(csd(
                        np.sqrt(
                            df[f'u_{height}m_{tower}']**2 + df[f'v_{height}m_{tower}']**2
                        ),
                        df[f'w_{height}m_{tower}'],
                        fs=20, #Hz
                        window='hann', #'hann' is the default,
                        nperseg=72000
                ))
            ))).assign(height=height).assign(tower=tower)
            # local_df['power spectrum'] = np.real(local_df['power spectrum'])
            local_df['cospectrum'] = local_df['power spectrum'].apply(lambda complex: complex.real)
            local_df['quadrature spectrum'] = local_df['power spectrum'].apply(lambda complex: complex.imag)
            local_df_list.append(local_df.drop(columns=['power spectrum']))
    momentum_copower_spectrum = pd.concat(local_df_list)

    # calculate sensible heat cospectra
    local_df_list = []
    for tower in towers:
        if tower == 'c':
            heights = [2, 3, 5, 10, 15, 20]
        else:
            heights = [3, 10]
        for height in heights:
            local_df = pd.DataFrame(dict(zip(
                ['frequency', 'power spectrum'],
                list(csd(
                        df[f'w_{height}m_{tower}'],
                        df[f'tc_{height}m_{tower}'],
                        fs=20, #Hz
                        window='hann', #'hann' is the default,
                        nperseg=72000
                ))
            ))).assign(height=height).assign(tower=tower)
            local_df['cospectrum'] = local_df['power spectrum'].apply(lambda complex: complex.real)
            local_df['quadrature spectrum'] = local_df['power spectrum'].apply(lambda complex: complex.imag)
            local_df_list.append(local_df.drop(columns=['power spectrum']))
    sensheat_copower_spectrum = pd.concat(local_df_list)

    # Calculate latent heat cospectra
    local_df_list = []
    for tower in towers:
        if tower == 'c':
            heights = [2, 3, 5, 10, 15, 20]
        else:
            heights = [3, 10]
        for height in heights:
            local_df = pd.DataFrame(dict(zip(
                ['frequency', 'power spectrum'],
                list(csd(
                        df[f'w_{height}m_{tower}'],
                        df[f'h2o_{height}m_{tower}'],
                        fs=20, #Hz
                        window='hann', #'hann' is the default,
                        nperseg=72000
                ))
            ))).assign(height=height).assign(tower=tower)
            local_df['cospectrum'] = local_df['power spectrum'].apply(lambda complex: complex.real)
            local_df['quadrature spectrum'] = local_df['power spectrum'].apply(lambda complex: complex.imag)
            local_df_list.append(local_df.drop(columns=['power spectrum']))
    latheat_copower_spectrum = pd.concat(local_df_list)

    variance_spectrum_spectrum_fn = os.path.join(output_dir, 'velocity', date + '.parquet')
    variance_spectrum_df = variance_spectrum_df.assign(date = date)
    variance_spectrum_df.to_parquet(variance_spectrum_spectrum_fn)

    momentum_copower_spectrum_fn = os.path.join(output_dir, 'momentum', date + '.parquet')
    momentum_copower_spectrum = momentum_copower_spectrum.assign(date = date)
    momentum_copower_spectrum.to_parquet(momentum_copower_spectrum_fn)

    sensheat_copower_spectrum_fn = os.path.join(output_dir, 'sensible_heat', date + '.parquet')
    sensheat_copower_spectrum = sensheat_copower_spectrum.assign(date = date)
    sensheat_copower_spectrum.to_parquet(sensheat_copower_spectrum_fn)

    latheat_copower_spectrum_fn = os.path.join(output_dir, 'latent_heat', date + '.parquet')
    latheat_copower_spectrum = latheat_copower_spectrum.assign(date = date)
    latheat_copower_spectrum.to_parquet(latheat_copower_spectrum_fn)

    return (
        variance_spectrum_spectrum_fn,
        momentum_copower_spectrum_fn,
        sensheat_copower_spectrum_fn,
        latheat_copower_spectrum_fn
    )


if __name__ == '__main__':
    dates_tqdm = tqdm(datelist)

    print(f"Beginning processing (parallelism = {PARALLELISM})")
    processed_results =  Parallel(n_jobs = PARALLELISM)(
        delayed(process_date)(date) for date in dates_tqdm
    )
    