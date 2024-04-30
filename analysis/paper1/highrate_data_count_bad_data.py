# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: arm
#     language: python
#     name: python3
# ---

# Iterate over all fast data, aggregate to 5 minute blocks, and count number of instantaneous measurements that had a flag (all types of flags).

# +
import numpy as np
import xarray as xr

import datetime as dt
import glob
import pandas as pd

import matplotlib.pyplot as plt
import os
import altair as alt
alt.data_transformers.enable('json')

from sublimpy import utils
from sublimpy import tidy
from sublimpy import extrautils
import glob
from joblib import Parallel, delayed

PARALLELISM = 8
# +
base_vars = [
    'base_time'
]

diagbits_vars = [
    'diagbits_10m_c',	
    'diagbits_10m_d',	
    'diagbits_10m_ue',	
    'diagbits_10m_uw',	
    'diagbits_15m_c',	
    'diagbits_1m_c',	
    'diagbits_1m_d',	
    'diagbits_1m_ue',	
    'diagbits_1m_uw',	
    'diagbits_20m_c',	
    'diagbits_2m_c',	
    'diagbits_3m_c',	
    'diagbits_3m_d',	
    'diagbits_3m_ue',	
    'diagbits_3m_uw',	
    'diagbits_5m_c',	
]    

irga_vars = [
    'irgadiag_10m_c',	
    'irgadiag_10m_d',	
    'irgadiag_10m_ue',	
    'irgadiag_10m_uw',	
    'irgadiag_15m_c',	
    'irgadiag_1m_c',	
    'irgadiag_1m_d',	
    'irgadiag_1m_ue',	
    'irgadiag_1m_uw',	
    'irgadiag_20m_c',	
    'irgadiag_2m_c',	
    'irgadiag_3m_c',	
    'irgadiag_3m_d',	
    'irgadiag_3m_ue',	
    'irgadiag_3m_uw',	
    'irgadiag_5m_c',	
]

ldiag_vars = [
    'ldiag_10m_c',
    'ldiag_10m_d',
    'ldiag_10m_ue',
    'ldiag_10m_uw',
    'ldiag_15m_c',
    'ldiag_1m_c',
    'ldiag_1m_d',
    'ldiag_1m_ue',
    'ldiag_1m_uw',
    'ldiag_20m_c',
    'ldiag_2m_c',
    'ldiag_3m_c',
    'ldiag_3m_d',
    'ldiag_3m_ue',
    'ldiag_3m_uw',
    'ldiag_5m_c',
]

all_vars = base_vars + diagbits_vars + irga_vars + ldiag_vars
path_to_hourly_nc_files = "/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast/"
all_diag_vars = diagbits_vars + irga_vars + ldiag_vars

def process_file(nc_file, output_file):
    print(f"Processing {nc_file} and saving to {output_file}")
    ds = xr.open_dataset(nc_file)
    all_vars_in_this_ds = [v for v in all_vars if v in ds]
    ds = ds[all_vars_in_this_ds]
    
    # create timestamp that makes sense
    df1 = pd.DataFrame({'time': np.unique(ds['time'])})
    df2 = pd.DataFrame({'base_time': np.unique(ds['base_time'])})
    df3 = pd.DataFrame({'sample': np.unique(ds['sample'])})
    (
        alt.Chart(df3).mark_tick(thickness=5).encode(
            alt.X("sample:Q").title(
                f'sample (n = {len(df3)})'
            )
        ).properties(width=600) & 

        alt.Chart(df1).mark_tick(thickness=1).encode(
            alt.X("time:T").axis(
                format='%H%M%p'
            ).title(
                f'time (n = {len(df1)})'
            )
        ).properties(width=600) & 

        alt.Chart(df2).mark_tick(thickness=5).encode(
            alt.X("base_time:T").title(
                f'base_time (n = {len(df2)})'
            )
        ).properties(width=600)
    )
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
    df = ds.to_dataframe().drop(columns=['base_time','sample'])
    df_counts = ( df>0 ).groupby(pd.Grouper(freq='5Min')).sum()
    df_counts.index = df_counts.index + dt.timedelta(minutes=2, seconds=30)
    df_counts.to_parquet(output_file)
    return output_file

if __name__ == '__main__':
    file_list = glob.glob(os.path.join(path_to_hourly_nc_files, "*.nc"))
    output_file_list = [
        f.replace('/sosqc_fast/', '/sosqc_fast_flagcounts/').replace(".nc", '.parquet')
        for f in file_list
    ]

    print(f"Beginning processing (parallelism = {PARALLELISM})")
    processed_results =  Parallel(n_jobs = PARALLELISM)(
        delayed(process_file)(in_file, out_file) for in_file, out_file in zip(file_list, output_file_list)
    )
    