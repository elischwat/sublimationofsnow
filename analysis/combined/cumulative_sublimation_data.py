# %%
import pandas as pd
import act
import glob
import numpy as np

import altair as alt
alt.data_transformers.enable('json')

from sublimpy import tidy
import datetime as dt
import os

start_date = '20221130'
end_date = '20230509'
start_date_lastseason = '20211101'
end_date_lastseason = '20220601'
ecor_gothic = 'guc30ecorM1.b1'
ecor_kp = 'guc30ecorS3.b1'
output_dir = '/data2/elilouis/sublimationofsnow/'

username = os.getenv("ARM_USERNAME")
token = os.getenv("ARM_TOKEN")

kp_sail_ecor_download_dir = os.path.join(output_dir, ecor_kp)
gothic_sail_ecor_download_dir = os.path.join(output_dir, ecor_gothic)

# %%
act.discovery.download_data(
    username,    token,    ecor_gothic,    
    start_date_lastseason,
    end_date,
    output = os.path.join(output_dir, ecor_gothic)
)
act.discovery.download_data(
    username,    token,    ecor_kp,    
    start_date_lastseason,
    end_date,
    output = os.path.join(output_dir, ecor_kp)
)

# %%
ecor_gothic_ds = act.io.armfiles.read_netcdf(
    glob.glob(os.path.join(output_dir, ecor_gothic, '*.cdf'))
)
ecor_gothic_ds_thisseason = ecor_gothic_ds.sel(time = slice(start_date, end_date))
ecor_gothic_ds_lastseason = ecor_gothic_ds.sel(time = slice(start_date_lastseason, end_date_lastseason))

# %%
ecor_kps_ds = act.io.armfiles.read_netcdf(
    glob.glob(os.path.join(output_dir, ecor_kp, '*.cdf'))
)
ecor_kps_ds_thisseason = ecor_kps_ds.sel(time = slice(start_date, end_date))
ecor_kps_ds_lastseason = ecor_kps_ds.sel(time = slice(start_date_lastseason, end_date_lastseason))

# %%
ecor_gothic_ds_thisseason.to_netcdf('cumulative_sublimation-ecor_gothic_ds_thisseason.cdf')
ecor_kps_ds_thisseason.to_netcdf('cumulative_sublimation-ecor_kps_ds_thisseason.cdf')
ecor_gothic_ds_lastseason.to_netcdf('cumulative_sublimation-ecor_gothic_ds_lastseason.cdf')
ecor_kps_ds_lastseason.to_netcdf('cumulative_sublimation-ecor_kps_ds_lastseason.cdf')