# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: arm
#     language: python
#     name: python3
# ---

# +
import act
from tempfile import TemporaryDirectory
import glob
import numpy as np
import pandas as pd
import datetime as dt
import altair as alt
pd.set_option('display.max_columns', 100)

import sys
sys.path.append("/home/elilouis/sublimationofsnow")
import os
import sosutils
import seaborn as sns
# -

# # Inputs

# +
username = os.getenv("ARM_USERNAME")
token = os.getenv("ARM_TOKEN")
DATE_FORMAT_STR = '%Y-%m-%d'
# start_date = "2022-01-01"
# end_date = "2022-12-31"
start_date = "2022-11-29"
end_date = "2023-05-10"

DL_DATA_STREAM = 'gucdlprofwind4newsM1.c1'
DL_DATA_STREAM_FILEEXT = '.nc'
DL_OUTPUT_DIR = os.path.join("/data2/elilouis/sublimationofsnow/", DL_DATA_STREAM)

SONDE_DATA_STREAM = 'gucsondewnpnM1.b1'
SONDE_DATA_STREAM_FILEEXT = '.cdf'
SONDE_OUTPUT_DIR = os.path.join("/data2/elilouis/sublimationofsnow/", SONDE_DATA_STREAM)
# -

# # Download DL data

act.discovery.download_data(username, token, DL_DATA_STREAM, start_date, end_date, output=DL_OUTPUT_DIR)

# # Download Sonde data

act.discovery.download_data(
            username, token, SONDE_DATA_STREAM, start_date, end_date, output = SONDE_OUTPUT_DIR)

# Open Sonde data

sonde_files = glob.glob(os.path.join(SONDE_OUTPUT_DIR,'*'+SONDE_DATA_STREAM_FILEEXT))
print(len(sonde_files))
sonde = act.io.armfiles.read_netcdf(sonde_files)

src_sonde = sonde.to_dataframe()



# # Save some synoptic wind information

# +
synoptic_winds_500 = src_sonde[np.abs(src_sonde['pres'] - 500) < 25]
synoptic_winds_500 = synoptic_winds_500[['deg', 'wspd']].groupby("time").median()
synoptic_winds_500_local = sosutils.modify_df_timezone(synoptic_winds_500.reset_index(), 'UTC', 'US/Mountain')
synoptic_winds_500_local.to_parquet("synoptic_winds_500_local.parquet")

synoptic_winds_700 = src_sonde[np.abs(src_sonde['pres'] - 700) < 25]
synoptic_winds_700 = synoptic_winds_700[['deg', 'wspd']].groupby("time").median()
synoptic_winds_700_local = sosutils.modify_df_timezone(synoptic_winds_700.reset_index(), 'UTC', 'US/Mountain')
synoptic_winds_700_local.to_parquet("synoptic_winds_500_local.parquet")