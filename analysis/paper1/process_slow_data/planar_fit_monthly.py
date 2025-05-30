"""
Produce a file of planar fits for each month
"""
# Adapted from https://www.eol.ucar.edu/content/sonic-tilt-corrections
import sys
sys.path.append('/home/elilouis/sublimationofsnow')
from sublimpy import extrautils
import xarray as xr
import numpy as np
import os

import altair as alt
import datetime, calendar
from sublimpy import utils

sos_files_dir = '/storage/elilouis/sublimationofsnow/sosqc/sos_isfs_qc_geo_tiltcor_5min_v20241227/'
OUTPUT_FILE = '/storage/elilouis/sublimationofsnow/monthly_planar_fits_10sectors.csv'
wind_dir_bins = np.arange(0, 396, 36)


with open(OUTPUT_FILE, "w") as file:
    variable_sets = [
        ('u_1m_c', 'v_1m_c', 'w_1m_c'),
        ('u_2m_c', 'v_2m_c', 'w_2m_c'),
        ('u_3m_c',   'v_3m_c',   'w_3m_c'),
        ('u_5m_c',   'v_5m_c',   'w_5m_c'),
        ('u_10m_c',   'v_10m_c',   'w_10m_c'),
        ('u_15m_c',   'v_15m_c',   'w_15m_c'),
        ('u_20m_c',   'v_20m_c',   'w_20m_c'),
        
        ('u_1m_uw', 'v_1m_uw', 'w_1m_uw'),
        ('u_2m_uw', 'v_2m_uw', 'w_2m_uw'),
        ('u_2_5m_uw', 'v_2_5m_uw', 'w_2_5m_uw'),
        ('u_3m_uw',   'v_3m_uw',   'w_3m_uw'),
        ('u_10m_uw',   'v_10m_uw',   'w_10m_uw'),
        
        ('u_1m_ue', 'v_1m_ue', 'w_1m_ue'),
        ('u_2m_ue', 'v_2m_ue', 'w_2m_ue'),
        ('u_3m_ue',   'v_3m_ue',   'w_3m_ue'),
        ('u_10m_ue',   'v_10m_ue',   'w_10m_ue'),

        ('u_1m_d', 'v_1m_d', 'w_1m_d'),
        ('u_2m_d', 'v_2m_d', 'w_2m_d'),
        ('u_3m_d',   'v_3m_d',   'w_3m_d'),
        ('u_10m_d',   'v_10m_d',   'w_10m_d'),
    ]
    VARIABLE_NAMES = list(np.array(variable_sets).flatten()) + ['dir_3m_c']

    DATE_FORMAT_STR = '%Y%m%d'

    file.write("month height tower bin_low bin_high a b c tilt tiltaz W_f_1 W_f_2 W_f_3\n")

    for month,year in [
        (11,2022),
        (12,2022),
        (1,2023),
        (2,2023),
        (3,2023),
        (4,2023),
        (5,2023),
        (6,2023),
    ]:
        print(f"Processing month: {month}/{year}")
        print(f"=================================")
        num_days = calendar.monthrange(year, month)[1]
        datelist = [datetime.date(year, month, day).strftime(DATE_FORMAT_STR) for day in range(1, num_days+1)]
        print("Opening data")
        datasets = []
        for date in datelist:
            try: 
                ds = xr.open_dataset(os.path.join(
                    sos_files_dir,
                    f"isfs_sos_qc_geo_tiltcor_5min_v2_{date}.nc"
                ))
                # this ensures we don't access variables that aren't in this dataset, which would throw an error
                ds_new = ds[set(ds.data_vars).intersection(VARIABLE_NAMES)]
                datasets.append(ds_new)
            except:
                print(f"Failed to open file for date {date}")

        sos_ds = xr.concat(datasets, dim='time')
        sos_ds = utils.fill_missing_timestamps(sos_ds)

        for bin_low, bin_high in list(zip(wind_dir_bins, wind_dir_bins[1:])):
            wind_dir = sos_ds[['dir_3m_c']].to_dataframe()
            wind_dir = wind_dir[wind_dir['dir_3m_c'] > bin_low][wind_dir['dir_3m_c'] <= bin_high]
            this_winddir_bin_ds = sos_ds.sel(time = wind_dir.index)
            for u_VAR, v_VAR, w_VAR in variable_sets:
                if (
                    (u_VAR in this_winddir_bin_ds) and (v_VAR in this_winddir_bin_ds) and (w_VAR in this_winddir_bin_ds)
                ) and (
                    len(this_winddir_bin_ds[u_VAR].dropna(dim='time'))  > 0 
                    and len(this_winddir_bin_ds[v_VAR].dropna(dim='time')) > 0 
                    and len(this_winddir_bin_ds[w_VAR].dropna(dim='time')) > 0
                ):
                    try:
                        (a,b,c), (tilt, tiltaz), W_f = extrautils.calculate_planar_fit(
                            this_winddir_bin_ds[u_VAR], 
                            this_winddir_bin_ds[v_VAR], 
                            this_winddir_bin_ds[w_VAR]
                        )
                    except ValueError:
                        print("why nee??")
                    try:
                        (u_streamwise, v_streamwise, w_streamwise) = extrautils.apply_planar_fit(
                            this_winddir_bin_ds[u_VAR], 
                            this_winddir_bin_ds[v_VAR], 
                            this_winddir_bin_ds[w_VAR], a, W_f
                        )
                    except ValueError:
                        print("why nee??")
                    [height, tower] = u_VAR[2:].split("m_")
                    file.write(f"{month} {height} {tower} {bin_low} {bin_high} {a} {b} {c} {np.rad2deg(tilt)} {np.rad2deg(tiltaz)} {W_f[0]} {W_f[1]} {W_f[2]}\n")