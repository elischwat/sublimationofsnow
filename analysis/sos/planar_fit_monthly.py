

# Adapted from https://www.eol.ucar.edu/content/sonic-tilt-corrections
import sys
sys.path.append('/home/elilouis/sublimationofsnow')
import sosutils
import xarray as xr
import numpy as np

import altair as alt
import datetime, calendar

# # Download multiple daily datasets and combine

# +
OUTPUT_FILE = '/data2/elilouis/sublimationofsnow/monthly_planar_fits.csv'

with open(OUTPUT_FILE, "w") as file:
    variable_sets = [
        ('u_1m_c', 'v_1m_c', 'w_1m_c'),
        ('u_3m_c',   'v_3m_c',   'w_3m_c'),
        ('u_10m_c',   'v_10m_c',   'w_10m_c'),
        ('u_20m_c',   'v_20m_c',   'w_20m_c'),
        
        ('u_1m_uw', 'v_1m_uw', 'w_1m_uw'),
        ('u_3m_uw',   'v_3m_uw',   'w_3m_uw'),
        ('u_10m_uw',   'v_10m_uw',   'w_10m_uw'),
        
        ('u_1m_ue', 'v_1m_ue', 'w_1m_ue'),
        ('u_3m_ue',   'v_3m_ue',   'w_3m_ue'),
        ('u_10m_ue',   'v_10m_ue',   'w_10m_ue'),

        ('u_1m_d', 'v_1m_d', 'w_1m_d'),
        ('u_3m_d',   'v_3m_d',   'w_3m_d'),
        ('u_10m_d',   'v_10m_d',   'w_10m_d'),
    ]
    VARIABLE_NAMES = list(np.array(variable_sets).flatten())

    sos_download_dir='/data2/elilouis/sublimationofsnow/sosnoqc'
    DATE_FORMAT_STR = '%Y%m%d'

    file.write("month height tower a b c tilt tiltaz\n")

    for month,year in [
        (11,2022),
        (12,2022),
        (1,2023),
    ]:
        print(f"Processing month: {month}/{year}")
        print(f"    with vars: {VARIABLE_NAMES}")
        print(f"=================================")
        num_days = calendar.monthrange(year, month)[1]
        datelist = [datetime.date(year, month, day).strftime(DATE_FORMAT_STR) for day in range(1, num_days+1)]
        print("Downloading data")
        # datasets = [xr.open_dataset(sosutils.download_sos_data_day(date, sos_download_dir))[VARIABLE_NAMES] for date in datelist]
        datasets = []
        for date in datelist:
            ds = xr.open_dataset(sosutils.download_sos_data_day(date, sos_download_dir))
            if all([var in ds.data_vars for var in VARIABLE_NAMES]):
                ds = ds[VARIABLE_NAMES]
                datasets += [ds]
            else:
                print(f"Skipping date: {date}")
        sos_ds = sosutils.merge_datasets_with_different_variables(datasets, dim='time')

        for u_VAR, v_VAR, w_VAR in variable_sets:
            (a,b,c), (tilt, tiltaz), W_f = sosutils.calculate_planar_fit(sos_ds[u_VAR], sos_ds[v_VAR], sos_ds[w_VAR])
            (u_streamwise, v_streamwise, w_streamwise) = sosutils.apply_planar_fit(sos_ds[u_VAR], sos_ds[v_VAR], sos_ds[w_VAR], a, W_f)
            [height, tower] = u_VAR[2:].split("m_")
            file.write(f"{month} {height} {tower} {a} {b} {c} {np.rad2deg(tilt)} {np.rad2deg(tiltaz)}\n")