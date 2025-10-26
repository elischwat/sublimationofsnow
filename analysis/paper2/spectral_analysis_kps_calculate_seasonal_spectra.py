import glob
import os
import numpy as np
from process_fast_data.fast_data_calculate_spectra_nomrd import calculate_mrd_for_df, fast_data_files_to_dataframe
from sublimpy import utils
import pandas as pd

### INPUTS
DATA_DIR = "/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast/"
OUTPUT_DIR = ""
START_DATE = '20221201'
END_DATE = '20230501'

### PREP LIST OF DATES
dates = pd.date_range(START_DATE, END_DATE, freq='D')
date_pairs = list(zip(
    dates[:-1],
    dates[1:]
))

for d1, d2 in date_pairs:
    print(f'Processing {d1}')
    dates_subset = [d1, d2]
    files = sorted(np.array([
        glob.glob(os.path.join(
            DATA_DIR, 
            f"isfs_sos_qc_geo_tiltcor_hr_v2_{d.strftime('%Y%m%d')}**.nc"
        )) for d in dates_subset
    ]).flatten())[6:6+25]
    fast_df_sos_all_data = fast_data_files_to_dataframe(
        files,
        rotation='none'
    )
    fast_df_sos_all_data = utils.modify_df_timezone(fast_df_sos_all_data, 'UTC', 'US/Mountain')
    fast_df_sos_all_data = fast_df_sos_all_data.set_index('time').loc[d1.strftime('%Y%m%d')]
    fast_df_sos = fast_df_sos_all_data[['u_3m_c', 'v_3m_c', 'w_3m_c']].rename(columns={
        'u_3m_c': 'u',
        'v_3m_c': 'v',
        'w_3m_c': 'w',
    })
    fast_df_sos = fast_df_sos.resample('0.1s').mean()
    fast_df_sos.index = fast_df_sos.index.get_level_values(1)
    fast_df_sos['u'] = fast_df_sos['u'].interpolate()
    fast_df_sos['v'] = fast_df_sos['v'].interpolate()
    fast_df_sos['w'] = fast_df_sos['w'].interpolate()
    mrd_uw_sos = calculate_mrd_for_df(
        fast_df_sos[['u', 'v', 'w']].reset_index(), 
        'u', 'v', 
        shift=6000, # 10 minute sliding window
        parallelism=20, 
        M=14, # 27.31 minute long calculations
    )
    mrd_uw_sos.to_parquet(os.path.join(OUTPUT_DIR, d1.strftime('%Y%m%d') + '.csv'))