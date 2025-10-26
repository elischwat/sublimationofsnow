import glob
import os
import numpy as np
from process_fast_data.fast_data_calculate_spectra_nomrd import calculate_mrd_for_df, fast_data_files_to_dataframe
from sublimpy import utils
import pandas as pd
from dask.distributed import Client

import time

### INPUTS
# DATA_DIR = "/Users/elischwat/Development/data/sublimationofsnow/sosqc_fast/"
DATA_DIR = "/storage/elilouis/sublimationofsnow/sosqc_fast"
OUTPUT_DIR = "mrds_uw"
START_DATE = '20221130'
END_DATE = '20230615'
VAR1='u'
VAR2='w'

def do_the_work():
    ### PREP LIST OF DATES
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    date_pairs = list(zip(
        dates[:-1],
        dates[1:]
    ))

    for d1, d2 in date_pairs:
        print(f'Processing {d1}')
        # Get the current time (first time)
        start_time = time.time()
        
        dates_subset = [d1, d2]
        files = sorted(np.array([
            glob.glob(os.path.join(
                DATA_DIR, 
                f"isfs_sos_qc_geo_tiltcor_hr_v2_{d.strftime('%Y%m%d')}**.nc"
            )) for d in dates_subset
        ]).flatten())[6:6+25]
        print(f'...opening files')
        
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
        print(f'......Elapsed time: {time.time() - start_time} seconds')
        print(f'...resampling')
        
        fast_df_sos = fast_df_sos.resample('0.1S').mean()
        # fast_df_sos.index = fast_df_sos.index.get_level_values(1)
        print(f'......Elapsed time: {time.time() - start_time} seconds')
        print(f'...interpolating')
        
        fast_df_sos = fast_df_sos.assign(
            u = fast_df_sos['u'].interpolate(),
            v = fast_df_sos['v'].interpolate(),
            w = fast_df_sos['w'].interpolate(),
        )
        print(f'......Elapsed time: {time.time() - start_time} seconds')
        print(f'...running mrds')
        
        mrd_uw_sos = calculate_mrd_for_df(
            fast_df_sos[['u', 'v', 'w']].reset_index(), 
            VAR1, VAR2,
            shift=6000, # 10 minute sliding window
            parallelism=22, 
            M=14, # 27.31 minute long calculations,
            double_rotate=True
        )
        print(f'......Elapsed time: {time.time() - start_time} seconds')
        print(f'...saving file')
        
        mrd_uw_sos.to_parquet(os.path.join(OUTPUT_DIR, d1.strftime('%Y%m%d') + '.csv'))
        print(f'......Elapsed time: {time.time() - start_time} seconds')
        print(f'...file saved, iteration finished')
        print()
        
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _ = do_the_work()