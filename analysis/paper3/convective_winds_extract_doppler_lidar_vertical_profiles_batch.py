"""_summary_
"""
import numpy as np
import pandas as pd
import act
import os
import datetime as dt
import datetime as dt
import altair as alt
alt.data_transformers.enable('json')


USERNAME = os.getenv("ARM_USERNAME")
TOKEN = os.getenv("ARM_TOKEN")
SAIL_DATA_STREAM = 'gucdlrhiM1.b1'
sail_download_path = f"/Users/elischwat/Development/data/sublimationofsnow/{SAIL_DATA_STREAM}/"
processed_output_path = f"/Users/elischwat/Development/data/sublimationofsnow/sail_processed/{SAIL_DATA_STREAM}/"
PARALLELISM = 8
VALLEY_INCLINE = 10
OFFSETS = [-1500, -1000, -500, -250, -125, 125, 250, 500, 1000, 1500]
SAMPLING_TOLERANCE = 50
SNR_THRESHOLD = 0.008
DATE_FORMAT = "%Y-%m-%d"
MAX_RANGE = 2000
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

dates = [
    dt.date(2023, 3, 3), # actual date of interest
    dt.date(2023, 3, 4), 
    dt.date(2023, 4, 16), # actual date of interest
    dt.date(2023, 4, 17),
    dt.date(2023, 5, 2), # actual date of interest
    dt.date(2023, 5, 3),
    # # dt.date(2023, 6, 9),  # actual date of interest
    # # dt.date(2023, 6, 10),
    # # dt.date(2023, 6, 11),  # actual date of interest
    # # dt.date(2023, 6, 12), 
]



# +
print("Beginning data download...")
def process_date(date):
    files = act.discovery.download_arm_data(
        USERNAME, 
        TOKEN, 
        SAIL_DATA_STREAM, 
        date.strftime('%Y%m%d'), 
        date.strftime('%Y%m%d'), 
        output=sail_download_path
    )

    print("Data download complete.")

    print("Opening files...")
    dl_rhi = act.io.arm.read_arm_netcdf(files)
    print("File opening complete.")

    print("Converting to dataframe...")
    src_rhi = dl_rhi.to_dataframe().reset_index()
    print("Conversion complete.")

    print("Preprocessing data")
    # Convert time zone
    src_rhi['time'] = src_rhi['time'].dt.tz_localize('UTC').dt.tz_convert('US/Mountain')
    src_rhi['time'] = pd.to_datetime(src_rhi['time'].dt.tz_localize(None))
    # Shrink dataset by removing extra data
    src_rhi = src_rhi.query(f"range < {MAX_RANGE}")
    # Filter with SNR
    src_rhi['SNR'] = src_rhi['intensity'] - 1
    src_rhi.loc[src_rhi.eval(f'SNR < {SNR_THRESHOLD}'), 'radial_velocity'] = np.nan
    # Remove extraneous scan data
    src_rhi = src_rhi.query("elevation != -0.01")
    # Add useful columns
    src_rhi['date'] = src_rhi['time'].dt.date
    src_rhi['hour'] = src_rhi['time'].dt.hour
    src_rhi['minute'] = src_rhi['time'].dt.minute
    src_rhi['second'] = src_rhi['time'].dt.second
    src_rhi['time_beginning_of_hour'] = src_rhi['time'].apply(lambda dt: dt.replace(minute=0, second=0, microsecond=0))
    # RHI: convert polar coordinates to rectangular coords with the radar at (0,0)
    src_rhi['x'] = src_rhi['range']*np.cos(np.deg2rad(src_rhi['elevation']))
    src_rhi['z'] = src_rhi['range']*np.sin(np.deg2rad(src_rhi['elevation']))

    # Calculate horizontal and streamwise velocities
    # Convert radial velocities to horizontal velocities. This makes all velocities positive down-valley.
    src_rhi['horizontal_velocity'] = ( src_rhi['radial_velocity']*np.cos(np.deg2rad(src_rhi['elevation'])))
    # Convert horizontal velocities to streamwise velocities. This a small adjustment for the slope of the valley.
    # Velocities are still positive down-valley.
    
    src_rhi['streamwise_velocity'] = src_rhi['horizontal_velocity'] / np.cos(np.deg2rad(VALLEY_INCLINE))
    print("Preprocessing complete")
    
    print("Isolating along-valley scans")
    # Separate cross valley and along valley scans
    ############################################################################################################
    ############################################################################################################
    scan_azimuth_valley_wise = 149
    scan_azimuth_valley_cross = 270
    # Isolate valley-wise RHI scans
    valley_rhi_df = src_rhi[np.abs(src_rhi['azimuth'] - scan_azimuth_valley_wise) < 1]

    # Label the 4 valley-wise scans that happen each hour
    # similarly to above
    valley_rhi_df['hourly_seconds'] = valley_rhi_df.apply(lambda row: row['minute']*60 + row['second'], axis=1)
    valley_rhi_df['hourly_scan_n'] = pd.cut(
        valley_rhi_df['hourly_seconds'],
        [22, 214, 1804, 2014, 3599],
        labels=['00.00','03.00','30.00','33.00']
    )

    valley_rhi_df.loc[:, 'scan_time'] = valley_rhi_df.apply(lambda row : row['time_beginning_of_hour']+ dt.timedelta(minutes = float(row['hourly_scan_n'])), axis=1)
    valley_rhi_df['scan_time'] = valley_rhi_df.apply(
        lambda row: row['scan_time'].replace(
            year = row['time'].year, 
            month = row['time'].month, 
            day = row['time'].day,
        ),
        axis=1
    )
    print("Isolating along-valley scans complete")
    
    print("Extracting profiles")
    ############################################################################################################
    ############################################################################################################
    # bin z so we have high res near surface (<100m), low res above
    z_bins = np.concatenate([
        np.linspace(0,100,11), 
        np.linspace(0,2000,41)[3:]
    ])

    valley_rhi_df['z_binned'] = pd.cut(valley_rhi_df['z'], bins=z_bins).apply(lambda bin: (bin.left + bin.right)/2).astype('float')

    df_list = []
    
    for x_offset in OFFSETS:
        this_offset_df = valley_rhi_df[np.abs(valley_rhi_df['x'] - x_offset) < SAMPLING_TOLERANCE]
        this_offset_df = this_offset_df.assign(x_offset = x_offset)
        df_list.append(this_offset_df)
        
    spatial_src_df = pd.concat(df_list)

    spatial_src_df = spatial_src_df.groupby(['z_binned', 'x_offset', 'scan_time'])[['horizontal_velocity', 'radial_velocity', 'streamwise_velocity']].median().reset_index()

    print("Saving to file")
    output_file = os.path.join(processed_output_path, str(date) + '.parquet')
    spatial_src_df.to_parquet(
        output_file
    )
    return output_file
    
if __name__ == '__main__':
    dates_tqdm = tqdm(dates)

    print(f"Beginning processing (parallelism = {PARALLELISM})")
    processed_results =  Parallel(n_jobs = PARALLELISM)(
        delayed(process_date)(date) for date in dates_tqdm
    )
    print(f"Finished processing, {len(processed_results)} files generated.")
    