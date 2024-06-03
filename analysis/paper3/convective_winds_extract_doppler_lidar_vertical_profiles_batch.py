"""This script aggregates doppler lidar data and saves out files. 
It downloads (or uses already downloaded) SAIL doppler lidar data, 
isolates along-valley RHI scan data and, for each scan:
1. aggregates the scan data in vertical and horizontal bins,
2. calculates mean, median, std dev, and count for each bin,
3. saves the data to parquet file.
"""

import numpy as np
import pandas as pd
import act
import os
import datetime as dt
from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# USER INPUTS
# Download files from SAIL servers. If false, looks in provided directories
DOWNLOAD = False
# don't change this one
SAIL_DATA_STREAM = 'gucdlrhiM1.b1'
sail_download_path = f"/storage/elilouis/sublimationofsnow/{SAIL_DATA_STREAM}/"
processed_output_path = f"/storage/elilouis/sublimationofsnow/sail_processed/{SAIL_DATA_STREAM}/"
PARALLELISM = 16 # 10 seems optimal on j-lunquist-3; 20 overloaded the memory
VALLEY_INCLINE = 10
OFFSETS = [-1500, -1000, -500, -250, -125, 125, 250, 500, 1000, 1500]
SAMPLING_TOLERANCE = 50
SNR_THRESHOLD = 0.008
MAX_RANGE = 2000
START_DATE = "20230201"
END_DATE = "20230616"

USERNAME = os.getenv("ARM_USERNAME")
TOKEN = os.getenv("ARM_TOKEN")

DATE_FORMAT = "%Y-%m-%d"
datelist = [timestamp.date() for timestamp in pd.date_range(START_DATE, END_DATE).tolist()]

def process_date(date):
    print(f"Processing {date}...")
    if DOWNLOAD:
        print("\tBeginning data download...")
        files = act.discovery.download_arm_data(
            USERNAME, 
            TOKEN, 
            SAIL_DATA_STREAM, 
            date.strftime('%Y%m%d'), 
            date.strftime('%Y%m%d'), 
            output=sail_download_path
        )

        print("\tData download complete.")
        print("\tOpening files...")
        dl_rhi = act.io.arm.read_arm_netcdf(files)
        print("\tFile opening complete.")
    else:
        print("\tSkipping data download... opening files locally")
        files = glob.glob(os.path.join(
            sail_download_path,
            f"{SAIL_DATA_STREAM}.{date.strftime('%Y%m%d')}.*.cdf"
        ))
        print(f"\tOpening files...{len(files)} of them")
        dl_rhi = act.io.arm.read_arm_netcdf(files)
        print("\tFile opening complete.")

    print("\tConverting to dataframe...")
    src_rhi = dl_rhi.to_dataframe().reset_index()
    print("\tConversion complete.")

    print("\tPreprocessing data")
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
    src_rhi['horizontal_velocity'] = ( src_rhi['radial_velocity'] / np.cos(np.deg2rad(src_rhi['elevation'])))
    # Convert horizontal velocities to streamwise velocities. This a small adjustment for the slope of the valley.
    # Velocities are still positive down-valley.
    
    src_rhi['streamwise_velocity'] = src_rhi['horizontal_velocity'] / np.cos(np.deg2rad(VALLEY_INCLINE))
    print("\tPreprocessing complete")
    
    print("\tIsolating along-valley scans")
    # Separate cross valley and along valley scans
    ############################################################################################################
    ############################################################################################################
    scan_azimuth_valley_wise = 149
    scan_azimuth_valley_cross = 270
    # Isolate valley-wise RHI scans
    valley_rhi_df = src_rhi[np.abs(src_rhi['azimuth'] - scan_azimuth_valley_wise) < 1]

    if len(valley_rhi_df) == 0:
        print(f"\tFound no along=valley RHI scans for date {date}")
        return None
    else:
        # Label the 4 valley-wise scans that happen each hour
        # similarly to above
        valley_rhi_df['hourly_seconds'] = valley_rhi_df.apply(lambda row: row['minute']*60 + row['second'], axis=1)
        valley_rhi_df['hourly_scan_n'] = pd.cut(
            valley_rhi_df['hourly_seconds'],
            [22, 214, 1804, 2014, 3599],
            labels=['00.00','03.00','30.00','33.00']
        )
        print(f"\t{len(valley_rhi_df)}")
        valley_rhi_df = valley_rhi_df.dropna(subset=['hourly_scan_n'])
        print(f"\t{len(valley_rhi_df)}")

        valley_rhi_df['scan_time'] = valley_rhi_df.apply(
            lambda row : row['time_beginning_of_hour']+ dt.timedelta(minutes = float(row['hourly_scan_n'])),
            axis=1
        )
        valley_rhi_df['scan_time'] = valley_rhi_df.apply(
            lambda row: row['scan_time'].replace(
                year = row['time'].year, 
                month = row['time'].month, 
                day = row['time'].day,
            ),
            axis=1
        )
        print("\tIsolating along-valley scans complete")
        
        print("\tExtracting profiles")
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

        spatial_src_df = spatial_src_df.groupby(
            ['z_binned', 'x_offset', 'scan_time']
        )[
            ['horizontal_velocity', 'radial_velocity', 'streamwise_velocity']
        ].agg(['mean', 'median', 'std', 'count']).reset_index()

        print("\tSaving to file")
        output_file = os.path.join(processed_output_path, str(date) + '.parquet')
        spatial_src_df.to_parquet(
            output_file
        )
        return output_file
    
if __name__ == '__main__':
    dates_tqdm = tqdm(datelist)

    print(f"Beginning processing (parallelism = {PARALLELISM})")
    # slow way
    # processed_results = [process_date(date) for date in dates_tqdm]
    # fast way
    processed_results =  Parallel(n_jobs = PARALLELISM)(
        delayed(process_date)(date) for date in dates_tqdm
    )
    # Remove Nones
    processed_results = [p for p in processed_results if p]
    print(f"Finished processing, {len(processed_results)} files generated.")
    