# -*- coding: utf-8 -*-
# %%
import numpy as np
import argparse
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
import act
import os
import sys
sys.path.append('/home/elilouis/sublimationofsnow')
import sosutils
from tempfile import TemporaryDirectory

# %%
USERNAME = os.getenv("ARM_USERNAME")
TOKEN = os.getenv("ARM_TOKEN")
SAIL_DATA_STREAM = 'gucdlrhiM1.b1'
SNR_THRESHOLD = 0.008
DATE_FORMAT = "%Y-%m-%d"


# %%
def create_dl_plots(output_path, date):
    startdate = date
    # also get data from the next day - UTC conversion
    enddate = (
        datetime.datetime.strptime(date, DATE_FORMAT) + datetime.timedelta(days=2)
    ).strftime(DATE_FORMAT)

    with TemporaryDirectory() as temp_dir:
        act.discovery.download_data(USERNAME, TOKEN, SAIL_DATA_STREAM, startdate, enddate, output=temp_dir)
        dl_rhi_files = glob.glob(''.join([temp_dir, '/', SAIL_DATA_STREAM,'*cdf']))
        print(len(dl_rhi_files))
        dl_rhi = act.io.armfiles.read_netcdf(dl_rhi_files)
        src_rhi = dl_rhi.to_dataframe().reset_index()
    
    src_rhi['time'] = src_rhi['time'].dt.tz_localize('UTC').dt.tz_convert('US/Mountain')
    src_rhi['date'] = src_rhi['time'].dt.date
    # get data for a complete, local time, day
    src_rhi = src_rhi[
        src_rhi['time'].dt.day == datetime.datetime.strptime(startdate, DATE_FORMAT).day
    ]

    
    # Filter with SNR
    src_rhi['SNR'] = src_rhi['intensity'] - 1
    src_rhi.loc[src_rhi.eval(f'SNR < {SNR_THRESHOLD}'), 'radial_velocity'] = np.nan

    # RHI: convert polar coordinates to rectangular coords with the radar at (0,0)
    src_rhi['x'] = src_rhi['range']*np.cos(np.deg2rad(src_rhi['elevation']))
    src_rhi['z'] = src_rhi['range']*np.sin(np.deg2rad(src_rhi['elevation']))

    src_rhi['time_beginning_of_hour'] = src_rhi['time'].apply(lambda dt: dt.replace(minute=0, second=0, microsecond=0))

    # Get ground profile
    upvalley_elev_profile_df = sosutils.get_radar_scan_ground_profile(
        lon =     dl_rhi['lon'].values[0],
        lat =     dl_rhi['lat'].values[0],
        bearing =     330,
        radius =     3, #km
        spacing = 10
    )
    downvalley_elev_profile_df = sosutils.get_radar_scan_ground_profile(
        lon =     dl_rhi['lon'].values[0],
        lat =     dl_rhi['lat'].values[0],
        bearing =     130,
        radius =     3, #km
        spacing = 10
    )

    # Only take the positive halfs of both, reverse the downvalley data, and connect the two
    upvalley_elev_profile_df = upvalley_elev_profile_df.query('distance > 0 ')
    downvalley_elev_profile_df = downvalley_elev_profile_df.query('distance > 0')
    downvalley_elev_profile_df['distance'] = - downvalley_elev_profile_df['distance']
    valleywise_elev_profile_df = pd.concat([upvalley_elev_profile_df, downvalley_elev_profile_df])

    src_downvalley = src_rhi[np.abs(src_rhi['azimuth'] - 130) < 1].query('range <= 3005')
    src_downvalley['x'] = - src_downvalley['x']
    src_upvalley = src_rhi[np.abs(src_rhi['azimuth'] - 330) < 1].query('range <= 3035')

    src = pd.concat([src_downvalley, src_upvalley])

    plot_hours = np.arange(0,24)

    fig, axes = plt.subplots(
        4, 
        6, 
        figsize=(30,10), 
        sharex=True, sharey=True
    )

    hexplot = None
    for i_day, day_and_hour in enumerate(sorted(src['time_beginning_of_hour'].unique())):
        local_src = src[src['time_beginning_of_hour'] == day_and_hour]
        ax = axes.flatten()[i_day]
        hexplot = ax.hexbin(local_src['x'], local_src['z'], C=local_src['radial_velocity'], cmap='RdYlBu', clim=(-10, 10))
        ax.title.set_text(pd.to_datetime(day_and_hour))
        ax.title.set_fontsize(8)
        ax.fill_between(
            valleywise_elev_profile_df['distance'],
            0,
            valleywise_elev_profile_df['elevation'],
            color='darkgrey'
        )
    fig.colorbar(hexplot, ax=axes.ravel().tolist())
    plt.suptitle(f"Valley-wise Scans (330˚ up-valley, 130˚ down-valley) for {date}")
    output_file = os.path.join(output_path, date + '.png')
    print(f'Saving figure to: {output_file}')
    plt.savefig(output_file)


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output file path for the plot of daily doppler lidar scans."
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,      
         #default b/c data is in UTC, converting to local time means we need 2 day old (the most recently available)
         # in addition to 3 day old data.
        default=(datetime.datetime.today() - datetime.timedelta(days=3)).strftime(DATE_FORMAT),
        help="Date you want data from, in format '%Y-%m-%d'. Default is 3 days before today, which is the most recent date data is usually posted for."
    )
    args = parser.parse_args()
    create_dl_plots(
        args.output_path,
        args.date
    )


# %%
if __name__ == "__main__":
    main()
