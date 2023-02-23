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
from matplotlib import patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %%
USERNAME = os.getenv("ARM_USERNAME")
TOKEN = os.getenv("ARM_TOKEN")
SAIL_DATA_STREAM = 'gucdlrhiM1.b1'
SNR_THRESHOLD = 0.008
DATE_FORMAT = "%Y-%m-%d"
MAX_RANGE = 2000

scan_azimuth_valley_wise = 149
scan_azimuth_valley_cross = 270


# %%
def create_dl_plots(output_path, date):
    startdate = date
    # also get data from the next day - UTC conversion
    enddate = (
        datetime.datetime.strptime(date, DATE_FORMAT) + datetime.timedelta(days=2)
    ).strftime(DATE_FORMAT)

    print("Beginning data download")
    with TemporaryDirectory() as temp_dir:
        act.discovery.download_data(USERNAME, TOKEN, SAIL_DATA_STREAM, startdate, enddate, output=temp_dir)
        print("Data download complete")
        dl_rhi_files = glob.glob(''.join([temp_dir, '/', SAIL_DATA_STREAM,'*cdf']))
        print(len(dl_rhi_files))
        print("Opening files")
        dl_rhi = act.io.armfiles.read_netcdf(dl_rhi_files)
        print("Converting to dataframe")
        src_rhi = dl_rhi.to_dataframe().reset_index()
    
    print("Preprocessing data")
    # Convert time zone
    src_rhi['time'] = src_rhi['time'].dt.tz_localize('UTC').dt.tz_convert('US/Mountain')

    # Shrink dataset
    #    by removing extra data
    src_rhi = src_rhi.query(f"range < {MAX_RANGE}")
    #    by isolating data to target day in local time
    src_rhi = src_rhi[
        src_rhi['time'].dt.day == datetime.datetime.strptime(startdate, DATE_FORMAT).day
    ]

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

    # Split dataset into valley-wise and cross-valley RHI scans
    valley_rhi_df = src_rhi[np.abs(src_rhi['azimuth'] - scan_azimuth_valley_wise) < 1]
    xvalley_rhi_df = src_rhi[np.abs(src_rhi['azimuth'] - scan_azimuth_valley_cross) < 1]
    
    # Label the 4 cross-valley scans that happen each hour
    # We do this by defining the "hourly seconds" (second for a data point where 0 seconds is 
    # at the beginning of the hour)
    # and saying that all data from after 913 (after 15:13 mm:ss) and before 1113 (before 18:13 
    # mm:ss) is the first scan, and so on - this may be imperfect
    xvalley_rhi_df['hourly_seconds'] = xvalley_rhi_df.apply(lambda row: row['minute']*60 + row['second'], axis=1)
    xvalley_rhi_df['hourly_scan_n'] = pd.cut(
        xvalley_rhi_df['hourly_seconds'],
        [913, 1113, 2713, 2913, 3599],
        labels=['15.00','18.00','45.00','48.00']
    )

    # Label the 4 valley-wise scans that happen each hour
    # similarly to above
    valley_rhi_df['hourly_seconds'] = valley_rhi_df.apply(lambda row: row['minute']*60 + row['second'], axis=1)
    valley_rhi_df['hourly_scan_n'] = pd.cut(
        valley_rhi_df['hourly_seconds'],
        [22, 214, 1804, 2014, 3599],
        labels=['00.00','03.00','30.00','33.00']
    )

    print("Retrieving ground profiles")
    # Get ground profiles
    xvalley_ground_profile = sosutils.get_radar_scan_ground_profile(
        lon =     dl_rhi['lon'].values[0],
        lat =     dl_rhi['lat'].values[0],
        bearing =     xvalley_rhi_df.azimuth.unique()[0],
        radius =     3, #km
        spacing = 100 #meters
    )
    
    valley_ground_profile = sosutils.get_radar_scan_ground_profile(
        lon =     dl_rhi['lon'].values[0],
        lat =     dl_rhi['lat'].values[0],
        bearing =     valley_rhi_df.azimuth.unique()[0],
        radius =     3, #km
        spacing = 100 #meters
    )

    print("Plotting")
    # Do Plots
    # X VALLEY
    hours = np.linspace(0,23,24)
    scan_n_options = ['15.00','18.00','45.00','48.00']
    fig, axes = plt.subplots(
        len(hours),
        len(scan_n_options),
        figsize=(30,96),
    )

    for i_hour, hour in enumerate(hours):
        for i_scan_n, scan_n in enumerate(scan_n_options):
            ax = axes[i_hour][i_scan_n]
            src = xvalley_rhi_df.query(f"hour == {hour}")
            src = src[src['hourly_scan_n'] == scan_n]
            hexplot = ax.hexbin(
                src['x'], 
                src['z'], 
                C=src['radial_velocity'], 
                cmap='RdYlBu',
                vmin=-5,
                vmax=5
            )
            ax.set_xlabel("East (negative) West (positive) Distance (m)")
            ax.xaxis.label.set_fontsize(12)
            ax.set_ylabel("Height above LiDAR (m)")
            ax.yaxis.label.set_fontsize(12)
            hour_str = f"{int(hour):02d}" if not np.isnan(int(hour)) else "NaT"
            minute_str = f"{int(src['minute'].min()):02d}" if not np.isnan(src['minute'].min()) else "NaT"
            ax.title.set_text(f"{hour_str}:{minute_str}")
            ax.title.set_fontsize(16)
            ax.title.set_weight("bold")
            ax.set_xlim(-MAX_RANGE, MAX_RANGE)
            ax.set_ylim(0, MAX_RANGE)
            axins = inset_axes(
                ax,
                width="33%",
                height="5%",
                loc="upper right",
                # bbox_to_anchor=(1.05, 0., 1, 1)
            )
            ax.fill_between(xvalley_ground_profile['distance'], 0, xvalley_ground_profile['elevation'])
            axins.xaxis.set_ticks_position("bottom")
            cbar = fig.colorbar(hexplot, cax=axins,  orientation="horizontal", ticks=[-5, 0, 5])
            [lab.set_fontweight(500) for lab in cbar.ax.get_xticklabels()]
            [lab.set_path_effects([pe.withStroke(linewidth=3, foreground="white")]) for lab in cbar.ax.get_xticklabels()]

    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f"Cross-valley Scans (270˚) on {date}")
    output_file = os.path.join(output_path, date + '-cross-valley' + '.png')
    print(f'Saving figure to: {output_file}')
    plt.savefig(output_file)


    # VALLEY WISE
    hours = np.linspace(0,23,24)
    scan_n_options = ['00.00','03.00','30.00','33.00']
    fig, axes = plt.subplots(
        len(hours),
        len(scan_n_options),
        figsize=(30,96),
    )

    for i_hour, hour in enumerate(hours):
        for i_scan_n, scan_n in enumerate(scan_n_options):
            ax = axes[i_hour][i_scan_n]
            src = valley_rhi_df.query(f"hour == {hour}")
            src = src[src['hourly_scan_n'] == scan_n]
            hexplot = ax.hexbin(
                src['x'], 
                src['z'], 
                C=src['radial_velocity'], 
                cmap='RdYlBu',
                vmin=-10,
                vmax=10
            )
            ax.set_xlabel("Upvalley (negative) Downvalley (positive) Distance (m)")
            ax.xaxis.label.set_fontsize(12)
            ax.set_ylabel("Height above LiDAR (m)")
            ax.yaxis.label.set_fontsize(12)
            hour_str = f"{int(hour):02d}" if not np.isnan(int(hour)) else "NaT"
            minute_str = f"{int(src['minute'].min()):02d}" if not np.isnan(src['minute'].min()) else "NaT"
            ax.title.set_text(f"{hour_str}:{minute_str}")
            ax.title.set_fontsize(16)
            ax.title.set_weight("bold")
            ax.set_xlim(-MAX_RANGE, MAX_RANGE)
            ax.set_ylim(0, MAX_RANGE)
            axins = inset_axes(
                ax,
                width="33%",
                height="5%",
                loc="upper right",
                # bbox_to_anchor=(1.05, 0., 1, 1)
            )
            ax.fill_between(valley_ground_profile['distance'], 0, valley_ground_profile['elevation'])
            axins.xaxis.set_ticks_position("bottom")
            cbar = fig.colorbar(hexplot, cax=axins,  orientation="horizontal", ticks=[-10, 0, 10])
            [lab.set_fontweight(500) for lab in cbar.ax.get_xticklabels()]
            [lab.set_path_effects([pe.withStroke(linewidth=3, foreground="white")]) for lab in cbar.ax.get_xticklabels()]
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f"Along-valley Scans (149˚) ons {date}")
    output_file = os.path.join(output_path, date + '-along-valley' + '.png')
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
