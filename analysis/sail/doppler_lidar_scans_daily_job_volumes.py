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
import numpy as np
import seaborn as sns
import geopy
import geopandas as gpd
from shapely import geometry
import contextily as ctx

## Assumes this scan plan:
# Volume scan: azi. 97-16˚9, elev. 6-31˚   31:18 --> 43
# 330˚ RHI-half:              01:17 --> 02:51
# 130˚ RHI-half:              03:09 --> 04:44
# 60˚ RHI-full:               05:13 --> 08:14

# # User Inputs
USERNAME = os.getenv("ARM_USERNAME")
TOKEN = os.getenv("ARM_TOKEN")
SAIL_DATA_STREAM = 'gucdlrhiM1.b1'
SNR_THRESHOLD = 0.008
DOWNVALLEY_XSECTION_DISTANCE = 1500
DOWNVALLEY_XSECTION_AVERAGING_DISTANCE = 100
VALLEY_PLANE_AZIMUTH = 129 # points down valley
sector_scan_hour = 21 # for an example plot of sector scans
sail_dl_pt = geometry.Point(-106.987900,38.956160)
kettle_ponds_pt = geometry.Point(-106.973006, 38.942005)


def create_dl_plots(output_path, date, downvalley_distance):
    startdate = date
    enddate = date

    ## Download data
    with TemporaryDirectory() as temp_dir:
        act.discovery.download_data(USERNAME, TOKEN, SAIL_DATA_STREAM, startdate, enddate, output=temp_dir)
        dl_rhi_files = glob.glob(''.join([temp_dir, '/', SAIL_DATA_STREAM,'*cdf']))
        print(len(dl_rhi_files))
        dl_rhi = act.io.armfiles.read_netcdf(dl_rhi_files)
    
    ## Preprocess data
    src_rhi = dl_rhi.to_dataframe().reset_index().query('range < 2100')
    src_rhi['time'] = src_rhi['time'].dt.tz_localize('UTC').dt.tz_convert('US/Mountain')
    src_rhi['date'] = src_rhi['time'].dt.date
    # RHI: convert polar coordinates to rectangular coords with the radar at (0,0)
    src_rhi['x'] = src_rhi['range']*np.cos(np.deg2rad(src_rhi['elevation']))
    src_rhi['z'] = src_rhi['range']*np.sin(np.deg2rad(src_rhi['elevation']))
    # Filter with SNR
    src_rhi['SNR'] = src_rhi['intensity'] - 1
    src_rhi.loc[src_rhi.eval(f'SNR < {SNR_THRESHOLD}'), 'radial_velocity'] = np.nan
    src_rhi['hour'] = src_rhi['time'].dt.hour
    src_rhi['minute'] = src_rhi['time'].dt.minute
    src_rhi['time_beginning_of_hour'] = src_rhi['time'].apply(lambda dt: dt.replace(minute=0, second=0, microsecond=0))
    src_rhi = src_rhi.set_index(['time', 'range'])

    ## Plot sector RHI for provided hour
    volume_df = src_rhi.query(
        f"hour == {sector_scan_hour}"
    ).query(
        "minute > 30"
    )
    np.linspace(97,169, 3)
    azimuths = [ 97., 100., 103., 106., 109., 112., 115., 118., 121., 124., 127.,
        130., 133., 136., 139., 142., 145., 148., 151., 154., 157., 160.,
        163., 166., 169.]
    src = volume_df[volume_df.azimuth.isin(azimuths)]
    fig, axes = plt.subplots(
            5, 
            5, 
            figsize=(30,10),
            sharex=True, sharey=True
        )
    hexplot = None
    axes_list = axes.flatten()
    for i, (azimuth, rhi_sector_scan) in enumerate(src.query('range < 2000').groupby(["azimuth"])):
        ax = axes_list[i]
        hexplot = ax.hexbin(rhi_sector_scan['x'], rhi_sector_scan['z'], C=rhi_sector_scan['radial_velocity'], cmap='RdYlBu', clim=(-10, 10))
        ax.set_title(f'{azimuth}')
    fig.colorbar(hexplot, ax=axes.ravel().tolist())
    out = os.path.join(
            output_path,
            f"{date}-sectors_hr{sector_scan_hour}.png"
        )
    print(f"Saving figure to path: {out}")
    fig.savefig(out)


    ## Calculate valley-wise coordinates given the provided VALLEY_PLANE_AZIMUTH
    valleywise_df = src_rhi.copy().reset_index()

    valleywise_df = valleywise_df.query(
        "minute > 30"
    )


    valleywise_df['x'] = valleywise_df['range']*np.sin(np.deg2rad(
        90 - valleywise_df['elevation']
    ))*np.cos(np.deg2rad(
        VALLEY_PLANE_AZIMUTH - valleywise_df['azimuth']
    ))
    valleywise_df['y'] = valleywise_df['range']*np.sin(np.deg2rad(
        90 - valleywise_df['elevation']
    ))*np.sin(np.deg2rad(
        VALLEY_PLANE_AZIMUTH - valleywise_df['azimuth']
    ))
    valleywise_df['z'] = valleywise_df['range']*np.cos(np.deg2rad(
        90 - valleywise_df['elevation']
    ))
    valleywise_df['valleywise_velocity'] = valleywise_df['radial_velocity']*np.cos(np.deg2rad(
        VALLEY_PLANE_AZIMUTH - valleywise_df['azimuth']
    ))*np.cos(np.deg2rad(
        valleywise_df['elevation']
    ))


    ## Plot a valley-xsection of valleywise velocities using the provided downvalley_distance and DOWNVALLEY_XSECTION_AVERAGING_DISTANCE
    valley_xsection_df = valleywise_df.copy()
    valley_xsection_df = valley_xsection_df[
        np.abs(valley_xsection_df.x - downvalley_distance) < DOWNVALLEY_XSECTION_AVERAGING_DISTANCE
    ]

    fig, axes = plt.subplots(
        4, 
        6, 
        figsize=(30,10), 
        sharex=True, sharey=True
    )
    hexplot = None
    axes_list = axes.flatten()
    for i_day, day_and_hour in enumerate(sorted(valley_xsection_df['time_beginning_of_hour'].unique())):
        src = valley_xsection_df[valley_xsection_df['time_beginning_of_hour'] == day_and_hour]
        ax = axes_list[i_day]
        hexplot = ax.hexbin(
            src['y'], 
            src['z'], 
            C=src['valleywise_velocity'], 
            cmap='RdYlBu', 
            clim=(-10, 10)
        )
        ax.title.set_text(pd.to_datetime(day_and_hour))
        ax.title.set_fontsize(8)
    fig.colorbar(hexplot, ax=axes.ravel().tolist())
    out = os.path.join(
        output_path,
        f"{date}-{DOWNVALLEY_XSECTION_DISTANCE}-volume.png"
    )
    print(f"Saving figure to path: {out}")
    fig.savefig(out)



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
        default=(datetime.datetime.today() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
        help="Date you want data from, in format '%Y-%m-%d'."
    )
    parser.add_argument(
        "-x",
        "--xsection-distance",
        type=int,
        default = DOWNVALLEY_XSECTION_DISTANCE,
        help="Distance in meters downvalley, in valleywise coordinates, to plot the cross section."
    )
    args = parser.parse_args()
    create_dl_plots(
        args.output_path,
        args.date,
        args.xsection_distance
    )

if __name__ == "__main__":
    main()