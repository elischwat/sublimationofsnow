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
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append("/home/elilouis/sublimationofsnow")
import sosutils
import argparse
import datetime as dt
# -

RESAMPLE_TIME = None

# # Inputs
sail_code_kazr = 'guckazrcfrgeM1.a1'
sail_code_ld = 'gucldM1.b1'
sail_code_ceil = 'gucceilM1.b1'
INPUT_DATE_FORMAT = "%Y-%m-%d"
DATE_FORMAT = "%Y%m%d"

output_dir_fmcw = "/data2/elilouis/noaa_psl_data/fmcw/"
output_dir_parsivel = "/data2/elilouis/noaa_psl_data/parsivel/"
output_dir_kazr = f"/data2/elilouis/{sail_code_kazr}/"
output_dir_ld = f"/data2/elilouis/{sail_code_ld}/"
output_dir_ceil = f"/data2/elilouis/{sail_code_ceil}/"

username = os.getenv("ARM_USERNAME")
token = os.getenv("ARM_TOKEN")


def create_plots(output_path, date):

    # # Download data
    print("Downloading data...")
    # ## FMCW Radar
    downloaded_files_fmcw = act.discovery.download_noaa_psl_data(
        site='kps',
        instrument='Radar FMCW Moment',
        startdate=date,
        # NOAA DATA includes the "enddate"
        enddate=dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=1), DATE_FORMAT),
        output=output_dir_fmcw
    )
    # ## Parsivel data
    downloaded_files_parsivel = act.discovery.download_noaa_psl_data(
        site='kps',
        instrument='Parsivel',
        startdate=date,
        # NOAA DATA includes the "enddate"
        enddate=dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=1), DATE_FORMAT),
        output= output_dir_parsivel
    )
    # ## Laser Disdrometer
    downloaded_files_ld = act.discovery.download_data(
        username,
        token,
        sail_code_ld,
        startdate=date,
        # SPLASH DATA does not include the "enddate"
        enddate=dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=1), DATE_FORMAT),
        output=output_dir_ld
    )
    # ## KAZR
    downloaded_files_kazr = act.discovery.download_data(
        username,
        token,
        sail_code_kazr,
        startdate=date,
        # SPLASH DATA does not include the "enddate"
        enddate=dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=2), DATE_FORMAT),
        output=output_dir_kazr
    )
    # ## Ceilometer
    downloaded_files_ceil = act.discovery.download_data(
        username,
        token,
        sail_code_ceil,
        startdate=date,
        # SPLASH DATA does not include the "enddate"
        enddate=dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=2), DATE_FORMAT),
        output=output_dir_ceil
    )

    # # Open data
    print("Opening data...")
    # ## FMCW
    fmcw_moment_ds = act.io.noaapsl.read_psl_radar_fmcw_moment([f for f in downloaded_files_fmcw if f.endswith(".raw")])
    # ## Parsivel
    parsivel_ds = act.io.noaapsl.read_psl_parsivel(downloaded_files_parsivel)
    # ## Laser disdrometer
    ld_ds = act.io.armfiles.read_netcdf(downloaded_files_ld)
    # ## KAZR
    kazr_ds = act.io.armfiles.read_netcdf(downloaded_files_kazr)
    # ## Ceilometer
    ceil_ds = act.io.armfiles.read_netcdf(downloaded_files_ceil)

    # # Plot Daily Data
    fmcw_moment_ds_local = sosutils.modify_xarray_timezone(fmcw_moment_ds, 'UTC', 'US/Mountain')
    print(fmcw_moment_ds['time'].min())
    print(fmcw_moment_ds['time'].max())
    parsivel_ds_local = sosutils.modify_xarray_timezone(parsivel_ds, 'UTC', 'US/Mountain')
    print(parsivel_ds['time'].min())
    print(parsivel_ds['time'].max())
    ld_ds_local = sosutils.modify_xarray_timezone(ld_ds, 'UTC', 'US/Mountain')
    print(ld_ds['time'].min())
    print(ld_ds['time'].max())
    kazr_ds_local = sosutils.modify_xarray_timezone(kazr_ds, 'UTC', 'US/Mountain')
    print(kazr_ds['time'].min())
    print(kazr_ds['time'].max())
    ceil_ds_local = sosutils.modify_xarray_timezone(ceil_ds, 'UTC', 'US/Mountain')
    print(ceil_ds['time'].min())
    print(ceil_ds['time'].max())

    plot_end_date = dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=1), DATE_FORMAT)
    if RESAMPLE_TIME:
        print("Resampling data...")
        fmcw_moment_ds_local_resampled = fmcw_moment_ds_local.resample(time=RESAMPLE_TIME).mean().sel(time=slice(date, plot_end_date))
        parsivel_ds_local_resampled = parsivel_ds_local.resample(time=RESAMPLE_TIME).mean().sel(time=slice(date, plot_end_date))
        ld_ds_local_resampled = ld_ds_local.resample(time=RESAMPLE_TIME).mean().sel(time=slice(date, plot_end_date))
        kazr_ds_local_resampled = kazr_ds_local.resample(time=RESAMPLE_TIME).mean().sel(time=slice(date, plot_end_date))
        ceil_ds_local_resampled = ceil_ds_local.resample(time=RESAMPLE_TIME).mean().sel(time=slice(date, plot_end_date))
    else:
        fmcw_moment_ds_local_resampled = fmcw_moment_ds_local.sel(time=slice(date, plot_end_date))
        parsivel_ds_local_resampled = parsivel_ds_local.sel(time=slice(date, plot_end_date))
        ld_ds_local_resampled = ld_ds_local.sel(time=slice(date, plot_end_date))
        kazr_ds_local_resampled = kazr_ds_local.sel(time=slice(date, plot_end_date))
        ceil_ds_local_resampled = ceil_ds_local.sel(time=slice(date, plot_end_date))

    # +
    # Create display object with both datasets
    print("Plotting data...")
    display_5min = act.plotting.TimeSeriesDisplay(
        {
            "Kettle Ponds, NOAA PSL Radar FMCW": fmcw_moment_ds_local_resampled,
            "Kettle Ponds, NOAA Parsivel": parsivel_ds_local_resampled,
            "Gothic, SAIL Laser Disdrometer": ld_ds_local_resampled,
            "Gothic, SAIL Laser Disdrometer Precip Rate": ld_ds_local_resampled,
            "Gothic, SAIL KAZR Radar": kazr_ds_local_resampled,
            "Gothic, SAIL Ceilometer": ceil_ds_local_resampled
        },
        subplot_shape=(6,), 
        figsize=(15, 15)
    )

    # Plot the subplots
    display_5min.plot(
        'reflectivity_uncalibrated', 
        dsname='Kettle Ponds, NOAA PSL Radar FMCW',
        cmap='act_HomeyerRainbow', 
        subplot_index=(0,)
    )
    display_5min.plot(
        'backscatter',
        dsname='Gothic, SAIL Ceilometer',
        cmap='act_HomeyerRainbow', 
        subplot_index=(1,),
        vmin=0, vmax=4000
    )
    display_5min.plot(
        'reflectivity', 
        dsname='Gothic, SAIL KAZR Radar',
        cmap='act_HomeyerRainbow', 
        subplot_index=(2,)
    )
    display_5min.plot(
        'number_density_drops', 
        dsname='Kettle Ponds, NOAA Parsivel',
        cmap='act_HomeyerRainbow', 
        subplot_index=(3,)
    )
    display_5min.plot(
        'number_density_drops', 
        dsname='Gothic, SAIL Laser Disdrometer',
        cmap='act_HomeyerRainbow', 
        subplot_index=(4,)
    )
    display_5min.plot(
        'precip_rate',
        dsname='Gothic, SAIL Laser Disdrometer',
        cmap='act_HomeyerRainbow', 
        subplot_index=(5,)
    )

    # Set limits
    display_5min.axes[0].set_ylim([0, 5000])
    display_5min.axes[1].set_ylim([0, 5000])
    display_5min.axes[2].set_ylim([0, 5000])
    display_5min.axes[3].set_ylim([0, 5])
    display_5min.axes[4].set_ylim([0, 5])
    for cb in display_5min.cbs:
        cb.ax.set_ylabel(cb.ax.get_ylabel(), labelpad=15)

    for ax in display_5min.axes:
        ax.set_xlim(
            # np.datetime64(dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT), INPUT_DATE_FORMAT)), 
            dt.datetime.strptime(date, DATE_FORMAT),
            # np.datetime64(dt.datetime.strftime(dt.datetime.strptime(enddate, DATE_FORMAT), INPUT_DATE_FORMAT))
            dt.datetime.strptime(
                dt.datetime.strftime(dt.datetime.strptime(date, DATE_FORMAT) + dt.timedelta(days=1), DATE_FORMAT),
                DATE_FORMAT
            )
        )
    display_5min.axes[-1].set_xlabel("time (US/Mountain)")    
    plt.subplots_adjust(hspace=0.40)
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
        default=(dt.datetime.today() - dt.timedelta(days=3)).strftime(INPUT_DATE_FORMAT),
        help="Date you want data from, in format '%Y-%m-%d'. Default is 3 days before today, which is the most recent date data is usually posted for."
    )
    args = parser.parse_args()
    create_plots(
        args.output_path,
        # Note that this differs from other daily jobs b/c the NOAA data requires dates in the format
        # defined above "DATE_FORMAT" while all SAIL data can take the either dates in the
        # INPUT_DATE_FORMAT or DATE_FORMAT
        dt.datetime.strftime(
            dt.datetime.strptime(args.date, INPUT_DATE_FORMAT),
            DATE_FORMAT
        )
    )


# %%
if __name__ == "__main__":
    main()
