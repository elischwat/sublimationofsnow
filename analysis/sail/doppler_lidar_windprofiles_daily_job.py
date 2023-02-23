# -*- coding: utf-8 -*-
# %%
import argparse
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
import act
import os
import sys
import xarray as xr
sys.path.append('/home/elilouis/sublimationofsnow')
import sosutils
from tempfile import TemporaryDirectory
import altair as alt
import altair_saver
alt.data_transformers.disable_max_rows()

# %%
USERNAME = os.getenv("ARM_USERNAME")
TOKEN = os.getenv("ARM_TOKEN")
SAIL_DATA_STREAM = 'gucdlppiM1.b1'
DATA_STREAM_FILEEXT = '.cdf'
DATE_FORMAT = "%Y-%m-%d"


# %%
def create_dl_plots(output_path, startdate):
    enddate = (
        datetime.datetime.strptime(startdate, DATE_FORMAT) + datetime.timedelta(days=2)
    ).strftime(DATE_FORMAT)

    with TemporaryDirectory() as temp_dir:
        act.discovery.download_data(USERNAME, TOKEN, SAIL_DATA_STREAM, startdate, enddate, output=temp_dir)
        ppi_files = glob.glob(''.join([temp_dir, '/', SAIL_DATA_STREAM, '*'+DATA_STREAM_FILEEXT]))
        print(len(ppi_files))
        ppi_ds = act.io.armfiles.read_netcdf(ppi_files)
    
        ppi_ds['time_hour_and_min']  = ppi_ds.time.to_series().apply(lambda dt: dt.replace(second=0, microsecond=0))

        objs = []
        wind_obj=None
        # Split ppi dataset into chunks with full scans
        # and calculate the winds for each gucdlppi dataset.
        for key, group in ppi_ds.groupby("time_hour_and_min"):
            wind_obj = act.retrievals.compute_winds_from_ppi(
                group, 
                # remove_all_missing=True, 
                # snr_threshold=0.008
            )
            objs.append(wind_obj)
        wind_obj = xr.merge(objs)

    src_prof = wind_obj.to_dataframe().reset_index()
    src_prof = src_prof.reset_index().set_index('time').tz_localize("UTC").tz_convert("US/Mountain").tz_localize(None).reset_index()
    # get data for a complete, local time, day
    src_prof = src_prof[
        src_prof['time'].dt.day == datetime.datetime.strptime(startdate, DATE_FORMAT).day
    ]

    src_prof = src_prof.query('height <= 2035')
    
    src_prof['day_hour'] = src_prof['time'].dt.strftime('%D %H')
    src_prof['minute'] = src_prof['time'].dt.minute
    src_prof['hour'] = src_prof['time'].dt.hour

    src_prof['hour_group'] = pd.cut(
        src_prof['hour'],
        4,
        labels=['0-5', '6-11', '12-17', '18-23']
    )

    # for hr_group in src_prof.hour_group.
    speed_chart = alt.Chart(src_prof).transform_filter(
        alt.datum.minute==14
    ).mark_circle(size=25).encode(
        alt.X('wind_speed:Q', title='wind speed (m/s)', sort='-y', scale=alt.Scale(domain=[0,25], clamp=True, nice=False)),
        alt.Y('height:Q', scale=alt.Scale(domain=[0,2000], clamp=True)),
        alt.Color("hour:O", scale=alt.Scale(scheme='turbo')),
    ).properties(width=150).facet(
        alt.Column("hour_group:O", sort=['0-5', '6-11', '12-17', '18-23'], title=startdate, header=alt.Header(labelExpr="''"))
    ).resolve_scale(color='independent')

    # for hr_group in src_prof.hour_group.
    wind_dir_chart = alt.Chart(src_prof).transform_filter(
        alt.datum.minute==14
    ).mark_circle(size=25).encode(
        alt.X('wind_direction:Q', sort='-y', axis=alt.Axis(values=[0, 90, 180, 270, 360]), scale=alt.Scale(domain=[0,360], clamp=True, nice=False)),
        alt.Y('height:Q', scale=alt.Scale(domain=[0,2000], clamp=True)),
        alt.Color("hour:O", scale=alt.Scale(scheme='turbo')),
    ).properties(width=150).facet(
        alt.Column("hour_group:O", sort=['0-5', '6-11', '12-17', '18-23'], title=startdate, header=alt.Header(labelExpr="''"))
    ).resolve_scale(color='independent')
    
    # for hr_group in src_prof.hour_group.

    speed_chart_path = os.path.join(output_path, f"{startdate}-speed.png")
    print(f"Saving: {speed_chart_path}")
    altair_saver.save(speed_chart, speed_chart_path, vega_cli_options=["-s 4"])

    wind_dir_chart_path = os.path.join(output_path, f"{startdate}-direction.png")
    print(f"Saving: {wind_dir_chart_path}")
    altair_saver.save(wind_dir_chart, wind_dir_chart_path, vega_cli_options=["-s 4"])


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
        help="Date you want data from, in format '%Y-%m-%d'."
    )
    args = parser.parse_args()
    create_dl_plots(
        args.output_path,
        args.date
    )


# %%
if __name__ == "__main__":
    main()
