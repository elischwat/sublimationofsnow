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
import altair as alt
import altair_saver
alt.data_transformers.disable_max_rows()

# %%
USERNAME = os.getenv("ARM_USERNAME")
TOKEN = os.getenv("ARM_TOKEN")
SAIL_DATA_STREAM = 'gucdlprofwind4newsM1.c1'
DATA_STREAM_FILEEXT = '.nc'


# %%
def create_dl_plots(output_path, date):
    startdate = date
    enddate = date

    with TemporaryDirectory() as temp_dir:
        act.discovery.download_data(USERNAME, TOKEN, SAIL_DATA_STREAM, startdate, enddate, output=temp_dir)
        dl_prof_files = glob.glob(''.join([temp_dir, '/', SAIL_DATA_STREAM, '*'+DATA_STREAM_FILEEXT]))
        print(len(dl_prof_files))
        dl_prof = act.io.armfiles.read_netcdf(dl_prof_files)
    
    src_prof = dl_prof.to_dataframe()
    src_prof = src_prof.reset_index().set_index('time').tz_localize("UTC").tz_convert("US/Mountain").tz_localize(None)

    src_prof = src_prof.reset_index()[['time', 'height', 'bound', 'scan_duration', 'elevation_angle', 'wind_speed', 'wind_speed_error',
        'wind_direction', 'wind_direction_error', 'mean_snr', 'snr_threshold']].query('bound==0')
    
    src_prof['day_hour'] = src_prof['time'].dt.strftime('%D %H')
    src_prof['minute'] = src_prof['time'].dt.minute
    src_prof['hour'] = src_prof['time'].dt.hour

    src_prof['hour_group'] = pd.cut(
        src_prof['hour'],
        [-1, 5, 11, 16, 24],
        labels=['0-5', '6-11', '12-16', '17-23']
    )

    # for hr_group in src_prof.hour_group.
    speed_chart = alt.Chart(src_prof).transform_filter(
        alt.FieldOneOfPredicate('minute', [0,1])
    ).mark_line().encode(
        alt.X('wind_speed:Q', sort='-y', scale=alt.Scale(domain=[0,25], clamp=True)),
        alt.Y('height:Q', scale=alt.Scale(domain=[0,2000], clamp=True)),
        alt.Color("hour:O", scale=alt.Scale(scheme='turbo')),
    ).properties(width=150).facet(
        alt.Column("hour_group:O", sort=['17-23', '0-5', '6-11', '12-16'])
    ).resolve_scale(color='independent')

    # for hr_group in src_prof.hour_group.
    wind_dir_chart = alt.Chart(src_prof).transform_filter(
        alt.FieldOneOfPredicate('minute', [0,1])
    ).mark_circle(size=25).encode(
        alt.X('wind_direction:Q', sort='-y', scale=alt.Scale(domain=[0,360], nice=False)),
        alt.Y('height:Q', scale=alt.Scale(domain=[0,2000], clamp=True)),
        alt.Color("hour:O", scale=alt.Scale(scheme='turbo')),
        alt.Facet("hour_group:O", sort=['17-23', '0-5', '6-11', '12-16'])
    ).properties(width=150).resolve_scale(color='independent')# for hr_group in src_prof.hour_group.
    wind_dir_chart = alt.Chart(src_prof).transform_filter(
        alt.FieldOneOfPredicate('minute', [0,1])
    ).mark_circle(size=25).encode(
        alt.X('wind_direction:Q', sort='-y', scale=alt.Scale(domain=[0,360], nice=False)),
        alt.Y('height:Q', scale=alt.Scale(domain=[0,2000], clamp=True)),
        alt.Color("hour:O", scale=alt.Scale(scheme='turbo')),
        alt.Facet("hour_group:O", sort=['17-23', '0-5', '6-11', '12-16'])
    ).properties(width=150).resolve_scale(color='independent')

    speed_chart_path = os.path.join(output_path, f"{date}-speed.png")
    print(f"Saving: {speed_chart_path}")
    altair_saver.save(speed_chart, speed_chart_path, vega_cli_options=["-s 4"])

    wind_dir_chart_path = os.path.join(output_path, f"{date}-direction.png")
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
        default=(datetime.datetime.today() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
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