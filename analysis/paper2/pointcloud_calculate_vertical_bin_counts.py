"""Opens all lidars from each 5-minute period, corrects them into a properly-vertical coordinate system, 
divides vertical coordinate into 10cm bins, and calculates number of points. Saves to file.

Returns:
    _type_: _description_
"""
import os
import glob
import numpy as np

# Suppress future warnings - must go BEFORE pandas import
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import pandas as pd
import datetime as dt

import correct_lidar
from tqdm import tqdm
from joblib import Parallel, delayed

# PARALLELISM = 8
# exclude_scanners = ['l4']
# lidar_base_path = "/Users/elischwat/Development/data/sublimationofsnow/lidar_raw/"
# dates = ['2022-12-12']

PARALLELISM = 10
exclude_scanners = ['l4']
lidar_base_path = "/Volumes/LaCie/lidar/"
dates = [
    '2022-12-12', # cloudy and warm overnight, becoming windy
    '2022-12-13', # Cloudy, windy and cold for the day with only a trace of new snow
    '2022-12-14', # cloudy, windy with a mild low temperature, then a slow but some gradual clearing through the morning but windy with very slow warming through the day.

    '2022-12-21',  # "cloudy and mild overnight with only very light, scattered snow.  Some clearing just after sunrise with wind picking up a bit early, then cloudy with much stronger wind mid-morning.  Wind becomes strong midafternoon as it turns nasty but with only very light snow"
    '2022-12-22',  # "dangerously strong wind, cooling temperature though only very light snow. Blizzard conditions exist overnight and into the morning"

    '2022-12-31', # Snow stops near sunrise then starts back 2 hours later but very light and off and on through the morning, starting back up in the afternoon, generally light but with short heavy periods.
    '2023-01-01', # "cloudy with moderate snow, and very dense, overnight while staying warm and calm.  Dry and calm through mid-morning with very light snow starting midday through the afternoon with little snow but again very dense"
    '2023-01-02', #  cloudy and mild with light to moderate wind at times and a bit of clearing near sunrise.  Sadly, that little bit of clearing (or maybe it was just in my mind?) was short lived with light snow to follow early morning but ending in just an about an hour before some actual clearing midafternoon.
    '2023-01-03', # cloudy and mild overnight with light snow much of the night, most coming after midnight, with no wind.  Cloudy and cool with only occasional very light off and on snow through the day and no wind until it starts up moderately strong in the afternoon turning things nasty.

    '2023-01-06', # cloudy with snow starting after midnight, light at first but picking up towards sunrise.  Generally moderate snow through midmorning before letting up, and then picking up some in the afternoon, heavy at times late afternoon, but thankfully with only light wind.  Snow stops at sunset.

    '2023-01-08', # clear and cool overnights before clouds move in a couple hours before sunrise, then clearing soon after with a gradual warming to just reach freezing on a calm and partly cloudy day.

    '2023-01-09', # partly to mostly cloudy and mild overnight, then clearing and cooling at sunrise before clouds build back becoming cloudy by late morning and remaining so through the day while calm and mild.

]



def get_vertical_bins_for_file(this_date_base_path, date, hour, minute):

    # Gather all paths for this timestamp (5 minute period)
    all_paths_this_timestamp = glob.glob(
        f"{this_date_base_path}_{hour:02}-{int(minute):02}-*.lvx"
    )

    # Remove paths for scanners we want to exclude (set with `exclude_scanners` variable, above)
    for scanner in exclude_scanners:
        all_paths_this_timestamp = [
            file for file in all_paths_this_timestamp if f"/{scanner}/" not in file
        ]
    if len(all_paths_this_timestamp) == 0:
        return pd.DataFrame()
    else:
        # Load points from all the scan files for this timestamp
        loaded_points = []
        for path in all_paths_this_timestamp:
            lidar_str = path.split(lidar_base_path)[1].split('/')[0]
            points = correct_lidar.load_file(path, lidar_str)
            points = points.T
            loaded_points.append(
                points   
            )
        combined_points = np.concatenate(loaded_points)

        # Put points into a dataframe, assign time columns
        columns_convert = {0: 'x', 1: 'y', 2: 'z'}
        df = pd.DataFrame(combined_points).rename(columns=columns_convert)
        
        # Separate into 10cm vertical bins, naming the bins using the mean height
        df['z_bin'] = pd.cut(
            df['z'], 
            np.linspace(-30, 1, 31*10+1)
        )
        df['z_bin'] = df['z_bin'].apply(lambda x: (x.left + x.right)/2)

        # Group by bins amd calculate point counts 
        df_bin_count = pd.DataFrame(df.groupby('z_bin').size()).rename(columns={0:'count'}).reset_index()
        this_datetime = dt.datetime(
            int(date.split('-')[0]), 
            int(date.split('-')[1]), 
            int(date.split('-')[2]), 
            hour, 
            int(minute)
        )
        df_bin_count = df_bin_count.assign(time = this_datetime)
        return df_bin_count

if __name__ == '__main__':
    # Create list of inputs
    inputs_list = []
    for date in dates:
        this_date_base_path = os.path.join(lidar_base_path, f"*/{date}/{date}")
        for hour in range(0,24):
            for minute in np.linspace(0,55, 12):
                inputs_list.append((this_date_base_path, date, hour, minute))
    
    inputs_list_tqdm = tqdm(inputs_list)

    # THE SLOW WAY
    # processed_results = []
    # for  (this_date_base_path, date, hour, minute) in inputs_list_tqdm:
    #     res = get_vertical_bins_for_file(this_date_base_path, date, hour, minute) 
    #     processed_results.append(res)

    # THE FAST WAY
    processed_results =  Parallel(n_jobs = PARALLELISM)(
        delayed(get_vertical_bins_for_file)(this_date_base_path, date, hour, minute) 
            for  (this_date_base_path, date, hour, minute) in inputs_list_tqdm
    )

    combined_results = pd.concat(processed_results)

    combined_results.to_parquet(f"binned_point_counts_{dt.datetime.now().timestamp()}.parquet")


