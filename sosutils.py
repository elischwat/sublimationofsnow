import pandas as pd 
import datetime
import xarray as xr
import os
import urllib
import geopy
import geopy.distance
import shapely.geometry
import py3dep
import pytz
import numpy as np
import rasterio
import geopandas as gpd
from sklearn import linear_model

from astral import LocationInfo
from astral.sun import sun

ROTATION_SUPPORTED_MEASUREMENTS = [
     'u',     'v',     
     'u_w_',     'v_w_', 
     'u_tc_',     'v_tc_',    
    'u_h2o_',     'v_h2o_',    
    ]


def download_sos_highrate_data_hour(date = '20221031', hour = '00', local_download_dir = 'hr_noqc_geo'):
    """Download a netcdf file from the ftp url provided by the Earth Observing Laboratory at NCAR.
    Data is the high-rate, 20hz data.

    Args:
        date (str, optional): String date. Defaults to '20221101'.
        hour (str, optional): String hour, two digits. Defaults to '00'.
        local_download_dir (str, optional): _description_. Defaults to 'hr_noqc_geo'.
    """
    base_url = 'ftp.eol.ucar.edu'
    path = 'pub/archive/isfs/projects/SOS/netcdf/hr_noqc_geo'
    file_example = f'isfs_geo_hr_{date}_{hour}.nc'

    os.makedirs(local_download_dir, exist_ok=True)

    full_file_path = os.path.join('ftp://', base_url, path, file_example)
    download_file_path = os.path.join(local_download_dir, file_example)

    urllib.request.urlretrieve(
        full_file_path,
        download_file_path   
    )
    
    return download_file_path



def download_sos_data_day(date = '20221101', local_download_dir = 'sosnoqc'):
    """Download a netcdf file from the ftp url provided by the Earth Observing Laboratory at NCAR.
    Data is the daily data reynolds averaged to 5 minutes.

    Args:
        date (str, optional): String version of a date. Defaults to '20221101'.
        local_download_dir (str, optional): Directory to which files will be downloaded. Defaults to 'sosnoqc'; this directory will be created if it does not already exist.

    Returns:
        _type_: _description_
    """
    base_url = 'ftp.eol.ucar.edu'
    path = 'pub/archive/isfs/projects/SOS/netcdf/noqc_geo'
    file_example = f'isfs_{date}.nc'

    os.makedirs(local_download_dir, exist_ok=True)

    full_file_path = os.path.join('ftp://', base_url, path, file_example)
    download_file_path = os.path.join(local_download_dir, file_example)

    urllib.request.urlretrieve(
        full_file_path,
        download_file_path   
    )

    return download_file_path

def open_datasets_as_dataframe(file_list, variables = None):
    """Create a dataframe from a list of files. Files have extension ".nc" and are from the EOL lab's ftp address

    ftp://ftp.eol.ucar.edu/pub/archive/isfs/projects/SOS/netcdf/noqc_geo

    Args:
        file_list (_type_): _description_
    """
    if variables:
        return pd.concat([
            xr.open_dataset(f)[variables].to_dataframe().reset_index() for f in file_list
        ])
    else:
        return pd.concat([
            xr.open_dataset(f).to_dataframe().reset_index() for f in file_list
        ])

def height_from_variable_name(name):
    """Parse instrument/sensor height from EOL variable names.

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    if '_1m_' in name:
        return 1
    elif '_2m_' in name:
        return 2
    elif '_3m_' in name:
        return 3
    elif '_4m_' in name:
        return 4
    elif '_5m_' in name:
        return 5
    elif '_6m_' in name:
        return 6
    elif '_7m_' in name:
        return 7
    elif '_8m_' in name:
        return 8
    elif '_9m_' in name:
        return 9
    elif '_10m_' in name:
        return 10
    elif '_11m_' in name:
        return 11
    elif '_12m_' in name:
        return 12
    elif '_13m_' in name:
        return 13
    elif '_14m_' in name:
        return 14
    elif '_15m_' in name:
        return 15
    elif '_16m_' in name:
        return 16
    elif '_17m_' in name:
        return 17
    elif '_18m_' in name:
        return 18
    elif '_19m_' in name:
        return 19
    elif '_20m_' in name:
        return 20
    # handle the soil moisture depths
    elif '_0_6cm' in name:
        return -0.006
    elif '_1_9cm' in name:
        return -0.019
    elif '_3_1cm' in name:
        return -0.031
    elif '_4_4cm' in name:
        return -0.044
    elif '_8_1cm' in name:
        return -0.081
    elif '_9_4cm' in name:
        return -0.094
    elif '_10_6cm' in name:
        return -.106
    elif '_11_9cm' in name:
        return -.119
    elif '_18_1cm' in name:
        return -.181
    elif '_19_4cm' in name:
        return -.194
    elif '_20_6cm' in name:
        return -.206
    elif '_21_9cm' in name:
        return -.219
    elif '_28_1cm' in name:
        return -.281
    elif '_29_4cm' in name:
        return -.294
    elif '_30_6cm' in name:
        return -.306
    elif '_31_9cm' in name:
        return -.319

def tower_from_variable_name(name):
    """Parse instrument/sensor tower from EOL variable names.

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    if name.endswith('_d'):
        return 'd'
    elif name.endswith('_c'):
        return 'c'
    elif name.endswith('_ue'):
        return 'ue'
    elif name.endswith('uw'):
        return 'uw'

def measurement_from_variable_name(name):
    """Provide plain text measurement name from EOL variable names.

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    if any([prefix in name for prefix in ['P_10m_', 'P_20m_']]):
        return 'pressure'
    elif any([prefix in name for prefix in ['dir_1m_','dir_2m_','dir_3m_','dir_5m_','dir_10m_','dir_15m_','dir_20m_']]):
        return 'wind direction'
    elif any([prefix in name for prefix in ['spd_1m_', 'spd_2m_', 'spd_3m_', 'spd_5m_', 'spd_10m_', 'spd_15m_', 'spd_20m_']]):
        return 'wind speed'
    elif any([prefix in name for prefix in ['u_1m_','u_2m_','u_3m_','u_5m_','u_10m_','u_15m_','u_20m_']]):
        return 'u'
    elif any([prefix in name for prefix in ['v_1m_','v_2m_','v_3m_','v_5m_','v_10m_','v_15m_','v_20m_']]):
        return 'v'
    elif any([prefix in name for prefix in ['w_1m_','w_2m_','w_3m_','w_5m_','w_10m_','w_15m_','w_20m_']]):
        return 'w'
    elif any([prefix in name for prefix in ['u_w__1m_','u_w__2m_','u_w__3m_','u_w__5m_','u_w__10m_','u_w__15m_','u_w__20m_']]):
        return 'u_w_'
    elif any([prefix in name for prefix in ['v_w__1m_','v_w__2m_','v_w__3m_','v_w__5m_','v_w__10m_','v_w__15m_','v_w__20m_']]):
        return 'v_w_'
    elif any([prefix in name for prefix in ['u_v__1m_','u_v__2m_','u_v__3m_','u_v__5m_','u_v__10m_','u_v__15m_','u_v__20m_']]):
        return 'u_v_'
    elif any([prefix in name for prefix in ['u_tc__1m_','u_tc__2m_','u_tc__3m_','u_tc__5m_','u_tc__10m_','u_tc__15m_','u_tc__20m_']]):
        return 'u_tc_'
    elif any([prefix in name for prefix in ['v_tc__1m_','v_tc__2m_','v_tc__3m_','v_tc__5m_','v_tc__10m_','v_tc__15m_','v_tc__20m_']]):
        return 'v_tc_'
    elif any([prefix in name for prefix in ['w_tc__1m_','w_tc__2m_','w_tc__3m_','w_tc__5m_','w_tc__10m_','w_tc__15m_','w_tc__20m_']]):
        return 'w_tc_'
    elif any([prefix in name for prefix in ['u_h2o__1m_','u_h2o__2m_','u_h2o__3m_','u_h2o__5m_','u_h2o__10m_','u_h2o__15m_','u_h2o__20m_']]):
        return 'u_h2o_'
    elif any([prefix in name for prefix in ['v_h2o__1m_','v_h2o__2m_','v_h2o__3m_','v_h2o__5m_','v_h2o__10m_','v_h2o__15m_','v_h2o__20m_']]):
        return 'v_h2o_'
    elif any([prefix in name for prefix in ['w_h2o__1m_','w_h2o__2m_','w_h2o__3m_','w_h2o__5m_','w_h2o__10m_','w_h2o__15m_','w_h2o__20m_']]):
        return 'w_h2o_'
    elif any([prefix in name for prefix in ['T_1m_', 'T_2m_', 'T_3m_', 'T_4m_', 'T_5m_', 'T_6m_', 'T_7m_', 'T_8m_', 'T_9m_', 'T_10m_', 'T_11m_', 'T_12m_', 'T_13m_', 'T_14m_', 'T_15m_', 'T_16m_', 'T_17m_', 'T_18m_', 'T_19m_', 'T_20m_']]):
        return 'temperature'
    elif any([prefix in name for prefix in ['RH_1m_', 'RH_2m_', 'RH_3m_', 'RH_4m_', 'RH_5m_', 'RH_6m_', 'RH_7m_', 'RH_8m_', 'RH_9m_', 'RH_10m_', 'RH_11m_', 'RH_12m_', 'RH_13m_', 'RH_14m_', 'RH_15m_', 'RH_16m_', 'RH_17m_', 'RH_18m_', 'RH_19m_', 'RH_20m_']]):
        return 'RH'
    elif any([prefix in name for prefix in ['tc_1m', 'tc_2m', 'tc_3m', 'tc_5m', 'tc_10m', 'tc_15m', 'tc_20m']]):
        return 'virtual temperature'
    # NOTE: Tpot IS NOT A SOSNOQC VARIABLE NAME
    elif any([prefix in name for prefix in ['Tpot_1m_', 'Tpot_2m_', 'Tpot_3m_', 'Tpot_4m_', 'Tpot_5m_', 'Tpot_6m_', 'Tpot_7m_', 'Tpot_8m_', 'Tpot_9m_', 'Tpot_10m_', 'Tpot_11m_', 'Tpot_12m_', 'Tpot_13m_', 'Tpot_14m_', 'Tpot_15m_', 'Tpot_16m_', 'Tpot_17m_', 'Tpot_18m_', 'Tpot_19m_', 'Tpot_20m_']]):
        return 'potential temperature'
    elif any([prefix in name for prefix in ['Tsoil_3_1cm_d', 'Tsoil_8_1cm_d', 'Tsoil_18_1cm_d', 'Tsoil_28_1cm_d', 'Tsoil_4_4cm_d', 'Tsoil_9_4cm_d', 'Tsoil_19_4cm_d', 'Tsoil_29_4cm_d', 'Tsoil_0_6cm_d',  'Tsoil_10_6cm_d', 'Tsoil_20_6cm_d', 'Tsoil_30_6cm_d', 'Tsoil_1_9cm_d', 'Tsoil_11_9cm_d', 'Tsoil_21_9cm_d', 'Tsoil_31_9cm_d']]):
        return 'soil temperature'
    elif name == 'Gsoil_d':
        return 'ground heat flux'
    elif name == 'Qsoil_d':   
        return 'soil moisture'
    elif name == 'Rsw_in_9m_d':
        return 'shortwave radiation incoming'
    elif name == 'Rsw_out_9m_d':
        return 'shortwave radiation outgoing'
    # these following Rlw variables do not actually come in the NCAR provided datasets, but are consistent with their naming schema
    elif name == 'Rlw_in_9m_d':
        return 'shortwave radiation incoming'
    elif name == 'Rlw_out_9m_d':
        return 'shortwave radiation outgoing'

        
def time_from_day_and_hhmm(
    day,
    hhmm,
    base_day = datetime.datetime(2022, 1, 1)
):
    """ Get a datetime object from an integer day and a an integer hhmm

    This was created for ingesting data from the Stretch (Tilden's) tower data.

    Args:
        day (int): a number indicating an ordinal day
        hhmm (int): an integer indicating the hour and minute during a single day. E.G., 12:01am is 1, 1:01am is 101, 1:32pm is 1332
        base_day (_type_, optional): _description_. Defaults to datetime.datetime(2022, 1, 1).

    Returns:
        _type_: _description_
    """
    hours = float(str(int(hhmm)).zfill(4)[:2])
    minutes = float(str(int(hhmm)).zfill(4)[2:])
    return base_day \
        + datetime.timedelta(days = day - 1) \
        + datetime.timedelta(hours = hours) \
        + datetime.timedelta(minutes = minutes)

# if already downloaded

def merge_datasets_with_different_variables(ds_list, dim):
    """ This gets slow with lots of datasets

    Args:
        ds_list (_type_): _description_
        dim (_type_): _description_
    """
    def _merge_datasets_with_different_variables(ds1, ds2, dim):
        vars1 = set(ds1.data_vars)
        vars2 = set(ds2.data_vars)
        in1_notin2 = vars1.difference(vars2)
        in2_notin1 = vars2.difference(vars1)
        # add vars with NaN values to ds1
        for v in in2_notin1:
            ds1[v] = xr.DataArray(coords=ds1.coords, dims=ds1.dims)
        # add vars with NaN values to ds2
        for v in in1_notin2:
            ds2[v] = xr.DataArray(coords=ds2.coords, dims=ds2.dims)
        return xr.concat([ds1, ds2], dim=dim)

    new_ds = ds_list.pop(0)
    while ds_list:
        new_ds = _merge_datasets_with_different_variables(
            new_ds,
            ds_list.pop(0),
            dim=dim
        )
    return new_ds

def modify_df_timezone(df, source_tz, target_tz):
    df = df.copy()
    df['time'] = df['time'].dt.tz_localize(source_tz).dt.tz_convert(target_tz).dt.tz_localize(None)
    return df

def modify_xarray_timezone(ds, source_tz, target_tz):
    time_utc = ds['time'].to_index().tz_localize(source_tz)
    tz_corrected = time_utc.tz_convert(target_tz).tz_localize(None)
    local_da=xr.DataArray.from_series(tz_corrected)
    ds.coords.update({'local_time':tz_corrected})
    return ds

def get_linestring(lon, lat, bearing, radius):
    radar_location = geopy.Point(lat, lon)
    radar_elevation = py3dep.elevation_bycoords(
        [(radar_location.longitude, radar_location.latitude)]
    )[0]

    positive_distance = geopy.distance.distance(
        kilometers=radius
    ).destination(
        point=radar_location, 
        bearing=bearing
    )
    negitive_distance = geopy.distance.distance(
        kilometers=radius
    ).destination(
        point=radar_location, 
        bearing=bearing-180
    )
    return shapely.geometry.LineString([
        shapely.geometry.Point(negitive_distance.longitude, negitive_distance.latitude),
        shapely.geometry.Point(radar_location.longitude, radar_location.latitude),
        shapely.geometry.Point(positive_distance.longitude, positive_distance.latitude),
        
    ])


def get_radar_scan_ground_profile(lon, lat, bearing, radius, spacing = 10):
    """Returns the ground profile relative to the provided point, i.e. the elevation at the
    provided lon/lat is equal to 0 in the returned dataset.

    Args:
        lon (_type_): _description_
        lat (_type_): _description_
        bearing (_type_): _description_
        radius (_type_): _description_
        spacing (int, optional): _description_. Defaults to 10.
    """
    radar_location = geopy.Point(lat, lon)
    radar_elevation = py3dep.elevation_bycoords(
        [(radar_location.longitude, radar_location.latitude)]
    )[0]
    line = get_linestring(lon, lat, bearing, radius)

    elevation_profile = py3dep.elevation_profile(line, spacing=spacing, crs='EPSG:4326')
    elevation_profile.values = elevation_profile.values - radar_elevation
    elevation_profile_df = elevation_profile.to_dataframe().reset_index()
    elevation_profile_df['distance'] = elevation_profile_df['distance'] - radius*1000
    elevation_profile_df['zero'] = 0

    return elevation_profile_df

def get_radar_scan_ground_profile_from_raster(dem_file, lon, lat, bearing, radius, n_points):
    """Returns the ground profile relative to the provided point, i.e. the elevation at the
    provided lon/lat is equal to 0 in the returned dataset.

    Args:
        lon (_type_): _description_
        lat (_type_): _description_
        bearing (_type_): _description_
        radius (float): radius in kilometers
        spacing (int, optional): _description_. Defaults to 10.
    """
    radar_location = geopy.Point(lat, lon)
    radar_elevation = py3dep.elevation_bycoords(
        [(radar_location.longitude, radar_location.latitude)]
    )[0]
    line = get_linestring(lon, lat, bearing, radius)

    line = gpd.GeoDataFrame(geometry=pd.Series([line])).set_crs('EPSG:4326').to_crs('EPSG:32613').geometry.iloc[0]
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]
    line = gpd.GeoDataFrame(geometry=[shapely.geometry.LineString(points)]).set_crs('EPSG:32613').to_crs('EPSG:4326').geometry.iloc[0]
    points = list(line.coords)
    dem = rasterio.open(dem_file)
    elevation_profile = rasterio.sample.sample_gen(dem, points)
    elevation_profile = [f[0]-radar_elevation for f in elevation_profile]

    return pd.DataFrame({
        'elevation': elevation_profile,
        'distance': pd.Series(distances) - radius*1000
    })


def get_nightime_df(timezone, lat, lon, dates):
    nighttime_df = pd.DataFrame()
    loc = LocationInfo(timezone=timezone, latitude=lat, longitude=lon)
    for date in dates:
        s = sun(loc.observer, date=date, tzinfo=loc.timezone)
        nighttime_df = pd.concat([
            nighttime_df,
            pd.DataFrame({
                'datetime': [date],
                'sunset': [s['sunset']],
                'sunrise': [s['sunrise']],
            } )
        ], ignore_index=True)
    # remove timezone info
    nighttime_df['sunset'] = nighttime_df['sunset'].dt.tz_localize(None)
    nighttime_df['sunrise'] = nighttime_df['sunrise'].dt.tz_localize(None)
    # add partial first night:
    nighttime_df = pd.concat([
            pd.DataFrame({
                'datetime': [np.nan],
                'sunset': [np.nan],
                'sunrise': [np.nan],
            }),
            nighttime_df,
            pd.DataFrame({
                'datetime': [np.nan],
                'sunset': [np.nan],
                'sunrise': [np.nan],
            })
        ], ignore_index=True)
    nighttime_df['sunrise'] = nighttime_df['sunrise'].shift(-1)
    nighttime_df.loc[0, 'sunset'] = pd.Series(sun(loc.observer, date=dates[0] - datetime.timedelta(days=1), tzinfo=loc.timezone)['sunset']).dt.tz_localize(None).iloc[0]
    nighttime_df.loc[len(dates), 'sunrise'] = dates[-1] + datetime.timedelta(days=1) 
    return nighttime_df.drop(nighttime_df.tail(1).index)


def get_tidy_dataset(ds, variable_names):
    """Convert an SoS netcdf xr.DataSet into a dataframe with time, height, tower, and measurement as indexes in a tidy dataset.

    Args:
        ds (_type_): Dataset to convert
        variable_names (_type_): Variable names that you want operated on. Variables may not be supported.
    """
    tidy_df = ds[variable_names].to_dataframe().reset_index().melt(id_vars='time', value_vars=variable_names)
    tidy_df['height'] = tidy_df['variable'].apply(height_from_variable_name)
    tidy_df['tower'] = tidy_df['variable'].apply(tower_from_variable_name)
    tidy_df['measurement'] = tidy_df['variable'].apply(measurement_from_variable_name)
    return tidy_df


def calculate_planar_fit(u,v,w):
    # remove nans from u,v,w
    temp_df = pd.DataFrame({'u': u, 'v': v, 'w': w}).dropna()
    u = temp_df['u']
    v = temp_df['v']
    w = temp_df['w']
    
    # Fit, using least-squares, the averaged wind components to the equation of the plane,
    # $ a = -bu - cv + w$
    X_data = np.array([u, v]).reshape((-1,2))
    Y_data = np.array(w)
    reg = linear_model.LinearRegression().fit(X_data, Y_data)
    a = reg.intercept_
    b,c = reg.coef_

    # Now define a new set of coordinates, streamwise coordinates, 
    # <$U_f$, $V_f$, $W_f$>
    # The normal vector to the plane defined by U_f and V_f is Wf, and is calculated
    W_f = np.array([-b, -c, 1]) / np.sqrt( b**2 + c**2 + 1**2 )
    tilt = np.arctan(np.sqrt(b**2 + c**2))
    tiltaz = np.arctan2(-c, -b)
    # Check that these two angles correctly define $W_f$ (floating point errors are ok)
    assert (
        all(W_f - ( np.sin(tilt)*np.cos(tiltaz), np.sin(tilt)*np.sin(tiltaz), np.cos(tilt)) < 10e-7)
    )

    return (a,b,c), (tilt, tiltaz), W_f

def apply_planar_fit(u, v, w, a, W_f):
    # # Define the sonic coordinate system unit vectors
    U_s = np.array([1,0,0])
    V_s = np.array([0,1,0])
    W_s = np.array([0,0,1])

    # Calculate the other axes of the mean flowwise coordinate system
    U_f_normal_vector = np.cross(np.cross(W_f, U_s), W_f)
    U_f = U_f_normal_vector / np.sqrt((U_f_normal_vector**2).sum())
    V_f = np.cross(W_f, U_f)
    
    # Transform velocity measurements in sonic coords to mean flowwise coords
    u_streamwise = np.dot(U_f, np.array([u, v, w - a]))
    v_streamwise = np.dot(V_f, np.array([u, v, w - a]))
    w_streamwise = np.dot(W_f, np.array([u, v, w - a]))

    return (u_streamwise, v_streamwise, w_streamwise)

def calculate_and_apply_planar_fit(u, v, w):
    """Provides planar fit adjusted velocity components for a given dataset of u,v,w.

    Args:
        u (_type_): _description_
        v (_type_): _description_
        w (_type_): _description_

    Returns:
       (float,float,float), (float, float), (np.array, np.array, np.array):
                 (a,b,c), (tilt (radians), tiltaz (rad)), (u_streamwise, v_streamwise, w_streamwise)
    """
    (a,b,c), (tilt, tiltaz), W_f = calculate_planar_fit(u, v, w)
    (u_streamwise, v_streamwise, w_streamwise) = apply_planar_fit(u, v, w, a, W_f)
    return (u_streamwise, v_streamwise, w_streamwise)

    


def streamwise_coordinates_xarray(ds):
    assert type(ds) == xr.DataSet

    """ToDo: Implement for the dataset as it comes in the SOSNOQC format"""

def streamwise_coordinates_single_rotation_tidy_df(original_tidy_df):
    """Modify multiple data variables of a tidy_df created from a SoS netcdf dataset (see 
    `get_tidy_dataset` abozve). Namely, the u and v velocity variables and all first and second 
    covariances. A single rotation is used (no adjustment to the  vertical axis) based on the mean
    u and v velocity over the provided time period. The returned dataset will have all relevant data 
    variables replaced with rotated values.
    Args:
        tidy_df (pd.DataFrame): In the same format as the result of the `get_tidy_dataset` function 
        above.
    """

    assert type(original_tidy_df) == pd.DataFrame

    tidy_df = original_tidy_df.copy()

    u_avg = tidy_df[tidy_df['measurement'] == 'u'].groupby(['tower', 'height'])['value'].mean()
    v_avg = tidy_df[tidy_df['measurement'] == 'v'].groupby(['tower', 'height'])['value'].mean()
    angles = np.arctan2(v_avg, u_avg)
    D = angles.to_dict()
    D

    variable_pairs = [
        ('u', 'v'), 
        ('u_w_', 'v_w_'), 
        ('u_tc_', 'v_tc_'), 
        ('u_h2o_', 'v_h2o_'), 
    ]
    for u_var, v_var in variable_pairs:
        for (tower, height), angle_rad in D.items():
            us = tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == u_var)
                ]['value'].reset_index(drop=True)
            vs = tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == v_var)
                ]['value'].reset_index(drop=True)
            new_u = np.array(us*np.cos(angle_rad) + vs*np.sin(angle_rad))
            new_v = np.array(-us*np.sin(angle_rad) + vs*np.cos(angle_rad))
            tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == u_var), 'value'] = new_u
            tidy_df.loc[
                    (tidy_df['tower'] == tower) & (tidy_df['height'] == height) & (tidy_df['measurement'] == v_var), 'value'] = new_v
    return tidy_df


    # u_avg = tidy_df[tidy_df['measurement'] == 'u'].groupby(['tower', 'height'])['value'].mean()
    # v_avg = tidy_df[tidy_df['measurement'] == 'v'].groupby(['tower', 'height'])['value'].mean()
    # angles = np.arctan(v_avg/u_avg)
    # print(np.rad2deg(angles))
    # D = angles.to_dict()
    
        
    # df_wide = tidy_df.pivot_table(index=['time','tower','height'], values='value', columns='measurement').reset_index()
    
    # # convert u velocity
    # df_wide['u'] = df_wide.apply(
    #     lambda row: row['u']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ) + row['v']*np.sin(
    #             D.get((row['tower'], row['height']), np.nan)
    #     ),
    #     axis=1
    # )

    # # # convert v velocity
    # df_wide['v'] = df_wide.apply(
    #     lambda row: - row['u']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['u_w_'] = df_wide.apply(
    #     lambda row: row['u_w_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)) 
    #     + row['v_w_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ),
    #     axis=1
    # )

    # df_wide['v_w_'] = df_wide.apply(
    #     lambda row: - row['u_w_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ) + row['v_w_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)
    #     ),
    #     axis=1
    # )

    # df_wide['u_tc_'] = df_wide.apply(
    #     lambda row: row['u_tc_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_tc_']*np.sin(
    #             D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['v_tc_'] = df_wide.apply(
    #     lambda row: - row['u_tc_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_tc_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['u_h2o_'] = df_wide.apply(
    #     lambda row: row['u_h2o_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_h2o_']*np.sin(
    #             D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # df_wide['v_h2o_'] = df_wide.apply(
    #     lambda row: - row['u_h2o_']*np.sin(
    #         D.get((row['tower'], row['height']), np.nan)) + row['v_h2o_']*np.cos(
    #         D.get((row['tower'], row['height']), np.nan)),
    #     axis=1
    # )

    # return df_wide.melt(
    #     id_vars=['time', 'tower', 'height'], 
    #     value_vars=tidy_df['measurement'].unique()
    # )