import pandas as pd 
import datetime
import xarray as xr
import os
import urllib

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
    if name in ['dir_1m_c','dir_2m_c','dir_3m_c','dir_5m_c','dir_10m_c','dir_15m_c','dir_20m_c']:
        return 'wind direction'
    elif name in ['spd_1m_c', 'spd_2m_c', 'spd_3m_c', 'spd_5m_c', 'spd_10m_c', 'spd_15m_c', 'spd_20m_c']:
        return 'wind speed'
    elif 'T_' in name:
        return 'temperature'
    elif name in ['u_1m_c','u_2m_c','u_3m_c','u_5m_c','u_10m_c','u_15m_c','u_20m_c']:
        return 'u'
    elif name in ['v_1m_c','v_2m_c','v_3m_c','v_5m_c','v_10m_c','v_15m_c','v_20m_c']:
        return 'v'
    elif name in ['u_w__1m_c','u_w__2m_c','u_w__3m_c','u_w__5m_c','u_w__10m_c','u_w__15m_c','u_w__20m_c']:
        return 'u_w_'
    elif name in ['v_w__1m_c','v_w__2m_c','v_w__3m_c','v_w__5m_c','v_w__10m_c','v_w__15m_c','v_w__20m_c']:
        return 'v_w_'
    elif name in ['u_tc__1m_c','u_tc__2m_c','u_tc__3m_c','u_tc__5m_c','u_tc__10m_c','u_tc__15m_c','u_tc__20m_c']:
        return 'u_tc_'
    elif name in ['v_tc__1m_c','v_tc__2m_c','v_tc__3m_c','v_tc__5m_c','v_tc__10m_c','v_tc__15m_c','v_tc__20m_c']:
        return 'v_tc_'
    elif name in ['w_tc__1m_c','w_tc__2m_c','w_tc__3m_c','w_tc__5m_c','w_tc__10m_c','w_tc__15m_c','w_tc__20m_c']:
        return 'w_tc_'
    elif name in ['w_h2o__1m_c','w_h2o__2m_c','w_h2o__3m_c','w_h2o__5m_c','w_h2o__10m_c','w_h2o__15m_c','w_h2o__20m_c']:
        return 'w_h2o_'
    elif name in [
        'T_1m_c',
        'T_2m_c',
        'T_3m_c',
        'T_4m_c',
        'T_5m_c',
        'T_6m_c',
        'T_7m_c',
        'T_8m_c',
        'T_9m_c',
        'T_10m_c',
        'T_11m_c',
        'T_12m_c',
        'T_13m_c',
        'T_14m_c',
        'T_15m_c',
        'T_16m_c',
        'T_17m_c',
        'T_18m_c',
        'T_19m_c',
        'T_20m_c'
     ]:
        return 'T'

def download_sos_data_day(date = '20221101', local_download_dir = 'sosnoqc'):

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