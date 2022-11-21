import pandas as pd 
import datetime
import xarray as xr
import os
import urllib

def download_sos_data_day(date = '20221101', local_download_dir = 'sosnoqc'):
    """Download a netcdf file from the ftp url provided by the Earth Observing Laboratory at NCAR.

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
    if any([prefix in name for prefix in ['dir_1m_','dir_2m_','dir_3m_','dir_5m_','dir_10m_','dir_15m_','dir_20m_']]):
        return 'wind direction'
    elif any([prefix in name for prefix in ['spd_1m_', 'spd_2m_', 'spd_3m_', 'spd_5m_', 'spd_10m_', 'spd_15m_', 'spd_20m_']]):
        return 'wind speed'
    elif any([prefix in name for prefix in ['u_1m_','u_2m_','u_3m_','u_5m_','u_10m_','u_15m_','u_20m_']]):
        return 'u'
    elif any([prefix in name for prefix in ['v_1m_','v_2m_','v_3m_','v_5m_','v_10m_','v_15m_','v_20m_']]):
        return 'v'
    elif any([prefix in name for prefix in ['u_w__1m_','u_w__2m_','u_w__3m_','u_w__5m_','u_w__10m_','u_w__15m_','u_w__20m_']]):
        return 'u_w_'
    elif any([prefix in name for prefix in ['v_w__1m_','v_w__2m_','v_w__3m_','v_w__5m_','v_w__10m_','v_w__15m_','v_w__20m_']]):
        return 'v_w_'
    elif any([prefix in name for prefix in ['u_tc__1m_','u_tc__2m_','u_tc__3m_','u_tc__5m_','u_tc__10m_','u_tc__15m_','u_tc__20m_']]):
        return 'u_tc_'
    elif any([prefix in name for prefix in ['v_tc__1m_','v_tc__2m_','v_tc__3m_','v_tc__5m_','v_tc__10m_','v_tc__15m_','v_tc__20m_']]):
        return 'v_tc_'
    elif any([prefix in name for prefix in ['w_tc__1m_','w_tc__2m_','w_tc__3m_','w_tc__5m_','w_tc__10m_','w_tc__15m_','w_tc__20m_']]):
        return 'w_tc_'
    elif any([prefix in name for prefix in ['w_h2o__1m_','w_h2o__2m_','w_h2o__3m_','w_h2o__5m_','w_h2o__10m_','w_h2o__15m_','w_h2o__20m_']]):
        return 'w_h2o_'
    elif any([prefix in name for prefix in ['T_1m_', 'T_2m_', 'T_3m_', 'T_4m_', 'T_5m_', 'T_6m_', 'T_7m_', 'T_8m_', 'T_9m_', 'T_10m_', 'T_11m_', 'T_12m_', 'T_13m_', 'T_14m_', 'T_15m_', 'T_16m_', 'T_17m_', 'T_18m_', 'T_19m_', 'T_20m_']]):
        return 'T'
    elif any([prefix in name for prefix in ['Tsoil_3_1cm_d', 'Tsoil_8_1cm_d', 'Tsoil_18_1cm_d', 'Tsoil_28_1cm_d', 'Tsoil_4_4cm_d', 'Tsoil_9_4cm_d', 'Tsoil_19_4cm_d', 'Tsoil_29_4cm_d', 'Tsoil_0_6cm_d',  'Tsoil_10_6cm_d', 'Tsoil_20_6cm_d', 'Tsoil_30_6cm_d', 'Tsoil_1_9cm_d', 'Tsoil_11_9cm_d', 'Tsoil_21_9cm_d', 'Tsoil_31_9cm_d']]):
        return 'soil temperature'
    elif 'T_' in name:
        return 'temperature'
    elif name == 'Gsoil_d':
        return 'ground heat flux'
    elif name == 'Qsoil_d':   
        return 'soil moisture'
    elif name == 'Rsw_in_9m_d':
        return 'shortwave radiation incoming'
    elif name == 'Rsw_out_9m_d':
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