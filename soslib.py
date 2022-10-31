import pandas as pd 
import datetime

def time_from_day_and_hhmm(
    day,
    hhmm,
    base_day = datetime.datetime(2022, 1, 1)
):
    """ Get a datetime object from an integer day and a an integer hhmm

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