# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: sublimationofsnow
#     language: python
#     name: python3
# ---

# %%
import act
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import glob

# %%
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units

# %% [markdown]
# date_start = input("One day of sondes.\nDate and Time for Radiosonde Start (format YYYY-MM-DD): ")
# date_end = input("Date and Time for Radiosonde End (format YYYY-MM-DD): ")
# one_or_two = input("Y/N for two plots to be produced: ")

# %%
date_start = "2023-04-16"
date_end = "2023-04-16"
one_or_two = "Y"


# %%
# Personal access necessary for downloading from the ARM portal, need an account to due so
username = os.getenv("ARM_USERNAME")
token = os.getenv("ARM_TOKEN")
radiosonde ='gucsondewnpnM1.b1'

# %%
start = date_start[0:10]
end = date_end[0:10]

# %%

# %%
p = sonde1.pres.values * units.hPa
t = sonde1.tdry.values * units.degC
td = mpcalc.dewpoint_from_relative_humidity(sonde1.tdry,sonde1.rh).values
u = sonde1.u_wind
v = sonde1.v_wind

# %%
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], t[0], td[0])


# %%
def plot_skewT(ds):
    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=30)

    p = ds.pres.values * units.hPa
    t = ds.tdry.values * units.degC
    td = mpcalc.dewpoint_from_relative_humidity(ds.tdry,ds.rh)
    u = ds.u_wind
    v = ds.v_wind

    # Calculate the LCL
    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], t[0], td[0])

    # Calculate the parcel profile.
    parcel_prof = mpcalc.parcel_profile(p, t[0], td[0]).to("degC")

    # Set spacing interval--Every 50 mb from pmax to 100 mb
    plot_interval = np.arange(100, max(p.magnitude), 50) * units('hPa')
    # Get indexes of values closest to defined interval
    ix = mpcalc.resample_nn_1d(p, plot_interval)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, t, 'r')
    skew.plot(p, td, 'g')
    skew.plot_barbs(p[ix], u[ix], v[ix])
    skew.ax.set_ylim(max(p), 100)
    skew.ax.set_xlim(min(t.magnitude)-10, max(t.magnitude)+40)

    # Plot LCL as black dot
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

    # Plot the parcel profile as a black line
    skew.plot(p, parcel_prof, 'k', linewidth=2)

    # Shade areas of CAPE and CIN
    skew.shade_cin(p, t, parcel_prof, td)
    skew.shade_cape(p, t, parcel_prof)

    # Plot dendritic growth zone
    skew.ax.axvline(-13, color='c', linestyle='--', linewidth=2)
    skew.ax.axvline(-19, color='c', linestyle='--', linewidth=2)
    skew.shade_area(p, -13,-19, color='c', alpha=0.3, label='DGZ')

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines(pressure=p)
    plt.legend(loc='upper left')

    # Create a hodograph
    # Create an inset axes object that is 40% width and height of the
    # figure and put it in the upper right hand corner.
    ax_hod = inset_axes(skew.ax, '40%', '40%', loc=1)
    h = Hodograph(ax_hod, component_range=60.,)
    h.add_grid(increment=20)
    h.plot_colormapped(u[ix], v[ix], ds.wspd[ix])  # Plot a line colored by wind speed

    start_hour = ds.time.dt.hour[0]
    # # Save the plot

    # plt.savefig(f'../figures/radiosondes/SAIL_sonde_{start}_{start_hour}UTC.png')
    # # Show the plot
    plt.show()
    return

# %%
# Download SAIL sonde data
# try:
sonde1_start = dt.datetime.strptime(date_start+'T11:00:00','%Y-%m-%dT%H:%M:%S')
sonde1_end = sonde1_start + dt.timedelta(hours=6)  


# %%
from tempfile import TemporaryDirectory
with TemporaryDirectory() as temp_dir:
    act.discovery.download_arm_data(
        username,    token,    radiosonde,    start, end,
        output = temp_dir
    )
    sonde_ds = act.io.read_arm_netcdf(glob.glob(os.path.join(temp_dir, '*.cdf')))

# %%
sonde1 = sonde_ds.sel(time=slice(sonde1_start,sonde1_end))

# %%
plot_skewT(sonde1)
if one_or_two == "Y":
    sonde2_start = sonde1_end + dt.timedelta(hours=6)
    sonde2_end = sonde2_start + dt.timedelta(hours=6)
    sonde2 = sonde_ds.sel(time=slice(sonde2_start,sonde2_end))
    plot_skewT(sonde2)
# except: 
#     print('Data not found, may not be loaded yet.')

# %%
