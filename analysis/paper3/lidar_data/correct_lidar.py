#!/usr/bin/env python

# import open3d as o3d
import lvx_reader
import glob
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np


# importlib.reload(time_series)
# d1 = time_series.load_first_file("l1")

#2022-11-04_01
# s1 = time_series.load_date("l1", "2022-11-04",hour="01")

# d1map = time_series.get_height(d1, xr=(-30,30), yr=(-30,30), zmax=-9, title="test", zero=0, get_zero=True, scatter_plot=False)

# s1map = time_series.get_height(s1, xr=(-30,30), yr=(-30,30), zmax=-9, title="test", zero=0, get_zero=True, scatter_plot=False)

pillow_locations = {"l1":{"x":[-3.2,-12.11],"y":[2.9,-1.48],"z":[-11.3,-12.16], "name":["C","UW"]},
                    "l2":{"x":[4.33,7.66],"y":[7.95,-11.55],"z":[-10.791, -12.25], "name":["UE","D"]},
                    "l3":{"x":[-3.0,8.16], "y":[2.6,-11.22], "z":[-11.32,-12.19], "name":["C","D"]},
                    "l4":{"x":[3.26,-4.02,-12.6], "y":[8.28, 2.66,-2.2], "z":[-11.22,-11.61,-12.312], "name":["UE","C","UW"]},
                    "l5":{"x":[3.93,-3.34],"y":[7.95,2.76],"z":[-10.87, -11.32], "name":["UE","C"]},
                    "l6":{"x":[-3.36,-12.3,7.73],"y":[2.83,-0.75,-11.34],"z":[-11.34,-12.15, -12.258], "name":["C","UW","D"]}
                    }


def angle_correction(x_true,y_true, x0,y0, x_obs, y_obs):
    true_angle = np.arctan((y_true-y0) / (x_true-x0))
    obs_angle = np.arctan((y_obs-y0) / (x_obs-x0))
    return np.rad2deg(obs_angle - true_angle)

# lidar_config = {
#     "l1": {"tower":"ue","azimuth":-61.5, "xr":(-5,50),   "yr":(-50,0.), "xoffset":-0.1, "yoffset":0.2, "zoffset":0},
#     "l2": {"tower":"ue","azimuth":0,     "xr":(0,50),    "yr":(-30,30), "xoffset":0,    "yoffset":0,   "zoffset":0},
#     "l3": {"tower":"uw","azimuth":64,    "xr":(3,60),    "yr":(-30,30), "xoffset":19.5, "yoffset":-28, "zoffset":-2.5},
#     "l4": {"tower":"uw","azimuth":120,   "xr":(3,60),    "yr":(-30,30), "xoffset":19.5, "yoffset":-28, "zoffset":-2.5},
#     "l5": {"tower":"d", "azimuth":-179,  "xr":(-40,30),  "yr":(-30,30), "xoffset":33.3, "yoffset":2.63,"zoffset":-2.8},
#     "l6": {"tower":"d", "azimuth":-113,  "xr":(0,40),    "yr":(-40,0.), "xoffset":33.42,"yoffset":2.44,"zoffset":-3.7}
#     }

center_point = 17.75, -8.1 # after referencing all lidars to lidar2 - what does this do exactly? Uses l2 as reference point?
global_rotation = -75 # after centering
# configuration with azimuth, what is elev_adj? rotation? xoffset, yoffset, zoffset?
lidar_config = {
    "l1": {"tower":"ue", "azimuth":-61.0,   "elev_adj":0,    "rot":0.0, "xr":(-5,50),   "yr":(-50,0.), "xoffset":-0.1,  "yoffset":-0.2,  "zoffset":-0.05},
    "l2": {"tower":"ue", "azimuth":0,       "elev_adj":0,    "rot":0.0, "xr":(0,50),    "yr":(-30,30), "xoffset":0,     "yoffset":0,     "zoffset":0},
    "l3": {"tower":"uw", "azimuth":62.8,    "elev_adj":0,    "rot":0.0, "xr":(3,60),    "yr":(-30,30), "xoffset":18.47, "yoffset":-27.5, "zoffset":-2.5},
    "l4": {"tower":"uw", "azimuth":120,     "elev_adj":0,    "rot":0.0, "xr":(3,40),    "yr":(-20,40), "xoffset":19.5,  "yoffset":-28,   "zoffset":-2.5},
    "l5": {"tower":"d",  "azimuth":-177.66, "elev_adj":0.2,  "rot":0.1, "xr":(-40,30),  "yr":(-30,30), "xoffset":33.05, "yoffset":2.47,  "zoffset":-2.45},
    "l6": {"tower":"d",  "azimuth":-113.2,  "elev_adj":-0.4, "rot":0.0, "xr":(0,40),    "yr":(-40,0.), "xoffset":33.27, "yoffset":2.29,  "zoffset":-3.9}
    }

# calibrate targets- where did these come from
cal_targets = {"l1": [
                    {"angle": 0.0853, "name":"uw", "x":(32.5, 33,5), "y":(2.4,  3.2),  "z":(-8,-5.5), "n":20},
                    {"angle": 0.6175, "name":"center_1", "x":(16.4, 16.8), "y":(11.6, 11.9), "z":(-7.6,-7), "n":20},
                    {"angle": 0.5962, "name":"center_2", "x":(16.1, 16.5), "y":(10.6, 11.2), "z":(-8.6,-8), "n":100},
                    {"angle":-0.3194, "name":"util", "x":(57.0, 58.0), "y":(-19.5,-18.5),"z":(-8.2, -3.7), "n":20},
                    ],
            "l2": [
                    {"angle":0.0743, "name":"dw", "x":(32.5, 34,5), "y":(1.4,  3.5),  "z":(-8,-4.5), "n":20},
                    {"angle":-0.424, "name":"center", "x":(17.2, 18.2), "y":(-8.6, -7.5), "z":(-4.8,-4), "n":20},
                    {"angle":-0.4564, "name":"center_2", "x":(18.1, 18.5), "y":(-9.4,-8.7), "z":(-7.8,-6.9), "n":20},
                    # {"angle":0, "name":"alter", "x":(37, 38.2), "y":(-24, -22), "z":(-12, -10.3), "n":200},
                    {"angle":0.386, "name":"michi", "x":(43.0, 44.5), "y":(17,18.5),"z":(-12.2, -9), "n":20},
                    ],
            "l3": [
                    {"angle":0.0149, "name":"dw", "x":(33.25, 34), "y":(-0.3,  1.1),  "z":(-6.5,-1.5), "n":20},
                    {"angle":0.5053, "name":"center", "x":(16.8, 17.5), "y":(9.2, 9.8), "z":(-5,-3.5), "n":20},
                    {"angle":0.4787, "name":"center_2", "x":(16.2, 16.8), "y":(8.2,8.9), "z":(-8.1,-7.2), "n":50},
                    {"angle":0.004516, "name":"post", "x":(44.8, 45.5), "y":(-0.2, 0.6), "z":(-8.6, -7.5), "n":10},
                    ],
            "l4":[
                   {"angle":0.142, "name":"ue", "x":(32.2, 33.5), "y":(4,  5.5),  "z":(-3,0), "n":200},
                   {"angle":-0.412, "name":"center", "x":(17.5, 18.3), "y":(-8.3, -7.4), "z":(-2.6,-2), "n":100},
                   {"angle":-0.412, "name":"center_2", "x":(17.5, 18.3), "y":(-8.3, -7.4), "z":(-3.6,-3), "n":50},
                   {"angle":-0.439, "name":"center_3", "x":(16.5, 17.0), "y":(-8.3,-7.6), "z":(-4.5,-3.9), "n":50},
                   {"angle":-0.3365, "name":"tilden", "x":(40,42), "y":(-15.5, -13.5), "z":(-3, 1), "n":50},
                    ],
            "l5": [
                    {"angle":0.051277, "name":"ue", "x":(33, 33.7), "y":(1,  2.2),  "z":(-3.5,0), "n":20},
                    {"angle":0.5999, "name":"center_2", "x":(15.7, 16.2), "y":(10.7,11.3), "z":(-6.1,-5.6), "n":20},
                    ],
            "l6": [
                    {"angle":-0.0561, "name":"uw", "x":(33, 33.7), "y":(-2.5,  -1.2),  "z":(-4.5,0), "n":20},
                    {"angle":-0.5221, "name":"center", "x":(16, 16.5), "y":(-9.7, -9), "z":(-3.7,-3.2), "n":20},
                    {"angle":0.1365, "name":"util", "x":(49, 51), "y":(6,8), "z":(-5, 1), "n":20},
                    ],
            }


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate_data(v=[3, 5, 0], axis=[0, 0, 1], theta=0.523598783, rotation=None):
    # v = 3D array of points
    # 0.523598783 = 30 * np.dtor

    if rotation is None:
        rotation = rotation_matrix(axis, theta)

    return np.dot(rotation, v)

def find_angle(points, location):
    """Find points in the point cloud that correspond to a region then
    compute the angle from the lidar to the mean of those points if they exceed a threshold number of points"""

    mask = ((points[0,:] > location["x"][0]) &
            (points[0,:] < location["x"][1]) &
            (points[1,:] > location["y"][0]) &
            (points[1,:] < location["y"][1]) &
            (points[2,:] > location["z"][0]) &
            (points[2,:] < location["z"][1]))

    count = np.count_nonzero(mask)
    if count > location["n"]:
        x,y = np.mean(points[0,mask]), np.mean(points[1,mask]) #z= np.mean(points[2,mask])
        angle = np.arctan(y/x)
    else:
        angle = np.nan

    return angle



def correct_points_background(points, background):
    '''
    Find known targets in the points cloud and rotate points to line up targets
    '''
    angles = []

    for loc in background:
        angles.append(loc["angle"] - find_angle(points, loc))

    angle = np.nanmean(angles)

    if np.isfinite(angle):
        corrected_points = rotate_data(v=points, axis=[0, 0, 1], theta=angle)
    else:
        corrected_points = points

    return corrected_points

def update_l5(d):
    a1 = np.arctan(1.6/33.48)
    a2 = np.arctan(2.13/33.49)
    # rotate about z axis
    d2=rotate_data(d,theta=(a1-a2)*2)
    # rotate about y axis
    d3=rotate_data(d2,axis=[0,1,0],theta=np.deg2rad(0.7))
    # rotate about x axis
    final=rotate_data(d3,axis=[1,0,0],theta=np.deg2rad(1))
    return final


def correct_data(points, azimuth, xoffset, yoffset, zoffset, roll=0, elevation_adjustment=0,
                 global_adjustment=True, background_correction=None, correct_l5=False):
    """
    Apply corrections to raw lidar data for known lidar positions/orientations and to correct background light effects.
    Optionally do not reposition points into a common reference across lidars.
    Background light correction only applied if calibration targets are specified.
    """

    # first take out 30deg downward pointing mount for all lidars
    if elevation_adjustment==0:
        pr = rotate_data(v=points[:,:3].T, axis=[0, 0, 1], theta=np.deg2rad(-30.0), rotation=None)   # 31 deg
    else:
        pr = rotate_data(v=points[:,:3].T, axis=[0, 0, 1], theta=np.deg2rad(-30.0+elevation_adjustment), rotation=None)   # 31 deg

    # Then account for mounting on a vertical mount
    if roll==0:
        pr = rotate_data(pr, axis=[1,0,0],theta=np.pi/2)
    else:
        pr = rotate_data(pr, axis=[1,0,0],theta=np.pi/2 + np.deg2rad(roll))

    if correct_l5:
        pr = update_l5(pr)

    if background_correction is not None:
        pr = correct_points_background(pr, background_correction)


    if global_adjustment:
        # Then rotate to a common map orientation
        pr = rotate_data(pr, axis=[0,0,1],theta=np.deg2rad(azimuth))

        # shift location based on lidar mount points
        pr[0,:] += (xoffset - center_point[0])
        pr[1,:] += (yoffset - center_point[1])
        pr[2,:] += zoffset

        # Then rotate to a North map orientation
        pr = rotate_data(pr, axis=[0,0,1],theta=np.deg2rad(global_rotation))


    return pr


def read_file(filename, file_number=0):
    """
    Read a livox lvx file into a numpy point cloud
    """
    try:
        datafile = glob.glob(filename)[file_number]
    except:
        datafile = filename
    d = lvx_reader.BinaryReaders.lvxreader(datafile)
    points = np.array(d.datapoints)
    return points

def get_date(filename):
    """
    Parse the date/time stamp from a filename
    """
    f = filename.split('/')[-1]
    print(f)
    year = int(f[:4])
    month = int(f[5:7])
    day = int(f[8:10])
    hour = int(f[11:13])
    minute = int(f[14:16])
    return datetime.datetime(year, month, day, hour, minute, 0)

def find_zero_height(x,y,z, nx=10, ny=10, dx=None, xr=None, yr=None):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    if xr is None:
        xr = (xmin,xmax)
    else:
        xmin, xmax = xr
    if yr is None:
        yr = (ymin,ymax)
    else:
        ymin, ymax = yr

    if dx is not None:
        nx = int((xr[1] - xr[0]) / dx)
        ny = int((yr[1] - yr[0]) / dx)

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    res = np.zeros((ny,nx))
    map_n = np.zeros((ny,nx))
    for j in range(ny):
        y0 = ymin + dy*j
        y1 = ymin + dy*(j+1)

        for i in range(nx):
            x0 = xmin + dx*i
            x1 = xmin + dx*(i+1)
            cur = (x>x0) & (x<x1) & (y>y0) & (y<y1)
            map_n[j,i] = np.count_nonzero(cur)
            if map_n[j,i]<10:
                if map_n[j,i]>0:
                    res[j,i] = np.min(z[cur])
                else:
                    res[j,i] = np.nan
            else:
                res[j,i] = np.percentile(z[cur], 5)


    return np.mean(res[np.isfinite(res)]), res, map_n

def make_map(x,y,z,title):
    print(x.shape)
    plt.clf()
    plt.scatter(x=x,y=y,c=z, cmap=plt.cm.Blues)
    #if not get_zero: plt.clim(0,0.3)
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"image_{title[:9]}.png")


def get_height(points, xr=(-1,5), yr=(-6,-3), zmax=-9.5, title="date", zero=-11.97, get_zero=False, scatter_plot=True, dx=0.5):
    # get the x, y, and z points (reveresed)
    x = points[0, ::1]
    y = points[1, ::1]
    z = points[2, ::1]
    # set values to xr and yr, but what are these values representing?
    x0, x1 = xr
    y0, y1 = yr
    # boolean array for depth values to find zero heights
    g = (z < zmax) & (x>x0) & (x<x1) & (y>y0) & (y<y1) & np.isfinite(z)

    if scatter_plot: make_map(x=x[g], y=y[g], z=z[g]-zero, title=title)

    if get_zero:
        #print("standard_height:",z[g].mean())
        return find_zero_height(x[g],y[g],z[g], dx=dx, xr=xr,yr=yr)

    print(np.count_nonzero(g), z[g].min(), z[g].max(), z[g].mean())
    height = z[g].mean() - zero



    return height

tower_locations = np.array([[0,0],
                            [-18.8, -5.8],
                            [14.2, -12.5],
                            [2.5, 19.5]])

def plot_map(this_map, x=None, y=None, tower_name="Center", title="title", clim=None, path=""):
    plt.clf()
    if x is None: x = (np.arange(this_map.shape[1])-(this_map.shape[1]/2)) * global_dx
    if y is None: y = (np.arange(this_map.shape[0])-(this_map.shape[0]/2)) * global_dx
    plt.pcolormesh(x, y, this_map, cmap="turbo", shading="auto")
    # plt.imshow(this_map, cmap="turbo", origin="lower")
    plt.xlabel(f"Distance relative to {tower_name} tower [m]")
    plt.ylabel(f"Distance relative to {tower_name} tower [m]")

    # plt.text(0, 0, "Center\nTower", ha="center", va="center", #size=50,
    #      bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8))
    #      )
    plt.text(0, 0, "C", ha="center", va="center", #size=50,
         bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8))
         )
    plt.text(14.2, -12.5, "d", ha="center", va="center", #size=50,
         bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8))
         )
    plt.text(2.5, 19.5, "ue", ha="center", va="center", #size=50,
         bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8))
         )
    plt.text(-18.8, -5.8, "uw", ha="center", va="center", #size=50,
         bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8))
         )
    # plt.text(40, 22, "Radar", ha="center", va="center", #size=50,
    #      bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8))
    #      )
    plt.plot(   4,  7,'o', color="green", label="Snow Pillow")
    plt.plot(  -4,  3,'o', color="green")
    plt.plot(   8,-13,'o', color="green")
    plt.plot( -12, -2,'o', color="green")

    plt.colorbar()
    plt.title(title)
    if clim is not None:
        plt.clim(clim)
    title = title.replace(" ","_").replace(":","-")
    plt.savefig(f"{path}map_{title}.png")

# daylight savings? MDT 2 UTC
dt = -1*datetime.timedelta(hours=6)


# xr = (26.1,26.8)
# yr = (-13.8,-12.8)
# xr = (24,35)
# yr = (-18,-10.0)
xr = (3,60)
yr = (-30,30.0)

# l1 xr=(-5,50); yr=(-50,0)
# l2 xr=(0,50); yr=(-30,30)
# l3 xr = (3,60); yr = (-30,30.0)
# l5 xr = (-40,30); yr = (-30,30.0)
# l6 xr = (0,40); yr = (-40,0)
# full range:
# xr = (-40,50)
# yr = (-40,30)

# xr = (26.2,26.8)
# yr = (-13,-12.0)
zmax=-9
global_dx = 0.5
# global_dx = 1.0


# l1uw = 18.08, -27.85
# l5ue = -0.25, 0.73 # should be 0,0

def point_mask(x,y,z,xr,yr,zr=None):
    if zr is None:
        return ((x>xr[0]) & (x<xr[1]) &
                (y>yr[0]) & (y<yr[1])
               )
    else:
        return ((x>xr[0]) & (x<xr[1]) &
                (y>yr[0]) & (y<yr[1]) &
                (z>zr[0]) & (z<zr[1])
               )

def load_file(filepath, lidar):
    """Get an np.array of points from a filepath to a .lvx file.

    Args:
        filepath (str): Filepath to .lvx file
        lidar (str): The lidar which scanned/created the point cloud.

    Returns:
        np.array: array of points
    """
    tower_name = lidar_config[lidar]["tower"]
    az   = lidar_config[lidar]["azimuth"]
    pitch= lidar_config[lidar]["elev_adj"]
    roll = lidar_config[lidar]["rot"]
    xoff = lidar_config[lidar]["xoffset"]
    yoff = lidar_config[lidar]["yoffset"]
    zoff = lidar_config[lidar]["zoffset"]

    return correct_data(read_file(filepath),
                            xoffset=xoff, yoffset=yoff, zoffset=zoff,
                            azimuth=az, elevation_adjustment=pitch, roll=roll,
                            background_correction=cal_targets[lidar])
    
def load_date(lidar, date, hour="01", minute="??", path="", load_all=False):
    file_search = f"{path}/{lidar}/{date}/{date}_{hour}-{minute}*.lvx"
    files = glob.glob(file_search)
    files.sort()
    

    tower_name = lidar_config[lidar]["tower"]
    az   = lidar_config[lidar]["azimuth"]
    pitch= lidar_config[lidar]["elev_adj"]
    roll = lidar_config[lidar]["rot"]
    xoff = lidar_config[lidar]["xoffset"]
    yoff = lidar_config[lidar]["yoffset"]
    zoff = lidar_config[lidar]["zoffset"]

    # print(file_search)
    if load_all:
        all_points = []
        for f in files:
            all_points.append(correct_data(read_file(f),
                            xoffset=xoff, yoffset=yoff, zoffset=zoff,
                            azimuth=az, elevation_adjustment=pitch, roll=roll,
                            background_correction=cal_targets[lidar]))
            points = np.concatenate(all_points, axis=1)

    
    else:
        points = correct_data(read_file(files[0]),
                            xoffset=xoff, yoffset=yoff, zoffset=zoff,
                            azimuth=az, elevation_adjustment=pitch, roll=roll,
                            background_correction=cal_targets[lidar])
    return points


def load_first_file(lidar, path="data", file_number=0, file_search=None, global_adjustment=True, correction=True):
    if file_search is None: 
        file_search = f"{path}/{lidar}/202?-??-??_??*.lvx"
    files = glob.glob(file_search)
    files.sort()

    tower_name = lidar_config[lidar]["tower"]
    az   = lidar_config[lidar]["azimuth"]
    pitch= lidar_config[lidar]["elev_adj"]
    roll = lidar_config[lidar]["rot"]
    xoff = lidar_config[lidar]["xoffset"]
    yoff = lidar_config[lidar]["yoffset"]
    zoff = lidar_config[lidar]["zoffset"]

    if correction:
        targets = cal_targets[lidar]
    else:
        targets = None

    points = correct_data(read_file(files[file_number]),
                            xoffset=xoff, yoffset=yoff, zoffset=zoff,
                            azimuth=az, elevation_adjustment=pitch, roll=roll,
                            global_adjustment=global_adjustment, background_correction=targets)
    return points

def load_files(lidar, path="data", xr=[-40,40], yr=[-40,40], dx=0.5, file_search="{path}/{lidar}/202?-??-??_*-00*.lvx"):
    file_search = file_search.format(lidar=lidar, path=path)
    files = glob.glob(file_search)
    files.sort()

    tower_name = lidar_config[lidar]["tower"]
    az   = lidar_config[lidar]["azimuth"]
    pitch= lidar_config[lidar]["elev_adj"]
    roll = lidar_config[lidar]["rot"]
    xoff = lidar_config[lidar]["xoffset"]
    yoff = lidar_config[lidar]["yoffset"]
    zoff = lidar_config[lidar]["zoffset"]

    all_points = []
    all_data = []
    all_maps = []
    all_n = []
    all_dates = []
    print(len(files))
    i=0
    for f in files:
        i += 1
        try:
            print(i, len(files), f)
            points = correct_data(read_file(f),
                                    xoffset=xoff, yoffset=yoff, zoffset=zoff,
                                    azimuth=az, elevation_adjustment=pitch, roll=roll,
                                    background_correction=cal_targets[lidar])
            data, this_map, map_n = get_height(points, xr=xr, yr=yr, dx=dx, zero=0, get_zero=True, scatter_plot=False)

            all_dates.append(get_date(f))
            all_points.append(points)
            all_data.append(data)
            all_maps.append(this_map)
            all_n.append(map_n)
        except:
            print(f"Error with file {f}")

    return all_points, all_data, all_maps, all_n, all_dates


def make_time_series(lidar="l3", path=""):
    # create a figure
    plt.figure(figsize=(20,15), dpi=100)
    # load in tower name
    tower_name = lidar_config[lidar]["tower"]
    # get azimuth angle and offsets for the specific lidar
    az   = lidar_config[lidar]["azimuth"]
    xoff = lidar_config[lidar]["xoffset"]
    yoff = lidar_config[lidar]["yoffset"]
    zoff = lidar_config[lidar]["zoffset"]
    elev_adj = lidar_config[lidar]["elev_adj"]
    roll = lidar_config[lidar]["rot"]

    # find files from the snow off period in October
    file_search = f"./{path}/{lidar}/2022-10-??_01*.lvx"
    files = glob.glob(file_search)
    files.sort()
    # Show how many files we have
    print(len(files))
    # set a pr variable
    pr = None
    # loop through the first 2 files and concatenate values
    for f in files[:2]:
        print(f)
        if pr is None:
            # correct values for the first file
            pr = correct_data(read_file(f), azimuth=az, xoffset=xoff, yoffset=yoff, zoffset=zoff,
                              roll=roll, elevation_adjustment=elev_adj,
                              background_correction=cal_targets[lidar])
        else:
            # correct and concatenate values in the second file. But why is this done?
            pr = np.concatenate([pr,
                 correct_data(read_file(f), azimuth=az, xoffset=xoff, yoffset=yoff, zoffset=zoff,
                              roll=roll, elevation_adjustment=elev_adj,
                              background_correction=cal_targets[lidar])
                 ], axis=1)
        # provide the shape of this 3D array
        print(pr.shape)
        # adjust date to the date of the 0 map
        zero_date = get_date(f) - dt
    h0, zero_map, map_n = get_height(pr, xr=xr,yr=yr, zmax=zmax, title=str(zero_date), zero=0, get_zero=True)
    print(f"zero snow height:{h0} "+str(zero_date))
    plot_map(zero_map, title=str(zero_date))

    # sys.exit()

    file_search = f"{path}/{lidar}/202?-*.lvx"
    file_list = glob.glob(file_search)
    file_list.sort()
    f=file_list[0]
    file_list = file_list[1:]
    print("Number of files found: "+str(len(file_list)))
    dates = []
    data = np.zeros(len(file_list))
    map_data = np.zeros(len(file_list))

    # maps = [zero_map]

    for i,f in enumerate(file_list):
        dates.append(get_date(f) - dt)
        print(str(dates[-1]))
        try:
            pr = correct_data(read_file(f), azimuth=az, xoffset=xoff, yoffset=yoff, zoffset=zoff,
                              background_correction=cal_targets[lidar],
                              correct_l5=((lidar=="l5") and (dates[-1].year == 2023)))
            data[i], this_map, map_n = get_height(pr, xr=xr, yr=yr, zmax=zmax, title=str(dates[-1]), zero=0, get_zero=True, scatter_plot=False)
            depth = this_map - zero_map
            data[i] -= h0
            plot_map(depth, title=str(dates[-1]), clim=(-0.1,0.3))
            if np.count_nonzero(np.isfinite(depth)) > 3000:
                map_data[i] = np.median(depth[np.isfinite(depth)])
            else:
                map_data[i] = np.nan
        except:
            data[i] = np.nan
            map_data[i] = np.nan
            print("error with last date!")

    # sys.exit()
    plt.figure(figsize=(20,5))
    plt.plot(dates, data)
    plt.ylabel("Snow Depth [m]")
    plt.xlabel("Date/time [UTC]")
    # make facecolor white
    plt.gcf().set_facecolor("white")
    plt.savefig("snow_depth_time.png")

    plt.clf()
    plt.plot(dates, map_data)
    plt.ylabel("Snow Depth [m]")
    plt.xlabel("Date/time [UTC]")
    plt.savefig("snow_depth_map_time.png")

if __name__ == '__main__':
    make_time_series(lidar="l4", path='data')


# pr1 = correct_data(read_file(l1file), azimuth=-61.5, xoffset=0, yoffset=0, zoffset=0)
# pr2 = correct_data(read_file(l2file), azimuth=0, xoffset=0, yoffset=0, zoffset=0)
# pr3 = correct_data(read_file(l3file), azimuth=64, xoffset=19.5, yoffset=-28, zoffset=-2.5)
# pr4 = correct_data(read_file(l4file), azimuth=120, xoffset=19.5, yoffset=-28, zoffset=-2.5)
# pr5 = correct_data(read_file(l5file), azimuth=-179, xoffset=33, yoffset=2.8, zoffset=-4)
# pr6 = correct_data(read_file(l6file), azimuth=-113, xoffset=33, yoffset=2.8, zoffset=-2.5)
