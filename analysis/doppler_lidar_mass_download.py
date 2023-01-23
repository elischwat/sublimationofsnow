import glob
import os
import act
import multiprocessing

ds_dl_fpt = 'gucdlfptM1.b1'

# startdate = '2021-12-01'
# enddate = '2022-04-30'
# output_path = "/data2/elilouis/sublimationofsnow/gucdlfptM1.b1/original"
# modified_path = "/data2/elilouis/sublimationofsnow/gucdlfptM1.b1/downsampled"

# startdate = '2022-05-01'
# enddate = '2022-06-30'
# output_path = "/data2/elilouis/sublimationofsnow/gucdlfptM1.b1/spring/original"
# modified_path = "/data2/elilouis/sublimationofsnow/gucdlfptM1.b1/spring/downsampled"

# startdate = '2022-01-01'
# enddate = '2022-12-31'
output_path = "/data2/elilouis/sublimationofsnow/gucdlfptM1.b1/alldatatodate/original/2022"
modified_path = "/data2/elilouis/sublimationofsnow/gucdlfptM1.b1/alldatatodate/downsampled/2022"

username='eschwat'
token='761cf7339f9adaec'

# os.makedirs(output_path, exist_ok=True)
os.makedirs(modified_path, exist_ok=True)

# act.discovery.download_data(username, token, ds_dl_fpt, startdate, enddate, output=output_path)

files = glob.glob(os.path.join(output_path, '*cdf'))    

def decrease_data_size(f):
    new_f = f.replace(output_path, modified_path)
    print(f"Modifiying: {f}")
    print(f"Placing it: {new_f}")
    ds = act.io.armfiles.read_netcdf([f])
    ds.coords['time'] = ds.time.dt.floor('1Min')
    ds = ds.groupby('time').mean()
    ds = ds.sel(range=slice(0,4000))
    ds.write.write_netcdf(path=new_f)

print("Processing files in parallel")
pool = multiprocessing.Pool(16)
pool.map(decrease_data_size, files)
print("Finished processing files.")