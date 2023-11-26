import os
import numpy as np
from netCDF4 import Dataset

path = '/Volumes/painswick/数据/u10+v10_2016-2022_6h_[0-45][100-160].nc'
# for i, nc_file in enumerate(nc_list):
# print(nc_file)
# nc_path = os.path.join(path, nc_file)
ds = Dataset(path)
# ds.variables['lat'][180:360]
# ds.variables['lon'][400:580]
# 经度是100-144.75,维度是45-0.25
u10_era5 = ds.variables['u10'][:, :180, :180].data
v10_era5 = ds.variables['v10'][:, :180, :180].data
u10_era5 = np.delete(u10_era5, 3507, axis=0)
u10_era5 = np.delete(u10_era5, 9194, axis=0)
u10_era5 = np.delete(u10_era5, 10097, axis=0)
v10_era5 = np.delete(v10_era5, 3507, axis=0)
v10_era5 = np.delete(v10_era5, 9194, axis=0)
v10_era5 = np.delete(v10_era5, 10097, axis=0)
# 前8767个是2016-2021年的数据

np.save('Dataset/u10_era5.npy', u10_era5)
np.save('Dataset/v10_era5.npy', v10_era5)