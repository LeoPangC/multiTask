import os
import numpy as np
from netCDF4 import Dataset

path = '/Volumes/painswick/数据/GFS_6h_forecast/GFS_nc'
nc_list = os.listdir(path)
nc_list.sort()
u10_gfs = np.array([])
v10_gfs = np.array([])
for i, nc_file in enumerate(nc_list):
    print(nc_file)
    nc_path = os.path.join(path, nc_file)
    ds = Dataset(nc_path)
    # ds.variables['lat'][180:360]
    # ds.variables['lon'][400:580]
    # 经度是100-144.75，维度是45-0.25
    u10 = ds.variables['U_GRD_L103'][:, :180, :180].data
    v10 = ds.variables['V_GRD_L103'][:, :180, :180].data
    try:
        u10_gfs = np.concatenate((u10_gfs, u10), axis=0)
        v10_gfs = np.concatenate((v10_gfs, v10), axis=0)
    except:
        u10_gfs = u10
        v10_gfs = v10
np.save('Dataset/u10_gfs.npy', u10_gfs)
np.save('Dataset/v10_gfs.npy', v10_gfs)