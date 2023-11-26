import numpy as np

era5v = np.load('./Dataset/v10_era5.npy')
np.save('./Dataset/era5v_train.npy', era5v[:, :96, :96])
gfsv = np.load('./Dataset/v10_gfs.npy')
np.save('./Dataset/gfsv_train.npy', gfsv[:, :96, :96])
era5u = np.load('./Dataset/u10_era5.npy')
np.save('./Dataset/era5u_train.npy', era5u[:, :96, :96])
gfsu = np.load('./Dataset/u10_gfs.npy')
np.save('./Dataset/gfsu_train.npy', gfsu[:, :96, :96])

# rmse = np.sqrt(np.sum((era5u[:2000]-gfsu[:2000])**2)/np.prod(era5u[:2000].shape))
# print('RMSE:', rmse)
# np.save('gfsu_test.npy', gfsu[2000:2500])