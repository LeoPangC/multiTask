import numpy as np

era5v = np.load('v10_era5.npy')
# u_max = era5u[8767:9000].max()
# u_min = era5u[8767:9000].min()
# print('max:', u_max, '\nmin:', u_min)
np.save('era5v_train.npy', era5v[:, :96, :96])
gfsv = np.load('v10_gfs.npy')
# g_max = gfsu[8767:9000].max()
# g_min = gfsu[8767:9000].min()
# print('max:', g_max, '\nmin:', g_min)
np.save('gfsv_train.npy', gfsv[:, :96, :96])

# rmse = np.sqrt(np.sum((era5u[:2000]-gfsu[:2000])**2)/np.prod(era5u[:2000].shape))
# print('RMSE:', rmse)
# np.save('gfsu_test.npy', gfsu[2000:2500])