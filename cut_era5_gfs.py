import numpy as np

era5v = np.load('./Dataset/v10_era5.npy')
# np.save('era5v_train.npy', era5v[:, :96, :96])
era5v = era5v[:, :96, :96]
gfsv = np.load('./Dataset/v10_gfs.npy')
# np.save('gfsv_train.npy', gfsv[:, :96, :96])
gfsv = gfsv[:, :96, :96]

era5u = np.load('./Dataset/u10_era5.npy')
era5u = era5u[:, :96, :96]
gfsu = np.load('./Dataset/u10_gfs.npy')
gfsu = gfsu[:, :96, :96]

deg = 180.0/np.pi
era5_speed = np.sqrt(era5u**2 + era5v**2)
gfs_speed = np.sqrt(gfsu**2 + gfsv**2)
era5_dir = 180.0 + np.arctan2(era5u, era5v) * deg
gfs_dir = 180.0 + np.arctan2(gfsu, gfsv) * deg

print(era5_dir.max(), era5_dir.min())
print(gfs_dir.max(), gfs_dir.min())

# np.save('./Dataset/era5_spd.npy', era5_speed)
# np.save('./Dataset/era5_dir.npy', era5_dir)
# np.save('./Dataset/gfs_spd.npy', gfs_speed)
# np.save('./Dataset/gfs_dir.npy', gfs_dir)
