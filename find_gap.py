import os
from datetime import datetime, date, timedelta


path = '/Volumes/painswick/数据/GFS_6h_forecast/GFS_nc'
nc_list = os.listdir(path)
nc_list.sort()
date_time = '2016010100'
gap = []

for nc in nc_list:
    if date_time == nc[9:19]:
        date_time = datetime.strptime(date_time, '%Y%m%d%H')
        date_time = date_time + timedelta(hours=6)
        date_time = date_time.strftime('%Y%m%d%H')
    else:
        gap.append(date_time)
        date_time = datetime.strptime(date_time, '%Y%m%d%H')
        date_time = date_time + timedelta(hours=12)
        date_time = date_time.strftime('%Y%m%d%H')

print(gap)