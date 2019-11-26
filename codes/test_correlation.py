
from datetime import datetime
import dask.array as da
import os
import time
import h5py
import numpy as np
import xarray as xr
import downscale as down

down.matplotlib_update_settings()
tmpa_dir = os.path.join('..', 'data', 'tmpa_conus_data')

'''--------------------------------------------------------------------------
load precipitation data over conus (3-hr totals)
and compute taylor for a given location
--------------------------------------------------------------------------'''

f = h5py.File(os.path.join(tmpa_dir, 'data_tmpa_3h.hdf5'), "r")
print(list(f.keys()))
# print(list(f.attrs()))
tmpalat = f['lat'][:]
tmpalon = f['lon'][:]
dates_int = f['dates'][:]
hours_int = f['hours'][:]
dset = f['prcp']
print('dataset shape = {}'.format(dset.shape)) # too large to fit in memory!


x = da.from_array(dset, chunks=(6, 6, 300))

# UTC time
dates = [datetime.strptime(str(integd)+str(inthour), '%Y%m%d%H')
                 for integd, inthour in zip(dates_int, hours_int)]

# create xarray
xrs0 = xr.DataArray(x,  coords={'lon':tmpalon, 'lat':tmpalat, 'time':dates},
                                                dims=('lon', 'lat', 'time'))

# set negative values to NaN (missing values)
xrs = xrs0.where(xrs0 >= -0.001)



# now extract a bounding box of interest:
clat = 34.785
clon = -98.125
dx = 0.25
npix = 3
buffer = 0.50*npix*dx
eps = 1e-4 # to make sure to include boundaires -> add an eps buffer
solat = clat - buffer - eps
nolat = clat + buffer + eps
ealon = clon + buffer + eps
welon = clon - buffer - eps


bcond = np.logical_and(
            np.logical_and( xrs.lat > solat, xrs.lat < nolat),
            np.logical_and( xrs.lon > welon, xrs.lon < ealon))

# selection in space
# pbox_3h = xrs.where(bcond, drop = True)
# load data to memory::
dt = 3 # timescale in hours
dx = 0.25
tmax = 48
# smax = 3



# XARRAY WITH VALUES LOADED IN MEMORY
box_3h = xrs.where(bcond, drop = True).load()

# aggregate to the daily scale propagating nans
# and only then remove nans
boxd0 = box_3h.resample(time ='D').sum(dim='time', skipna=False)


boxd = boxd0.dropna(dim='time', how='any')

# ts1 = boxd.loc[dict(lat = 34.875, lon = -98.125)]

# downscaling:



# compute correlation between gridded precipitation cells





res = down.grid_corr(boxd0, plot=False)

# res['fig'].show()
vdist = res['vdist']
vcorr = res['vcorr']
# res['fig'].show()
# res['d0']
# res['mu0']
# res['alpha']
# res['epsilon']
print(res)




acftype = 'mar'
init_time = time.time()
# npoints = 5
resdown = down.down_corr(vdist, vcorr, 25.3, acf=acftype,
                         use_ave=True, opt_method = 'genetic',
                         toll=0.005, plot=True, disp=True)
print(resdown)
end_time = time.time()
exec_time = end_time - init_time
print('exec_time = {} minutes'.format(exec_time/60))

# xx5 = np.linspace(np.min(vdist), np.max(vdist), npoints)
# fig2 = res['fig']
# epsD = 50
# alpD = 0.25

# resdown['eps_d'] = resdown.x[0]
# resdown['alpha_d'] = resdown.x[1]



# try to integrate bacl the downscaled correlation::

resdown['fig'].show()


xx = np.linspace(0.5, 100)
corrL = down.int_corr(xx, (resdown['eps_d'], resdown['alp_d']), acftype, 25.3)


# plt.figure()
# plt.plot(xx, down.epl_fun(xx, res['eps_s'], res['alp_s']), 'r')
# # plt.plot(xx5, down.epl_fun(xx5, res['epsilon'], res['alpha']), 'ok')
# plt.plot(vdist, vcorr, 'or')
# plt.plot(xx, down.epl_fun(xx, resdown['eps_d'], resdown['alp_d']), 'b')
# plt.plot(xx, down.epl_fun(xx, 26.87, 0.229), '--b')
# plt.plot(xx, corrL, 'sc')
# plt.show()

# plt.figure()
# plt.plot(xx, down.str_exp_fun(xx, res['d0'], res['mu0']), 'r')
# # plt.plot(xx5, down.epl_fun(xx5, res['epsilon'], res['alpha']), 'ok')
# plt.plot(vdist, vcorr, 'or')
# plt.plot(xx, down.str_exp_fun(xx, resdown['eps_d'], resdown['alpha_d']), 'b')
# # plt.plot(xx, down.epl_fun(xx, 26.87, 0.229), '--b')
# plt.plot(xx, corrL, 'sc')
# plt.show()


#
# def block_ave(x, window=1):
#     xmin = np.min(x)
#     xmax = np.max(x)



# vdist_ave, vcorr_ave, vd, cd, cluster = block_ave(vdist, vcorr, toll = 0.2)
# res = block_ave_corr(vdist, vcorr, toll = 0.2)









