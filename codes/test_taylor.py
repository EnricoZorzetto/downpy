
import os
from datetime import datetime
import dask.array as da
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



# XARRAY WITH VALUES
pbox_3h = xrs.where(bcond, drop = True).load()


# NUMPY ARRAY WITH VALUES
np3h = pbox_3h.values


thresh = 1

L1 = 25 # pixel linear size (in km)
target_x = 0.001 # km
target_t = 24    # hrs
ninterp = 1000 # number of points in t-dimension
Nt = 365.25
origin_x = 25 # km
origin_t = 24 # hours

pwets, xscales, tscales = down.compute_pwet_xr(pbox_3h, thresh) # *= keyword only argument

res_taylor = down.Taylor_beta(pwets, xscales, tscales, L1=25, target_x=0.001,
                              target_t=24,
                      origin_x=25, origin_t=24, ninterp = 1000, plot = True)

beta = res_taylor['beta']
res_taylor['fig'].show()
res_taylor['contour'].show()


