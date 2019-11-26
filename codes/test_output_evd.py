import os
import numpy as np
import h5py
# import pandas as pd
import conusfun as cfun
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# from datetime import datetime
import xarray as xr
# import dask.array as da
import matplotlib as mpl

evd_years_world_file = 'tmpa_mev_global_quants.hdf5'
evd_hdfco_world_file = 'evd_conus_map_world.hdf5'

with h5py.File(os.path.join(cfun.outdir_data, evd_hdfco_world_file), "r") as fr:
    qmev = fr['qmev'][:]
    qgev = fr['qgev'][:]
    Tr = fr['Tr'][()]
    maxmiss = fr['maxmiss'][()]
    thresh = fr['thresh'][()]
    tmpalat = fr['lat'][:]
    tmpalon = fr['lon'][:]

with h5py.File(os.path.join(cfun.outdir_data, evd_years_world_file), "r") as fr:
    qmev2 = fr['qmev'][:]
    qgev2 = fr['qgev'][:]
    Tr2 = fr['Tr'][()]
    maxmiss2 = fr['nmax_miss'][()]
    thresh2 = fr['thresh'][()]
    tmpalat2 = fr['lat'][:]
    tmpalon2 = fr['lon'][:]
    min_n_complete_years2 = fr['min_n_complete_years'][()]
    num_complete_years2 = fr['num_complete_years'][()]



ncres = cfun.load_results_netcdf(ncname='ncres_cluster_30580.nc')
list(ncres.variables)

mev_s_all = ncres['mev_s_all']
gev_s_all = ncres['gev_s_all']

# plt.figure()
# plt.plot(qmev[:,:], qmev2[:,:,2],'o')
# plt.plot([0, 800], [0, 800], 'k')
# plt.show()
#
#
# plt.figure()
# plt.plot(qgev[:,:], qgev2[:,:,2],'o')
# plt.plot([0, 800], [0, 800], 'k')
# plt.show()



xmev = xr.DataArray(qmev, coords=[tmpalon, tmpalat], dims=['lon', 'lat'])
xgev = xr.DataArray(qgev, coords=[tmpalon, tmpalat], dims=['lon', 'lat'])


xmev2 = xr.DataArray(qmev2[:,:,2], coords=[tmpalon, tmpalat], dims=['lon', 'lat'])
xgev2 = xr.DataArray(qgev2[:,:,2], coords=[tmpalon, tmpalat], dims=['lon', 'lat'])


conuscond = np.logical_and(
    np.logical_and(xmev.lat > cfun.solat, xmev.lat < cfun.nolat),
    np.logical_and(xmev.lon > cfun.welon, xmev.lon < cfun.ealon))
# Load in memory the bounding box of interest
xmev_conus = xmev.where(conuscond, drop=True).load()
xgev_conus = xgev.where(conuscond, drop=True).load()


xmev_conus2 = xmev2.where(conuscond, drop=True).load()
xgev_conus2 = xgev2.where(conuscond, drop=True).load()


plt.figure()
# plt.imshow(xmev_conus2.values)
plt.imshow(mev_s_all.values[:,:,2] - xmev_conus2.values)
plt.colorbar()
plt.show()

plt.figure()
plt.plot(xmev_conus2.values, mev_s_all.values[:,:,2], 'o')
plt.plot([0, 800], [0, 800], 'k')
# plt.plot(xmev_conus2.values, mev_s_all.values, 'o')
plt.show()
