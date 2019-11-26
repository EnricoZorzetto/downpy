# this file reads an hdf file with a precipitation dataset
# over a given domain, and perform EV analysis with
# two models (GEV and MEV) for each grid cell in the dataset

import os
import numpy as np
import h5py
import pandas as pd
import conusfun as cfun
import downscale as down
from datetime import datetime
import xarray as xr
import dask.array as da


# relevant quantities:
# TODO: get them from conusfun
thresh = cfun.pixelkwargs['thresh']
maxmiss = cfun.pixelkwargs['maxmiss']
TR = cfun.Tr
domain = 'conus'
outname = "evd_conus_map_{}.hdf5".format(domain)
land_sea_mask = os.path.join(cfun.tmpa_dir, 'TRMM_TMPA_LandSeaMask.2.nc4')

# TODO: get them from conusfun
if domain == 'conus':
    nb = 50.0
    sb = 22.0
    eb = -60.0
    wb = -130.0
    tmpa_hdf_file = os.path.join(cfun.tmpa_dir, 'data_tmpa_daily.hdf5')
elif domain == 'world':
    nb = 50.0
    sb = -50.0
    eb = 180.0
    wb = -180.0
    tmpa_hdf_file = os.path.join(cfun.tmpa_dir, 'data_tmpa_world_daily.hdf5')
else:
    print('main_evd_maps ERROR:: must specify a valid domain!')


# read dask array with all daily precipitation data
f = h5py.File(tmpa_hdf_file, "r")
print(list(f.keys()))
tmpalat = f['lat'][:]
tmpalon = f['lon'][:]
nlat = np.size(tmpalat)
nlon = np.size(tmpalon)
dates_int = f['dates'][:]
# hours_int = f['hours'][:]
dset = f['prcp']
# print('dataset shape = {}'.format(dset.shape))
x = da.from_array(dset, chunks=(6, 6, 300))
# UTC time
dates = [datetime.strptime(str(integd), '%Y%m%d') for integd in dates_int]
xconus = xr.DataArray(x,
      coords={'lon':tmpalon, 'lat':tmpalat, 'time':dates},
      dims=('lon', 'lat', 'time'))
xconus = xconus.where(xconus >= -0.001)
### end reading prcp dataset ###


# for each grid cell do the following:
ntr = np.size(TR)
Fi = 1 - 1 / TR
qmev = np.zeros((nlon, nlat, ntr))
qgev = np.zeros((nlon, nlat, ntr))
for ii, clon in enumerate(tmpalon):
    print(ii)
    for jj, clat in enumerate(tmpalat):
        xpixel = xconus.sel(lat=clat, lon=clon).dropna(
                    dim='time', how='any').load()
        ts = xpixel.values
        years = xpixel.time.dt.year.values
        df = pd.DataFrame({'PRCP': ts, 'YEAR': years})
        df = down.remove_missing_years(df, maxmiss)[0]
        Ny, Cy, Wy = down.mev_fit(df, thresh=thresh)
        x0 = 9.0 * np.mean(Cy)
        qmev[ii, jj, :] = down.mev_quant(Fi, x0, Ny, Cy, Wy, thresh=thresh)[0]
        # fit GEV and compute quantiles
        XIemp, Fiemp, TRemp = down.tab_rain_max(df)
        csi, psi, mu = down.gev_fit_lmom(XIemp)  # fit to annual maxima
        qgev[ii, jj, :] = down.gev_quant(Fi, csi, psi, mu)


with h5py.File(os.path.join(cfun.outdir_data, outname), "w") as fr:
    fr['qmev'] = qmev
    fr['qgev'] = qgev
    fr['Tr'] = TR
    fr['maxmiss'] = maxmiss
    fr['thresh'] = thresh
    fr['lat'] = tmpalat
    fr['lon'] = tmpalon

