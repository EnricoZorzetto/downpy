from datetime import datetime
import dask.array as da
import os
import time
import h5py
import numpy as np
import xarray as xr
import downscale as down
import matplotlib.pyplot as plt


down.matplotlib_update_settings()

'''--------------------------------------------------------------------------
load precipitation data over conus (3-hr totals)
And test the functions is the downscaling package
--------------------------------------------------------------------------'''

# load data from the hdf file with TMPA 3b42 accumulations:
def extract_sample_data(inputfile, outputfile,
                        clat=34.875, clon=-98.125, npix=3, dx=0.25):
    '''------------------------------------------------------------------------
    Extract a given bounding box centered in (clat, clon) with size npix
    grid cells from an input hdf file
    and save it as nc dataset for further analysis
    ------------------------------------------------------------------------'''
    # f = h5py.File(os.path.join(cdat.tmpa_dir, 'data_tmpa_3h.hdf5'), "r")
    f = h5py.File(inputfile, "r")
    # print(list(f.keys()))
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
    # clat = 34.875
    # clon = -98.125
    # dx = 0.25
    # npix = 3
    buffer = 0.50*npix*dx
    eps = 1e-4 # to make sure to include boundaires -> add an eps buffer
    # eps = 0 # to make sure to include boundaires -> add an eps buffer
    solat = clat - buffer + eps
    nolat = clat + buffer + eps
    ealon = clon + buffer + eps
    welon = clon - buffer + eps
    bcond = np.logical_and(
                np.logical_and( xrs.lat > solat, xrs.lat < nolat),
                np.logical_and( xrs.lon > welon, xrs.lon < ealon))
    # XARRAY WITH VALUES LOADED IN MEMORY
    box_3h = xrs.where(bcond, drop = True).load()
    print(box_3h.shape)
    box_3h.to_netcdf(outputfile)


extract_box = True # do it only once
# inputfile = os.path.join('..', cdat.tmpa_dir, 'data_tmpa_3h.hdf5')
tmpa_dir = os.path.join('..', 'data', 'tmpa_conus_data')
inputfile = os.path.join(tmpa_dir, 'data_tmpa_3h.hdf5')
outputdir = os.path.join('..', 'data', 'sample_data')
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
# outputfile = os.path.join('..', 'sample_data','okla.nc')
outputfile = os.path.join(outputdir,'okla.nc')
clat = 34.875
clon = -98.125
dx = 0.25
npix = 3
if extract_box:
    extract_sample_data(inputfile, outputfile,
                        clat=clat, clon=clon, npix=npix,dx=dx)


# box_3h = xr.open_dataset(outputfile)
box_3h = xr.open_dataarray(outputfile)
Tr = 100

res = {}  # initialize dictionary for storing results

# ######
# xdata = box_3h
# tscale = 24
# xdata = xdata.where(xdata >= -0.001)  # set negative values to np.nan if any
# xdaily0 = xdata.resample(time='{}H'.format(tscale)).sum(
#     dim='time', skipna=False)
# xdaily = xdaily0.dropna(dim='time', how='any')
# lons = xdata.lon.values
# lats = xdata.lat.values
# nlon = np.size(lons)
# nlat = np.size(lats)
# dx = np.abs(lons[1] - lons[0])
# if nlon != nlat:
#     print('downscale warning: box sizes are not equal')
# if nlon % 2 == 0:
#     print('downscale warning: at least one box size has even length')
# # TODO: if passed clat / clon, overwrite this
# if (bool(clat) and bool(clon) and clat in lats and clon in lons):
#     clat = lats[np.argmin(np.abs(clat - lats))]
#     clon = lons[np.argmin(np.abs(clon - lons))]
#     # otherwise us the one provided by the
# else:
#     clat = lats[np.argmin(np.abs(np.mean(lats) - lats))]
#     clon = lons[np.argmin(np.abs(np.mean(lons) - lons))]
# L1 = 25
# thresh = 1
# tsc = xdaily.loc[dict(lat=clat, lon=clon)]
# c_excesses = np.maximum(tsc.values - thresh, 0.0)
# ######


# downscaling:
print('Example: running the downscaling function')
init_time = time.time()
downres = down.downscale(box_3h, Tr, thresh=1.0, L0=0.0001, toll=0.005,
                         acf='mar', save_yearly=True,
                         maxmiss=36, clat=clat, clon=clon,
                         opt_method='genetic', plot=True)
end_time = time.time()
exec_time = end_time - init_time
print('Downscaling, execution time was = {} minutes'.format(exec_time/60))

downres['Taylor_contour'].show()
downres['corr_plot'].show()



plt.figure()
plt.plot(downres['CYs'], downres['CYd'], 'o')
plt.plot(downres['CYs'], downres['CYs'], 'k')
plt.show()


plt.figure()
plt.plot(downres['WYs'], downres['WYd'], 'o')
plt.plot(downres['WYs'], downres['WYs'], 'k')
plt.show()



