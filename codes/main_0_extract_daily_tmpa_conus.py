# given the HDF file with 3-hr rainfall data,
# construct the one with daily accumulations

import os
import time
import numpy as np
# from pyhdf.SD import SD, SDC
import h5py


# Office_PC = False
# Cluster = True
# if Office_PC:
#     datadir ='C:\\Users\\ez23\\Desktop\\data_3B42_3hourly'
#     outdir  = 'C:\\Users\\ez23\\Desktop\\data_TMPA_conus_time_series'
# elif Cluster:
outdir  = os.path.join('..', 'data', 'tmpa_conus_data')
start_time = time.time()

onlyconus=False


if onlyconus: # for CONUS domain
    inputfile = 'data_tmpa_3h.hdf5'
    outputfile = 'data_tmpa_daily.hdf5'
else: # for WORLD domain
    inputfile = 'data_tmpa_world_3h.hdf5'
    outputfile = 'data_tmpa_world_daily.hdf5'


with h5py.File( os.path.join(outdir, inputfile), 'r') as f:
    trset = f['prcp']
    mylat = f['lat'][:]
    mylon = f['lon'][:]
    alldates = f['dates'][:]
    # also read and write attributes of interest
    nb = trset.attrs['north_bound']
    sb = trset.attrs['south_bound']
    wb = trset.attrs['west_bound']
    eb = trset.attrs['east_bound']
    sd = trset.attrs['start_date']
    ed = trset.attrs['end_date']
    sr =  trset.attrs['space_res']
    fc = trset.attrs['first_corner']
    rows = trset.attrs['rows']
    cols = trset.attrs['cols']

    nblon = np.size(mylon)
    nblat = np.size(mylat)

    # we compute daily totals only where all 8 3-hr values are non -missing
    # and otherwise mark the day with -9999
    # note: data must be already in 3-hr accumulations
    mydates = np.unique(alldates)
    ndates = np.size(mydates)

    with h5py.File( os.path.join(outdir,outputfile), 'w') as ff:
        dset = ff.create_dataset('prcp', (nblon, nblat, ndates),
                                           chunks = (1, 1, ndates) , dtype = 'f')
        dset2 = ff.create_dataset('lat', (nblat,), dtype = 'f')
        dset2[:] = mylat
        dset3 = ff.create_dataset('lon', (nblon,), dtype = 'f')
        dset3[:] = mylon
        dset4 = ff.create_dataset('dates', (ndates,), dtype = 'int32')
        dset4[:] = mydates

        for ii in range(nblon):
            for jj in range(nblat):
                sample = trset[ii,jj,:]
                daily = np.zeros(ndates)
                for kk in range(ndates):
                    mydate = mydates[kk]
                    mysample = sample[alldates == mydate]
                    non_missing = np.logical_and(mysample > -0.1, ~np.isnan(mysample) )
                    mysample2 = mysample[non_missing]
                    if np.size(mysample2) == 8:
                        daily[kk] = np.sum(mysample2)
                        # dset[ii,jj,kk] = np.sum(mysample2)
                    else:
                        daily[kk] = -9999
                        # dset[ii,jj,kk] = -9999
                dset[ii,jj,:] = daily

        dset.attrs['north_bound'] = nb
        dset.attrs['south_bound'] = sb
        dset.attrs['west_bound'] = wb
        dset.attrs['east_bound'] = eb
        dset.attrs['start_date'] =sd
        dset.attrs['end_date'] = ed
        dset.attrs['variable'] = 'PRCP Daily Accumulation [mm/day]'
        dset.attrs['time_res'] = 'daily'
        dset.attrs['space_res'] = sr
        dset.attrs['chunks_shape'] = '1 x 1 x ndates'
        dset.attrs['first_corner'] = fc
        dset.attrs['rows'] = rows
        dset.attrs['cols'] = cols







# TIME of EXECUTION of the script
execution_time = time.time() - start_time
print('extract_bounding_box:')
print("---execution time was %s minutes ---" % (execution_time/60))


