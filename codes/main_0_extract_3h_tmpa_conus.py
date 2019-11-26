# load daily totals from TMPA 3b42
# and extract time series for a given bounding box.
# Note this may lead to intensive memory usage:
# limit to small bounding box sizes


import os
import time
import numpy as np
from pyhdf.SD import SD, SDC
import h5py


# # boundaries of the selected bounding box
box_name = 'Conus'
solat = 22    # south bound
nolat = 50    # north
welon = -130  # west
ealon = - 60   # east


outdir  = os.path.join('..', 'data', 'tmpa_conus_data')
datadir  = os.path.join('..', 'data', 'tmpa_raw_data')


chnunkshape = (1, 1, 1000)
start_time  = time.time() # track execution time
filenames   = sorted([f for f in os.listdir(datadir) if f.endswith('.HDF')], 
                key = lambda name: name[5:13]+name[14:16])
# filenames = filenames[:1000:]
numfiles    = np.size(filenames)
# range_files = np.arange(numfiles)
#
# years_all   =np.array( [int(filenames[ii][5:9]) for ii in range(numfiles) ] )
# years = np.unique(years_all)
# nyears      = np.size(years)
# print('Hello World')

# dates  = np.array([int(filenames[ii][5:13]) for ii in range(numfiles) ] )
# hours   = np.array([int(filenames[ii][14:16]) for ii in range(numfiles) ] )

###############################################################################
# insert lat and long manually - see
# http://hdfeos.org/zoo/GESDISC/TRMM_3B42_precipitation_scan0.py
# rem go slightly beyond last step in python
lat    = np.arange(-49.875, 49.876, 0.25) # South to North
lon   = np.arange(-179.875, 179.876, 0.25) # West to East
nlon        = np.size(lon)
nlat        = np.size(lat)
###############################################################################

# mask arrays for selected  bounding box
bblat = np.logical_and(lat >= solat, lat <= nolat)
bblon = np.logical_and(lon >= welon, lon <= ealon)

boxlat = lat[bblat]
boxlon = lon[bblon]

boxx = np.arange(nlon, dtype=int)[bblon]
boxy = np.arange(nlat, dtype=int)[bblat]

nblat = np.size(boxy)
nblon = np.size(boxx)

with h5py.File( os.path.join(outdir, 'data_tmpa_3h.hdf5'), 'w') as f:
    for tt in range(numfiles):
        print(tt, filenames[tt])
        # read
        fullname = os.path.join(datadir, filenames[tt])
        hdf      = SD(fullname, SDC.READ)
        # read only prcp over conus
        prcpmat_rates = hdf.select('precipitation')[int(boxx[0]):int(
                                boxx[-1]+1), int(boxy[0]):int(boxy[-1]+1)]
        prcpmat = prcpmat_rates*3 # accumulations
        if tt == 0:
            dset = f.create_dataset('prcp', (nblon, nblat, numfiles),
                                       chunks = chnunkshape , dtype = 'f')
            dset[ :,:, tt] = prcpmat # save accumulations

            dset2 = f.create_dataset('lat', (nblat,), dtype = 'f')
            dset2[:] = boxlat
            dset3 = f.create_dataset('lon', (nblon,), dtype = 'f')
            dset3[:] = boxlon
            dset4 = f.create_dataset('dates', (numfiles,), dtype = 'int32')
            # dset4[:] = hours
            dset4[tt]=int(filenames[tt][5:13])
            dset5 = f.create_dataset('hours', (numfiles,), dtype = 'int32')
            # dset5[:] = dates
            dset5[tt]=int(filenames[tt][14:16])
            dset.attrs['north_bound'] = nolat
            dset.attrs['south_bound'] = solat
            dset.attrs['west_bound'] = welon
            dset.attrs['east_bound'] = ealon
            dset.attrs['start_date'] =filenames[0][5:13]
            dset.attrs['start_time'] =filenames[0][14:16]
            dset.attrs['end_date'] =filenames[-1][5:13]
            dset.attrs['end_time'] =filenames[-1][14:16]
            dset.attrs['variable'] = 'PRCP 3-hr ACCUMULATION [mm]'
            dset.attrs['time_res'] = '3h'
            dset.attrs['space_res'] = '0.25deg'
            dset.attrs['chunks_shape'] = '1x1x1000'
            dset.attrs['first_corner'] = 'south_west as in original dataset'
            dset.attrs['rows'] = 'longitude (as in the original TMPA dataset)'
            dset.attrs['cols'] = 'latitude (as in the original TMPA datset)'
            print(dset.shape)
        else:
            dset[ :,:, tt] = prcpmat
            dset4[tt]=int(filenames[tt][5:13])
            dset5[tt]=int(filenames[tt][14:16])



# TIME of EXECUTION of the script
execution_time = time.time() - start_time
print('extract_bounding_box:')
print("---execution time was %s minutes ---" % (execution_time/60))
