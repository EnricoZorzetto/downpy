
'''----------------------------------------------------------------------------
Main analysis for the conus project
----------------------------------------------------------------------------'''
import os
import numpy as np
import time
import pandas as pd
import conusfun as cfun
import pickle




if __name__ == '__main__':
    init_time = time.time()

    # remove output file with number of jobs if already there
    numjobsfile = "numberofjobs.txt"
    # if os.path.exists(numjobsfile):
    #     os.remove(numjobsfile)

    # create folder with results if it does not exists already
    if not os.path.exists(cfun.outdir_data):
        os.makedirs(cfun.outdir_data)


    # read list of stations and bounding box coordinates:
    sdf = pd.read_csv( cfun.stat_list_file, index_col = 0)
    xconus = cfun.read_gridded_data(cfun.tmpa_hdf_file)
    lats = xconus.lat.values
    lons = xconus.lon.values

    # Extreme value analysis for all gauges
    if cfun.do_evd_all_gauges:
        cfun.gauge_evd(cfun.Tr, cfun.gauges_dir,
                       cfun.stat_list_file, cfun.outdir_data,
                       nyears_min=cfun.pixelkwargs['min_nyears_pixel'],
                       maxmiss=cfun.pixelkwargs['maxmiss'],
                       thresh=cfun.pixelkwargs['thresh'])

    # TODO: add here TMPA EVD analysis over CONUS / WORLD


    nlats = np.size(lats)
    nlons = np.size(lons)
    INPUT = []
    for ii in range(1, nlons-1):
        print('lon_ii', ii)
        for jj in range(1, nlats-1):
            clon = lons[ii]
            clat = lats[jj]
            myargs = (ii, jj, clon, clat,
                      cfun.Tr,
                      cfun.stat_list_file,
                      cfun.tmpa_hdf_file,
                      cfun.gauges_dir)
            INPUT.append((myargs, cfun.pixelkwargs))

    if not os.path.exists(cfun.pickletemp):
        os.makedirs(cfun.pickletemp)

    pickle.dump( INPUT, open( os.path.join(
                                cfun.pickletemp, "inputlist.p"), "wb" ) )

    pickle.dump( lons, open( os.path.join(
                                cfun.pickletemp, "lons.p"), "wb" ) )

    pickle.dump( lats, open( os.path.join(
                                cfun.pickletemp, "lats.p"), "wb" ) )

    with open(os.path.join(cfun.outdir_data, 'jobindex.txt'), 'w') as file1:
        file1.write('Summary from sbatch 2 \n')

    with open(os.path.join(numjobsfile), 'w') as file1:
        file1.write(str(len(INPUT)))



