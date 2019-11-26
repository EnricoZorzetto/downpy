
'''----------------------------------------------------------------------------
Main analysis for the conus project
Using parallel processes for a single machine (eg my laptop)
----------------------------------------------------------------------------'''
import os
import numpy as np
import time
import pandas as pd
import conusfun as cfun
import xarray as xr
from multiprocessing import Pool


def analyze_cell_wrapper(arg):
    args, kwargs = arg
    return cfun.analyze_cell(*args, **kwargs)




def typemat(value, nlons, nlats):
    '''return a numpy array of size nlons*nlats*np.size(value)
       of the same type as value (int, bool or float)
       if value is array, return that as the third dimension'''
    nd = np.ndim(value)
    # print(nd)
    if nd >= 1:
        m = np.size(value)
        mytype = type(value[0])
        # print(m)
        # print(mytype)
        res = np.zeros((nlons, nlats, m), dtype=mytype)
        if np.issubdtype(mytype, np.number):
            res = res*np.nan
    elif nd == 0:
        mytype = type(value)
        res = np.zeros((nlons, nlats), dtype=mytype)
        if np.issubdtype(mytype, np.number):
            res = res*np.nan
    else: # case ndim   > 1:
        print('typemat ERROR: value must be scalar or 1D array')
        res = np.array([])
    return res


if __name__ == '__main__':
    init_time = time.time()

    if not os.path.exists(cfun.outdir_data):
        os.makedirs(cfun.outdir_data)
    # if not os.path.exists(cfun.pickletemp):
    #     os.makedirs(cfun.pickletemp)

    # read list of stations and bounding box coordinates:
    sdf = pd.read_csv( cfun.stat_list_file, index_col = 0)
    xconus = cfun.read_gridded_data(cfun.tmpa_hdf_file)
    lats = xconus.lat.values
    lons = xconus.lon.values
    lats = lats[ np.logical_and(lats > 33.00, lats < 35.50)]
    lons = lons[ np.logical_and(lons > -99.00, lons < -97.50)]

    # extreme value analysis for all gauges
    cfun.gauge_evd(cfun.Tr, cfun.gauges_dir,
                   cfun.stat_list_file, cfun.outdir_data)

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

    p = Pool(processes=cfun.nprocesses)
    RES = p.map(analyze_cell_wrapper, INPUT)
    # RES = [analyze_cell_wrapper(argi) for argi in INPUT] # if not parallel

    ninput = len(INPUT)
    # save results in a single pandas data frame for stats analyses:
    dfres = pd.DataFrame(RES)
    to_drop = ['mev_s', 'mev_g', 'mev_d', 'mev_s_all', 'gev_s_all', 'Tr',
               'NYs', 'NYd', 'CYs', 'CYd', 'WYs', 'WYd',
                                    'NYg', 'CYg', 'WYg'] # remove


    for elem in to_drop:
        if elem in dfres.columns:
            dfres.drop(elem, inplace=True, axis=1)

    dfres.to_csv(os.path.join(cfun.outdir_data,
              'dfres_laptop_parallel_{}.csv'.format(ninput)))



    # # now save from list to dictionary of arrays - scalars only
    mydict = {key:typemat(value, nlons, nlats) for key, value in zip(
                RES[0].keys(), RES[0].values())}

    # now repeat for the first complete pixel:
    for item in RES:
        if 'Cd' in item.keys(): # if TRMM values too, add them!
            mydict = {key:typemat(value, nlons, nlats) for key, value in zip(
                            item.keys(), item.values())}
            break

    for item in RES:
        for key, value in zip(item.keys(), item.values()):
            if np.isscalar(value):
                mydict[key][item['i'], item['j']] = item[key]
            else: # array of size 1
                mydict[key][item['i'], item['j'], :] = item[key]

    mycoords = ['lon', 'lat']
    dset = xr.Dataset({key:(mycoords, value) for key, value in
                  zip(mydict.keys(), mydict.values()) if np.ndim(value) == 2},
                  coords = {'lon':lons,
                            'lat':lats})

    # add dataset variables with 3 coordinates:
    # only return times for now
    # for key, value in zip(mydict.keys(), mydict.values()):
    #     if np.ndim(value) == 3:
    #         dset[key] = xr.DataArray(value,
    #                                  coords = [lons, lats, cfun.Tr],
    #                                  dims = ['lon', 'lat', 'TR'])

    quant_vars = ['mev_s', 'mev_g', 'mev_d', 'mev_s_all', 'gev_s_all']
    for key, value in zip(mydict.keys(), mydict.values()):
        if np.ndim(value) == 3 and key in quant_vars:
            dset[key] = xr.DataArray(value,
                                     coords = [lons, lats, cfun.Tr],
                                     dims = ['lon', 'lat', 'TR'])

    yearly_vars = ['NYs', 'NYd', 'CYs', 'CYd', 'WYs', 'WYd',
                                        'NYg', 'CYg', 'WYg'] # remove

    for key, value in zip(mydict.keys(), mydict.values()):
        if np.ndim(value) == 3 and key in yearly_vars:
            nyears_tmpa = np.shape(value)[2]
            dset[key] = xr.DataArray(value,
                         coords = [lons, lats, np.arange(nyears_tmpa)],
                         dims = ['lon', 'lat', 'years'])



    dset.to_netcdf( os.path.join(cfun.outdir_data,
                         'ncres_laptop_parallel_{}.nc'.format(ninput)), mode='w')

    # with open(os.path.join(cfun.outdir_data, 'numberofjobs.txt'), 'w') as file1:
    #     file1.write(str(ninput))

    exec_time = time.time() - init_time
    print('Execution time was  = {} minutes'.format(exec_time/60))


