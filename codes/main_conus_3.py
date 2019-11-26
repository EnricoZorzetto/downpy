
'''----------------------------------------------------------------------------
Main analysis for the conus project
----------------------------------------------------------------------------'''
import os
import numpy as np
import time
import pandas as pd
import conusfun as cfun
import xarray as xr
import pickle


# def typemat(value, nlons, nlats):
#     '''return a numpy array of size nlons*nlats
#        of the same type as value (int, bool or float)'''
#     mytype = type(value)
#     # print(mytype)
#     res = np.zeros((nlons, nlats), dtype=mytype)*np.nan
#     return res

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


    INPUT = pickle.load( open( os.path.join(
                                cfun.pickletemp, "inputlist.p"), "rb" ) )

    lons = pickle.load( open( os.path.join(
                                cfun.pickletemp, "lons.p"), "rb" ) )

    lats = pickle.load( open( os.path.join(
                                cfun.pickletemp, "lats.p"), "rb" ) )

    nlons = np.size(lons)
    nlats = np.size(lats)


    ninput = len(INPUT)

    RES = []
    for i in range(ninput):
        try:
            resdict = pickle.load(open(
                    os.path.join(cfun.pickletemp, "resdict_{}.p"
                    .format(i)), "rb" ) )
        except:
            print('missing_file_{}'.format(i))
            resdict = {}

        RES.append(resdict)

    # # for debugging purposes only
    # with open(os.path.join(cfun.outdir_data, 'keys.txt'), 'w') as file1:
    #     for i in range(ninput):
    #         mylist = list(RES[i].keys())
    #         file1.write("%s " % i)
    #         for item in mylist:
    #             file1.write("%s " % item)
    #         file1.write("\n")

    # save results in a single pandas data frame for stats analyses:
    # but first let us drop the variables with too many dimensions
    dfres = pd.DataFrame(RES)
    to_drop = ['mev_s', 'mev_g', 'mev_d', 'mev_s_all', 'gev_s_all', 'Tr',
               'NYs', 'NYd', 'CYs', 'CYd', 'WYs', 'WYd',
                                    'NYg', 'CYg', 'WYg']# remove

    for elem in to_drop:
        if elem in dfres.columns:
            dfres.drop(elem, inplace=True, axis=1)

    dfres.to_csv(os.path.join(cfun.outdir_data,
              'dfres_cluster_{}.csv'.format(ninput)))


    # # now save from list to dictionary of arrays - scalars only
    mydict = {key:typemat(value, nlons, nlats) for key, value in zip(
                RES[0].keys(), RES[0].values())}
    print(list(mydict.keys()))
    print("****************************")

    # now repeat for the first complete pixel:
    for item in RES:
        if 'Cd' in item.keys(): # if TRMM values too, add them!
            mydict = {key:typemat(value, nlons, nlats) for key, value in zip(
                            item.keys(), item.values())}
            break
    print(list(mydict.keys()))

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
    # for quantiles: mev_s, mev_s_all, mev_g, mev_d:
    quant_vars = ['mev_s', 'mev_g', 'mev_d', 'mev_s_all', 'gev_s_all']
    for key, value in zip(mydict.keys(), mydict.values()):
        if np.ndim(value) == 3 and key in quant_vars:
            dset[key] = xr.DataArray(value,
                                     coords = [lons, lats, cfun.Tr],
                                     dims = ['lon', 'lat', 'TR'])

    # quant_vars = ['mev_s', 'mev_g', 'mev_d', 'mev_s_all']
    # for key, value in zip(mydict.keys(), mydict.values()):
    #     if np.ndim(value) == 3 and key in quant_vars:
    #         dset[key] = xr.DataArray(value,
    #                                  coords = [lons, lats, cfun.Tr],
    #                                  dims = ['lon', 'lat', 'TR'])

    yearly_vars = ['NYs', 'NYd', 'CYs', 'CYd', 'WYs', 'WYd',
                   'NYg', 'CYg', 'WYg']# remove
    for key, value in zip(mydict.keys(), mydict.values()):
        if np.ndim(value) == 3 and key in yearly_vars:
            nyears_tmpa = np.shape(value)[2]
            dset[key] = xr.DataArray(value,
                         coords = [lons, lats, np.arange(nyears_tmpa)],
                         dims = ['lon', 'lat', 'years'])



    dset.to_netcdf( os.path.join(cfun.outdir_data,
                         'ncres_cluster_{}.nc'.format(ninput)), mode='w')

    exec_time = time.time() - init_time
    print('Execution time was  = {} minutes'.format(exec_time/60))


