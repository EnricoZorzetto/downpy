# main functions for the conus project

# TODO: move the main_0 codes to the download folder,
# TODO: or add them to the main workflow
# TODO: add options for EV of gauges and trmm-domain

# TODO: separate QRF results and plots
# TODO: check that mev_s_all, years and hdf quantiles are consistent
# TODO: fix or remove the multiprocessing code

# TODO: for testing: move sample data to its own folder
# TODO: improve testing
# TODO: speed up EV analysis - multiprocessing

import os
import h5py
import dask.array as da
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import downscale as down
from datetime import datetime
import xarray as xr


# project directories
# tmpa_dir = os.path.join('..', 'data', 'tmpa_conus_data')
tmpa_dir = os.path.join('..', 'data', 'tmpa_conus_data')
outdir_data = os.path.join('..', 'output', 'pixel_stats')
outplot = os.path.join('..', 'output', 'outplot')
# list_gauges_dir = os.path.join(tmpa_dir, 'gauges_noaa_hpd')
list_gauges_dir = os.path.join('..', 'data', 'data_noaa_hpd_gauges')
gauges_dir = os.path.join(list_gauges_dir, 'daily_csv')
stat_list_file = os.path.join(list_gauges_dir, 'HOURLY_LOC_NYEARS.csv')
tmpa_hdf_file = os.path.join(tmpa_dir, 'data_tmpa_3h.hdf5')
pickletemp = os.path.join('..','output','pickletemp')
elev_dir = os.path.join('..', 'data', 'data_elevation')


#### QUANTITIES TO SET FOR ANALYSIS: ####
Tr = np.array([10, 20, 50, 100]) # return times for extreme value analysis
evd_domain = 'world' # (can be = 'conus' or 'world' for EV analysis only)
do_evd_all_gauges = True # do EV for all gauges in the dataset
# do_evd_all_tmpa = True   # do EV for all pixels in evd_domain (WORLD or CONUS)
do_trmm_evd = True # to fit MEV to each grid cell over CONUS
do_trmm = True # to downscale trmm where gauges are available
do_gauges = True # to compute gauge stats where there are enough


################# added for reading elevation::
# Boundaries of the CONUS domain and TRMM grid coordinates
solat = 22    # south bound
nolat = 50    # north
welon = -130  # west
ealon = - 60   # east
dx = 0.25
lats = np.arange(-49.875, 49.876, 0.25) # south to North
lons = np.arange(-179.875, 179.876, 0.25) # West to East
nlats = np.size(lats)
nlons = np.size(lons)
#################

# kwargs in input for the function analyze_cell
pixelkwargs = {
        'npix':3, # box size for analysis, in number of grid cells
        'npix_gauges':5, # size of the box for selecting gauges, in grid cells
        'dx':dx, # size of a grid cell (dx = 0.25 for TMPA)
        'minstat_bb':4, # number of gauges required over npix_gauges**2 area
        'minstat_pixel':1, # number of gauges required over the grid cell
        'min_nyears_pixel':10, # min record length (years) to select a gauge
        'min_overlap_corr':2000, # compute correlation if at least 2000 obs.
        'thresh':1, # [mm] precipitation magnitude threshold
        'L0':0.0001, # rain gauge point scale [km]
        'acf':'mar', # autocorrelation function used
        'dt':3, # hours - timescale of TMPA dataset
        'plot':False, # plot results (set False when running on the Cluster)
        'tscale':24, # daily timescale to perform analysis
        'save_yearly':True, # save yearly Weibull parameters
        'toll':0.005, # for optimization algorithm for correlation downscaling
        'maxmiss':36, # max number of missing daily data in each year of record
        'opt_method':'genetic', # opt algorithm for corr. downscaling
        'do_trmm_evd':do_trmm_evd,   # to fit MEV to each grid cell over CONUS
        'do_gauges':  do_gauges,   # to compute gauge stats where there are enough
        'do_trmm':    do_trmm,   # to downscale trmm where gauges are available
        'do_smoke':   False   # to test the pixel positions
        }


def tmpa_evd(clon, clat, tmpa_hdf_file, Tr, *,
             thresh=1, maxmiss=36):
    """ extreme value analysis for a tmpa grid cell pixel
    load the pixel centered in (clon, clat) from the
    dask - xarray stored in the hdf file named tmpa_hdf_file,
    and compute quantiles for the range of return times in the array Tr
    optional: thresh = 1 theshold for MEV
              maxmiss = 36 max number of missing data / year
    Do not perform analysis if dry years have less than 2 events"""
    res_evd = {}
    Fi = 1 - 1 / Tr
    xconus = read_gridded_data(tmpa_hdf_file)
    # xdata = load_bounding_box(xconus, clon, clat, 1)
    # print(xdata.shape)
    xconus = xconus.where(xconus >= -0.001)
    xpixel = xconus.sel(lat=clat, lon=clon).resample(time='D').sum(
        dim='time', skipna=False).dropna(dim='time',
                                         how='any').load()
    ts = xpixel.values
    years = xpixel.time.dt.year.values
    df = pd.DataFrame({'PRCP': ts, 'YEAR': years})
    df, ny2, ny1 = down.remove_missing_years(df, maxmiss)
    Ny, Cy, Wy = down.mev_fit(df, thresh=thresh)
    x0 = 9.0 * np.mean(Cy)
    mevq = down.mev_quant(Fi, x0, Ny, Cy, Wy, thresh=thresh)[0]
    XIemp, Fiemp, TRemp = down.tab_rain_max(df)
    csi, psi, mu = down.gev_fit_lmom(XIemp)  # fit to annual maxima
    gevq = down.gev_quant(Fi, csi, psi, mu)
    res_evd['mev_s_all'] = mevq
    res_evd['gev_s_all'] = gevq
    return res_evd


def gauge_evd(Tr, gauges_dir, stat_list_file, output_dir,
              *, nyears_min=10, maxmiss=36, thresh=1):
    '''----------------
    compute evd statistics for all gauges in the dataset
    with enough complete years of data
    ----------------'''
    sdf = pd.read_csv(stat_list_file, index_col = 0)
    nstats = np.shape(sdf)[0]
    ntr = np.size(Tr)
    Fi = 1-1/Tr
    nyearsg = np.zeros(nstats, dtype = int)
    mev_g_all = np.zeros((nstats, ntr))*np.nan
    gev_g_all = np.zeros((nstats, ntr))*np.nan
    for i in range(nstats):
        df0 = pd.read_csv( os.path.join(gauges_dir,
                 '{}.csv'.format(sdf['ID'].values[i])))
        df = df0[(df0['PRCP'] > -0.1) & (~np.isnan(df0['PRCP']))]
        df, ny2, ny1 = down.remove_missing_years(df, maxmiss)
        XIemp, Fiemp, TRemp = down.tab_rain_max(df)
        nyearsg[i] = ny2
        if nyearsg[i] >= nyears_min:
            Ny, Cy, Wy = down.mev_fit(df, thresh=thresh)
            x0 = 6.0*np.mean(Cy)
            mev_g_all[i, :] = down.mev_quant(Fi, x0, Ny, Cy, Wy,
                                        thresh=thresh)[0]
            csi, psi, mu = down.gev_fit_lmom(XIemp)
            gev_g_all[i,:] = down.gev_quant(Fi, csi, psi, mu)
    # subset dataframe keeping only long enough time series:
    sdf['nyearsg'] = nyearsg
    for i in range(ntr):
        sdf['mev_{}'.format(Tr[i])] = mev_g_all[:,i]
        sdf['gev_{}'.format(Tr[i])] = gev_g_all[:,i]
    sdf = sdf[sdf['nyearsg'] >= nyears_min]
    ngauges = np.shape(sdf)[0]
    sdf.to_csv(os.path.join(output_dir, 'dfres_gauges_{}.csv'.format(ngauges)))
    return sdf


def gauge_stats(clon, clat, df, Tr, gauges_dir, *, npix=5, dx=0.25,
                minstat_bb=4,
                minstat_pixel=1, thresh=1,
                min_nyears_pixel=10, min_overlap_corr=2000,
                maxmiss=36):
    '''------------------------------------------------------------------------
    gauge_stats:
    Computes the statistics for longest-record gauge in the pixel (clon, clat)
    if it is at least min_nyears_pixel years long,
    and compute correlation between gauges in a npix*npix bounding box
    if there are at least minstat_bb gauges with min common record of
    min_overlap correlation.
    Returns dictionary with corrlation and local gauge Weibull C, W, N
    and if there were enough gauges in the pixel / bounding box
    INPUT:
    clon = longitude central pixel point
    clat - latitude central pixel point
    df = data frame with list of stations, extracted by NOAA HPD, daily scale
    ------------------------------------------------------------------------'''

    # default values of output variables if not enough stations at the ground:
    enough_gauges_bb = False
    enough_gauges_pixel = False
    alpha = np.nan
    epsilon = np.nan
    d0 = np.nan
    mu0 = np.nan
    pwet = np.nan
    C = np.nan
    W = np.nan
    N = np.nan
    # Cy = np.zeros(min_nyears_pixel)*np.nan
    # Wy = np.zeros(min_nyears_pixel)*np.nan
    # Ny = np.zeros(min_nyears_pixel)*np.nan
    gam_g = np.nan
    nyearsg = np.nan
    mev_g = np.zeros(np.size(Tr))*np.nan
    # read stations within the box
    wb = clon - npix/2*dx
    eb = clon + npix/2*dx
    nb = clat + npix/2*dx
    sb = clat - npix/2*dx
    wbpix = clon - 1/2*dx # pixel
    ebpix = clon + 1/2*dx
    nbpix = clat + 1/2*dx
    sbpix = clat - 1/2*dx
    # stations within the central pixel and the bounding box
    mydf = df[ (df['LAT'] < nb) & (df['LAT'] > sb) &
               (df['LON'] > wb) & (df['LON'] < eb) ]
    mydfc = df[ (df['LAT'] < nbpix) & (df['LAT'] > sbpix)
                & (df['LON'] > wbpix) & (df['LON'] < ebpix) ]
    nstations_bb = np.shape(mydf)[0] # numebr of stats in bounding box
    nstations_pixel = np.shape(mydfc)[0] # number of stats in central pixel
    # compute empirical correlation
    if nstations_bb >= minstat_bb:
        vdist = []
        vcorr = []
        for iii in range(nstations_bb):
            dfi0 = pd.read_csv( os.path.join(gauges_dir,
                 '{}.csv'.format(mydf['ID'].values[iii])))
            dfi = dfi0[(dfi0['PRCP'] > -0.1) & (~np.isnan(dfi0['PRCP']))]
            dates_ii = dfi['DATE'].values
            for jjj in range(iii + 1, nstations_bb):
                dfj0 = pd.read_csv( os.path.join(gauges_dir,
                     '{}.csv'.format(mydf['ID'].values[jjj])))
                dfj = dfj0[(dfj0['PRCP'] > -0.1) & (~np.isnan(dfj0['PRCP']))]
                dates_jj = dfj['DATE'].values
                commondates = np.intersect1d(dates_ii, dates_jj)
                sample_ii = dfi['PRCP'].values[dfi['DATE'].isin(commondates)]
                sample_jj = dfj['PRCP'].values[dfj['DATE'].isin(commondates)]
                if np.size(sample_ii) > min_overlap_corr:
                    excesses_ii = np.maximum(sample_ii - thresh, 0.0)
                    excesses_jj = np.maximum(sample_jj - thresh, 0.0)
                    vcorr.append(np.corrcoef(excesses_ii, excesses_jj)[0,1])
                    vdist.append(down.haversine(mydf['LAT'].values[iii],
                                                mydf['LAT'].values[jjj],
                                                mydf['LON'].values[iii],
                                                mydf['LON'].values[jjj]
                                                ))
        # fit acf function
        if len(vdist) >= minstat_bb:
            try:
                popt0, pcov0 = curve_fit(down.epl_fun,
                             np.array(vdist), np.array(vcorr),
                             p0 = np.array([50.0, 1.0]),
                             bounds = ((0.0, 0.0), (+np.inf, +np.inf)))
                epsilon = popt0[0]
                alpha = popt0[1]
                enough_gauges_bb = True
                L = down.area_lat_long(clat, clon, dx, dx)[0]
                L0 = 0.0001
                gam_g = down.vrf(L, L0, (epsilon, alpha), acf='mar')
                popt1, pcov1 = curve_fit(down.str_exp_fun,
                             np.array(vdist), np.array(vcorr),
                             p0 = np.array([50.0, 1.0]),
                             bounds = ((0.0, 0.0), (+np.inf, +np.inf)))
                d0 = popt1[0]
                mu0 = popt1[1]
            except:
                print('gauge_stats WARNING: \n'
                      'pass - not possible to compute correlation reliably')
    # fit Weibull to the longest station in the central pixel
    if nstations_pixel >= minstat_pixel:
        vec_nyears = mydfc['NYEARS'].values
        if np.max(vec_nyears) >= min_nyears_pixel: # at least 10 years of data
            # enough_gauges_pixel = True
            long_index = np.argmax(mydfc['NYEARS'].values)
            dfl0 = pd.read_csv( os.path.join(gauges_dir,
                         '{}.csv'.format(mydfc['ID'].values[long_index])))
            dfl = dfl0[ (dfl0['PRCP'] > -0.1) & (~np.isnan(dfl0['PRCP']) )]
            sample = dfl['PRCP'].values
            excesses = sample[sample > thresh] - thresh
            NCWg = down.wei_fit(excesses)
            pwet = NCWg[0]/np.size(sample)
            C = NCWg[1]
            W = NCWg[2]
            N = np.int(np.rint(pwet*365.25))
            # fit MEV
            # TODOS: add option to save yearly parameters here if needed
            dfl, ny2, ny1 = down.remove_missing_years(dfl, maxmiss)
            if ny2 >= min_nyears_pixel:
                enough_gauges_pixel = True
                Ny, Cy, Wy = down.mev_fit(dfl, thresh=thresh)
                # nyearsg = np.size(Ny)
                Fi = 1-1/Tr
                x0 = 6.0*np.mean(Cy)
                mev_g = down.mev_quant(Fi, x0, Ny, Cy, Wy, thresh=thresh)[0]

    # save results in dictionary:
    res_gauges = {'Cg':C,
                  'Wg':W,
                  'Ng':N,
                  'pwg':pwet,
                  'enough_gauges_pixel':enough_gauges_pixel,
                  'enough_gauges_bb':enough_gauges_bb,
                  'ngauges_bb':nstations_bb,
                  'ngauges_pixel':nstations_pixel,
                  'nyears_gauge':nyearsg,
                  'alp_g':alpha,
                  'eps_g':epsilon,
                  'd0_g':d0,
                  'mu0_g':mu0,
                  'Tr':Tr,
                  'gam_g':gam_g,
                  'mev_g':mev_g
                 }
    return res_gauges


def read_gridded_data(tmpa_hdffile):
    # f = h5py.File(os.path.join(datadir, 'data_tmpa_3h.hdf5'), "r")
    f = h5py.File(tmpa_hdffile, "r")
    # print(list(f.keys()))
    tmpalat = f['lat'][:]
    tmpalon = f['lon'][:]
    dates_int = f['dates'][:]
    hours_int = f['hours'][:]
    dset = f['prcp']
    # print('dataset shape = {}'.format(dset.shape))
    x = da.from_array(dset, chunks=(6, 6, 300))
    dates = [datetime.strptime(str(integd)+str(inthour), '%Y%m%d%H')
             for integd, inthour in zip(dates_int, hours_int)] # UTC time
    xconus = xr.DataArray(x,
                      coords={'lon':tmpalon, 'lat':tmpalat, 'time':dates},
                      dims=('lon', 'lat', 'time'))
    return xconus


def load_bounding_box(xconus, clon, clat, npix, dropna = False):
    ''' load data within the bounding box in memory
    from an out-of-memory xarray + dask array
    DOES NOT REMOVE MISSING DATA, BUT SET THEM TO NANS'''
    xconus = xconus.where(xconus >= -0.001)
    lons = xconus.lon.values
    dx = np.abs(lons[1] - lons[0])
    buffer = 0.50*npix*dx
    eps = 1e-4 # to make sure to include boundaires -> add an eps buffer
    solat = clat - buffer + eps
    nolat = clat + buffer + eps
    ealon = clon + buffer + eps
    welon = clon - buffer + eps
    bcond = np.logical_and(
                np.logical_and( xconus.lat > solat, xconus.lat < nolat),
                np.logical_and( xconus.lon > welon, xconus.lon < ealon))
    # Load in memory the bounding box of interest
    if dropna:
        xdata = xconus.where(bcond, drop = True).load()
    else:
        xdata = xconus.where(bcond, drop = True
                             ).dropna(dim='time', how='any').load()

    return xdata


def analyze_cell(i, j, clon, clat, Tr, stat_list_file, tmpa_hdf_file,
                gauges_dir, *,
                npix=3, npix_gauges=5, dx=0.25,
                minstat_bb=4, minstat_pixel=1,
                min_nyears_pixel=10, min_overlap_corr=2000,
                thresh=1,
                L0=0.0001,
                acf='mar', dt=3, plot=False, tscale=24,
                save_yearly = True, toll=0.005, maxmiss=36,
                opt_method='genetic',
                do_smoke = True,
                do_trmm = True,
                do_gauges = True,
                do_trmm_evd = True):
    '''------------------------------------------------------------------------
    analyze gauge data and gridded qpes for a bouding box of size npix
    centered in clat, clon (indexes i, j respectively)
    sdf = list of station coordinates and names
    ------------------------------------------------------------------------'''
    # compute some basic statistics, as NCW
    res_smoke = {}
    if do_smoke:
        res_smoke['clon'] = clon
        res_smoke['clat'] = clat

    res_evd = {}
    if do_trmm_evd:
        res_evd = tmpa_evd(clon, clat, tmpa_hdf_file, Tr,
                           thresh=thresh, maxmiss=maxmiss)

    res_gauges = {}
    res_tmpa = {}
    if do_gauges:
        sdf = pd.read_csv(stat_list_file, index_col=0)
        res_gauges = gauge_stats(clon, clat, sdf, Tr, gauges_dir,
                    npix=npix_gauges, dx=dx,  thresh=thresh,
                    minstat_bb=minstat_bb, minstat_pixel=minstat_pixel,
                    min_nyears_pixel=min_nyears_pixel,
                    min_overlap_corr=min_overlap_corr, maxmiss=maxmiss)

        if res_gauges['enough_gauges_bb'] and res_gauges['enough_gauges_pixel']:
            res_gauges['complete_pixel'] = True
        else:
            res_gauges['complete_pixel'] = False

        # by default analyze trmm only where enough gauges
            ## CANCELLED FOR CONUS ANALYSIS _ LONG!
        # if res_gauges['complete_pixel']:
        if do_trmm:

            xconus = read_gridded_data(tmpa_hdf_file)
            xdata = load_bounding_box(xconus, clon, clat, npix,
                                      dropna=False)

            res_tmpa = down.downscale(xdata, Tr, thresh=thresh, L0=L0,
                                      acf=acf, dt=dt,
            plot=plot, tscale=tscale, save_yearly=save_yearly, toll=toll,
            maxmiss=maxmiss, clat=clat, clon=clon, opt_method=opt_method)

    res = {'i':i, 'j':j, **res_gauges, **res_tmpa, **res_evd, **res_smoke}
    return res


def load_results_df(csvname='dfres.csv'):
    ''' load results from main analysis and add elevation data'''
    dfres = pd.read_csv(os.path.join(outdir_data, csvname), index_col=0)
    dfres['esa_d'] = dfres['eps_d']/dfres['alp_d']
    dfres['esa_s'] = dfres['eps_s']/dfres['alp_s']
    dfres['esa_g'] = dfres['eps_g']/dfres['alp_g']
    dfres['etaC'] = (dfres['Cd']-dfres['Cg'])/dfres['Cg']
    dfres['etaW'] = (dfres['Wd']-dfres['Wg'])/dfres['Wg']
    dfres['etaN'] = (dfres['Nd']-dfres['Ng'])/dfres['Ng']

    # load elevation and its stdv and add it to the dataset
    with h5py.File(os.path.join(tmpa_dir, "elev.hdf5"), "r") as fr:
        # print(list(fr.keys()))
        mean_el_conus = fr['mean_el'][:]
        stdv_el_conus = fr['stdv_el'][:]
        # npix_stdv_conus = fr['npix_stdv']
        elev_lat = fr['lat'][:]
        elev_lon = fr['lon'][:]

    nelem = dfres.shape[0]
    dfres['melev'] = np.zeros(nelem)*np.nan
    dfres['selev'] = np.zeros(nelem)*np.nan
    for i in range(nelem):
        if dfres['complete_pixel'].loc[i]:
            clon = dfres['clon'].loc[i]
            clat = dfres['clat'].loc[i]
            ii = np.argmin(np.abs(clon-elev_lon))
            jj = np.argmin(np.abs(clat-elev_lat))
            dfres.at[i, 'melev'] = mean_el_conus[ii,jj]
            dfres.at[i, 'selev'] = stdv_el_conus[ii,jj]

    # cond1 = np.logical_and(dfres['complete_pixel'].values == True, True)
    # dfresc = dfres[cond1]
    # cond2 = np.logical_and(dfresc['esa_d'].values > 25.0,
    #                        dfresc['corr_down_funval'].values < 1)
    # dfresc = dfresc[cond2]

    cond1 = dfres['complete_pixel'].values == True
    dfresc = dfres[cond1]
    num_tot_pixels = np.shape(dfresc)[0]
    print('total number of complete pixel loaded = {}'.format(num_tot_pixels))
    cond21 = dfresc['esa_d'].values > 25.0
    cond_large = dfresc['esa_d'].values > 200.0
    num_esad_above_200 = np.size(cond_large[cond_large==True])
    # cond22 = dfresc['corr_down_funval'].values < 1

    num_esad_below_25 = np.size(cond21[cond21==False])
    # num_downfun_below_1 = np.size(cond22[cond22==False])
    print(' number of complete pixels where e/a down < 25: '
          '{}'.format(num_esad_below_25))


    print(' number of complete pixels where e/a down > 200: '
          '{}'.format(num_esad_above_200))
    # print(' number of complete pixels where down fun > 1: '
    #       '{}'.format(num_downfun_below_1))
    # cond2 = np.logical_and(cond21, cond22)
    dfresc = dfresc[cond21]
    dfresc.reset_index(inplace=True)

    return dfresc


def load_results_netcdf(ncname='ncres.nc', elevname='elev.hdf5'):
    ''' load results from netcdf file in x-array
        and add elevation '''
    # load hdf file with mean elevation and its stdv and add it to the dataset
    with h5py.File(os.path.join(tmpa_dir, elevname), "r") as fr:
        mean_el_conus = fr['mean_el'][:]
        stdv_el_conus = fr['stdv_el'][:]
        elev_lat = fr['lat'][:]
        elev_lon = fr['lon'][:]
    melev = xr.DataArray(mean_el_conus,
                         coords=[elev_lon, elev_lat], dims=['lon', 'lat'])
    selev = xr.DataArray(stdv_el_conus,
                         coords=[elev_lon, elev_lat], dims=['lon', 'lat'])
    ncres = xr.open_dataset(os.path.join(outdir_data, ncname))
    ncres['melev'] = melev
    ncres['selev'] = selev
    ncres['esa_d'] = (ncres['eps_d'])/ncres['alp_d']
    ncres['esa_s'] = (ncres['eps_s'])/ncres['alp_s']
    ncres['esa_g'] = (ncres['eps_g'])/ncres['alp_g']
    ncres['etaC'] = (ncres['Cd'] - ncres['Cg'])/ncres['Cg']
    ncres['etaW'] = (ncres['Wd'] - ncres['Wg'])/ncres['Wg']
    ncres['etaN'] = (ncres['Nd'] - ncres['Ng'])/ncres['Ng']
    ncres['etaGAM'] = (ncres['gam_d'] - ncres['gam_g'])/ncres['gam_g']
    ncres['etaESA'] = (ncres['esa_d'] - ncres['esa_g'])/ncres['esa_g']

    # set values with epsilon/alpha < 25 to NaN


    return ncres
