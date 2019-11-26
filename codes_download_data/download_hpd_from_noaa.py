# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:47:19 2019

@author: ez23

Download hourly precipitation data from NOAA
"""

import tarfile
import ftplib
import os, os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import mevpy as mev

# Cluster = True
# Office_PC = False

compute_daily = True
compute_hourly = True
download_data = True


pass_file = 'user_noaa.txt'
usp = [line.rstrip('\n') for line in open(pass_file) if line.strip(' \n') != '']
user_name = usp[0]

# if Office_PC:
#     output_dir = os.path.join('C:\\', 'Users', 'ez23','Desktop','NOAA_Hourly_prcp_conus2')
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
# elif Cluster:
#     output_dir = os.path.join('..', '..', 'data', 'data_noaa_hpd_gauges')
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)


output_dir = os.path.join('..', 'data', 'data_noaa_hpd_gauges')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

outdir_daily = os.path.join(output_dir, 'daily_csv')
if not os.path.exists(outdir_daily):
        os.mkdir(outdir_daily)

outdir_hourly = os.path.join(output_dir, 'hourly_csv')
if not os.path.exists(outdir_hourly):
        os.mkdir(outdir_hourly)

   


def download_noaa_hpd(output_dir, user_name):
    parent_url = 'ftp.ncdc.noaa.gov'
    folder_2_down = '/pub/data/hpd/auto/v1/beta/'
    ftp = ftplib.FTP(parent_url)
    ftp.login('ftp', user_name)
    ftp.cwd(folder_2_down)    
    filenames = ftp.nlst(folder_2_down)    
    onlyfiles = [name for name in filenames if (len(name) > 4)
        and ( name[-4] == '.' or name[-3] == '.')]
    for filename in onlyfiles:
        # try:
        print('downloading file: {}'.format(filename))        
        local_filename = os.path.join(output_dir, filename.split('/')[-1])
        file = open(local_filename, 'wb')
        ftp.retrbinary('RETR {0}'.format(filename), file.write)
        file.close()
    ftp.quit()    
    unzip = True
    if unzip == True:
        fname = os.path.join(output_dir, 'hpd_all.tar.gz')
        print('unzipping tar.gz')
        tar = tarfile.open(fname, "r:gz")
        tar.extractall( path =  os.path.join(output_dir))
        tar.close()


# first download the files:
if download_data:
    download_noaa_hpd(output_dir, user_name)
    
# after downloading the files:        
files = os.listdir( os.path.join(output_dir, 'all'))
nfiles = len(files)    


# statmat = np.loadtxt(os.path.join(datadir, 'hpd-stations.txt'))
hdp_stations = pd.read_csv( os.path.join(output_dir,
             'hpd-stations.txt'), sep = ',', header = None, names = ['ALL'])

fields =  {'ID':[1,11],
           'LATITUDE':[13, 20],
           'LONGITUDE':[22, 30],
           'ELEVATION':[32, 37],
           'STATE':[39, 40],
           'NAME':[42, 122],
           'WMO ID':[124, 128],
           'NOMINAL SAMPLING INTERVAL':[130, 133],
           'N HOURS OFFSET FROM GMT':[135, 139]}


for key in fields.keys():
    # indexing: remove 1 for python 0-based, add 1 to the end
    # point bc in would be excluded
    hdp_stations[key] = hdp_stations['ALL'].apply(lambda x:
                                          x[fields[key][0]-1:fields[key][1]])



nstats = np.shape(hdp_stations)[0]

mywidths = [11, 4, 2, 2, 4] + [5, 1, 1, 1, 1]*24
mynames = ['STATION','YEAR', 'MONTH', 'DAY', 'ELEMENT'] +\
          ['VALUE', 'MFLAG', 'QFLAG', 'SFLAG', 'S2FLAG']*24
# print(sum(mywidths))

# write in a file the locations of the stations
# list all station positions in the GHCN network
hours24 = np.arange(0, 24)
# nhours = np.size(hours24)
bad_flags = ['X', 'N', 'Y', 'K', 'G', 'O', 'Z', 'A', 'M', 'D']


nyears = np.zeros(nstats)
start = np.zeros(nstats)
end = np.zeros(nstats)
# LatLon = np.zeros((nstats, 2))
# nstats =10
print('Saving daily and hourly time series in csv')
for ii in range(nstats):
    # read station
    # compute and save daily totals
    file_ii = os.path.join(output_dir, 'all',
                           '{}.hly'.format(hdp_stations['ID'][ii]))
    df  = pd.read_fwf(file_ii, widths=mywidths, header = None)
    df.columns = mynames
    # Hourly Precipitation Total (Hundredth of Inches)
    prcp = df[df['ELEMENT'] == 'HPCP']
    prcpmat = prcp['VALUE'].values*0.254 # in mm
    qflagmat = prcp['QFLAG'].values

    years = prcp['YEAR'].values
    months = prcp['MONTH'].values
    days = prcp['DAY'].values
    daily_dates = years*10000 + months*100 + days

    ndays = np.shape(prcpmat)[0]
    daily_prcp = np.zeros(ndays)


    
    for jj in range(ndays):
        sample = prcpmat[jj,:]
        flags =  qflagmat[jj, :]
        # remove rainfall with flags above:
        # of with negative values
        cond_1 = sample > -0.1
        cond_2 = np.ones(24, dtype = 'bool') # True by default
        for kk in range(24):
            # if qflagmat[jj,kk].isin(bad_flags):
            if type(qflagmat[jj,kk]) != float: # non empty-flag
                cond_2[kk] = False
        cond12 = np.logical_and(cond_1, cond_2)
        sample_1 = sample[cond12]
        # compute daily totals only if no missing data
        if np.size(sample_1) == 24:
            daily_prcp[jj] = np.sum(sample_1)
        else:
            daily_prcp[jj] = -9999
    # build array of dates
    ddf = pd.DataFrame({'DATE':daily_dates, 'YEAR':years, 'PRCP':daily_prcp})
    # write the extracted time series in csv files
    dailyname = hdp_stations['ID'][ii].replace('dly', 'csv')
    # print(statname[:11])
    outname_daily = os.path.join(outdir_daily, dailyname)
    ddf.to_csv( '{}.csv'.format(outname_daily))



    dates = np.repeat( daily_dates, 24)
    years_all = np.repeat( years, 24)
    hours = np.tile(hours24, ndays)

    # dictionary and data frame for the given time series}
    ts = pd.DataFrame({'DATE': dates, 'YEAR':years_all,
                       'HOUR':hours, 'PRCP': prcpmat.flatten(),
                       'QFLAG': qflagmat.flatten()})

    # also save number of years for each station
    nyears[ii] = np.size(np.unique(years))
    start[ii] = ts['YEAR'].values[0]
    end[ii] = ts['YEAR'].values[-1]


    # write the extracted time series in csv files
    statname =hdp_stations['ID'][ii].replace('dly', 'csv')
    print(statname[:11])
    outname = os.path.join(outdir_hourly, statname)
    ts.to_csv('{}.csv'.format(outname))

    # LatLon[ii, 0] = hdp_stations['LATITUDE'][ii]
    # LatLon[ii, 1] = hdp_stations['LONGITUDE'][ii]

sll_df = hdp_stations.copy()
sll_df.rename(columns = {'LATITUDE':'LAT', 'LONGITUDE':'LON'}, inplace = True)
sll_df['NYEARS'] = nyears
sll_df['START'] = start
sll_df['END'] = end
sll_df.to_csv(os.path.join(output_dir, 'HOURLY_LOC_NYEARS.csv')) 
    



