

Fractional Blowing Snow Depth Estimation Algorithm

Files: 'fractional_depth_algorithm.ipynb' - Jupyter Notebook 'fractional_depth_algorithm.py' - Python Script 
Both files do the same thing, just available in different forms. The jupyter notebook includes flowcharts, detailed
descriptions, and examples.

    Required module imports:
        import xarray as xr
        import numpy as np
        import os
        import pandas as pd
        from os import listdir
        from os.path import isfile, join
        from datetime import datetime, timedelta
        import math as m

    Inputs:
        ceil_path = string containing path to directory of ceilometer data files
        kazr_path = string containing path to directory of KAZR data files
        mpl_path = string containing path to directory of MPL data files
        hsrl_path = string containing path to directory of HSRL data files
        skyrad_path = string containing path to directory of SKYRAD60s data files
        detection_output = string containing path to csv file of output from the
                           5-min+MET algorithm
        ceil_clr_file = string containing path to csv containing ceilometer clear
                        sky profile
        mpl_clr_file = string containing path to csv containing MPL clear
                       sky profile
        hsrl_clr_file = string containing path to csv containing HSRL clear
                        sky profile
        mpl_calibration = string containing path to csv containing MPL
                          calibration profiles
    
    Other Input Parameters:
        h_bins: array of bins to average altitude over [in meters]
                (default = np.arange(0,2100,30)) 
        rad_thres: radiation threshold for day/night for MPL calibration
                   in W/m^2 (default = 20)
        dates: list of dates to analyze (default = None)
               if None - all dates in ceilometer data folder
               if a subset - list of strings in the form of ['yyyymmdd']
                             ex. ['20160612','20160430']
        radar_wavelength: wavelength of the radar (KAZR) in meters for
                          calculation of color ratio
                          (default = 0.0085655)
        avg_period: averaging period in minutes (default = 5)
                    * must match the BLSN detection input file 
                      averaging period
        smooth_num: number of observation periods to apply a running mean
                    to for the final BLSN top - must be an odd integer >= 1
                    (default = 5)

        
    Output:
        Pandas dataframe containing the following:
            col 0: 'datetime' - time of the start of averaging period in the
                    format ('YYYY-MM-DD hh:mm:ss') 
            col 1: 'category' - category of detected BLSN [0 = no BLSN detected, 
                    1 = clear sky BLSN, 2 = cloud/precipitation with BLSN, 
                    3 = intense mixed event, 4 = fog]
            col 2: 'ceil_blsn_top' - altitude of top of BLSN plume based on 
                    5-min + MET algorithm [m]
            col 3: 'smoothed_blsn_top' - altitude of top of BLSN plume based on
                    fractional BLSN depth algorithm [m]
            col 4: '10m_windspeed' - 10 m wind speed from MET [m/s]
            col 5: '10m_winddir' - 10 m wind direction from MET [degrees]
            col 6: '2m_visibility' - 2 m visibility from MET [m]
            col 7: '2m_temperature' - 2 m temperature from MET [degC]
            col 8: 'ceil_30m' - altitude of top of BLSN plume based on 30 m CEIL
            col 9: 'mpl_bs' - altitude of top of BLSN plume based on MPL BS
            col 10: 'mpl_ldr' - altitude of top of BLSN plume based on MPL LDR
            col 11: 'hsrl_bs' - altitude of top of BLSN plume based on HSRL BS
            col 12: 'hsrl_ldr' - altitude of top of BLSN plume based on HSRL LDR
            col 13: 'cr' - altitude of top of BLSN plume based on Color Ratio
            col 14: 'valid_criteria' - number of algorithms whose criteria are
                    met for each time period

        
        missing value code = -99999999.9

Reference: Loeb and Kennedy (in prep) For extra assistance, email nicoleloeb4@gmail.com
