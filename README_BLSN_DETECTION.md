Blowing Snow Detection Algorithm

Files: 'BLSN_detection_algorithm.ipynb' - Jupyter Notebook
       'BLSN_detection_algorithm.py' - Python Script
Both files do the same thing, just available in different forms. The jupyter notebook includes flowchart
and examples. 

        Required module imports:
            import xarray as xr
            import numpy as np
            import pandas as pd
            import os
    
        Inputs:
            ceil_path = string containing path to ceilometer data files
            met_path = string containing path to MET data files
            ceil_clr_file = string containing path and filename of csv containing
                            ceilometer clear sky profile
        
        Other Input Parameters:
            avg_period: averaging period in minutes (default = 5)
            date_range: if None - all dates in ceilometer data folder
                        if a subset: [str(start_date),str(end_date)]
                                     ex. ['2016-01-01','2017-01-01']
                               inclusive of start date, not end date
                        (default = None)
            wind_thres: threshold 10 m wind speed in m s^-1 (default = 3.0)
            vis_thres: threshold 2 m visibility in m (default = 10000.0)
            rh_thres: threshold relative humidity in % (default = 90.0)
            
        Output:
            Pandas dataframe containing the following: 
            col 0: 'datetime' - time of the start of averaging period in the
                    format ('YYYY-MM-DD hh:mm:ss') 
            col 1: 'top_of_detected_blsn' - height of top of BLSN plume, if 
                    detected [m]
            col 2: 'category' - category of detected BLSN [0 = no BLSN detected, 
                    1 = clear sky BLSN, 2 = cloud/precipitation with BLSN, 
                    3 = intense mixed event, 4 = fog]
            col 3: 'lowest_bin_backscatter' - backscatter coefficient in lowest 
                    usable bin (10-15 m AGL) [sr^-1 km^-1 10^-4]
            col 4: 'decreasing_profile' - results of check for decreasing 
                    backscatter profile with height [1 = yes, 0 = no]
            col 5: 'above_clear_sky' - results of check that backscatter in 
                    lowest usable bin > clear sky signal [1 = yes, 0 = no]
            col 6: '10m_windspeed' - 10 m wind speed average for selected 
                    averaging period [m s^-1]
            col 7: '10m_winddirection' - average 10 m wind direction for selected 
                    averaging period [deg]
            col 8: '2m_temperature' - average 2 m temperature for selected
                    averaging period [degC]
            col 9: '2m_rel_hum' - average 2 m relative humidity with respect 
                    to liquid water for selected averaging period [%]
            col 10: '2m_visibility' - average 2 m meteorological optical range 
                    visibility for selected averaging period [m]
            
            missing value code = 99999999.9

Reference: Loeb and Kennedy (in review)
For extra assistance, email nicoleloeb4@gmail.com
