# This script unzips the GEFCom2014 data zip file and extracts the 'extended'
# load forecasting competition data. Data is saved in energy.csv


import zipfile
import os
import sys
import pandas as pd

data_dir = 'data/'
GEFCom_dir = os.path.join(data_dir, 'GEFCom2014', 'GEFCom2014 Data')

if not os.path.exists(os.path.join(data_dir, 'GEFCom2014.zip')):
    
    sys.exit("Download GEFCom2014.zip from https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0 and save it to the 'data' directory.")

# unzip root directory
zip_ref = zipfile.ZipFile(os.path.join(data_dir, 'GEFCom2014.zip'), 'r')
zip_ref.extractall(os.path.join(data_dir, 'GEFCom2014'))
zip_ref.close()

# extract the extended competition data
zip_ref = zipfile.ZipFile(os.path.join(GEFCom_dir, 'GEFCom2014-E_V2.zip'), 'r')
zip_ref.extractall(os.path.join(data_dir, 'GEFCom2014-E'))
zip_ref.close()

# load the data from Excel file
data = pd.read_excel(os.path.join(data_dir, 'GEFCom2014-E', 'GEFCom2014-E.xlsx'), parse_date='Date')

# create timestamp variable from Date and Hour
data['timestamp'] = data['Date'].add(pd.to_timedelta(data.Hour - 1, unit='h'))
data = data[['timestamp', 'load', 'T']]
data = data.rename(columns={'T':'temp'})

# remove time period with no load data
data = data[data.timestamp >= '2012-01-01']

# save to csv
data.to_csv(os.path.join(data_dir, 'energy.csv'), index=False)