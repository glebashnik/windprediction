import pandas as pd
import numpy as np
import datetime
import sys
import getopt

from functools import reduce
from os import listdir, path

def excel_date_to_normal_date(excel_date):
    out_format = '%d/%m/%Y %I:%M:%S %p'
    origin_date = datetime.datetime(1900, 1, 1)
    date_offset = datetime.timedelta(days=excel_date)
    new_datetime = origin_date + date_offset
    
    if new_datetime.minute >= 30:
        return new_datetime.replace(second=0, microsecond=0, minute=0, hour=new_datetime.hour+1).strftime(out_format)
    else:
        return new_datetime.replace(second=0, microsecond=0, minute=0).strftime(out_format)

def format_tek_files(data_path, out_path):
    try: 
        bess_windvel = pd.read_csv(path.join(data_path, 'WindSpeedNacelleBessaker.csv'), sep=';')
        bess_prod = pd.read_csv(path.join(data_path, 'ProduksjonBessaker.csv'), sep=';')
        bess_error = pd.read_csv(path.join(data_path, 'ErrorCodesBessaker.csv'), sep=';')
        bess_heading = pd.read_csv(path.join(data_path, 'NoseHeadingBessaker.csv'), sep=';')
        bess_status = pd.read_csv(path.join(data_path, 'StatusBessaker.csv'), sep=';')

        vals_prod = pd.read_csv(path.join(data_path, 'ProduksjonValsneset.csv'), sep=';')
        vals_error = pd.read_csv(path.join(data_path, 'ErrorCodesValsneset.csv'), sep=';')
        vals_heading = pd.read_csv(path.join(data_path, 'NoseHeadingValsneset.csv'), sep=';')
        vals_status = pd.read_csv(path.join(data_path, 'StatusValsneset.csv'), sep=';')

        storm_windvel = pd.read_csv(path.join(data_path, 'WindSpeedForecast.csv'), sep=';')
        storm_airtemp = pd.read_csv(path.join(data_path, 'AirTemperaturForecast.csv'), sep=';')
        storm_winddir = pd.read_csv(path.join(data_path, 'WindDirectionForecast.csv'), sep=';')
    except:
        print('One or more files is missing from ' + data_path)
    
    all_dfs =[bess_prod, bess_error, bess_heading, bess_status, vals_prod, vals_error, vals_heading, vals_status, storm_windvel, storm_airtemp, storm_winddir]
    
    bess_windvel['ExcelDato'] = bess_windvel['ExcelDato'].apply(excel_date_to_normal_date)
    bess_windvel.set_index('ExcelDato')

    for df in all_dfs:
        df['ExcelDato'] = df['ExcelDato'].apply(excel_date_to_normal_date)
        bess_windvel = pd.concat([bess_windvel, df.drop('ExcelDato', axis=1)], axis=1)
    #print(bess_windvel.info(verbose=True))
    bess_windvel.to_csv(out_path, sep=';', index=False)

def main(argv):
    data_path = ''
    out_path = ''
    
    try:
        opts, args = getopt.getopt(argv, 'h', ['datapath=', 'out='])
    except getopt.GetoptError:
        print('Usage: tek_data_formatter.py --datapath <directory containing input files> --out <output file>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: tek_data_formatter.py --datapath <directory containing input files> --out <output file>')
            sys.exit()
        elif opt in ('', '--datapath'):
            data_path = arg
        elif opt in ('', '--out'):
            out_path = arg

    print('Using datapath ' + data_path)
    print('Saving to ' + out_path)
    print('')
    format_tek_files(data_path, out_path)

if __name__ == '__main__':
   main(sys.argv[1:])
   #print(excel_date_to_normal_date(41456.0000))
   #print(excel_date_to_normal_date(43159.9583))