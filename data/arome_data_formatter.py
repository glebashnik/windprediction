import pandas as pd
import numpy as np
import sys
import getopt
from collections import defaultdict
from os import listdir, path


arome_col_names = ['timestamp', 'location', 'airtemp', 'windvel', 'winddir', 'airpress']
retain_arome_locations = [
    '64.21|10.42',
    '64.21|10.35', 
    '64.27|10.35', 
    '64.21|10.50', 
    '64.18|10.31', 
    '64.74|9.62', 
    '64.72|11.41', 
    '63.73|9.62', 
    '63.73|11.37', 
    '65.23|8.71', 
    '65.23|12.33', 
    '63.24|8.71', 
    '63.22|12.33'
]

station_dict = {
    'bessaker': '', 
    'valsneset': '',

    'buholmraasa': 'DNMI_71990',
    'buholmråsa fyr': 'DNMI_71990',

    'halten': 'DNMI_71850',
    'halten fyr': 'DNMI_71850',

    'namsos': 'DNMI_72580',
    'namsos lufthavn': 'DNMI_72580',

    'roervik': 'DNMI_75220',
    'rørvik lufthavn': 'DNMI_75220',

    'oerlandet': 'DNMI_71550',
    'ørland iii': 'DNMI_71550',

    'vaernes': 'DNMI_69100',
    'værnes': 'DNMI_69100',
    
    'sula': 'DNMI_65940',
    'namsskogan': 'DNMI_74350',
    'nordøyan fyr': 'DNMI_75410',
    'sklinna fyr': 'DNMI_75550',
    'sømna-kvaløyfjellet': 'DNMI_76240', 
    'brønnøysund lufthavn': 'DNMI_76330',  
    'vega-vallsjø': 'DNMI_76450',   
    
    'skomaker': '',
}

col_dict = {
    'airtemp': 'T0017A3-0114', 
    'windvel': 'T0015A3-0120', 
    'winddir': 'T0014A3-0113', 
    'airpress': 'T0004A3-0116',
}

def get_closest_arome(df, arome):
    lat = float(arome.split('|')[0])
    lon = float(arome.split('|')[1])
    cols = df['location'].unique()

    for delta in range(-1, 2, 1):
        delta_val = delta * 0.01
        
        if str(lat + delta_val) + '|' + str(lon) in cols:
            return str(lat + delta * delta_val) + '|' + str(lon)
        elif str(lat) + '|' + str(lon + delta_val) in cols:
            return str(lat) + '|' + str(lon + delta_val)
        elif str(lat + delta_val) + '|' + str(lon + delta_val) in cols:
            return str(lat + delta_val) + '|' + str(lon + delta_val)  

def format_arome_col(col, location):
    lat, long = location.split('|')
    long_missing_leading_zero = len(long.split('.')[0]) == 1
    if long_missing_leading_zero:
        return 'arome_{}_{}_0{}'.format(col, lat.replace('.', ''), long.replace('.', ''))
    return 'arome_{}_{}_{}'.format(col, lat.replace('.', ''), long.replace('.', ''))

def format_weather_station_col(col, station):
    station_code = station_dict[station]
    col_code = col_dict[col]
    if len(station_code) > 0 and len(col_code) > 0:
        return '{}...........{}'.format(station_code, col_code)
    return ''

def arome_from_raw_df(arome_df, hours_to_keep, cols_to_keep):
    cols = defaultdict(list)
    for timestamp in arome_df['timestamp'].unique()[:hours_to_keep]:
        cols['timestamp'].append(timestamp)
    
    prev_timestamp = arome_df['timestamp'].unique()[0]
    hours_kept = 1
    for index, row in arome_df.iterrows():
        current_timestamp = row['timestamp']
        location = row['location'].lower()
        
        if prev_timestamp != current_timestamp:
            prev_timestamp = current_timestamp
            hours_kept += 1
        
        if hours_kept > hours_to_keep:
            break
        
        #if location not in cols_to_keep and location not in station_dict.keys():
        if location not in station_dict.keys():
            continue
 
        for col in arome_df.columns[2:]:
            col_key = format_weather_station_col(col, location) if location in station_dict.keys() else format_arome_col(col, location)
            if len(col_key) > 0: 
                cols[col_key].append(row[col])

    new_df = pd.DataFrame(cols)
    new_df.set_index('timestamp')
    return new_df

def format_arome_files(data_path, out_path, keep_first_hours=6):
    filenames = list(map(lambda filename: path.join(data_path, filename), listdir(data_path)))
    filenames = list(filter(lambda filename: filename.endswith('.csv'), filenames))
    
    formatted_dfs = []
    for index, filename in enumerate(sorted(filenames)[0:2]):
        print('Formatting {}, file number {} of {}'.format(filename, index + 1, len(filenames)))
        raw_df = pd.read_csv(filename, names=arome_col_names, sep=';', low_memory=False)
        #cols_to_keep = map(lambda arome: get_closest_arome(raw_df, arome), retain_arome_locations)
        cols_to_keep = retain_arome_locations
        arome_df = arome_from_raw_df(raw_df, keep_first_hours, cols_to_keep)
        print(arome_df.info())
        formatted_dfs.append(arome_df)

    all_dfs = pd.concat(formatted_dfs)
    all_dfs.reset_index(drop=True, inplace=True)
    all_dfs.to_csv(out_path, sep=';', index=False)

def main(argv):
    data_path = ''
    out_path = ''
    
    try:
        opts, args = getopt.getopt(argv, 'h', ['datapath=', 'out='])
    except getopt.GetoptError:
        print('Usage: arome_data_formatter.py --datapath <directory containing input files> --out <output file>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: arome_data_formatter.py --datapath <directory containing input files> --out <output file>')
            sys.exit()
        elif opt in ('', '--datapath'):
            data_path = arg
        elif opt in ('', '--out'):
            out_path = arg

    print('Using datapath ' + data_path)
    print('Saving to ' + out_path)
    print('')
    format_arome_files(data_path, out_path)

if __name__ == '__main__':
   main(sys.argv[1:])