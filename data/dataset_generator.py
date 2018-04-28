import pandas as pd
import numpy as np
import os

def Bessaker_dataset(data_path):

    try:
        df = pd.read_csv(data_path, sep=';', header=0,
                         decimal=',', low_memory=False)
    except:
        print('No data found on: ' + tek_path)
        exit(1)
    return pd.concat([

        # Produksjon Bessaker
        df.filter(regex='BESS-Bessakerfj\.-G[^S].*-0104', axis=1),
        # Nacelle
        df.filter(regex='BESS-Bessakerfj.*-0120', axis=1),

        # Skomaker stasj
        df.filter(like='SKOM', axis=1),

        # Værnes
        # df['DNMI_69100...........T0015A3-0120'],
        # Alle værstasjoner med alle målinger
        df.filter(like='DNMI', axis=1),

        # ØRLAND III (Koordinater: 63.705, 9.611)
        # df['DNMI_71550...........T0015A3-0120'],

        # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        # df['DNMI_71850...........T0015A3-0120'],

        # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        # df['DNMI_71990...........T0015A3-0120'],

        # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        # df['DNMI_72580...........T0015A3-0120'],

        # Arome values
        df.filter(like='arome_wind', axis=1),

        # Arome airtemp
        df.filter(like='arome_airtemp', axis=1),

        # Storm vind måling
        df.filter(like='STORM-Bess', axis=1).shift(-2),

        # Sum produksjon
        df['BESS-Straum066KV-ut-T4045A3A-0106'],
        df['TargetBessaker'].astype('d')
    ], axis=1).iloc[:-2, :]

def Bessaker_dataset_sparse(data_path):

    try:
        df = pd.read_csv(data_path, sep=';', header=0,
                         decimal=',', low_memory=False)
    except:
        print('No data found on: ' + tek_path)
        exit(1)
    return pd.concat([

        # Produksjon Bessaker
        df.filter(regex='BESS-Bessakerfj\.-G[^S].*-0104', axis=1).iloc[:,0].astype('d'),# - df.filter(regex='BESS-Bessakerfj\.-G[^S].*-0104', axis=1).iloc[:,0].astype('d').shift(1),
        # Nacelle
        df.filter(regex='BESS-Bessakerfj.*-0120', axis=1).iloc[:,0].astype('d'), # - df.filter(regex='BESS-Bessakerfj.*-0120', axis=1).iloc[:,0].astype('d').shift(1),

        # Skomaker stasj
        df.filter(like='SKOM', axis=1).astype('d'),# - df.filter(like='SKOM', axis=1).astype('d').shift(1),

        # Værnes
        df['DNMI_69100...........T0015A3-0120'].astype('d'),# - df['DNMI_69100...........T0015A3-0120'].astype('d').shift(1),
        # Alle værstasjoner med alle målinger
        # df.filter(like='DNMI', axis=1),

        # ØRLAND III (Koordinater: 63.705, 9.611)
        # df['DNMI_71550...........T0015A3-0120'],

        # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        # df['DNMI_71850...........T0015A3-0120'],

        # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        # df['DNMI_71990...........T0015A3-0120'],

        # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        # df['DNMI_72580...........T0015A3-0120'],

        # Arome values
        df.filter(like='arome_wind', axis=1).iloc[:,0:2].astype('d'),# - df.filter(like='arome_wind', axis=1).astype('d').iloc[:,0:2].shift(1),

        # Storm vind måling
        df.filter(like='STORM-Bess', axis=1).astype('d').shift(-2),# - df.filter(like='STORM-Bess', axis=1).astype('d').shift(-1),

        # Sum produksjon
        df['BESS-Straum066KV-ut-T4045A3A-0106'],
        df['TargetBessaker'].astype('d')
    ], axis=1).iloc[:-2, :]


def Valsnes_dataset(data_path):
    try:
        df = pd.read_csv(data_path, sep=';', header=0,
                         decimal=',', low_memory=False)
    except:
        print('No data found on: ' + tek_path)
        exit(1)

    # with open('features.txt', 'w') as f:
    #     [f.write(column + '\n') for column in df.columns]

    return pd.concat([

        # Produksjon o.l. Valsnes
        df.filter(like='VALS', axis=1),
        # Nacelle
        df.filter(regex='BESS-Bessakerfj.*-0120', axis=1),

        # Skomaker stasjon
        df.filter(like='SKOM', axis=1),

        # Værnes
        # df['DNMI_69100...........T0015A3-0120'],
        # Alle værstasjoner med alle målinger
        df.filter(like='DNMI', axis=1),

        # ØRLAND III (Koordinater: 63.705, 9.611)
        # df['DNMI_71550...........T0015A3-0120'],
        df.filter(like='DNMI_71550', axis=1),

        # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        # df['DNMI_71850...........T0015A3-0120'],

        # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        # df['DNMI_71990...........T0015A3-0120'],

        # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        # df['DNMI_72580...........T0015A3-0120'],

        # Arome values
        # df.filter(like='arome_wind', axis=1),

        # Arome airtemp
        # df.filter(like='arome_airtemp', axis=1),

        # Storm vind måling
        df.filter(like='STORM-Vals', axis=1).shift(-2),

        # Sum produksjon
        df["## =-%'VALS-Valsneset..-GS-T4015A3 -0104'*0.96"],
        df['TargetValsnes'].astype('d')
    ], axis=1).iloc[:-2, :]



##
## @brief      Creates a dataset with history.
##
## @param      dataframe       The input dataframe.
## @param      history_length  How far back in time the output dataframe should go.
##
## @return     data            3D numpy array with data with the form (row, time, column) that matches
##                             the order used for keras conv1D nets.
## @return     target          Pandas series with target data.
def create_dataset_history(dataframe, history_length, future_length):

    history_data = []
    # Create a third dimension to represent time. Target has to be dropped
    # to stay 1 dimensional. Here I assume the last column is target.

    for i in range(-history_length, future_length+1):
        history_data.append(dataframe.shift(-i).as_matrix())
    data = np.stack(history_data, axis=1).astype(np.float64)

    # If original data was only one column the third dimension disappears.
    # Adding it back here
    if(len(data.shape)) == 2:
        data = np.expand_dims(data, axis=2)

    return data

# 1. Split dataset
# 2. Create relevant histories (12 for prod), (+-3 for weather)
# 3. Create relevant conv network structure

def add_production_and_forecast_history_bessaker(dataframe, y, production_length, forecast_start, forecast_stop):
    forecastDF = pd.concat([dataframe.filter(like='STORM-Bess', axis=1).shift(-2)]) ##, dataframe.filter(like='arome_wind', axis=1).shift(-2)
    productionDF = dataframe['BESS-Straum066KV-ut-T4045A3A-0106']

    forecast = create_dataset_history(forecastDF, forecast_start, forecast_stop)
    production = create_dataset_history(productionDF, production_length, 0)

    # dataframe.drop(labels='BESS-Straum066KV-ut-T4045A3A-0106', axis=1, inplace = True)
    # dataframe.drop(labels=forecastDF.columns, axis=1, inplace = True)

    lower_bound = forecast_start if (forecast_start > production_length) else production_length
    upper_bound = forecast_stop + 2 # To account for the original shift

    forecast = forecast[lower_bound+1:-upper_bound,:,:]
    production = production[lower_bound+1:-upper_bound,:,:]
    dataframe = dataframe.iloc[lower_bound+1:-upper_bound,:]
    y = y.iloc[lower_bound+1:-upper_bound,:]

    return production, forecast, dataframe, y