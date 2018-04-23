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
        df['TS-Straum066_BessVind_Inn'],
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
        df.filter(regex='BESS-Bessakerfj\.-G[^S].*-0104', axis=1).iloc[:,0],
        # Nacelle
        df.filter(regex='BESS-Bessakerfj.*-0120', axis=1).iloc[:,0],

        # Skomaker stasj
        df.filter(like='SKOM', axis=1),

        # Værnes
        df['DNMI_69100...........T0015A3-0120'],
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
        df.filter(like='arome_wind', axis=1).iloc[:,0:2],

        # Storm vind måling
        df.filter(like='STORM-Bess', axis=1).shift(-2),

        # Sum produksjon
        df['TS-Straum066_BessVind_Inn'],
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
        df.filter(like='DNMI_71550',axis=1),

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
        df['TS-Straum066_BessVind_Inn'],
        df['Target'].astype('d')
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
def create_dataset_history(dataframe, history_length):


    history_data = []

    # Create a third dimension to represent time. Target has to be dropped
    # to stay 1 dimensional. Here I assume the last column is target.
    data_without_target = dataframe.drop([dataframe.columns[-1]],axis=1)
    history_data.append(data_without_target.as_matrix())

    for i in range(1,history_length):
        history_data.append(data_without_target.shift(i).as_matrix())

    data = np.stack(history_data, axis=1).astype(np.float64)
    target = dataframe[dataframe.columns[-1]]

    
    # Remove rows with nan from data
    rows_with_nan = np.argwhere(np.isnan(data))[:,0]
    rows_with_nan = np.unique(rows_with_nan)

    data = np.delete(data, rows_with_nan, axis=0)


    # Remove the corresponding rows from target
    index_with_nan = target.index[[rows_with_nan]]
    target = target.drop(index_with_nan)
    
    return data, target