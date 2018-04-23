import pandas as pd
import os


def generate_skomaker_dataset():
    try:
        df_tek = pd.read_csv(tek_path, sep=';', header=0,
                             decimal=',', low_memory=False)
        df_arome = pd.read_csv(arome_path, sep=';',
                               header=0, decimal='.', low_memory=False)
    except:
        print('No data found on: ' + tek_path + ' or ' + arome_path)
        exit(1)

    return pd.concat([
        # Produksjonsverdier
        df_tek.filter(regex='SKOM-Skomakerfj.*-0104', axis=1),
        df_tek.filter(regex='SKOM-Skomakerfj.*-0120', axis=1),

        # Aromestasjoner
        df_arome.filter(like='6347_1092').shift(-2),
        df_arome.filter(like='6372_0961').shift(-2),
        df_arome.filter(like='6413_0933').shift(-2),
        df_arome.filter(like='6440_1047').shift(-2),
        df_arome.filter(like='6447_1156').shift(-2),

        # Værnes
        df_tek['DNMI_69100...........T0015A3-0120'],
        df_arome['/arome_windvel_6347_1092'],

        # ØRLAND III (Koordinater: 63.705, 9.611)
        df_tek['DNMI_71550...........T0015A3-0120'],
        df_arome['/arome_windvel_6372_0961'],

        # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        df_tek['DNMI_71850...........T0015A3-0120'],
        df_arome['/arome_windvel_6413_0933'],

        # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        df_tek['DNMI_71990...........T0015A3-0120'],
        df_arome['/arome_windvel_6440_1047'],

        # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        df_tek['DNMI_72580...........T0015A3-0120'],
        df_arome['/arome_windvel_6447_1156'],

        df_tek.filter(like='STORM-Bess', axis=1).shift(-2),

        df_tek['SKOM-Skomakerfj.-GS-T4015A3 -0104'],
        df_tek['SKOM-Skomakerfj.-GS-T4015A3 -0104'].shift(-2).rename(
            'Target', inplace=True),
    ], axis=1).iloc[:-2, :]

# def generate_bessaker_dataset(tek_path, arome_path):


def generate_bessaker_dataset(tek_path, arome_path):
    try:
        df_tek = pd.read_csv(tek_path, sep=';', header=0,
                             decimal=',', low_memory=False)
        df_arome = pd.read_csv(arome_path, sep=';',
                               header=0, decimal=',', low_memory=False)
    except:
        print('No data found on: ' + tek_path + ' or ' + arome_path)
        exit(1)

    return pd.concat([
        df_tek.filter(regex='BESS-Bessakerfj\.-G[^S].*-0104', axis=1),
        df_tek.filter(regex='BESS-Bessakerfj.*-0120', axis=1),

        df_arome.filter(like='6421_1035').shift(-2),
        df_arome.filter(like='6422_1040').shift(-2),

        # Værnes
        df_tek['DNMI_69100...........T0015A3-0120'],
        df_arome['/arome_windvel_6347_1092'],

        # ØRLAND III (Koordinater: 63.705, 9.611)
        df_tek['DNMI_71550...........T0015A3-0120'],
        df_arome['/arome_windvel_6372_0961'],


        # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        df_tek['DNMI_71850...........T0015A3-0120'],
        df_arome['/arome_windvel_6413_0933'],

        # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        df_tek['DNMI_71990...........T0015A3-0120'],
        df_arome['/arome_windvel_6440_1047'],

        # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        df_tek['DNMI_72580...........T0015A3-0120'],
        df_arome['/arome_windvel_6447_1156'],

        # Storm vind måling
        df_tek.filter(like='STORM-Bess', axis=1).shift(-2),

        # Sum produksjon
        df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'],
        df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].astype(
            'd').shift(-2).rename('Target', inplace=True),
    ], axis=1).iloc[:-2, :]


# Generates dataset for predicting change in production
def generate_bessaker_delta_target_dataset(tek_path, arome_path):
    try:
        df_tek = pd.read_csv(tek_path, sep=';', header=0,
                             decimal=',', low_memory=False)
        df_arome = pd.read_csv(arome_path, sep=';',
                               header=0, decimal='.', low_memory=False)
    except:
        print('No data found on: ' + tek_path + ' or ' + arome_path)
        exit(1)

    # Error between nacelle and storm wind speed
    forecast_error = pd.DataFrame(data=df_tek.filter(
        regex='BESS-Bessakerfj.*-0120', axis=1))
    cols_name = []
    index = 0
    for column in forecast_error:
        forecast_error[column] = forecast_error[column] - \
            df_tek['STORM-Bess-Vindhast-25km']
        cols_name.append('Error_nacelle_storm_single_{}'.format(index))
        forecast_error[column].rename(
            'Error_nacelle_storm_single_{}'.format(index), axis='columns')
        index += 1

    forecast_error.columns = cols_name

    # df_arome['/arome_windvel_6347_1092'].head()
    # df_tek['DNMI_71850...........T0015A3-0120'].head()
    # exit(0)

    return pd.concat([
        # To do:
        # Create error features (nacelle vs storm)
        # Create derivative feature (production change or/and windchange)

        # Single prod
        df_tek.filter(regex='BESS-Bessakerfj.*-0104', axis=1),
        # Nacelle
        df_tek.filter(regex='BESS-Bessakerfj.*-0120', axis=1),

        # df_arome.filter(like='6421_1035').shift(-2),
        # df_arome.filter(like='6422_1040').shift(-2),

        # # Værnes
        # df_tek['DNMI_69100...........T0015A3-0120'],
        # df_arome['/arome_windvel_6347_1092'],

        # # ØRLAND III (Koordinater: 63.705, 9.611)
        # df_tek['DNMI_71550...........T0015A3-0120'],
        # df_arome['/arome_windvel_6372_0961'],

        # # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        # df_tek['DNMI_71850...........T0015A3-0120'],
        # df_arome['/arome_windvel_6413_0933'],

        # # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        # df_tek['DNMI_71990...........T0015A3-0120'],
        # df_arome['/arome_windvel_6440_1047'],

        # # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        # df_tek['DNMI_72580...........T0015A3-0120'],
        # df_arome['/arome_windvel_6447_1156'],

        # # Storm vind måling
        # df_tek.filter(like='STORM-Bess', axis=1).shift(-2),

        # Wind error pred
        # forecast_error,

        # Storm wind delta
        # (df_tek['STORM-Bess-Vindhast-25km'].shift(-2) - \
        #  df_tek['STORM-Bess-Vindhast-25km'].shift(-1)).rename('Storm delta now - 1 hour', inplace=True),

        # (df_tek['STORM-Bess-Vindhast-25km'].shift(-1) - \
        #  df_tek['STORM-Bess-Vindhast-25km']).rename(
        #      'Storm delta 1 hour - 2 hour', inplace=True),

        # # production change from the previous hour
        # (df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'] - \
        #  df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(1)).rename('Production_delta', inplace=True),

        # # Prod forrige 1 og 2 time
        # df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(
        #     2).rename('production t-2', inplace=True),

        # df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(
        #     1).rename('production t-1', inplace=True),

        # Sum produksjon
        # df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'],
        df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(-2).rename(
            'Target', inplace=True),

    ], axis=1).iloc[:-2, :]


def generate_bessaker_dataset_single_target(tek_path, arome_path):
    try:
        df_tek = pd.read_csv(tek_path, sep=';', header=0,
                             decimal=',', low_memory=False)
        df_arome = pd.read_csv(arome_path, sep=';',
                               header=0, decimal=',', low_memory=False)
    except:
        print('No data found on: ' + tek_path + ' or ' + arome_path)
        exit(1)

    # Error between nacelle and storm wind speed
    single_target = pd.DataFrame(data=df_tek.filter(
        regex='BESS-Bessakerfj.*-0104', axis=1).shift(-2))
    cols_name = []
    index = 0
    for column in single_target:

        cols_name.append('Single_target_{}'.format(index+1))
        # single_target[column].rename(
        #     'Error_nacelle_storm_single_{}'.format(index), axis='columns')
        index += 1
    cols_name[-1] = 'Target'

    single_target.columns = cols_name

    # df_arome['/arome_windvel_6347_1092'].head()
    # df_tek['DNMI_71850...........T0015A3-0120'].head()
    # exit(0)

    return pd.concat([
        # To do:
        # Create error features (nacelle vs storm)
        # Create derivative feature (production change or/and windchange)

        # Bessaker molle produksjon
        df_tek.filter(regex='BESS-Bessakerfj\.-G[^S].*-0104', axis=1),

        # Bessaker mølle vindhastighet
        df_tek.filter(regex='BESS-Bessakerfj.*-0120', axis=1),

        df_arome.filter(like='6421_1035').shift(-2),
        df_arome.filter(like='6422_1040').shift(-2),
        # Værnes
        df_tek['DNMI_69100...........T0015A3-0120'],
        df_arome['/arome_windvel_6347_1092'],

        # ØRLAND III (Koordinater: 63.705, 9.611)
        df_tek['DNMI_71550...........T0015A3-0120'],
        df_arome['/arome_windvel_6372_0961'],

        # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        df_tek['DNMI_71850...........T0015A3-0120'],
        df_arome['/arome_windvel_6413_0933'],

        # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        df_tek['DNMI_71990...........T0015A3-0120'],
        df_arome['/arome_windvel_6440_1047'],

        # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        df_tek['DNMI_72580...........T0015A3-0120'],
        df_arome['/arome_windvel_6447_1156'],

        # Storm vind måling
        df_tek.filter(like='STORM-Bess', axis=1).shift(-2),

        # Wind error pred
        # forecast_error,

        # Storm wind delta
        (df_tek['STORM-Bess-Vindhast-25km'].shift(-2) - \
         df_tek['STORM-Bess-Vindhast-25km'].shift(-1)).rename('Storm delta now - 1 hour', inplace=True),

        (df_tek['STORM-Bess-Vindhast-25km'].shift(-1) - \
         df_tek['STORM-Bess-Vindhast-25km']).rename(
             'Storm delta 1 hour - 2 hour', inplace=True),




        # production change from the previous hour
        (df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'] - \
         df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(1)).rename('Production_delta', inplace=True),

        # Sum produksjon
        # df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'],

        single_target,
        # df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(-2).rename(
        #     'Target', inplace=True),

    ], axis=1).iloc[:-2, :]


def generate_bessaker_large_dataset_scratch(tek_path):
    try:

        WindSpeedNacelleBessaker_path = os.path.join(
            tek_path, 'WindSpeedNacelleBessaker.txt')
        WindSpeedNacelleBessaker = pd.read_csv(WindSpeedNacelleBessaker_path, sep=';', header=0,
                                               decimal='.', low_memory=False)

        WindSpeedForecast_path = os.path.join(
            tek_path, 'WindSpeedForecast.txt')
        WindSpeedForecast = pd.read_csv(WindSpeedForecast_path, sep=';', header=0,
                                        decimal='.', low_memory=False)

        WindDirectionForecast_path = os.path.join(
            tek_path, 'WindDirectionForecast.txt')
        WindDirectionForecast = pd.read_csv(WindDirectionForecast_path, sep=';', header=0,
                                            decimal='.', low_memory=False)

        AirTemperaturForecast_path = os.path.join(
            tek_path, 'AirTemperaturForecast.txt')
        AirTemperaturForecast = pd.read_csv(AirTemperaturForecast_path, sep=';', header=0,
                                            decimal='.', low_memory=False)

        StatusBessaker_path = os.path.join(
            tek_path, 'StatusBessaker.txt')
        StatusBessaker = pd.read_csv(StatusBessaker_path, sep=';', header=0,
                                     decimal='.', low_memory=False)

        NoseHeadingBessaker_path = os.path.join(
            tek_path, 'NoseHeadingBessaker.txt')
        NoseHeadingBessaker = pd.read_csv(NoseHeadingBessaker_path, sep=';', header=0,
                                          decimal='.', low_memory=False)

        ProduksjonBessaker_path = os.path.join(
            tek_path, 'ProduksjonBessaker.txt')
        ProduksjonBessaker = pd.read_csv(ProduksjonBessaker_path, sep=';', header=0,
                                         decimal='.', low_memory=False)

        windspeedmeasure_path = os.path.join(
            tek_path, 'out.csv')
        windspeedmeasure = pd.read_csv(windspeedmeasure_path, sep=';', header=0,
                                       decimal='.', low_memory=False)

    except:
        print('No data found on: ' + tek_path)
        exit(1)

    # windspeedmeasure.info()
    # exit(0)
    windspeedmeasure.head()
    windspeedmeasure['ExcelDato'].head()
    exit(0)

    merged = WindSpeedNacelleBessaker.merge(
        WindSpeedForecast, on='ExcelDato')
    merged = merged.merge(
        WindDirectionForecast, on='ExcelDato')

    merged = merged.merge(
        WindSpeedForecast, on='ExcelDato')

    merged = merged.merge(
        AirTemperaturForecast, on='ExcelDato')

    merged = merged.merge(
        StatusBessaker, on='ExcelDato')

    merged = merged.merge(
        NoseHeadingBessaker, on='ExcelDato')

    merged = merged.merge(
        ProduksjonBessaker, on='ExcelDato')

    return pd.concat([
        # Single prod
        df_tek.filter(regex='BESS-Bessakerfj.*-0104', axis=1),
        # Nacelle
        df_tek.filter(regex='BESS-Bessakerfj.*-0120', axis=1),

        # df_arome.filter(like='6421_1035').shift(-2),
        # df_arome.filter(like='6422_1040').shift(-2),

        # # Værnes
        # df_tek['DNMI_69100...........T0015A3-0120'],
        # df_arome['/arome_windvel_6347_1092'],

        # # ØRLAND III (Koordinater: 63.705, 9.611)
        # df_tek['DNMI_71550...........T0015A3-0120'],
        # df_arome['/arome_windvel_6372_0961'],


        # # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        # df_tek['DNMI_71850...........T0015A3-0120'],
        # df_arome['/arome_windvel_6413_0933'],

        # # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        # df_tek['DNMI_71990...........T0015A3-0120'],
        # df_arome['/arome_windvel_6440_1047'],

        # # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        # df_tek['DNMI_72580...........T0015A3-0120'],
        # df_arome['/arome_windvel_6447_1156'],

        # Storm vind måling
        df_tek.filter(like='STORM-Bess', axis=1).shift(-2),


        # Sum produksjon
        # df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'],

        df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(-2).rename(
            'Target', inplace=True),

    ], axis=1).iloc[:-2, :]


def generate_bessaker_large_dataset(tek_path, history_length=1):
    try:

        df_tek = pd.read_csv(os.path.join(tek_path), sep=';', header=0,
                             decimal='.', low_memory=False)

    except:
        print('No data found on: ' + tek_path)
        exit(1)

    # Creates 'history_length' columns of production history
    df_production_history = df_tek['/TS-Straum066_BessVind_Inn'].rename(
        'Produksjon-{}-Timer-Siden'.format(0), inplace=True)
    for i in range(1, history_length):
        df_production_history = pd.concat([df_production_history, df_tek['/TS-Straum066_BessVind_Inn'].astype(
            'd').shift(i).rename('Produksjon-{}-Timer-Siden'.format(i), inplace=True)], axis=1)

    # df_production_history_copy = df_production_history.shift(2)
    # df_production_history = df_production_history-df_production_history_copy

    return pd.concat([

        # Bessaker molle produksjon
        df_tek.filter(regex='BESS-Bessakerfj\.-G[^S].*-0104', axis=1),

        # Bessaker mølle vindhastighet (nacelle)
        df_tek.filter(regex='BESS-Bessakerfj.*-0120', axis=1),

        # Nose heading
        df_tek.filter(regex='BESS-BessakerfNP-V*', axis=1),

        # Mølle status
        # df_tek.filter(regex='RRS.S2464.Gunit.M1-7*', axis=1),

        # Air temperature forecast (på skomakerfjellet)

        df_tek['/SKOM-SfjHydLt30mMid-T0018A3 -0114'],

        ###############################################
        #           Værstasjoner                      #
        ###############################################

        # # Værnes
        # df_tek['DNMI_69100...........T0015A3-0120'],

        # # ØRLAND III (Koordinater: 63.705, 9.611)
        # df_tek['DNMI_71550...........T0015A3-0120'],

        # # HALTEN FYR ( Kordinater: 64.173, 9.405 )
        # df_tek['DNMI_71850...........T0015A3-0120'],

        # # BUHOLMRÅSA FYR (kordinater: 64.401, 10.455)
        # df_tek['DNMI_71990...........T0015A3-0120'],

        # # NAMSOS LUFTHAVN (Koordinater: 64.471, 11.571)
        # df_tek['DNMI_72580...........T0015A3-0120'],

        # Storm måling vindhastighet og retning
        df_tek.filter(like='STORM-Bess', axis=1).shift(-2),

        # Sum produksjon
        df_production_history,
        (df_tek['/TS-Straum066_BessVind_Inn'].astype(
            'd').shift(-2) - df_tek['/TS-Straum066_BessVind_Inn'].astype(
            'd')).rename('Target', inplace=True)
    ], axis=1).iloc[:-2, :]


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

        # Skomaker stasj
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
        df['TS-Straum066_BessVind_Inn'],
        df['TargetValsnes'].astype('d')
    ], axis=1).iloc[:-2, :]
