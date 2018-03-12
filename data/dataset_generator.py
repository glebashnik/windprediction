import pandas as pd

def generate_skomaker_dataset():
    try:
        df_tek = pd.read_csv(tek_path, sep=';', header=0, decimal=',', low_memory=False)
        df_arome = pd.read_csv(arome_path, sep=';',  header=0, decimal='.', low_memory=False)
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
        df_tek['SKOM-Skomakerfj.-GS-T4015A3 -0104'].shift(-2).rename('Target', inplace=True),
    ], axis=1).iloc[:-2,:]

def generate_bessaker_dataset(tek_path, arome_path):
    try:
        df_tek = pd.read_csv(tek_path, sep=';', header=0, decimal=',', low_memory=False)
        df_arome = pd.read_csv(arome_path, sep=';',  header=0, decimal='.', low_memory=False)
    except:
        print('No data found on: ' + tek_path + ' or ' + arome_path)
        exit(1)    

    return pd.concat([
        df_tek.filter(regex='BESS-Bessakerfj.*-0104', axis=1),
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
        
        df_tek.filter(like='STORM-Bess', axis=1).shift(-2),
        
        #Sum produksjon
        df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'],
        df_tek['BESS-Bessakerfj.-GS-T4015A3 -0104'].shift(-2).rename('Target', inplace=True),
    ], axis=1).iloc[:-2,:]