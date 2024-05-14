import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

class ProcessData():

    def __init__(self,  df,  pathcsv=None, extract_fromcsv=False, busdata=False):
        
        self.df  = df

        if extract_fromcsv:
            self.Df_VF_SF = df
            self.get_splitdata_PV_PQ()
            self.get_processdata_region()
        else:
            if busdata:
                df1 = pd.read_csv('RECURSOS/GeoINFO_BusesSIN.csv',sep=';')
                #***************************************************** Merge com o DATA FRAME COMPLETO ******************************************************
                # Dataframe geral 
                columns = ['BUS_ID', 'BUS_NAME', 'VBASEKV', 'TP', 'ARE', 'MODV_PU', 'ANGV_DEG', 'BASE_MVA', 'PG_MW', 'QG_MVAR', 'PMAX_MW', 'PMIN_MW', 'QMX_MVAR',
                            'QMN_MVAR', 'Ger_Units','Ger_Active_Units', 'PL_MW', 'QL_MVAR', 'TC', 'VMAX_PU', 'VMIN_PU', 'BCO_ID', 'B0_MVAR', 'ST', 'SHUNT_INST_IND', 'SHUNT_INST_CAP', 'Dia','Hora']
                Df_VF_novo = self.df[columns].merge(df1[['BUS_ID','Gen_Type','U_FED','REG', 'Latitude','Longitude']], on='BUS_ID', how='left')
                Df_VF_novo.drop(Df_VF_novo[Df_VF_novo['REG'] == np.nan].index)
                # Dataframe geral sem filtro
                self.Df_VF_SF = Df_VF_novo
            else:
                self.get_processdata()
            
            self.get_splitdata_PV_PQ()
            self.get_processdata_region()
            print(f'*** Salvando Dataframe com coordenadas ***')
            self.Df_VF_SF.to_csv(pathcsv, sep=';', index=False)
        
    def get_processdata(self):

        def labelBUS_UF_GT(data, keys, num: int, sf=None):
            Analize_patterns = [analize + r'\d{3}' for analize in keys]
            condition = data['BUS_NAME'].str.slice(-num).str.contains('|'.join(Analize_patterns))
            data.loc[condition, 'U_FED' if num == 5 else 'Gen_Type'] = data.loc[condition, 'BUS_NAME'].str[-num:-3]
            if sf:
                data.loc[condition, 'REG'] = sf
            
            return data

        def add_estados(data, condition, lista_1, lista_2):
            array_1 = np.array(lista_1)
            array_2 = np.array([(lat, lon) for lat, lon, _ in lista_2])
            distances = cdist(array_1, array_2, metric='euclidean')
            min_indices = np.argmin(distances, axis=1)
            asociaciones = [(lista_1[i], lista_2[min_indices[i]][2]) for i in range(len(lista_1))]
            data.loc[condition, 'U_FED'] = [label for _, label in asociaciones]
            return data

        def regiao(data):
            state_region_mapping = {
                'AC': 'AC-RO',
                'RO': 'AC-RO',
                'AM': 'Norte',
                'AP': 'Norte',
                'PA': 'Norte',
                'TO': 'Norte',
                'MA': 'Norte',
                'AL': 'Nordeste',
                'BA': 'Nordeste',
                'CE': 'Nordeste',
                'PB': 'Nordeste',
                'PE': 'Nordeste',
                'PI': 'Nordeste',
                'RN': 'Nordeste',
                'SE': 'Nordeste',
                'DF': 'Sudeste-Centro-Oeste',
                'GO': 'Sudeste-Centro-Oeste',
                'MT': 'Sudeste-Centro-Oeste',
                'MS': 'Sudeste-Centro-Oeste',
                'ES': 'Sudeste-Centro-Oeste',
                'MG': 'Sudeste-Centro-Oeste',
                'RJ': 'Sudeste-Centro-Oeste',
                'SP': 'Sudeste-Centro-Oeste',
                'PR': 'Sul',
                'RS': 'Sul',
                'SC': 'Sul'
            }  

            data.loc[data['U_FED'].isin(state_region_mapping), 'REG'] = data['U_FED'].map(state_region_mapping)
            return data
        
        column_rename_mapping = {
            'NB': 'BUS_ID',
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        }
        BarraGeo = pd.read_excel('RECURSOS/LATITUDE_LONGITUDE_SIN_ATUALIZADO.xlsx', sheet_name='Planilha1', header=0)
        BarraGeo.rename(columns=column_rename_mapping, inplace=True)

        #*********************************************************************************************************************************************

        Df_VF = self.df
        Df_ = Df_VF[(Df_VF['Dia']==Df_VF['Dia'].unique()[0]) & (Df_VF['Hora']==Df_VF['Hora'].unique()[0])][['BUS_ID', 'BUS_NAME']].copy()

        Df_.insert(1, 'U_FED', np.nan)
        Df_.insert(2, 'Gen_Type', np.nan)
        Df_.insert(3, 'REG', np.nan)

        labels_to_apply = [
            (['AC', 'RO'], 5, 'AC-RO'),
            (['AM', 'AP', 'PA', 'TO', 'MA'], 5, 'Norte'),
            (['AL', 'BA', 'CE', 'PB', 'PE', 'PI', 'RN', 'SE'], 5, 'Nordeste'),
            (['DF', 'GO', 'MT', 'MS', 'ES', 'MG', 'RJ', 'SP'], 5, 'Sudeste-Centro-Oeste'),
            (['PR', 'RS', 'SC'], 5, 'Sul'),
        ]

        def process_labels(label_args):
            keys, num, sf = label_args
            labelBUS_UF_GT(Df_, keys, num, sf)
            
        for labelarg in labels_to_apply:
            process_labels(labelarg)
            
        labelBUS_UF_GT(Df_,['UHE', 'UTE', 'UNE', 'PCH', 'EOL', 'UFV', 'BIO', 'SIN'], 6)

        # UNIR COORDENADAS
        Df_ = Df_.merge(BarraGeo[['BUS_ID', 'Latitude', 'Longitude']], on='BUS_ID', how='left')

        semcordenadas = Df_[Df_['Latitude'].isna()].shape[0]
        # bus_ngeo = Df_[Df_['Latitude'].isna()]['BUS_ID'].values
        print(f'Existe um total de, {semcordenadas} barras modeladas sem coordenadas segundo a base de dados usada')
        Df_ = Df_.dropna(subset=['Latitude'])
        sem_estado = Df_[Df_['REG'].isna()].shape[0]

        print(f'A partir da base de dados filtrada por barras com coordenadas existe, {sem_estado} barras modeladas sem região ou estado asociado pelo nome')
        print(f'*** ETAPA: Asignação de estado e região pelas coordenadas geograficas ***')

        lista_1 = Df_[Df_['U_FED'].isna()][['Latitude','Longitude']].values
        lista_2 = Df_[~Df_['U_FED'].isna()][['Latitude','Longitude','U_FED']].values
        condition = Df_['U_FED'].isna()

        Df_ = add_estados(Df_, condition, lista_1, lista_2)
        Df_ = regiao(Df_)

        sem_estado = Df_[Df_['REG'].isna()].shape[0]
        print(f'O número de barras sem estado associado foi reduzido para {sem_estado}')

        #***************************************************** Merge com o DATA FRAME COMPLETO ******************************************************
        # Dataframe geral 
        
        columns = ['BUS_ID', 'BUS_NAME', 'VBASEKV', 'TP', 'ARE', 'MODV_PU', 'ANGV_DEG',
                    'BASE_MVA', 'PG_MW', 'QG_MVAR', 'PMAX_MW', 'PMIN_MW', 'QMX_MVAR',
                    'QMN_MVAR', 'Ger_Units','Ger_Active_Units', 'PL_MW', 'QL_MVAR', 'TC', 'VMAX_PU', 'VMIN_PU',
                    'BCO_ID', 'B0_MVAR', 'ST', 'SHUNT_INST_IND', 'SHUNT_INST_CAP', 'Dia',
                    'Hora']
        
        Df_VF_novo = Df_VF[columns].merge(Df_[['BUS_ID','U_FED','Gen_Type','REG', 'Latitude','Longitude']], on='BUS_ID', how='left')
        Df_VF_novo.drop(Df_VF_novo[Df_VF_novo['REG'] == np.nan].index)

        # Dataframe geral sem filtro
        self.Df_VF_SF = Df_VF_novo

    def get_splitdata_PV_PQ(self):
        # Read DBAR.csv into DataFrame
        df_buscode = pd.read_csv('RECURSOS/DBAR.csv', sep=';')

        # Filter Df_VF based on BUS_ID
        barra_ids = [7050, 7051, 7061, 7062, 7064, 7055, 7053, 7063, 7060, 7056, 7065]
        Df_VF = self.Df_VF_SF
        Df_VF['REG'] = np.where(Df_VF['BUS_ID'].isin(barra_ids), 'Sudeste-Centro-Oeste', Df_VF['REG'])
        print(f"Trocando de Região as barras do compplexo madeira: {barra_ids}") 

        # Drop rows with NaN Latitude
        Df_VF.dropna(subset=['Latitude'], inplace=True)

        # Filter df_buscode and Df_VF based on 'Code' and 'BUS_ID' respectively
        dfcode = df_buscode[df_buscode['Code'] == 0]
        Df_VF = Df_VF[Df_VF['BUS_ID'].isin(dfcode['BusID'].unique())]

        # Print number of unique BUS_IDs
        print(f"Numero de Barras no pwf sem aplicar o filtro:  {df_buscode['BusID'].nunique()}") 
        print(f"Numero de Barras no pwf filtrando barras com indice 0: {dfcode['BusID'].nunique()}") 
        print(f"Numero de Barras no ntw sem filtro de barras: {self.df['BUS_ID'].nunique()}") 
        print(f"Numero de Barras no ntw filtrando barras com indice 0: {Df_VF['BUS_ID'].nunique()}") 

        self.Df_VF = Df_VF

        # Barras PV
        df_Final_ger = Df_VF[Df_VF['TP'].isin([2, 3])].copy()
        online = df_Final_ger[df_Final_ger['PG_MW'] != 0].groupby('BUS_ID')['PG_MW'].count().rename('Online').reset_index()
        compsyn = df_Final_ger[(df_Final_ger['PG_MW'] == 0) & (df_Final_ger['QG_MVAR'] != 0)].groupby('BUS_ID')['QG_MVAR'].count().rename('Compsync').reset_index()
        dff_Ger_map = df_Final_ger.groupby('BUS_ID').agg(
            BUS_NAME=('BUS_NAME', 'first'),
            VBASEKV=('VBASEKV', 'first'),
            U_FED=('U_FED', 'first'),
            REG=('REG', 'first'),
            Gen_Type=('Gen_Type', 'first'),
            Latitude=('Latitude', 'first'),
            Longitude=('Longitude', 'first'),
            Dia=('Dia', list),
            Hora=('Hora', list),
            MODV_PU=('MODV_PU', list),
            BASE_MVA=('BASE_MVA', 'mean'),
            Ger_Units=('Ger_Units', 'first'),
            PG_MW=('PG_MW', 'mean'),
            QG_MVAR=('QG_MVAR', 'mean'),
            PMAX_MW=('PMAX_MW', 'mean'),
            PMIN_MW=('PMIN_MW', 'mean'),
            QMX_MVAR=('QMX_MVAR', 'mean'),
            QMN_MVAR=('QMN_MVAR', 'mean'),
        ).reset_index()
        dff_Ger_map = dff_Ger_map.merge(online, on='BUS_ID', how='left').merge(compsyn, on='BUS_ID', how='left').fillna({'Online': 0, 'Compsync': 0})

        # Barras PQ
        df_Final_nt = Df_VF[Df_VF['TP'].isin([0, 1])].copy()
        dff_NT_map = df_Final_nt.groupby('BUS_ID').agg(
            BUS_NAME=('BUS_NAME', 'first'),
            VBASEKV=('VBASEKV', 'first'),
            U_FED=('U_FED', 'first'),
            REG=('REG', 'first'),
            Gen_Type=('Gen_Type', 'first'),
            Latitude=('Latitude', 'first'),
            Longitude=('Longitude', 'first'),
            MODV_PU=('MODV_PU', list),
            SHUNT=('B0_MVAR', 'first'),
            SHUNT_INST_IND=('SHUNT_INST_IND', 'first'),
            SHUNT_INST_CAP=('SHUNT_INST_CAP', 'first'),
        ).reset_index()

        # Calculate 'ReservaINDshunt' and 'ReservaCAPshunt' using vectorized operations
        df_Final_nt['ReservaINDshunt'] = np.where(df_Final_nt['B0_MVAR'] < 0, df_Final_nt['SHUNT_INST_IND'] - df_Final_nt['B0_MVAR'], df_Final_nt['SHUNT_INST_IND'])
        df_Final_nt['ReservaCAPshunt'] = np.where(df_Final_nt['B0_MVAR'] > 0, df_Final_nt['SHUNT_INST_CAP'] - df_Final_nt['B0_MVAR'], df_Final_nt['SHUNT_INST_CAP'])

        # Calculate 'Qmin' and 'Qmax' using vectorized operations
        df_Final_ger['Qmin'] = (df_Final_ger['QMN_MVAR'] / df_Final_ger['Ger_Units']) * df_Final_ger['Ger_Active_Units']
        df_Final_ger['Qmax'] = (df_Final_ger['QMX_MVAR'] / df_Final_ger['Ger_Units']) * df_Final_ger['Ger_Active_Units']
        df_Final_ger['ReservaIND'] = np.where(df_Final_ger['QG_MVAR'] < 0, df_Final_ger['Qmin'] - df_Final_ger['QG_MVAR'], df_Final_ger['Qmin'])
        df_Final_ger['ReservaCAP'] = np.where(df_Final_ger['QG_MVAR'] > 0, df_Final_ger['Qmax'] - df_Final_ger['QG_MVAR'], df_Final_ger['Qmax'])

        self.df_Final_ger = df_Final_ger
        self.df_Final_nt = df_Final_nt
        self.dff_Ger_map = dff_Ger_map
        self.dff_NT_map = dff_NT_map

    def get_processdata_region(self):

        def discriminador(valor_lista):
            indutivo = sum(valor for valor in valor_lista if valor < 0)
            capacitivo = sum(valor for valor in valor_lista if valor > 0)
            return indutivo, capacitivo

        def separar_shunt(data):
            data[['Shunt_Ind', 'Shunt_Cap']] = data['B0_MVAR'].apply(lambda x: pd.Series(discriminador(x)))
            return data

        def process_generation_data(df_gerbar, generation_type, group_columns):
            df_generation = df_gerbar[df_gerbar['Gen_Type'].isin(generation_type)]
            return df_generation.groupby(by=group_columns).agg(
                PG_MW=('PG_MW', 'sum'),
                QG_MVAR=('QG_MVAR', 'sum'),
                NUM_USINAS=('PG_MW', 'count')
            )

        def fill_nan_columns(df, columns):
            df[columns] = df[columns].fillna(0)
            return df

        print(f'*** ETAPA: COMENÇO DA CRIAÇÃO DE DATAFRAME COM INFO REGIONAL ***')

        Df_UHE = process_generation_data(self.df_Final_ger, ['UHE', 'PCH'], ['Dia', 'Hora', 'REG'])
        Df_UTE = process_generation_data(self.df_Final_ger, ['UTE', 'UNE'], ['Dia', 'Hora', 'REG'])
        Df_FERV_EOL = process_generation_data(self.df_Final_ger, ['EOL'], ['Dia', 'Hora', 'REG'])
        Df_FERV_SOL = process_generation_data(self.df_Final_ger, ['UFV'], ['Dia', 'Hora', 'REG'])
        Df_FERV_BIO = process_generation_data(self.df_Final_ger, ['BIO'], ['Dia', 'Hora', 'REG'])
        Df_FERV_SIN = process_generation_data(self.df_Final_ger, ['SIN'], ['Dia', 'Hora', 'REG'])

        DF_REGIONAL_GER = self.df_Final_ger.groupby(by=['Dia', 'Hora', 'REG']).agg({
            'BUS_ID': 'unique', 'MODV_PU': list, 'B0_MVAR': list, 'PG_MW': 'sum', 'QG_MVAR': 'sum',
            'PL_MW': 'sum', 'QL_MVAR': 'sum', 'SHUNT_INST_IND': 'sum', 'SHUNT_INST_CAP': 'sum', 'ReservaIND': 'sum',
            'ReservaCAP': 'sum'
        })

        for df, df_name in zip([Df_UHE, Df_UTE, Df_FERV_EOL, Df_FERV_SOL, Df_FERV_BIO, Df_FERV_SIN],
                            ['PG_UHE', 'PG_UTE', 'PG_EOL', 'PG_SOL', 'PG_BIO', 'PG_SIN']):
            DF_REGIONAL_GER[df_name] = df['PG_MW']

        for df, df_name in zip([Df_UHE, Df_UTE, Df_FERV_EOL, Df_FERV_SOL, Df_FERV_BIO, Df_FERV_SIN],
                            ['QG_UHE', 'QG_UTE', 'QG_EOL', 'QG_SOL', 'QG_BIO', 'QG_SIN']):
            DF_REGIONAL_GER[df_name] = df['QG_MVAR']

        for df, df_name in zip([Df_UHE, Df_UTE, Df_FERV_EOL, Df_FERV_SOL, Df_FERV_BIO, Df_FERV_SIN],
                            ['Num_Usinas_UHE', 'Num_Usinas_UTE', 'Num_Usinas_EOL', 'Num_Usinas_SOL',
                                'Num_Usinas_BIO', 'Num_Usinas_SIN']):
            DF_REGIONAL_GER[df_name] = df['NUM_USINAS']

        DF_REGIONAL_GER = fill_nan_columns(DF_REGIONAL_GER, ['PG_UHE', 'PG_UTE', 'PG_EOL', 'PG_SOL', 'PG_BIO', 'PG_SIN',
                                                            'QG_UHE', 'QG_UTE', 'QG_EOL', 'QG_SOL', 'QG_BIO', 'QG_SIN',
                                                            'Num_Usinas_UHE', 'Num_Usinas_UTE', 'Num_Usinas_EOL',
                                                            'Num_Usinas_SOL', 'Num_Usinas_BIO', 'Num_Usinas_SIN'])

        DF_REGIONAL_PQ = self.df_Final_nt.groupby(by=['Dia', 'Hora', 'REG']).agg({
            'BUS_ID': 'unique', 'MODV_PU': list, 'B0_MVAR': list, 'PG_MW': 'sum', 'QG_MVAR': 'sum', 'PL_MW': 'sum',
            'QL_MVAR': 'sum', 'SHUNT_INST_IND': 'sum', 'SHUNT_INST_CAP': 'sum'
        })

        print(f'*** ETAPA: SEPARAÇÃO DE SHUNT ***')

        DF_REGIONAL_GER = separar_shunt(DF_REGIONAL_GER)
        DF_REGIONAL_PQ = separar_shunt(DF_REGIONAL_PQ)

        DF_REGIONAL_PQ = fill_nan_columns(DF_REGIONAL_PQ,
                                        ['PG_MW', 'QG_MVAR', 'PL_MW', 'QL_MVAR', 'Shunt_Ind', 'Shunt_Cap',
                                            'SHUNT_INST_IND', 'SHUNT_INST_CAP'])
        DF_REGIONAL_GER = fill_nan_columns(DF_REGIONAL_GER, ['Shunt_Ind', 'Shunt_Cap', 'SHUNT_INST_IND', 'SHUNT_INST_CAP'])

        DF_REGIONAL_GER['PG_Dist'] = DF_REGIONAL_GER['PG_MW'] - (
                    DF_REGIONAL_GER['PG_UHE'] + DF_REGIONAL_GER['PG_UTE'] + DF_REGIONAL_GER['PG_EOL'] +
                    DF_REGIONAL_GER['PG_SOL'] + DF_REGIONAL_GER['PG_BIO']) + DF_REGIONAL_PQ['PG_MW']
        DF_REGIONAL_GER['QG_Dist'] = DF_REGIONAL_GER['QG_MVAR'] - (
                    DF_REGIONAL_GER['QG_UHE'] + DF_REGIONAL_GER['QG_UTE'] + DF_REGIONAL_GER['QG_EOL'] +
                    DF_REGIONAL_GER['QG_SOL'] + DF_REGIONAL_GER['QG_BIO']) + DF_REGIONAL_PQ['QG_MVAR']

        # Sumo o shunt que existe en el dataframe de barras PV e barras PQ
        DF_REGIONAL_GER[['PL_MW', 'QL_MVAR', 'Shunt_Ind', 'Shunt_Cap', 'SHUNT_INST_IND', 'SHUNT_INST_CAP']] += \
            DF_REGIONAL_PQ[['PL_MW', 'QL_MVAR', 'Shunt_Ind', 'Shunt_Cap', 'SHUNT_INST_IND', 'SHUNT_INST_CAP']]

        self.DF_REGIONAL_GER = DF_REGIONAL_GER
        self.DF_REGIONAL_PQ = DF_REGIONAL_PQ
        
        print(f'*** ETAPA: FINAL DO PROCESSAMENTO DE DADOS ***')


# dataF = pd.read_csv('DataframeInfoSys.csv')
# print('dataframe leido')
# processdata = ProcessData(dataF, None, False)

