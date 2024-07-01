import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import dask.dataframe as dd
from NTW_Reader import NTW_Reader
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


class ReadScenarios:

    def __init__(self, path, cenario, PO, pathcsv=None):
        self.path = path
        self.PO = PO
        self.csv = pathcsv
        self.cenario = cenario
        self.folders = []
        pass

    @staticmethod
    def get_unique_bus(df):
        df_modified_BUS = df[0].groupby('BUS_ID').first().reset_index()
        df_modified_GEN = df[1].groupby('BUS_ID').sum().reset_index()
        df_modified_GEN.rename(columns={'ST': 'Ger_Active_Units'}, inplace=True)
        df_GEN = df[1].groupby('BUS_ID')['PG_MW'].count().reset_index().rename(columns={'PG_MW': 'Ger_Units'})
        df_modified_GEN = pd.merge(df_modified_GEN, df_GEN, on='BUS_ID', how='left')
        df_modified_LOAD = df[2].groupby('BUS_ID').agg({
            'PL_MW': 'sum',
            'QL_MVAR': 'sum',
        }).reset_index()

        return df_modified_BUS, df_modified_GEN, df_modified_LOAD

    def read_file(self, file_path):
        try:
            Caso_name = os.path.basename(file_path).replace('.ntw', '').replace('.txt', '')
            Dia_name = self.folders if self.PO else os.path.basename(os.path.dirname(os.path.dirname(file_path)))[-2:]

            NetData = NTW_Reader(file_path)

            Bus = NetData.bus_data[['BUS_ID', 'BUS_NAME', 'VBASEKV', 'TP', 'ARE', 'MODV_PU', 'ANGV_DEG']]
            Gen = NetData.gen_data[['BUS_ID', 'ST', 'PG_MW', 'QG_MVAR', 'BASE_MVA', 'PMAX_MW', 'PMIN_MW', 'QMX_MVAR', 'QMN_MVAR']]
            Load = NetData.load_data[['BUS_ID', 'PL_MW', 'QL_MVAR']]
            Shunt = NetData.DF_shunt

            df = [Bus, Gen, Load]
            BUS_grouped, GEN_grouped, LOAD_grouped = self.get_unique_bus(df)

            df_merge_0 = pd.merge(BUS_grouped, GEN_grouped, on='BUS_ID', how='outer').sort_values('BUS_ID')
            df_merge_1 = pd.merge(df_merge_0, LOAD_grouped, on='BUS_ID', how='outer').sort_values('BUS_ID')
            merged_df = pd.merge(df_merge_1, Shunt, on='BUS_ID', how='outer').sort_values('BUS_ID')

            merged_df['Dia'] = str(Dia_name)
            merged_df['Hora'] = str(Caso_name[-5:])

            return merged_df
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def get_data_extract(self):
        if not self.PO:
            path_arquivo = [os.path.join(self.path, directory, 'Output') for directory in self.folders]
            directories_files = [os.listdir(path) for path in path_arquivo]

            files_path = [os.path.join(path_arquivo[i], file_name)
                          for i in range(len(path_arquivo))
                          for file_name in directories_files[i]
                          if file_name.endswith('.ntw')]
        else:
            files_path = [self.path]

        print('Extracting the Data of ntw files...')

        results = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.read_file, file_path): file_path for file_path in files_path}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        print('Concatenating ...')
        if results:  # Check if results is not empty before concatenating
            df_Cvg = pd.concat(results, axis=0, sort=False)
            self.Df_Cases = df_Cvg.sort_values(by=['Dia', 'Hora'])
            self.OpPointsC = len(results)
            print('The total number of analyzed cases is:', self.OpPointsC)
        else:
            self.Df_Cases = pd.DataFrame()
            self.OpPointsC = 0
            print('No data was processed.')

    def get_dataframes_csv(self):
        self.Df_Cases = pd.read_csv(self.csv, sep=';')
        self.Df_Cases['Dia'] = self.Df_Cases['Dia'].astype(str).str.zfill(2)
        results = self.Df_Cases.groupby(['Dia', 'Hora'])['BUS_ID'].first()
        self.OpPointsC = results.shape[0]
        print('The total number of analyzed cases is: ', self.OpPointsC)
        
# ======================================================================================================================
#                                                AC & DC LINES & RESERVE INFO EXTRACTION 
# ======================================================================================================================
        
    def get_Networkinfo(self, linhas=False, Reserva=False, Intercambios=False, hour=None):

        print(f'*** LEITURA DOS PWF_.CSV GERADOS ***')
        
        dtype_dict_linhas = {
            'From#': 'int32', ' From Name': 'object', ' To# - Circ#': 'object', ' To Name': 'object', ' Type': 'object',
            ' MVA': 'float32', ' % L1': 'float32', ' L1(MVA)': 'float32', ' MW:From-To': 'float32', ' Mvar:From-To': 'float32',
            ' Mvar:Losses': 'float32', ' MW:To-From': 'float32', ' Power Factor:From-To': 'float32', ' Power Factor:To-From': 'float32'
        }
        col_list_lines = [
            'From#', ' From Name', ' To# - Circ#', ' To Name', ' Type', ' MVA', ' % L1', ' L1(MVA)', ' MW:From-To',
            ' Mvar:From-To', ' Mvar:Losses', ' MW:To-From', ' Power Factor:From-To', ' Power Factor:To-From'
        ]
        col_list_hvdc = ['Bus #', ' Bus Name', ' Type', ' Pole #', ' P(MW)', ' Q(Mvar)', ' Status']
        col_list_reserve = ['Bus', ' Group', ' Bus Name', ' Area', ' Zone', ' V (pu)', ' Pg(MW)', ' Qg(Mvar)', ' Reserve', ' Units']

        def add_dia_hora(df, path, dia, hour=None):
            df['Dia'] = dia
            df['Hora'] = hour if hour else os.path.basename(path).split('_')[-1].split('.')[0]
            return df

        def get_files_path(base_path, pattern):
            return [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith(pattern + '.csv')]

        if self.PO:
            files_path = get_files_path(os.path.dirname(self.path), hour)
        else:
            files_path = []
            for day in sorted(self.folders):
                folder = os.path.join(self.path, day, 'Output')
                files_path.extend(get_files_path(folder, ''))

        def read_files(pattern, col_list, dtype_dict=None):
            filtered_files = [file for file in files_path if pattern in os.path.basename(file)]
            if not filtered_files:
                return []
            try:
                dfs = [dd.read_csv(file, sep=';', skiprows=[0], usecols=col_list, dtype=dtype_dict).pipe(add_dia_hora, file, os.path.basename(os.path.dirname(os.path.dirname(file)))[-2:], hour) for file in filtered_files]
            except ValueError:
                dfs = [dd.read_csv(file, sep=';', skiprows=[0], dtype=dtype_dict).pipe(add_dia_hora, file, os.path.basename(os.path.dirname(os.path.dirname(file)))[-2:], hour) for file in filtered_files]
                
            return dfs

        PWFs_sep = read_files('PWF16_', col_list_lines) if linhas else []
        DCLinks_sep = read_files('PWF25_', col_list_hvdc) if Intercambios else []
        SGN01_sep = read_files('SGN01_', col_list_reserve) if Reserva else []

        def concat_and_compute(dfs):
            if dfs:
                return dd.concat(dfs, ignore_index=True).compute()
            return pd.DataFrame()

        if linhas:
            PWF16_concatenados = concat_and_compute(PWFs_sep)

            PWF16_concatenados.rename(columns={'From#':'From#', ' From Name': 'From Name', ' To# - Circ#':'To# - Circ#', ' To Name':'To Name', ' Type':'Type', ' MVA':'MVA', ' MW:From-To':'MW:From-To', ' Mvar:From-To':'Mvar:From-To',' % L1':'% L1', ' L1(MVA)':'L1(MVA)',  ' Mvar:Losses':'Mvar:Losses', ' MW:To-From':'MW:To-From', ' Power Factor:From-To':'Power Factor:From-To', ' Power Factor:To-From':'Power Factor:To-From'}, inplace=True)

            if not PWF16_concatenados.empty:
                print("Concatenação das linhas")
                PWF16_concatenados[['To#', 'Circ#']] = PWF16_concatenados["To# - Circ#"].str.split(' # ', expand=True)
                PWF16_concatenados['From#'] = PWF16_concatenados['From#'].astype('int32')
                PWF16_concatenados['To#'] = PWF16_concatenados['To#'].astype('int32')
                PWF16_concatenados.drop(columns=["To# - Circ#"], inplace=True)
                self.linesInfo = PWF16_concatenados
                if not self.PO:
                    print("Salvando Dataframe das linhas")
                    PWF16_concatenados.to_csv(os.path.join(self.path, 'LinhasInfo.csv'), index=None)
                    print("Final da leitura das Linhas")
                if Intercambios:
                    self.get_Intercambios()

        if Intercambios:
            print("Concatenação da info do HVDC")
            DCLinks_concatenados = concat_and_compute(DCLinks_sep)
            if not DCLinks_concatenados.empty:
                self.HVDCInfo = DCLinks_concatenados
                if not self.PO:
                    print("Salvando Dataframe do HVDC")
                    DCLinks_concatenados.to_csv(os.path.join(self.path, 'HVDCInfo.csv'), index=None)
                    print("Final da leitura do HVDC")

        if Reserva:
            print("Concatenação da Reserva")
            SGN01_concatenados = concat_and_compute(SGN01_sep)
            if not SGN01_concatenados.empty:
                SGN01_concatenados['Bus'] = SGN01_concatenados['Bus'].astype(int)
                SGN01_concatenados[' Pg(MW)'] = SGN01_concatenados[' Pg(MW)'].astype(float)
                SGN01_concatenados[' Qg(Mvar)'] = SGN01_concatenados[' Qg(Mvar)'].astype(float)
                SGN01_concatenados[' Reserve'] = SGN01_concatenados[' Reserve'].astype(float)
                SGN01_concatenados[' Units'] = SGN01_concatenados[' Units'].astype(int)
                self.ReserveInfo = SGN01_concatenados
                if not self.PO:
                    print("Salvando Dataframe da Reserva")
                    SGN01_concatenados.to_csv(os.path.join(self.path, 'ReservaInfo.csv'), index=None)
                    print("Final da leitura da Reserva")

    def generate_script(self, path = None):

        # Rodar só uma vez para gerar o arquivo de texto para simular no organon
        if path == None:
            folder1 = self.path
        else:
            folder1 = path

        files_and_directories = os.listdir(folder1)
        var = [i.find("DS20") for i in files_and_directories]
        dias = []
        for i in range(len(files_and_directories)):
            if var[i] >= 0:
                dias.append(files_and_directories[i])
        dias.sort() 

        folder_aux = []
        path_script = folder1 + "script_savePWF.txt"
        filesgeral = []
        for i in dias:
            # folder = folder1 + i +'/' # caso de melhorprevisão
            folder = folder1 + i + '/Output/'
            folder_aux.append(folder)
            files_and_directories = os.listdir(folder) # Abre a pasta do dia
            files = []
            var = [i.find(".ntw") for i in files_and_directories] # Seleciona os PWFs
            for i in range(len(files_and_directories)):
                if var[i] > 0:
                    files.append(files_and_directories[i].replace(".ntw", ""))
            files.sort()
            filesgeral.append(files)

        with open(path_script, 'w') as f:

            for idx, i in enumerate(folder_aux):
                for j in filesgeral[idx]:
                    f.write('OPEN "' + i + 'SCN.prm"')
                    f.write('\n')
                    f.write('OPEN "' + i + j + '.ntw"')
                    f.write('\n')
                    f.write('OPEN "' + i + 'Model.dyn"')
                    f.write('\n')
                    f.write('NEWTON')
                    f.write('\n')
                    f.write('CSV PWF16')
                    f.write('\n')
                    f.write('CSV SGN01')
                    f.write('\n')
                    f.write('CSV PWF25')
                    f.write('\n')
                    f.write('COPY PWF16.csv PWF16_' + j +'.csv')
                    f.write('\n')
                    f.write('COPY PWF25.csv PWF25_' + j +'.csv')
                    f.write('\n')
                    f.write('COPY SGN01.csv SGN01_' + j +'.csv')
                    f.write('\n')

        print('Script para rodar fluxos gerado exitosamente!')

    def get_Intercambios(self, df=None):
        print(f'*** OBTENÇÃO DOS INTERCAMBIOS ***')
        if df is None:
            PWF16_concatenados = self.linesInfo
        else:
            PWF16_concatenados = df

        linhas_expNE = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_EXPNE.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_expNE_flip = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_EXPNE_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNS = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_FNS.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNESE = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_FNESE.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNESE_flip = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_FNESE_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNEN = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_FNEN.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNEN_flip = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_FNEN_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FSULSECO = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_FSULSECO.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FSULSECO_flip = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_FSULSECO_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_RSUL = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_RSUL.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_RSUL_flip = pd.read_csv('Static-Analysis/RECURSOS/LINHAS/buses_RSUL_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        
        mask = PWF16_concatenados[['From#', 'To#']].apply(tuple, axis=1)

        print(f'******')
        EXPNE_grouped = PWF16_concatenados[mask.isin(set(linhas_expNE.index))]
        EXPNE_grouped.loc[EXPNE_grouped[['From#', 'To#']].apply(tuple, axis=1).isin(set(linhas_expNE_flip.index)),'MW:From-To'] *= -1
        EXPNE_grouped = EXPNE_grouped.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_NS = PWF16_concatenados[mask.isin(set(linhas_FNS.index))]
        Fluxo_NS_grouped = Fluxo_NS.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_NESE = PWF16_concatenados[mask.isin(set(linhas_FNESE.index))]
        Fluxo_NESE.loc[Fluxo_NESE[['From#', 'To#']].apply(tuple, axis=1).isin(set(linhas_FNESE_flip.index)), 'MW:From-To'] *= -1
        Fluxo_NESE_grouped = Fluxo_NESE.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_NEN = PWF16_concatenados[mask.isin(set(linhas_FNEN.index))]
        Fluxo_NEN.loc[Fluxo_NEN[['From#', 'To#']].apply(tuple, axis=1).isin(set(linhas_FNEN_flip.index)), 'MW:From-To'] *= -1
        Fluxo_NEN_grouped = Fluxo_NEN.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_SULSECO = PWF16_concatenados[mask.isin(set(linhas_FSULSECO.index))]
        Fluxo_SULSECO.loc[Fluxo_SULSECO[['From#', 'To#']].apply(tuple, axis=1).isin(set(linhas_FSULSECO_flip.index)), 'MW:From-To'] *= -1
        Fluxo_SULSECO_grouped = Fluxo_SULSECO.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})
        print(f'******')
        Fluxo_RSUL = PWF16_concatenados[mask.isin(set(linhas_RSUL.index))]
        Fluxo_RSUL.loc[Fluxo_RSUL[['From#', 'To#']].apply(tuple, axis=1).isin(set(linhas_RSUL_flip.index)), 'MW:From-To'] *= -1
        Fluxo_RSUL_grouped = Fluxo_RSUL.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})
        Fluxo_RSUL_grouped['MW:From-To'] *= -1
        
        print(f'*** Concatenating ... ***')
        self.DF_Intercambios = pd.concat([EXPNE_grouped,Fluxo_NESE_grouped, Fluxo_NS_grouped, Fluxo_SULSECO_grouped, Fluxo_NEN_grouped, Fluxo_RSUL_grouped], axis=0, keys=['EXP_NE', 'Fluxo_NE-SE', 'Fluxo_N-S' ,'Fluxo_SUL-SECO', 'Fluxo_NE-N', 'Fluxo_RSUL'])

        print(f'*** FINAL OBTENÇÃO DOS INTERCAMBIOS ***')


# ======================================================================================================================
#                                                   CONVERGENCE INFO EXTRACTION
# ======================================================================================================================

    def get_convergence_data(self):

        import re
        days = self.folders
        days.sort() 
        def getOPFdata(path):

            with open(path, 'r') as file:
                lines = file.readlines()
            data = []
            pattern = re.compile(r'([TF]\s+){6}')  # Padrão para encontrar linhas com 6 valores True/False

            for i in range(len(lines)):
                if (i>9):
                    line = lines[i].strip()
                    # Encontrar linhas que correspondem ao padrão
                    if pattern.match(line):
                        name = lines[i + 17].strip().split()[0]  # Pegar o nome com final .ntw
                        values = line.split()[:6]  # Pegar os primeiros 6 valores True/False
                        data.append([name, *values])

            # Criar o DataFrame
            df = pd.DataFrame(data, columns=['Nome', 'Valor1', 'Valor2', 'Valor3', 'Valor4', 'Valor5', 'Valor6'])
            df = df.drop(df.index[-1])
            return df

        dtfrs = []
        for i in days:
            folder = self.path + i  + '/Output/'
            arquivos_REP= [arquivo for arquivo in os.listdir(folder) if arquivo.startswith('SCD.rep')]
            for arquivo in arquivos_REP:
                caminho_arquivo = os.path.join(folder, arquivo)
                df = getOPFdata(caminho_arquivo)
                df['Dia'] = i[-2:]
                df['Hora'] = df['Nome'].apply(lambda x: x.replace('.ntw','')[-5:])
                dtfrs.append(df)

        OPFs_concatenados = pd.concat(dtfrs, ignore_index=True)
        self.OPF = OPFs_concatenados

        print(' => Informação da convergencia dos casos:')
        self.OPF_NC = OPFs_concatenados[(OPFs_concatenados['Valor4'] == 'F')]
        self.PWF_NC = OPFs_concatenados[(OPFs_concatenados['Valor5'] == 'F')]
        print('=============================================')
        print('Numero de casos não Convergidos no OPF: ' + str(len(self.OPF_NC)) + '=> ' + str(round(len(self.OPF_NC)/1344*100,2)))
        print('Numero de casos não Convergidos no PWF: ' + str(len(self.PWF_NC)) + '=> ' +  str(round(len(self.PWF_NC)/1344*100,2)))
        print('==============================================')

        self.OPF_NC[['Dia','Hora']].to_csv(self.cenario+'/Data/Geral/OPF_NC.csv', index=None)
        self.PWF_NC[['Dia','Hora']].to_csv(self.cenario+'/Data/Geral/PWF_NC.csv', index=None)

class ProcessData():

    def __init__(self, cenario, options):
        
        self.cenario = cenario
        self.options = options
        self.mapsdata = options['MapasPlots']
        pass

    def get_processdata(self, Df_VF):

    # ========================== Define main funtions for this method ===================================================================
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
    # =====================================================================================================================
        column_rename_mapping = {
            'NB': 'BUS_ID',
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        }
        file = os.path.abspath('Static-Analysis/RECURSOS/LATITUDE_LONGITUDE_SIN_ATUALIZADO.xlsx')
        BarraGeo = pd.read_excel(file, sheet_name='Planilha1', header=0)
        BarraGeo.rename(columns=column_rename_mapping, inplace=True)

        # dataframe com um unico ponto de operação
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
        print(f'Existe um total de, {semcordenadas} barras modeladas sem coordenadas asociadas segundo a base de dados usada')
        print('... filtrando só as barras que aparecem com coordenadas ...')

        Df_ = Df_.dropna(subset=['Latitude'])
        sem_estado = Df_[Df_['REG'].isna()].shape[0]
        print(f'A partir da base de dados filtrada por barras com coordenadas, existe {sem_estado} barras modeladas sem região ou estado asociado pelo nome')

        print(f'*** ETAPA: Asignação de estado e região pelas coordenadas geograficas ***')
        lista_1 = Df_[Df_['U_FED'].isna()][['Latitude','Longitude']].values
        lista_2 = Df_[~Df_['U_FED'].isna()][['Latitude','Longitude','U_FED']].values
        condition = Df_['U_FED'].isna()
        Df_ = add_estados(Df_, condition, lista_1, lista_2)
        Df_ = regiao(Df_)

        sem_estado = Df_[Df_['REG'].isna()].shape[0]
        print(f'O número de barras sem estado associado foi reduzido para {sem_estado}')

        #***************************************************** Merge com o DATA FRAME COMPLETO ******************************************************
        columns = ['BUS_ID', 'BUS_NAME', 'VBASEKV', 'TP', 'ARE', 'MODV_PU', 'ANGV_DEG',
                    'BASE_MVA', 'PG_MW', 'QG_MVAR', 'PMAX_MW', 'PMIN_MW', 'QMX_MVAR',
                    'QMN_MVAR', 'Ger_Units','Ger_Active_Units', 'PL_MW', 'QL_MVAR', 'TC',
                    'VMAX_PU', 'VMIN_PU', 'BCO_ID', 'B0_MVAR', 'ST', 'SHUNT_INST_IND', 
                    'SHUNT_INST_CAP', 'Dia', 'Hora']
        # Dataframe geral 
        self.Df_VF_SF = Df_VF[columns].merge(Df_[['BUS_ID','U_FED','Gen_Type','REG', 'Latitude','Longitude']], on='BUS_ID', how='left')
        self.Df_VF_SF.drop(self.Df_VF_SF[self.Df_VF_SF['REG'] == np.nan].index)

    @staticmethod
    def add_key(data):
        from datetime import datetime, timedelta
        fechas = [dia for dia in range(1, 30)]
        semihoras_dia = [(datetime(2022, 10, dia, 0, 0) + timedelta(minutes=30*i)).strftime('%d-%H-%M') for dia in fechas for i in range(48)]
        df = pd.DataFrame({'key': semihoras_dia})
        df[['Dia', 'Hora']] = df['key'].str.split('-', n=1, expand=True)
        df['key'] = df['key'].str.replace('-','_')
        df['key'] = 'D_' + df['key'].str.slice(0,2) + '_H' + df['key'].str.slice(2) 
        df['Dia'] = df['Dia'].astype(str)
        data = data.merge(df, on=['Dia','Hora'], how='inner')
        return data
    
    def get_splitdata_PV_PQ(self, df):

        # Read DBAR.csv into DataFrame
        file = os.path.abspath('Static-Analysis\RECURSOS\DBAR.csv')
        df_buscode = pd.read_csv(file, sep=';')

        # complexo madeira buses
        barra_ids = [7050, 7051, 7061, 7062, 7064, 7055, 7053, 7063, 7060, 7056, 7065]
        print(f"Trocando de Região as barras do complexo madeira: {barra_ids}")
        self.Df_VF_SF['REG'] = np.where(self.Df_VF_SF['BUS_ID'].isin(barra_ids), 'Sudeste-Centro-Oeste', self.Df_VF_SF['REG'])
        
        # Usinas Eolicas buses
        barra_ids =['MSULD3-EOL22', 'CLEMNTEOL-66', 'CLEMNTEOL-60', 'MSULD1-EOL27', 'MSULD2-EOL27', 'MSULD4-EOL08'] 
        print(f"Asignando tipo de geração nas usinas eolicas faltantes: {barra_ids}")
        self.Df_VF_SF['Gen_Type'] = np.where(self.Df_VF_SF['BUS_NAME'].isin(barra_ids), 'EOL', self.Df_VF_SF['Gen_Type'])
        
        if self.options['extract_fromcsv']==False:
            # Asignando Chave
            print("Asignando uma chave no dataframe ...")
            self.Df_VF_SF = self.add_key(self.Df_VF_SF)

        # Drop rows with NaN Latitude
        self.Df_VF_SF = self.Df_VF_SF[self.Df_VF_SF['Latitude'].notna()]
        # Filter df_buscode and self.Df_VF_SF based on 'Code' and 'BUS_ID' respectively
        dfcode = df_buscode[df_buscode['Code'] == 0]
        self.Df_VF_SF = self.Df_VF_SF.merge(dfcode[['BusID']], left_on='BUS_ID', right_on='BusID', how='inner').drop(columns=['BusID'])

        # Print number of unique BUS_IDs
        print(f"Numero de Barras no pwf sem aplicar o filtro:  {df_buscode['BusID'].nunique()}") 
        print(f"Numero de Barras no pwf filtrando barras com indice 0: {dfcode['BusID'].nunique()}") 
        print(f"Numero de Barras no ntw sem filtro de barras: {df['BUS_ID'].nunique()}") 
        print(f"Numero de Barras no ntw filtrando barras com indice 0: {self.Df_VF_SF['BUS_ID'].nunique()}") 

        # Barras PV
        df_Final_ger = self.Df_VF_SF[self.Df_VF_SF['TP'].isin([2, 3])].copy()
        # Barras PQ
        df_Final_nt = self.Df_VF_SF[self.Df_VF_SF['TP'].isin([0, 1])].copy()

        # Calculate 'Qmin' and 'Qmax' using vectorized operations
        df_Final_ger['Qmin'] = (df_Final_ger['QMN_MVAR'] / df_Final_ger['Ger_Units']) * df_Final_ger['Ger_Active_Units']
        df_Final_ger['Qmax'] = (df_Final_ger['QMX_MVAR'] / df_Final_ger['Ger_Units']) * df_Final_ger['Ger_Active_Units']
        df_Final_ger['ReservaIND'] = np.where(df_Final_ger['QG_MVAR'] < 0, df_Final_ger['Qmin'] - df_Final_ger['QG_MVAR'], df_Final_ger['Qmin'])
        df_Final_ger['ReservaCAP'] = np.where(df_Final_ger['QG_MVAR'] > 0, df_Final_ger['Qmax'] - df_Final_ger['QG_MVAR'], df_Final_ger['Qmax'])
        # Calculate 'ReservaINDshunt' and 'ReservaCAPshunt' using vectorized operations
        df_Final_nt['ReservaINDshunt'] = np.where(df_Final_nt['B0_MVAR'] < 0, df_Final_nt['SHUNT_INST_IND'] - df_Final_nt['B0_MVAR'], df_Final_nt['SHUNT_INST_IND'])
        df_Final_nt['ReservaCAPshunt'] = np.where(df_Final_nt['B0_MVAR'] > 0, df_Final_nt['SHUNT_INST_CAP'] - df_Final_nt['B0_MVAR'], df_Final_nt['SHUNT_INST_CAP'])

        self.df_Final_ger = df_Final_ger
        self.df_Final_nt = df_Final_nt

        if self.mapsdata:
            online = df_Final_ger[df_Final_ger['PG_MW'] != 0].groupby('BUS_ID')['PG_MW'].count().rename('Online').reset_index()
            compsyn = df_Final_ger[(df_Final_ger['PG_MW'] == 0) & (df_Final_ger['QG_MVAR'] != 0)].groupby('BUS_ID')['QG_MVAR'].count().rename('Compsync').reset_index()
            self.dff_Ger_map = df_Final_ger.groupby('BUS_ID').agg(
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
            self.dff_Ger_map = self.dff_Ger_map.merge(online, on='BUS_ID', how='left').merge(compsyn, on='BUS_ID', how='left').fillna({'Online': 0, 'Compsync': 0})
            
            self.dff_NT_map = df_Final_nt.groupby('BUS_ID').agg(
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

        print(f'*** ETAPA: CRIAÇÃO DE DATAFRAME COM INFO REGIONAL ***')

        Df_UHE = process_generation_data(self.df_Final_ger, ['UHE', 'PCH'], ['Dia', 'Hora', 'REG'])
        Df_UTE = process_generation_data(self.df_Final_ger, ['UTE', 'UNE'], ['Dia', 'Hora', 'REG'])
        Df_FERV_EOL = process_generation_data(self.df_Final_ger, ['EOL'], ['Dia', 'Hora', 'REG'])
        Df_FERV_SOL = process_generation_data(self.df_Final_ger, ['UFV'], ['Dia', 'Hora', 'REG'])
        Df_FERV_BIO = process_generation_data(self.df_Final_ger, ['BIO'], ['Dia', 'Hora', 'REG'])
        Df_FERV_SIN = process_generation_data(self.df_Final_ger, ['SIN'], ['Dia', 'Hora', 'REG'])

        DF_Regional_Ger = self.df_Final_ger.groupby(by=['Dia', 'Hora', 'REG']).agg({
            'BUS_ID': 'unique', 'MODV_PU': list, 'B0_MVAR': list, 'PG_MW': 'sum', 'QG_MVAR': 'sum',
            'PL_MW': 'sum', 'QL_MVAR': 'sum', 'SHUNT_INST_IND': 'sum', 'SHUNT_INST_CAP': 'sum', 'ReservaIND': 'sum',
            'ReservaCAP': 'sum', 'key':'first'
        })

        for df, df_name in zip([Df_UHE, Df_UTE, Df_FERV_EOL, Df_FERV_SOL, Df_FERV_BIO, Df_FERV_SIN],['PG_UHE', 'PG_UTE', 'PG_EOL', 'PG_SOL', 'PG_BIO', 'PG_SIN']):
            DF_Regional_Ger[df_name] = df['PG_MW']

        for df, df_name in zip([Df_UHE, Df_UTE, Df_FERV_EOL, Df_FERV_SOL, Df_FERV_BIO, Df_FERV_SIN],['QG_UHE', 'QG_UTE', 'QG_EOL', 'QG_SOL', 'QG_BIO', 'QG_SIN']):
            DF_Regional_Ger[df_name] = df['QG_MVAR']

        for df, df_name in zip([Df_UHE, Df_UTE, Df_FERV_EOL, Df_FERV_SOL, Df_FERV_BIO, Df_FERV_SIN],['Num_Usinas_UHE', 'Num_Usinas_UTE', 'Num_Usinas_EOL', 'Num_Usinas_SOL',  'Num_Usinas_BIO', 'Num_Usinas_SIN']): 
            DF_Regional_Ger[df_name] = df['NUM_USINAS']

        DF_Regional_Ger = fill_nan_columns(DF_Regional_Ger, ['PG_UHE', 'PG_UTE', 'PG_EOL', 'PG_SOL', 'PG_BIO', 'PG_SIN',
                                                            'QG_UHE', 'QG_UTE', 'QG_EOL', 'QG_SOL', 'QG_BIO', 'QG_SIN',
                                                            'Num_Usinas_UHE', 'Num_Usinas_UTE', 'Num_Usinas_EOL',
                                                            'Num_Usinas_SOL', 'Num_Usinas_BIO', 'Num_Usinas_SIN'])

        DF_Regional_PQ = self.df_Final_nt.groupby(by=['Dia', 'Hora', 'REG']).agg({
                        'BUS_ID': 'unique', 'MODV_PU': list, 'B0_MVAR': list, 'PG_MW': 'sum', 'QG_MVAR': 'sum', 'PL_MW': 'sum',
                        'QL_MVAR': 'sum', 'SHUNT_INST_IND': 'sum', 'SHUNT_INST_CAP': 'sum','key':'first'
                    })

        print(f'*** ETAPA: SEPARAÇÃO DE SHUNT ***')

        DF_Regional_Ger = separar_shunt(DF_Regional_Ger)
        DF_Regional_PQ = separar_shunt(DF_Regional_PQ)
        DF_Regional_PQ = fill_nan_columns(DF_Regional_PQ,
                                        ['PG_MW', 'QG_MVAR', 'PL_MW', 'QL_MVAR', 'Shunt_Ind', 'Shunt_Cap',
                                            'SHUNT_INST_IND', 'SHUNT_INST_CAP'])
        DF_Regional_Ger = fill_nan_columns(DF_Regional_Ger, ['Shunt_Ind', 'Shunt_Cap', 'SHUNT_INST_IND', 'SHUNT_INST_CAP'])
        DF_Regional_Ger['PG_Dist'] = DF_Regional_Ger['PG_MW'] - (
                    DF_Regional_Ger['PG_UHE'] + DF_Regional_Ger['PG_UTE'] + DF_Regional_Ger['PG_EOL'] +
                    DF_Regional_Ger['PG_SOL'] + DF_Regional_Ger['PG_BIO']) + DF_Regional_PQ['PG_MW']
        DF_Regional_Ger['QG_Dist'] = DF_Regional_Ger['QG_MVAR'] - (
                    DF_Regional_Ger['QG_UHE'] + DF_Regional_Ger['QG_UTE'] + DF_Regional_Ger['QG_EOL'] +
                    DF_Regional_Ger['QG_SOL'] + DF_Regional_Ger['QG_BIO']) + DF_Regional_PQ['QG_MVAR']

        # Sumo o shunt que existe en el dataframe de barras PV e barras PQ
        DF_Regional_Ger[['PL_MW', 'QL_MVAR', 'Shunt_Ind', 'Shunt_Cap', 'SHUNT_INST_IND', 'SHUNT_INST_CAP']] += \
            DF_Regional_PQ[['PL_MW', 'QL_MVAR', 'Shunt_Ind', 'Shunt_Cap', 'SHUNT_INST_IND', 'SHUNT_INST_CAP']]
        
        DF_Regional_Ger['QG/QL'] = DF_Regional_Ger['QG_MVAR']/DF_Regional_Ger['QL_MVAR']
        DF_Regional_Ger['PG/PL'] = DF_Regional_Ger['PG_MW']/DF_Regional_Ger['PL_MW']
        DF_Regional_Ger['PG_FERV'] =  (DF_Regional_Ger['PG_EOL'] + DF_Regional_Ger['PG_SOL'])/DF_Regional_Ger['PL_MW']
        DF_Regional_Ger['ReservaINDshunt'] = DF_Regional_Ger['SHUNT_INST_IND'] - DF_Regional_Ger['Shunt_Ind']
        DF_Regional_Ger['ReservaCAPshunt'] = DF_Regional_Ger['SHUNT_INST_CAP'] - DF_Regional_Ger['Shunt_Cap']

        self.DF_REGIONAL_GER = DF_Regional_Ger
        self.DF_REGIONAL_PQ = DF_Regional_PQ
        
        print(f'*** FINAL DO PROCESSAMENTO DE DADOS ***')


# if __name__ == "__main__":
#     import time
#     path = 'D:/MPV_(FNS Lim)_RC/'
#     options=    {
#                 'gen_script4lines' : False,
#                 'extract_fromcsv' : False,
#                 'ConvergenceAnalise' : True,
#                 'busdata' : True,
#                 }


