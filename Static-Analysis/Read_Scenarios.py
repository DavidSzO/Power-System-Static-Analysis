from multiprocessing import Pool, cpu_count
import dask.dataframe as dd
from NTW_Reader import *
import os

class Read_Scenarios():

    def __init__(self, path, pathcsv = None, RST=False, PO = False, genscript = False):
        
        self.path  = path
        self.PO = PO
        self.csv = pathcsv
        self.RST = RST
        self.noConverged = [] 

        #******************* Cria pasta para salvar os graficos para cada cenario *******************
        # Solicitar al usuario la ruta del directorio
        user_specified_dir = input("Please enter the directory path where you want to save the files: ")
        # Asegurarse de que la ruta especificada es absoluta
        notebook_dir = os.path.abspath(user_specified_dir)
        # Obtener el nombre del cenario (suponiendo que es el último directorio en la ruta)

        cenario = path.split('/')[-2]

        # Crear la ruta del directorio principal y los subdirectorios
        folder_path = os.path.join(notebook_dir, cenario)
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'Mapas'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'Intercambios'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'Indice'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'Potencia'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'Boxplot'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'CriticalBuses'), exist_ok=True)

        self.cenario = folder_path

        print(f"The directories have been created in: {folder_path}")


        if genscript ==  False:
            if PO == False:
                archivos=os.listdir(path)
                self.folders = [nomes_archivos for nomes_archivos in archivos if 'DS202' in nomes_archivos]  
                self.folders.sort()
                if pathcsv != None:
                    self.getDataFrames_csv()
                    self.get_ConvergenceData()
                else:
                    self.getDataExtract()
                    self.get_ConvergenceData()
            else:
                self.folders = 'PO'
                self.getDataExtract()
        else:
            self.generatescript()

# ======================================================================================================================
#                                                   MAIN EXTRACTION
# ======================================================================================================================

    def getDataFrames_csv(self): #Recupera los datos de archivos .csv previamente guardados


        df_Cvg = dd.read_csv(self.csv, sep=';').compute()
        df_Cvg['Dia'] = df_Cvg['Dia'].astype(str)
        df_Cvg['Dia'] = df_Cvg['Dia'].str.zfill(2)

        self.DfAnalysis = df_Cvg 
        results = df_Cvg.groupby(by= ['Dia', 'Hora'])['BUS_ID'].first()
        self.OpPointsNC = results.shape[0] - results.shape[0]
        self.OpPointsC = results.shape[0]
        print('O numero total de casos analisados é: ', self.OpPointsC)

    def getUniqueBus(self, df): #Recupera los datos de archivos .csv previamente guardados

        df_modified_BUS = df[0].groupby(['BUS_ID']).agg({
        'BUS_NAME': 'first',
        'VBASEKV': 'first',
        'TP': 'first',
        'ARE': 'first',
        'MODV_PU': 'first',
        'ANGV_DEG': 'first',
        }).reset_index()

        df_modified_GEN = df[1].groupby(['BUS_ID']).agg({
        'BASE_MVA': 'sum',
        'ST': 'sum',
        'PG_MW': 'sum',
        'QG_MVAR': 'sum',
        'PMAX_MW': 'sum',
        'PMIN_MW': 'sum',
        'QMX_MVAR': 'sum',
        'QMN_MVAR': 'sum'
        }).reset_index()
        df_modified_GEN.rename(columns={'ST': 'Ger_Active_Units'}, inplace=True)
        df_GEN = df[1].groupby(['BUS_ID']).agg({'PG_MW': 'count'}).reset_index().rename(columns={'PG_MW': 'Ger_Units'})
        df_modified_GEN = pd.merge(df_modified_GEN, df_GEN,on='BUS_ID', how='left')
        
        df_modified_LOAD = df[2].groupby(['BUS_ID']).agg({
        'PL_MW': 'sum',
        'QL_MVAR': 'sum',
        }).reset_index()

        return df_modified_BUS, df_modified_GEN, df_modified_LOAD

    def extract_scenario_data(self, file_path):

        Caso_name = file_path.strip().split('/')[-1].replace('.ntw', '').replace('.txt', '')
        # Caso_name = file_path.strip().split('/')[-1].replace('.NTW', '').replace('.txt', '') #descomentar para rodados_serialmente
        if self.PO == False:
            # Dia_name = file_path.strip().split('/')[-2][-2:] # caso de melhorprevisão
            Dia_name = file_path.strip().split('/')[-3][-2:]
            # Dia_name = file_path.strip().split('/')[-4][-2:] #descomentar para rodados_serialmente
        else: 
            Dia_name = self.folders

        NetData = NTW_Reader(file_path)

        BUS = NetData.bus_data[['BUS_ID', 'BUS_NAME', 'VBASEKV', 'TP', 'ARE', 'MODV_PU','ANGV_DEG']]
        GEN = NetData.gen_data[['BUS_ID','ST', 'PG_MW', 'QG_MVAR', 'BASE_MVA', 'PMAX_MW', 'PMIN_MW', 'QMX_MVAR', 'QMN_MVAR']]
        LOAD = NetData.load_data[['BUS_ID','PL_MW', 'QL_MVAR']]
        SHUNT = NetData.DF_shunt

        df = [BUS, GEN, LOAD]

        BUS_grouped, GEN_grouped, LOAD_grouped = self.getUniqueBus(df)

        df_merge_0 = pd.merge(BUS_grouped, GEN_grouped, on='BUS_ID', how='outer').sort_values('BUS_ID')
        df_merge_1 = pd.merge(df_merge_0, LOAD_grouped, on='BUS_ID', how='outer').sort_values('BUS_ID')
        merged_df = pd.merge(df_merge_1, SHUNT, on='BUS_ID', how='outer').sort_values('BUS_ID')
        merged_df['Dia']=str(Dia_name)
        merged_df['Hora']=str(Caso_name[-5:])

        return merged_df

    def getDataExtract(self):
        
        # CREACION DE DIRECTORIOS (PATHS) PARA ACCESO DE DATOS
        if self.PO == False:

            path_arquivo = [self.path + directory + '/' + 'Output' for directory in self.folders]
            directories_files = [os.listdir(path) for path in path_arquivo]

            files_path = [path_arquivo[i] + '/' + file_name
                    for i in range(len(path_arquivo))
                    for file_name in directories_files[i]
                    if '.ntw' in file_name]
            
        else:
            files_path = [self.path]

        # EXTRACCION DE DATOS
        with Pool() as pool:
            results = pool.map(self.extract_scenario_data, files_path)

        # results = []
        # for unique_path in files_path:
        #     results.append(self.extract_scenario_data(unique_path))

        self.results = list(filter(lambda elemento: elemento is not None, results))

        df_Cvg = pd.concat(results, axis=0, sort=False)
        self.DfAnalysis = df_Cvg.sort_values(by=['Dia', 'Hora'])
        self.OpPointsC = len(results)

        print('O numero total de casos analisados é: ', self.OpPointsC)

# ======================================================================================================================
#                                                AC & DC & RESERVE INFO EXTRACTION 
# ======================================================================================================================
        
    def get_Networkinfo(self, linhas = True, Reserva = False, Intercambios = False):

        files_and_directories = os.listdir(self.path)
        days = [nomes_archivos for nomes_archivos in files_and_directories if 'DS20' in nomes_archivos] 
        days.sort() 

        PWFs_sep = []
        dtype_dict_linhas = {'From#':'int32', ' From Name':'object', ' To# - Circ#':'object', ' To Name':'object', ' Type':'object', ' MVA':'float32', ' % L1':'float32', ' L1(MVA)':'float32', ' MW:From-To':'float32', ' Mvar:From-To':'float32',  ' Mvar:Losses':'float32', ' MW:To-From':'float32', ' Power Factor:From-To':'float32', ' Power Factor:To-From':'float32'}
        col_list = ['From#', ' From Name', ' To# - Circ#', ' To Name', ' Type', ' MVA', ' % L1', ' L1(MVA)', ' MW:From-To', ' Mvar:From-To',  ' Mvar:Losses', ' MW:To-From', ' Power Factor:From-To', ' Power Factor:To-From']
        # col_list = ['From#', ' From Name', ' To# - Circ#', ' To Name', ' Type', ' MVA', ' % L1', ' L1(MVA)',  ' Mvar:Losses']
        DCLinks_sep = []
        # col_list_hvdc = ['Bus #', ' Bus Name', ' Type', ' Pole #', ' P(MW)', ' Q(Mvar)', ' Satus']
        col_list_hvdc = ['Bus #', ' Bus Name', ' Type', ' Pole #', ' P(MW)', ' Q(Mvar)', ' Status']
        SGN01_sep = []
        col_list_reserve = ['Bus', ' Group', ' Bus Name', ' Area', ' Zone', ' V (pu)', ' Pg(MW)', ' Qg(Mvar)', ' Reserve', ' Units']

        print(f'*** ETAPA: LEITURA DE INFORMAÇÃO DAS LINHAS ***')
    
        for i in days:
            folder = os.path.join(self.path, i, 'Output')
            files = [file for file in os.listdir(folder) if file.endswith('.csv')]

            for file in files:
                caminho_arquivo = os.path.join(folder, file)
                if file.startswith('PWF16_') and linhas:
                    df = dd.read_csv(caminho_arquivo, sep=';', skiprows=[0], usecols=col_list, dtype=dtype_dict_linhas)
                    df['Dia'] = i[-2:]
                    df['Hora'] = file.split('_')[-1].split('.')[0]
                    PWFs_sep.append(df)
                elif file.startswith('PWF25_') and Intercambios:
                    df = dd.read_csv(caminho_arquivo, sep=';', skiprows=[0], usecols=col_list_hvdc)
                    df['Dia'] = i[-2:]
                    df['Hora'] = file.split('_')[-1].split('.')[0]
                    DCLinks_sep.append(df)
                elif file.startswith('SGN01_') and Reserva:
                    df = dd.read_csv(caminho_arquivo, sep=';', skiprows=[0], usecols=col_list_reserve)
                    df['Dia'] = i[-2:]
                    df['Hora'] = file.split('_')[-1].split('.')[0]
                    SGN01_sep.append(df)

        if linhas:
            PWF16_concatenados = dd.concat(PWFs_sep, ignore_index=True).compute()
            PWF16_concatenados.rename(columns={'From#':'From#', ' From Name': 'From Name', ' To# - Circ#':'To# - Circ#', ' To Name':'To Name', ' Type':'Type', ' MVA':'MVA', ' MW:From-To':'MW:From-To', ' Mvar:From-To':'Mvar:From-To',
                                                ' % L1':'% L1', ' L1(MVA)':'L1(MVA)',  ' Mvar:Losses':'Mvar:Losses', ' MW:To-From':'MW:To-From', ' Power Factor:From-To':'Power Factor:From-To', ' Power Factor:To-From':'Power Factor:To-From',
                                               }, inplace=True)
            
            # Splitting "To# - Circ#" column and creating new columns
            print("Concatenação das linhas")
            PWF16_concatenados[['To#', 'Circ#']] = PWF16_concatenados["To# - Circ#"].str.split(' # ', expand=True)
            PWF16_concatenados['From#'] = PWF16_concatenados['From#'].astype('int32')
            PWF16_concatenados['To#'] = PWF16_concatenados['To#'].astype('int32')
            PWF16_concatenados.drop(columns=["To# - Circ#"], inplace=True)
            self.linesInfo = PWF16_concatenados
            print("Salvando Dataframe das linhas")
            PWF16_concatenados.to_csv(self.path+'/LinhasInfo.csv', index=None)
            print("Final da leitura das Linhas")

            if Intercambios:
                self.get_Intercambios()

        if Intercambios:
            print("Concatenação da info do HVDC")
            DCLinks_concatenados = dd.concat(DCLinks_sep, ignore_index=True).compute()
            self.HVDCInfo = DCLinks_concatenados
            print("Salvando Dataframe do HVDC")
            DCLinks_concatenados.to_csv(self.path+'/HVDCInfo.csv', index=None)
            print("Final da leitura do HVDC")
        
        if Reserva:
            print("Concatenação da Reserva")
            SGN01_concatenados = dd.concat(SGN01_sep, ignore_index=True).compute()
            SGN01_concatenados['Bus']= SGN01_concatenados['Bus'].astype(int)
            SGN01_concatenados[' Pg(MW)']= SGN01_concatenados[' Pg(MW)'].astype(float)
            SGN01_concatenados[' Qg(Mvar)']= SGN01_concatenados[' Qg(Mvar)'].astype(float)
            SGN01_concatenados[' Reserve']= SGN01_concatenados[' Reserve'].astype(float)
            SGN01_concatenados[' Units']= SGN01_concatenados[' Units'].astype(int)
            self.ReserveInfo = SGN01_concatenados
            print("Salvando Dataframe da Reserva")
            SGN01_concatenados.to_csv(self.path+'/ReservaInfo.csv', index=None)
            print("Final da leitura da Reserva")


    def generatescript(self, path = None):

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

        print(f'*** ETAPA: OBTENÇÃO DOS INTERCAMBIOS ***')
        if df is None:
            PWF16_concatenados = self.linesInfo
        else:
            PWF16_concatenados = df
            
        PWF16_concatenados = PWF16_concatenados.set_index(['From#', 'To#'])
        # PWF16_concatenados = PWF16_concatenados[(PWF16_concatenados['Type'] == ' TL')]  #Importante ver si afecta!!

        linhas_expNE = pd.read_csv('RECURSOS/LINHAS/buses_EXPNE.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_expNE_flip = pd.read_csv('RECURSOS/LINHAS/buses_EXPNE_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNS = pd.read_csv('RECURSOS/LINHAS/buses_FNS.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNESE = pd.read_csv('RECURSOS/LINHAS/buses_FNESE.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNESE_flip = pd.read_csv('RECURSOS/LINHAS/buses_FNESE_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNEN = pd.read_csv('RECURSOS/LINHAS/buses_FNEN.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FNEN_flip = pd.read_csv('RECURSOS/LINHAS/buses_FNEN_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FSULSECO = pd.read_csv('RECURSOS/LINHAS/buses_FSULSECO.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_FSULSECO_flip = pd.read_csv('RECURSOS/LINHAS/buses_FSULSECO_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_RSUL = pd.read_csv('RECURSOS/LINHAS/buses_RSUL.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])
        linhas_RSUL_flip = pd.read_csv('RECURSOS/LINHAS/buses_RSUL_flip.csv',sep=';', skipinitialspace=True).set_index(['De', 'Para'])


        EXPNE_grouped = PWF16_concatenados[PWF16_concatenados.index.isin(linhas_expNE.index)]
        linhas_para_inverter = EXPNE_grouped[EXPNE_grouped.index.isin(linhas_expNE_flip.index)]
        EXPNE_grouped.loc[linhas_para_inverter.index, 'MW:From-To'] *= -1
        EXPNE_grouped = EXPNE_grouped.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_NS = PWF16_concatenados[PWF16_concatenados.index.isin(linhas_FNS.index)]
        Fluxo_NS_grouped = Fluxo_NS.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_NESE = PWF16_concatenados[PWF16_concatenados.index.isin(linhas_FNESE.index)]
        Fluxo_NESE_invertido = Fluxo_NESE[Fluxo_NESE.index.isin(linhas_FNESE_flip.index)]
        Fluxo_NESE.loc[Fluxo_NESE_invertido.index, 'MW:From-To'] *= -1
        Fluxo_NESE_grouped = Fluxo_NESE.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_NEN = PWF16_concatenados[PWF16_concatenados.index.isin(linhas_FNEN.index)]
        Fluxo_NEN_invertido = Fluxo_NEN[Fluxo_NEN.index.isin(linhas_FNEN_flip.index)]
        Fluxo_NEN.loc[Fluxo_NEN_invertido.index, 'MW:From-To'] *= -1
        Fluxo_NEN_grouped = Fluxo_NEN.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_SULSECO = PWF16_concatenados[PWF16_concatenados.index.isin(linhas_FSULSECO.index)]
        Fluxo_SULSECO_invertido = Fluxo_SULSECO[Fluxo_SULSECO.index.isin(linhas_FSULSECO_flip.index)]
        Fluxo_SULSECO.loc[Fluxo_SULSECO_invertido.index, 'MW:From-To'] *= -1
        Fluxo_SULSECO_grouped = Fluxo_SULSECO.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})

        Fluxo_RSUL = PWF16_concatenados[PWF16_concatenados.index.isin(linhas_RSUL.index)]
        Fluxo_RSUL_invertido = Fluxo_RSUL[Fluxo_RSUL.index.isin(linhas_RSUL_flip.index)]
        Fluxo_RSUL.loc[Fluxo_RSUL_invertido.index, 'MW:From-To'] *= -1
        Fluxo_RSUL_grouped = Fluxo_RSUL.groupby(['Dia', 'Hora']).agg({'MW:From-To':'sum', 'Mvar:From-To':'sum'})
        Fluxo_RSUL_grouped.loc[:, 'MW:From-To'] *= -1

        self.DF_Intercambios = pd.concat([EXPNE_grouped,Fluxo_NESE_grouped, Fluxo_NS_grouped, Fluxo_SULSECO_grouped, Fluxo_NEN_grouped, Fluxo_RSUL_grouped], axis=0, keys=['EXP_NE', 'Fluxo_NE-SE', 'Fluxo_N-S' ,'Fluxo_SUL-SECO', 'Fluxo_NE-N', 'Fluxo_RSUL'])
        print(f'*** ETAPA: Salvando dados dos Intercambios ***')
        self.DF_Intercambios.to_csv(self.cenario+'/DF_Intercambios.csv')
        print(f'*** ETAPA: FINAL DA OBTENÇÃO DE INTERCAMBIOS ***')

# ======================================================================================================================
#                                                RST MAIN INFO EXTRACTION
# ======================================================================================================================

    def get_rstDF(self):

        import numpy as np
        path = self.path

        def get_rstinfo(path, dia, hora):
        
            namecolumns = []
            datos = []
            data_dict = {}

            with open(path, 'r') as file:
                lines = file.readlines()[54:61]  # Lines 55 to 61

            for line in lines:
                data_array = line.strip().split()
                namecolumns.append(data_array[::2])
                values = data_array[1::2]
                data = [[i.replace(':',''), float(values[idx].replace(';',''))]  for idx, i in  enumerate(data_array[::2])]
                datos.append(data)

            for sublist in datos:
                for value in sublist:
                    key = value[0]
                    val = value[1]
                    if key in data_dict.keys():
                        data_dict[key].append(val)
                    else:
                        data_dict[key] = [val]
                        
            length = 3 # Limita la longitud máxima de las listas a 3
            for key in data_dict:
                data_dict[key] = data_dict[key][:length] # Recorta las listas a la longitud máxima
            max_length = max(len(lst) for lst in data_dict.values()) # Encuentra la longitud máxima de las listas
            for key in data_dict:
                data_dict[key] += [np.nan] * (max_length - len(data_dict[key])) # Rellena las listas con NaN hasta la longitud máxima

            dataframe = pd.DataFrame(data_dict)
            dataframe['Dia'] = dia
            dataframe['Hora'] = hora

            return dataframe


        df_list = []
        folders_main = os.listdir(self.path)
        semanas = [nomes_archivos for nomes_archivos in folders_main if 'RST' in nomes_archivos] 
        self.semanas = semanas
        try:
            for semana in semanas:
                path_semana = os.path.join(self.path , semana)
                folders_dias = os.listdir(path_semana)
                for dia in folders_dias:
                    path_dia = os.path.join(path_semana, dia)
                    files_rst = os.listdir(path_dia)
                    rstfiles = [nomes_archivos for nomes_archivos in files_rst if  nomes_archivos.endswith(".rst")] 
                    for rst in rstfiles:
                        path_rst = os.path.join(path_dia, rst)
                        hora = rst.replace('.rst','')[-5:]
                        df_list.append(get_rstinfo(path_rst,dia[-6:],hora))

            df_rst = pd.concat(df_list, ignore_index=True)
            self.df_rst = df_rst

        except:
            print('Problema no directorio RST, revisar a disposição dos arquivos')

# ======================================================================================================================
#                                                CONVERGENCE INFO EXTRACTION
# ======================================================================================================================

    def get_ConvergenceData(self):

        import re
        days = self.folders
        days.sort() 

        def getOPFdata(path):

            # Ler o arquivo e armazenar as linhas em uma lista
            with open(path, 'r') as file:
                lines = file.readlines()

            data = []
            pattern = re.compile(r'([TF]\s+){6}')  # Padrão para encontrar linhas com 6 valores True/False

            # Iterar pelas linhas do arquivo
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
            # folder = self.path + i + '/' # caso de melhorprevisão
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
        print(' => A informação da convergencia de cenários foi obtida')

# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F2_RESP_FNS_lim_rev1_2026/'
# cases = Read_Scenarios(path_folder, RST=False, pathcsv = None)


