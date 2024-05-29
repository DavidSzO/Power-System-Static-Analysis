import os
import sys
from computeDPI import *
from Maps import *
from Handle_Plots_Static import *
from Read_Process_Cases import *

class AnalyzeStaticCases:

    def __init__(self, path, Options) -> None:

        self.path_folder = path
        self.Options = Options
        

        user_specified_dir = input("Please enter the directory path where you want to save the files: ")
        # user_specified_dir = "D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1/Simulation Results/p_2/"
        # user_specified_dir = "D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1/Simulation Results/p_supremo"
        notebook_dir = os.path.abspath(user_specified_dir)
        folder_path = os.path.join(notebook_dir, os.path.basename(os.path.normpath(self.path_folder)))
        user_question = input("Do you want to read? (1). All cases or (2). Just One Case, Please input the corresponding number:\n")
        # user_question = '1'
        readjustONEcase = True if user_question.strip().replace("(","").replace(")","")  == '2' else False

        self.readjustONEcase = readjustONEcase

        if readjustONEcase:

            Options['extract_fromcsv'] = False
            Options['savecsv'] = False
            Options['linhascsv'] = False
            Options['reservacsv'] =False
            Options['HVDCcsv'] = False
            Options['ConvergenceAnalise'] = False
            self.day = input("Write the specific Day in this format 01:\n")
            self.hour = input("Write the specific Hour in this format 00-00:\n")
            # self.day = '01'
            # self.hour = '00-00'
            folder_path = os.path.join(folder_path, self.day + '_'+ self.hour)
            archivos = os.listdir(self.path_folder)
            folder = [nomes_archivos for nomes_archivos in archivos if nomes_archivos.endswith(self.day)][0]
            relativepath = os.path.join(self.path_folder, folder, 'Output')
            files_path = [os.path.join(relativepath, file_name) for file_name in os.listdir(relativepath) if file_name.endswith(self.hour+'.ntw')]
            self.path_folder = os.path.join(self.path_folder, files_path[0])


        if not readjustONEcase:
            subfolders_1 = ['Plots', 'Data']
            subfolders_11 = ['Indice', 'Perfil Potência Ativa', 'Perfil Potência Reativa', 'Reserva', 'BoxPlot Tensão', 'Intercambios AC-DC', 'Mapas']
            subfolders_12 = ['Geral', 'Fluxo em Ramos', 'Potencia', 'Indice']
        else:
            subfolders_1 = ['Plots', 'Data']
            subfolders_11 = ['Indice', 'Mapas', 'Potencia', 'BoxPlot Tensão']
            subfolders_12 = ['Geral', 'Fluxo em Ramos', 'Potencia', 'Indice']

        for subfolder in subfolders_1:
            folders = subfolders_11 if subfolder == 'Plots' else subfolders_12 
            for folder in folders:
                os.makedirs(os.path.join(folder_path, subfolder, folder), exist_ok=True)
            
        print(f"The directories have been created in: {folder_path}")
        self.folder_path = folder_path

    def extraction_process(self):

        if self.Options['extract_fromcsv']:
            pathcsv1 = os.path.join(self.path_folder, 'ProcessedDataBase.csv')
            pathcsv2 = None
        else:
            pathcsv1 = None
            pathcsv2 = os.path.join(self.path_folder, 'ProcessedDataBase.csv')

        if self.Options['generatescript']:
            ReadScenarios(self.path_folder, self.folder_path, PO = self.readjustONEcase,  pathcsv=pathcsv1,  genscript=self.Options['generatescript'])
            sys.exit()
        else:
            print('******************** EXTRAÇÃO DE DADOS ********************')
            self.cases = ReadScenarios(self.path_folder, self.folder_path, PO = self.readjustONEcase, pathcsv=pathcsv1, genscript=self.Options['generatescript'])
            self.cenario = self.folder_path
            if self.Options['ConvergenceAnalise']:
                self.cases.get_convergence_data()

            print('******************** PROCESSAMENTO DE DADOS ********************')
            self.processdata = ProcessData(df= self.cases.Df_Cases, cenario = self.cenario, pathcsv = pathcsv2, extract_fromcsv = self.Options['extract_fromcsv'], savecsv=self.Options['savecsv'], busdata = self.Options['busdata'])

        if not self.readjustONEcase:
            if self.Options['ConvergenceAnalise']:
                self.bool_PWF_NConv = self.cases.PWF_NC[['Dia', 'Hora']].apply(tuple, axis=1)
            self.hour = None

        self.df_Final_ger = self.processdata.df_Final_ger
        self.df_Final_nt = self.processdata.df_Final_nt
        self.dff_Ger_map = self.processdata.dff_Ger_map
        self.dff_Ger_map.loc[self.dff_Ger_map['Gen_Type']=='UNE','Gen_Type'] = 'UTE' # cambia de designación de usinas nucleares a termicas para ser plotadas en el mapa
        self.dff_NT_map = self.processdata.dff_NT_map
        self.DF_REGIONAL_GER = self.processdata.DF_REGIONAL_GER
        self.DF_REGIONAL_PQ = self.processdata.DF_REGIONAL_PQ

        self.plots_static = Plots_Static(self.cenario, svg=False, PO=self.readjustONEcase)
        
        ## ***************** (O código seguinte obtem as informações das linhas AC e DC e reserva por maquina) *****************
        if self.Options['LinhaAnalise']:

            if self.Options['linhascsv'] and self.Options['LinhaAnalise']:
                self.PWF16_concatenados = dd.read_csv(self.path_folder + '/LinhasInfo.csv', sep=',').compute()
                self.PWF16_concatenados['Dia'] = self.PWF16_concatenados['Dia'].astype(str).str.zfill(2)
                self.cases.get_Intercambios(df=self.PWF16_concatenados)
                self.DF_Intercambios = self.cases.DF_Intercambios

            if self.Options['HVDCcsv'] and self.Options['HVDCAnalise']:
                self.DCLinks_concatenados = dd.read_csv(self.path_folder + '/HVDCInfo.csv', sep=',').compute()
                self.DCLinks_concatenados['Dia'] = self.DCLinks_concatenados['Dia'].astype(str).str.zfill(2)

            if self.Options['reservacsv'] and self.Options['ReservaAnalise']:
                self.SGN01_concatenados = dd.read_csv(self.path_folder + '/ReservaInfo.csv', sep=',').compute()
                self.SGN01_concatenados['Dia'] = self.SGN01_concatenados['Dia'].astype(str).str.zfill(2)

            if not (self.Options['linhascsv'] and self.Options['reservacsv'] and self.Options['HVDCcsv']):
                self.cases.get_Networkinfo(linhas=not self.Options['linhascsv'], Reserva=not self.Options['reservacsv'], Intercambios=not self.Options['HVDCcsv'], hour = self.hour)

                if not self.Options['linhascsv'] and self.Options['LinhaAnalise']:
                    self.PWF16_concatenados = self.cases.linesInfo
                    self.DF_Intercambios = self.cases.DF_Intercambios

                if not self.Options['reservacsv'] and self.Options['ReservaAnalise']:
                    try:
                        self.SGN01_concatenados = self.cases.ReserveInfo
                    except Exception as e:
                        print(f"Error obtaining Reserve: {e}")
                        pass

                if not self.Options['HVDCcsv'] and self.Options['HVDCAnalise']:
                    self.DCLinks_concatenados = self.cases.HVDCInfo

# =============================================================================================================================
#                                                LEITURA LINHAS E RESERVA
# =============================================================================================================================

    def LinhaAnalise(self):

        if self.Options['LinhaAnalise']:

            def addUF_linha(from_bus, to_bus, bus_info_map, vbasekv_map):

                reg1 = bus_info_map.get(from_bus, np.nan)
                Vbase1 = vbasekv_map.get(from_bus, np.nan)
                reg2 = bus_info_map.get(to_bus, np.nan)

                if reg1 == reg2 and reg1 is not np.nan:
                    return reg1, Vbase1
                else:
                    return np.nan, np.nan

            def Main_linha_addREG(PWF16_concatenados):

                Df_VF_SF = self.processdata.Df_VF_SF
                InfoBarras = Df_VF_SF[(Df_VF_SF['Dia']==Df_VF_SF['Dia'].iloc[0]) & (Df_VF_SF['Hora']==Df_VF_SF['Hora'].iloc[0])][['BUS_ID','BUS_NAME', 'ARE','VBASEKV','REG', 'U_FED', 'Gen_Type','Latitude', 'Longitude']]
                PWF16_concatenados_d1 = PWF16_concatenados[(PWF16_concatenados['Dia'] == PWF16_concatenados['Dia'].iloc[0]) & (PWF16_concatenados['Hora'] == PWF16_concatenados['Hora'].iloc[0])].groupby(by=['From#','To#']).first().reset_index().copy()

                # Create a dictionary to map 'BUS_ID' to 'REG' and 'VBASEKV'
                bus_info_map = dict(zip(InfoBarras['BUS_ID'], InfoBarras['REG']))
                vbasekv_map = dict(zip(InfoBarras['BUS_ID'], InfoBarras['VBASEKV']))

                # Use the apply function to create 'REG' and 'VBASEKV' columns in PWF16_concatenados_d1
                PWF16_concatenados_d1['REG'], PWF16_concatenados_d1['VBASEKV'] = zip(*PWF16_concatenados_d1.apply(lambda row: addUF_linha(row['From#'], row['To#'], bus_info_map, vbasekv_map), axis=1))

                PWF16_concatenados_R = PWF16_concatenados.merge(PWF16_concatenados_d1[['From#','To#','REG','VBASEKV']], on=['From#','To#'], how='left')
                PWF16_Filt_linhas = PWF16_concatenados_R[(PWF16_concatenados_R['Type'] == ' TL') & ~(PWF16_concatenados_R['REG'].isna())]
                PWF16_Filt_TRAFO = PWF16_concatenados_R[(PWF16_concatenados_R['Type'] == ' TRAFO') & ~(PWF16_concatenados_R['REG'].isna())]

                return PWF16_Filt_linhas, PWF16_Filt_TRAFO

            print('Saving CSV of lines and trafos: ...')
            self.PWF16_Filt_linhas, self.PWF16_Filt_TRAFO = Main_linha_addREG(self.PWF16_concatenados)
            
            if self.Options['IntercambiosAnalise']:

                ## ========================================== ELOS SEPARADOS POR BIPOLOS: ==========================================

                # pole_mapping = {1: 'Bipolo1', 2: 'Bipolo1', 3: 'Bipolo2', 4: 'Bipolo2'}
                # dfelo1 = DCLinks_concatenados[DCLinks_concatenados['Bus #'] == 85].groupby(by=['Dia', 'Hora', ' Pole #']).agg({' P(MW)': sum, ' Q(Mvar)': sum})
                # dfelo1['Nome Elo'] = 'Elo_FOZ-IBIUNA'
                # dfelo1['Bipole'] = dfelo1.index.get_level_values(' Pole #').map(pole_mapping)
                # dfelo2 = DCLinks_concatenados[DCLinks_concatenados['Bus #'] == 7055].groupby(by=['Dia', 'Hora', ' Pole #']).agg({' P(MW)': sum, ' Q(Mvar)': sum})
                # dfelo2['Nome Elo'] = 'Elo_PVEL-ARARQ'
                # dfelo2['Bipole'] = dfelo2.index.get_level_values(' Pole #').map(pole_mapping)
                # dfelo3 = DCLinks_concatenados[DCLinks_concatenados['Bus #'] == 7059].groupby(by=['Dia', 'Hora', ' Pole #']).agg({' P(MW)': sum, ' Q(Mvar)': sum})
                # dfelo3['Nome Elo'] = 'Elo_CPVBTB-PVEL'
                # dfelo3['Bipole'] = dfelo3.index.get_level_values(' Pole #').map(pole_mapping)
                # dfelo4 = DCLinks_concatenados[(DCLinks_concatenados['Bus #'] == 8100)].groupby(by=['Dia', 'Hora', ' Pole #']).agg({' P(MW)': sum, ' Q(Mvar)': sum})
                # dfelo4['Nome Elo'] = 'Elo_XINGU-SE'
                # dfelo4['Bipole'] = dfelo4.index.get_level_values(' Pole #').map(pole_mapping)
                # dfelo1.reset_index().groupby(['Dia', 'Hora', 'Bipole']).agg({' P(MW)': sum, 'Nome Elo': 'first'}).to_csv('HVDC_FOZ_IBIUNA.csv')
                # dfelo2.reset_index().groupby(['Dia', 'Hora', 'Bipole']).agg({' P(MW)': sum, 'Nome Elo': 'first'}).to_csv('HVDC_PVEL-ARARQ.csv')

                ## ========================================== ELOS SEM SEPARAÇÃO POR POLOS: ==========================================
                print('Filtering lines of Interconnections and Plotting:...')
                dfelo1 = self.DCLinks_concatenados[self.DCLinks_concatenados['Bus #'] == 85].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
                dfelo1['Nome Elo'] = 'Elo_FOZ-IBIUNA'
                dfelo2 = self.DCLinks_concatenados[self.DCLinks_concatenados['Bus #'] == 7055].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
                dfelo2['Nome Elo'] = 'Elo_PVEL-ARARQ'
                dfelo3 = self.DCLinks_concatenados[self.DCLinks_concatenados['Bus #'] == 7059].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
                dfelo3['Nome Elo'] = 'Elo_CPVBTB-PVEL'
                dfelo4 = self.DCLinks_concatenados[(self.DCLinks_concatenados['Bus #'] == 8100)].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
                dfelo4['Nome Elo'] = 'Elo_XINGU-SE'

                # dfelo4 = self.DCLinks_concatenados[(self.DCLinks_concatenados['Bus #'] == 8100) & (self.DCLinks_concatenados[' Pole #'].isin([1,2]))].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
                # dfelo4['Nome Elo'] = 'Elo_XINGU-ESTREI'
                # dfelo5 = DCLinks_concatenados[(DCLinks_concatenados['Bus #'] == 8100) & (DCLinks_concatenados[' Pole #'].isin([3,4]))].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
                # dfelo5['Nome Elo'] = 'Elo_XINGU-T.RIO'
                # Merge all dataframes
                # df_HVDC = pd.concat([dfelo1, dfelo2, dfelo3, dfelo4, dfelo5], axis=0, keys=['Elo_FOZ-IBIUNA', 'Elo_PVEL-ARARQ', 'Elo_CPVBTB-PVEL' ,'Elo_XINGU-ESTREI', 'Elo_XINGU-T.RIO'])
                
                df_HVDC = pd.concat([dfelo1, dfelo2, dfelo3, dfelo4], axis=0, keys=['Elo_FOZ-IBIUNA', 'Elo_PVEL-ARARQ', 'Elo_CPVBTB-PVEL' ,'Elo_XINGU-SE'])
                self.df_HVDC = df_HVDC
                
                if self.Options['PlotIntercambios'] == True and not self.readjustONEcase:
                    self.plots_static.plot_Intercambio (self.DF_Intercambios, df_HVDC , '(MW)', 'Exportação (N-S, NE-SE) e  Elo Xingu-SE', ['Fluxo_N-S', 'Fluxo_NE-SE'], ['Elo_XINGU-SE'], )
                    self.plots_static.plot_Intercambio (self.DF_Intercambios, df_HVDC , '(MW)', 'Exportação N-S e  Elo Xingu-SE', ['Fluxo_N-S'], ['Elo_XINGU-SE'], )
                    self.plots_static.plot_Intercambio (self.DF_Intercambios, df_HVDC , '(MW)', 'Comparativo Exportação NE-N e Elo Xingu-SE', ['Fluxo_NE-N'], ['Elo_XINGU-SE'], Xlimites=None)
                    self.plots_static.plot_Intercambio (self.DF_Intercambios, df_HVDC , '(MW)', 'Comparativo Exportação NE-SE e Elo FOZ-IBIUNA', ['Fluxo_NE-SE'], ['Elo_FOZ-IBIUNA'], Xlimites=None)
                    self.plots_static.plot_Intercambio (self.DF_Intercambios, df_HVDC , '(MW)', 'Comparativo Exportação SUL-SECO e Elo FOZ-IBIUNA', ['Fluxo_SUL-SECO'], ['Elo_FOZ-IBIUNA'], Xlimites=None)

#=============================================================================================================================
#                                                    RESERVA REGIONAL
#=============================================================================================================================
    def ReservaAnalise(self):
        if self.Options['ReservaAnalise'] == True:

            print('Reserve Analysis and Plots: ...')
            if (self.SGN01_concatenados.empty == False):
                dia = self.df_Final_ger['Dia'].iloc[0]
                hora = self.df_Final_ger['Hora'].iloc[0]
                df_Final_ger_mod = self.df_Final_ger[(self.df_Final_ger['Dia'] == dia) & (self.df_Final_ger['Hora'] == hora)][['BUS_ID', 'Gen_Type', 'U_FED', 'REG']]
                self.SGN01_concatenados.rename(columns={'Bus':'BUS_ID', }, inplace=True)
                self.SGN01_concatenados['BUS_ID'] = self.SGN01_concatenados['BUS_ID'].astype(float)
                Df_Reserva = self.SGN01_concatenados.merge(df_Final_ger_mod, how = 'left', on='BUS_ID')

                REG_groupReserve = Df_Reserva.groupby(by = ['Dia','Hora', 'REG']).agg({' Reserve': 'sum'})
                GroupReserve = Df_Reserva.groupby(by = ['Dia','Hora']).agg({' Reserve': 'sum'})
                self.dffreservaPO_REG_MW = REG_groupReserve
                self.dffreservaPO_MW = GroupReserve

                if not self.readjustONEcase:
                    self.plots_static.plot_Potencia(GroupReserve[' Reserve'], '(MW)', 'RESERVA (MW) - SIN', limites=None)
                    self.plots_static.plot_reserva_reg (REG_groupReserve, '(MW)', 'Reserva por Região', 'RESERVA POR REGIÃO', ' Reserve', xlimites=None,ylimites=None, order = False)

            # ======================ESSE DATAFRAME É SÓ DA RESERVA DAS MAQUINAS COM MODELO DO GERADOR PARA O CONTROLE DE FREQ
            # ===========================================================================================================================
            dff_reserva = Df_Reserva.merge(self.df_Final_ger[['BUS_ID','Dia', 'Hora', 'QMX_MVAR', 'QMN_MVAR', 'Ger_Units','QG_MVAR']], on=['BUS_ID','Dia', 'Hora'], how='left')
            dff_reserva['Qmin'] = (dff_reserva['QMN_MVAR'] / dff_reserva['Ger_Units']) * dff_reserva[' Units']
            dff_reserva['Qmax'] = (dff_reserva['QMX_MVAR'] / dff_reserva['Ger_Units']) * dff_reserva[' Units']
            dff_reserva['ReservaIND'] = np.where(dff_reserva['QG_MVAR'] < 0, dff_reserva['Qmin'] - dff_reserva['QG_MVAR'], dff_reserva['Qmin'])
            dff_reserva['ReservaCAP'] = np.where(dff_reserva['QG_MVAR'] > 0, dff_reserva['Qmax'] - dff_reserva['QG_MVAR'], dff_reserva['Qmax'])
            # ============================================================================================================================
            dffreservaPO = self.df_Final_ger.groupby(['Dia', 'Hora']).agg({'QG_MVAR': 'sum', 'ReservaIND':'sum', 'ReservaCAP':'sum'})
            dffreservaPO_REG = self.df_Final_ger.groupby(['Dia', 'Hora', 'REG']).agg({'QG_MVAR': 'sum', 'ReservaIND':'sum', 'ReservaCAP':'sum'})
            self.dffreservaPO_MVAR = dffreservaPO
            self.dffreservaPO_REG_MVAR = dffreservaPO_REG
            #=============================================================================================================================
            #                                                                   PLOTS RESERVA MVAR
            #=============================================================================================================================
            if not self.readjustONEcase:
                self.plots_static.plot_reserva_reg (dffreservaPO_REG, '(MVAR)', 'Reserva Capacitiva por Região MVAR', 'RESERVA CAPACITIVA POR REGIÃO MVAR', 'ReservaCAP', xlimites=None,ylimites=None, order = False)
                self.plots_static.plot_reserva_reg (dffreservaPO_REG, '(MVAR)', 'Reserva Indutiva por Região MVAR', 'RESERVA INDUTIVA POR REGIÃO MVAR', 'ReservaIND', xlimites=None,ylimites=None, order = False)

                fig, ax = plt.subplots(figsize=(20,10))
                dffreservaPO['ReservaCAP'].plot(figsize=(20,10), grid=True, title='RESERVA CAPACITIVA (Mvar)',legend='RESERVA')
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)
                ax.set_xlabel('PO',fontsize = 15)
                ax.set_ylabel('(MVAR)',fontsize = 15)
                ax.set_title('RESERVA CAPACITIVA (Mvar)', fontsize = 20)
                ax.legend(fontsize = 15)
                nome = self.cenario + '/Plots/Reserva/Reserva_cap_mvar.png'
                plt.savefig(nome, bbox_inches = 'tight')

                fig, ax = plt.subplots(figsize=(20,10))
                dffreservaPO['ReservaIND'].plot(figsize=(20,10), grid=True, title='RESERVA INDUTIVA (Mvar)',legend='RESERVA')
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)
                ax.set_xlabel('PO',fontsize = 15)
                ax.set_ylabel('(MVAR)',fontsize = 15)
                ax.set_title('RESERVA INDUTIVA (Mvar)', fontsize = 20)
                ax.legend(fontsize = 15)
                nome = self.cenario + '/Plots/Reserva/Reserva_ind_mvar.png'
                plt.savefig(nome, bbox_inches = 'tight')
            else: 
                dia= 'PO'
                hora = self.DF_REGIONAL_GER.index.to_frame()['Hora'].unique()[0]
                self.plots_static.analise_regiao_plot(self.DF_REGIONAL_GER.loc[dia,hora],'PowerPlot')

#=============================================================================================================================
#                                                POTENCIA ATIVA E REATIVA
#=============================================================================================================================

    def ActiveReactivePower(self):

        print('Active and Reactive Power Analysis and Plots:...')
        regioes = self.DF_REGIONAL_GER.reset_index()['REG'].unique()
        df_pg = self.DF_REGIONAL_GER.reset_index(level=['Dia','Hora', 'REG'])[['PG_MW','PL_MW','PG_EOL','PG_SOL', 'Dia', 'Hora', 'MODV_PU', 'QG_MVAR']]
        df_pg['PG_FERV'] =  (df_pg['PG_EOL'] + df_pg['PG_SOL'])/df_pg['PL_MW']
        df_grouped = df_pg.groupby(by = ['Dia', 'Hora'])[['PG_FERV', 'MODV_PU', 'QG_MVAR','PG_MW', 'PL_MW' ]].sum(numeric_only=True)
        self.df_grouped = df_grouped

        if self.Options['PlotGeralPotencia'] and not self.readjustONEcase:
            self.plots_static.plot_Potencia(df_grouped['QG_MVAR'], '(MVAR)', 'MW POTÊNCIA REATIVA GERADA - SIN', limites=None)
            self.plots_static.plot_Potencia(df_grouped['PG_MW'], '(MW)', 'MVAR POTÊNCIA ATIVA GERADA - SIN', limites=None)
            self.plots_static.plot_Potencia(df_grouped['PL_MW'], '(MW)', 'MW POTÊNCIA ATIVA DEMANDA BRUTA - SIN', limites=None)

            typeGenDic = {'QG_UHE':'Num_Usinas_UHE', 'QG_UTE':'Num_Usinas_UTE', 'QG_EOL':'Num_Usinas_EOL','QG_SOL':'Num_Usinas_SOL', 'QG_BIO':'Num_Usinas_BIO'}
            typeGenRegDic = {'Norte':['QG_UHE','QG_EOL','QG_SOL','QG_UTE'],'Nordeste':['QG_UHE','QG_EOL','QG_SOL','QG_UTE'],'Sudeste-Centro-Oeste':['QG_UHE','QG_EOL','QG_SOL','QG_UTE','QG_BIO'],'Sul':['QG_UHE','QG_EOL','QG_UTE','QG_BIO'], 'AC-RO':['QG_UHE','QG_UTE']}
            typeGenDic_MW = {'PG_UHE':'Num_Usinas_UHE', 'PG_UTE':'Num_Usinas_UTE', 'PG_EOL':'Num_Usinas_EOL','PG_SOL':'Num_Usinas_SOL', 'PG_BIO':'Num_Usinas_BIO'}
            typeGenRegDic_MW = {'Norte':['PG_UHE','PG_EOL','PG_SOL','PG_UTE'],'Nordeste':['PG_UHE','PG_EOL','PG_SOL','PG_UTE'],'Sudeste-Centro-Oeste':['PG_UHE','PG_EOL','PG_SOL','PG_UTE','PG_BIO'],'Sul':['PG_UHE','PG_EOL','PG_UTE','PG_BIO'], 'AC-RO':['PG_UHE','PG_UTE']}

            for reg in regioes:
                self.plots_static.plot_Potencia(self.DF_REGIONAL_GER.loc[:,:,reg]['QG_MVAR'], '(MVAR)', 'MVAR POTÊNCIA REATIVA GERADA - ' + reg, limites=None)
                self.plots_static.plot_Potencia(self.DF_REGIONAL_GER.loc[:,:,reg]['PG_MW'], '(MW)', 'MW POTÊNCIA ATIVA GERADA - ' + reg, limites=None)
                self.plots_static.plot_Potencia(self.DF_REGIONAL_GER.loc[:,:,reg]['PL_MW'], '(MW)', 'MW POTÊNCIA ATIVA DEMANDA BRUTA - ' + reg, limites=None)
                for tog in typeGenRegDic[reg]:
                    numUsinas = self.DF_REGIONAL_GER.loc[:,:,reg][typeGenDic[tog]].iloc[0]
                    nome = str('MVAR ' + reg.replace('-',' ')  + ' (' + tog.replace('_','-') + ') - Número de Usinas ' + str(int(numUsinas)))
                    self.plots_static.plot_Potencia(self.DF_REGIONAL_GER.loc[:,:,reg][tog], '(MVAR)', nome , limites=None)

                for tog in typeGenRegDic_MW[reg]:
                    numUsinas = self.DF_REGIONAL_GER.loc[:,:,reg][typeGenDic_MW[tog]].iloc[0]
                    nome = str('MW ' + reg.replace('-',' ')  + ' (' + tog.replace('_','-') + ') - Número de Usinas ' + str(int(numUsinas)))
                    self.plots_static.plot_Potencia(self.DF_REGIONAL_GER.loc[:,:,reg][tog], '(MW)', nome , limites=None)

#=============================================================================================================================
#                                                        TENSÃO
#=============================================================================================================================

    def Plot_Tensao_Geral(self):

        if self.Options['Plot_Tensao_Geral']:
            print('Voltage General BoxPlots: ...')
            def boxplot_barrasGeracao(Df_VF):
                grouped_UF = Df_VF.groupby('Gen_Type').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                data_UF = [grouped_UF.at[gen_type, 'MODV_PU'] for gen_type in ['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO', 'SIN']]
                Nbarras_UF = [grouped_UF.at[gen_type, 'BUS_ID'] for gen_type in ['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO', 'SIN']]
                labels_UF = ['Hydro', 'Thermal', 'SHP', 'Wind', 'Solar', 'Bio']
                
                self.plots_static.plot_boxplot(data_UF, labels_UF, 'Bus Voltage Distribution by Type of Generation', 
                                            'Type of Generation', 'Voltage (pu)', text=True, nbarra=Nbarras_UF)
                
                grouped_Reg = Df_VF.groupby('REG').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                data_Reg = [grouped_Reg.at[region, 'MODV_PU'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                Nbarras_Reg = [grouped_Reg.at[region, 'BUS_ID'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                labels_Reg = ['North', 'Northeast', 'SE-CW', 'South', 'AC-RO']
                
                self.plots_static.plot_boxplot(data_Reg, labels_Reg, 'Bus Voltage Distribution by Region for Voltage-Controlled Buses', 
                                            'Region', 'Voltage (pu)', text=True, nbarra=Nbarras_Reg)

            def boxplot_barrasCarga(Df_Vfpt):
                grouped_VBASEKV = Df_Vfpt.groupby('VBASEKV').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                voltage_levels = [230, 345, 440, 500, 525, 765]
                data_VBASEKV = [grouped_VBASEKV.at[level, 'MODV_PU'] for level in voltage_levels]
                Nbarras_VBASEKV = [grouped_VBASEKV.at[level, 'BUS_ID'] for level in voltage_levels]
                labels_VBASEKV = ['230', '345', '440', '500', '525', '765']
                
                self.plots_static.plot_boxplot(data_VBASEKV, labels_VBASEKV, 'Bus Voltage Distribution by Voltage Level', 
                                            'Voltage Level (kV)', 'Voltage (pu)', text=True, nbarra=Nbarras_VBASEKV)

                grouped_Reg = Df_Vfpt.groupby('REG').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                data_Reg = [grouped_Reg.at[region, 'MODV_PU'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                Nbarras_Reg = [grouped_Reg.at[region, 'BUS_ID'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                labels_Reg = ['North', 'Northeast', 'SE-CW', 'South', 'AC-RO']
                
                self.plots_static.plot_boxplot(data_Reg, labels_Reg, 'Bus Voltage Distribution by Region for Load Buses', 
                                            'Region', 'Voltage (pu)', text=True, nbarra=Nbarras_Reg)

            def plottensaoG():
                Df_VF = self.processdata.Df_VF
                filter_condition = (Df_VF['VBASEKV'].isin([230, 345, 440, 500, 525, 765]) | 
                                    Df_VF['Gen_Type'].isin(['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO']))
                DFF_Geral = Df_VF[filter_condition]
                
                if self.Options['ConvergenceAnalise']:
                    filter_non_converged = ~DFF_Geral[['Dia', 'Hora']].apply(tuple, axis=1).isin(self.bool_PWF_NConv)
                    DFF_Geral_PWFC = DFF_Geral[filter_non_converged]
                else:
                    DFF_Geral_PWFC = DFF_Geral

                data = [DFF_Geral_PWFC['MODV_PU'].values]
                labels = ['G. Sincrona']
                
                self.plots_static.plot_boxplot(data, labels, 'Bus Voltage Distribution of the System', 
                                            'Voltage (pu)', 'Bus Voltages', vert=False, text=True, rotation=0)

            def plottensaoPR():
                df_ger = self.df_Final_ger[self.df_Final_ger['Gen_Type'].isin(['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO','SIN'])]
                df_nt = self.df_Final_nt[self.df_Final_nt['VBASEKV'].isin([230, 345, 440, 500, 525, 765])]

                if self.Options['ConvergenceAnalise']:
                    filter_non_converged_ger = ~df_ger[['Dia', 'Hora']].apply(tuple, axis=1).isin(self.bool_PWF_NConv)
                    df_Final_ger_PWFC = df_ger[filter_non_converged_ger]
                    filter_non_converged_nt = ~df_nt[['Dia', 'Hora']].apply(tuple, axis=1).isin(self.bool_PWF_NConv)
                    df_Final_nt_PWFC = df_nt[filter_non_converged_nt]
                else:
                    df_Final_ger_PWFC = df_ger
                    df_Final_nt_PWFC = df_nt
                
                boxplot_barrasCarga(df_Final_nt_PWFC)
                boxplot_barrasGeracao(df_Final_ger_PWFC)

                return df_Final_ger_PWFC, df_Final_nt_PWFC

            plottensaoG()
            df_Final_ger_PWFC, df_Final_nt_PWFC = plottensaoPR()
            self.df_Final_ger_PWFC = df_Final_ger_PWFC
            self.df_Final_nt_PWFC = df_Final_nt_PWFC

#=============================================================================================================================
#                                                         MAPAS
#=============================================================================================================================
    def MapasPlots(self):
        if self.Options['MapasPlots']:
            Df_VF = self.processdata.Df_VF
            options = {'Limit Violations All': True, 'Mean and Variance': True, 'Limit Violations by Group': True, 'Limit Violations PO': self.readjustONEcase}
            Maps(Df_VF, self.dff_NT_map, self.dff_Ger_map, self.cenario, options)

#=============================================================================================================================
#                                             DPI (DECOMPOSED PERFORMANCE INDEX)
#=============================================================================================================================
    def ComputeDPI(self):

        if self.Options['ComputeDPI']:
            print('Computing the DPI for all cases: ...')
            ts, tb, n = 0.8, 1, 2
            VVI = computeDPI(self.df_Final_nt, self.df_Final_ger, ts, tb, p_norm=n, p_inf=False, NBcv=True)

            dfPQ_CSI, dfPV_CSI = VVI.dfPQ_CSI, VVI.dfPV_CSI
            df_PQ_reg, df_PV_reg = VVI.df_PQ_reg, VVI.df_PV_reg
            df_busPQ, df_busPV = VVI.df_busPQ, VVI.df_busPV

            def group_dataframes():
                return (
                    dfPQ_CSI.groupby(['Dia', 'Hora', 'REG']).first(),
                    dfPV_CSI.groupby(['Dia', 'Hora', 'REG']).first(),
                    df_PQ_reg.groupby(['Dia', 'Hora', 'REG', 'VBASEKV']).agg({'CSI_INF':'first', 'CSI_SUP':'first'}),
                    df_PV_reg.groupby(['Dia', 'Hora', 'REG', 'Gen_Type']).agg({'CSI_INF':'first', 'CSI_SUP':'first'})
                )

            dfPQ_CSI, dfPV_CSI, dffPQgb, dffPVgb = group_dataframes()


            if self.Options['ConvergenceAnalise']:
                for index in self.bool_PWF_NConv:
                    dfPQ_CSI.drop((index[0], index[1]), inplace=True)
                    dfPV_CSI.drop((index[0], index[1]), inplace=True)
                    dffPQgb.drop((index[0], index[1]), inplace=True)
                    dffPVgb.drop((index[0], index[1]), inplace=True)
                filtro1 = (df_busPQ[['Dia', 'Hora']].apply(tuple, axis=1).isin(self.bool_PWF_NConv))
                df_busPQ_mod = df_busPQ[~filtro1].copy()
                filtro2 = (df_busPV[['Dia', 'Hora']].apply(tuple, axis=1).isin(self.bool_PWF_NConv))
                df_busPV_mod = df_busPV[~filtro2].copy()
            else:
                df_busPQ_mod = df_busPQ
                df_busPV_mod = df_busPV

            self.dffPQgb = dffPQgb
            self.dffPVgb = dffPVgb

            def calculate_DPI(df, n):
                df['DPI_2N_INF'] = df['CSI_INF_FINAL'].pow(2 * n)
                df['DPI_2N_SUP'] = df['CSI_SUP_FINAL'].pow(2 * n)
                return df

            dfPQ_CSI = calculate_DPI(dfPQ_CSI, n)
            dfPV_CSI = calculate_DPI(dfPV_CSI, n)

            def aggregate_DPI(df, key_cols, agg_cols, sufix):
                ddf = df.reset_index().groupby(by=key_cols).agg({col: 'sum' for col in agg_cols})
                ddf[f"{sufix}_lower"] = ddf[agg_cols[0]].pow(1 / (2 * n))
                ddf[f"{sufix}_upper"] = ddf[agg_cols[1]].pow(1 / (2 * n))
                return ddf

            ddf_pq = aggregate_DPI(dfPQ_CSI, ['Dia', 'Hora'], ['DPI_2N_INF', 'DPI_2N_SUP'], "PQ")
            ddf_pv = aggregate_DPI(dfPV_CSI, ['Dia', 'Hora'], ['DPI_2N_INF', 'DPI_2N_SUP'], "PV")
            
            DF_DPI_pq_pv_ul = pd.concat([ddf_pv[['PV_lower', 'PV_upper']], ddf_pq[['PQ_lower', 'PQ_upper']]], axis=1)
            self.DF_DPI_pq_pv_ul = DF_DPI_pq_pv_ul

            df_DPI_PO = dfPQ_CSI[['DPI_2N_INF', 'DPI_2N_SUP']].sum(axis=1) + dfPV_CSI[['DPI_2N_INF', 'DPI_2N_SUP']].sum(axis=1)
            df_DPI_PO = pd.DataFrame(df_DPI_PO).reset_index().groupby(by=['Dia', 'Hora']).agg(DPI_PO=(0, 'sum'))
            df_DPI_PO['DPI_PO_final'] = df_DPI_PO['DPI_PO'].pow(1 / (2 * n))
            self.df_DPI_PO = df_DPI_PO

            self.plots_static.plot_indice_0(df_DPI_PO, r'$\mathrm{DPI}$', 'DPI_PO_final', '', 'DPI_PO_final', order=True, ylimites=[-0.05, 1.5])
            
            print('Ploting DPI Analysis: ...')
            if self.Options['plotDPI']:
                def plot_indices(df, prefix, label, column):
                    self.plots_static.plot_indice(df, label, f'DPI_({prefix})_{label}', '', column, order=True, ylimites=[0, 1])

                plot_indices(dfPQ_CSI, 'u', 'PQ', 'CSI_SUP_FINAL')
                plot_indices(dfPQ_CSI, 'l', 'PQ', 'CSI_INF_FINAL')
                plot_indices(dfPV_CSI, 'u', 'PV', 'CSI_SUP_FINAL')
                plot_indices(dfPV_CSI, 'l', 'PV', 'CSI_INF_FINAL')

                def plot_indice_1_per_region(df_pq, df_pv, name, region):
                    self.plots_static.plot_indice_1(df_pv, df_pq, 'DPI', name, region, order=False)

                regions = {
                    'North': 'Norte',
                    'Northeast': 'Nordeste',
                    'Southeast Center West': 'Sudeste-Centro-Oeste',
                    'AC-RO': 'AC-RO',
                    'South': 'Sul'
                }
                for name, region in regions.items():
                    plot_indice_1_per_region(dfPQ_CSI, dfPV_CSI, name, region)

                def main_plot_indice_2(dffPQgb, dffPVgb):
                    regioes = df_PQ_reg['REG'].unique()
                    region_map = {'Nordeste': 'Northeast', 'Norte': 'North', 'Sudeste-Centro-Oeste': 'SE-CW', 'Sul': 'South', 'AC-RO': 'AC-RO'}
                    for reg in regioes:
                        for indice in ['CSI_INF', 'CSI_SUP']:
                            self.plots_static.plot_indice_2(dffPQgb, f'$\mathrm{{DPI}}_\mathrm{{PQ}}^\mathrm{{l}}$', f'DPI_(l)_PQ_{region_map[reg]}', region_map[reg], reg, indice, 'VBASEKV', limites=[0, 2.5])
                            self.plots_static.plot_indice_2(dffPVgb, f'$\mathrm{{DPI}}_\mathrm{{PV}}^\mathrm{{l}}$', f'DPI_(l)_PV_{region_map[reg]}', region_map[reg], reg, indice, 'Gen_Type', limites=[0, 2.5])
                            self.plots_static.plot_indice_2(dffPQgb, f'$\mathrm{{DPI}}_\mathrm{{PQ}}^\mathrm{{u}}$', f'DPI_(u)_PQ_{region_map[reg]}', region_map[reg], reg, indice, 'VBASEKV', limites=[0, 2.5])
                            self.plots_static.plot_indice_2(dffPVgb, f'$\mathrm{{DPI}}_\mathrm{{PV}}^\mathrm{{u}}$', f'DPI_(u)_PV_{region_map[reg]}', region_map[reg], reg, indice, 'Gen_Type', limites=[0, 2.5])

                main_plot_indice_2(dffPQgb, dffPVgb)

            if self.Options['Plot_Boxplot_DPI'] and not self.readjustONEcase:
                def boxplot_plot_PB(dff_filtered_PQ, dff_filtered_PV, df_ind, condition):
                    dff_PQ =  dff_filtered_PQ.groupby(by=['REG','VBASEKV','BUS_NAME']).agg(Ocurrencies = ('VBASEKV','count'), 
                                                                                        MODV_PU = ('MODV_PU', list),
                                                                                        MIN = ('MODV_PU', 'min'),
                                                                                        MAX = ('MODV_PU', 'max')
                                                                                        )
                    dff_PV =  dff_filtered_PV.groupby(by=['REG','Gen_Type','BUS_NAME']).agg(Ocurrencies = ('Gen_Type','count'), 
                                                                                        MODV_PU = ('MODV_PU', list),
                                                                                        MIN = ('MODV_PU', 'min'),
                                                                                        MAX = ('MODV_PU', 'max')
                                                                                        )
                    DF_dfss = [dff_PQ, dff_PV]
                    name = ['PQ', 'PV']
                    for idx, dff in enumerate(DF_dfss):
                        regions = dff.index.get_level_values(0).unique()
                        for Region in regions:
                            nt = dff.loc[Region].index.get_level_values(0).unique()
                            for vb in nt:
                                df_boxplot = dff.loc[Region,vb].sort_values(['Ocurrencies'], ascending = False)[:15]
                                minimo  = df_boxplot['MIN'].min() - 0.01
                                maximo =  df_boxplot['MAX'].max() + 0.01
                                numbuses =  condition + ' ' + Region + ' - ' + df_ind  + ' - ' +  name[idx] + ' - ' +  str(vb)  + ' - Buses with voltage problems = ' + str(dff.loc[Region,vb].shape[0]) 
                                self.plots_static.plot_boxplot(df_boxplot['MODV_PU'], df_boxplot.index, numbuses , 'BUSES', 'VOLTAGE (pu)', text = True, rotation=45, limites=[minimo,maximo])
                                
                def boxplot_problematic_buses(df_busPQ,df_busPV):
                    dicIndice = ['IndiceInf','IndiceSup']
                    for df_ind in dicIndice:

                        dff_filtered_PQ = df_busPQ[df_busPQ[df_ind]>1].sort_values(by=['REG','VBASEKV', 'BUS_NAME', 'MODV_PU'], ascending=[True, True, False, True])
                        dff_filtered_PV = df_busPV[df_busPV[df_ind]>1].sort_values(by=['REG','Gen_Type', 'BUS_NAME', 'MODV_PU'], ascending=[True, True, False, True])
                        boxplot_plot_PB(dff_filtered_PQ, dff_filtered_PV, df_ind, 'Inseguro')

                        dff_filtered_PQ = df_busPQ[(df_busPQ[df_ind]<=1) & (df_busPQ[df_ind]>0)].sort_values(by=['REG','VBASEKV', 'BUS_NAME', 'MODV_PU'], ascending=[True, True, False, True])
                        dff_filtered_PV = df_busPV[(df_busPV[df_ind]<=1) & (df_busPV[df_ind]>0)].sort_values(by=['REG','Gen_Type', 'BUS_NAME', 'MODV_PU'], ascending=[True, True, False, True])
                        boxplot_plot_PB(dff_filtered_PQ, dff_filtered_PV, df_ind, 'Alarme')

                boxplot_problematic_buses(df_busPQ_mod,df_busPV_mod)

            if self.Options['resumoIndice']:
                
                print('Saving DPI Information files:')
                df_busPQ_mod['BUS_ID'] = df_busPQ_mod['BUS_ID'].astype(int)
                df_busPV_mod['BUS_ID'] = df_busPV_mod['BUS_ID'].astype(int)
                self.df_busPQ_mod = df_busPQ_mod
                self.df_busPV_mod = df_busPV_mod

                def discriminarIndice2(x):
                    if x>1:
                        return 'Inseguro'
                        # return 3
                    elif (x<=1) & (x>0):
                        return 'Alarme'
                        # return 2
                    elif x == 0:
                        return 'Seguro'
                        # return 1

                Df_IndiceT2 = pd.concat([dfPQ_CSI[['CSI_SUP_FINAL','CSI_INF_FINAL']],dfPV_CSI[['CSI_SUP_FINAL','CSI_INF_FINAL']]], axis=0, keys=['DPI_PQ', 'DPI_PV'])
                Df_IndiceT2['OV condition'] = Df_IndiceT2['CSI_SUP_FINAL'].apply(lambda x: discriminarIndice2(x))
                Df_IndiceT2['UV condition'] = Df_IndiceT2['CSI_INF_FINAL'].apply(lambda x: discriminarIndice2(x))
                Df_IndiceT2.rename(columns={'CSI_SUP_FINAL':'OV DPI','CSI_INF_FINAL':'UV DPI'}, inplace=True)
                self.Df_IndiceT2 = Df_IndiceT2

    def save_csv(self):

        if not self.readjustONEcase:
                
                self.df_grouped[['PG_MW','QG_MVAR','PL_MW']].to_csv(self.cenario + '/Data/Potencia/Df_MW-MVAR_PO.csv', header=True, index=True)
                
                if self.Options['Plot_Tensao_Geral']:

                    self.df_Final_ger_PWFC.to_csv(f'{self.cenario}/Data/Geral/Df_ger.csv', index=False, columns=['BUS_ID', 'ARE', 'MODV_PU', 'ANGV_DEG', 'PG_MW', 'QG_MVAR', 'Dia', 'Hora', 'U_FED', 'Gen_Type', 'REG', 'B0_MVAR', 'ST', 'SHUNT_INST_IND', 'SHUNT_INST_CAP', 'ReservaIND', 'ReservaCAP'])
                    self.df_Final_nt_PWFC.to_csv(f'{self.cenario}/Data/Geral/Df_nt.csv', index=False, columns=['BUS_ID', 'ARE', 'MODV_PU', 'ANGV_DEG', 'VBASEKV', 'PL_MW', 'QL_MVAR', 'Dia', 'Hora', 'U_FED', 'REG', 'B0_MVAR', 'ST', 'SHUNT_INST_IND', 'SHUNT_INST_CAP', 'ReservaINDshunt', 'ReservaCAPshunt'])

                if self.Options['LinhaAnalise']:

                    self.DF_REGIONAL_GER[['PG_MW', 'QG_MVAR', 'PL_MW', 'QL_MVAR','Shunt_Ind', 'Shunt_Cap','SHUNT_INST_IND', 'SHUNT_INST_CAP', 'ReservaIND', 'ReservaCAP','PG_UHE', 'PG_UTE', 'PG_EOL', 'PG_SOL', 'PG_BIO', 'PG_Dist', 'QG/QL', 'PG/PL', 'PG_FERV', 'ReservaINDshunt', 'ReservaCAPshunt']].to_csv(self.cenario + '/Data/Potencia/DF_POT_Reg.csv')
                    self.PWF16_Filt_linhas[['From#','To#','From Name','To Name','% L1', 'L1(MVA)', 'Mvar:Losses','Dia', 'Hora','REG', 'VBASEKV','MVA', 'MW:From-To', 'MW:To-From','Power Factor:From-To','Power Factor:To-From']].to_csv(self.cenario+'/Data/Fluxo em Ramos/Df_Linhas.csv', index=None)
                    self.PWF16_Filt_TRAFO[['From#','To#','From Name','To Name','% L1', 'L1(MVA)', 'Mvar:Losses','Dia', 'Hora','REG', 'VBASEKV','MVA', 'MW:From-To', 'MW:To-From','Power Factor:From-To','Power Factor:To-From']].to_csv(self.cenario+'/Data/Fluxo em Ramos/Df_Trafo.csv', index=None)
                    print(f'*** ETAPA: Salvando dados dos Intercambios ***')
                    self.DF_Intercambios.to_csv(self.cenario + '/Data/Fluxo em Ramos/DF_Intercambios.csv')
                    self.df_HVDC.to_csv(self.cenario + '/Data/Fluxo em Ramos/DF_HVDC.csv')

                if self.Options['ReservaAnalise']:

                    self.dffreservaPO_MVAR.to_csv(self.cenario + '/Data/Potencia/Df_Reserva_PO_MVAR.csv', header=True, index=True)
                    self.dffreservaPO_REG_MVAR.to_csv(self.cenario + '/Data/Potencia/Df_Reserva_REG_MVAR.csv', header=True, index=True)
                    self.dffreservaPO_REG_MW[' Reserve'].to_csv(self.cenario + '/Data/Potencia/Df_Reserva_REG_MW.csv', header=True, index=True)
                    self.dffreservaPO_MW[' Reserve'].to_csv(self.cenario + '/Data/Potencia/Df_Reserva_PO_MW.csv', header=True, index=True)

                if self.Options['ComputeDPI']:

                    self.df_DPI_PO['DPI_PO_final'].to_csv(f"{self.cenario}/Data/Indice/Df_DPI_S4.csv")
                    self.DF_DPI_pq_pv_ul.to_csv(f"{self.cenario}/Data/Indice/Df_DPI_S3.csv")
                    self.dffPQgb.to_csv(f"{self.cenario}/Data/Indice/Df_PQ_DPI_S1.csv", index=True)
                    self.dffPVgb.to_csv(f"{self.cenario}/Data/Indice/Df_PV_DPI_S1.csv", index=True)

                    Df_IndiceT2 = self.Df_IndiceT2
                    Df_PQ_OV = Df_IndiceT2.loc['DPI_PQ'][~((Df_IndiceT2.loc['DPI_PQ']['OV DPI']==0) & (Df_IndiceT2.loc['DPI_PQ']['UV DPI']>0))].sort_values('OV DPI', ascending=False)[['OV condition', 'OV DPI']]
                    Df_PQ_UV = Df_IndiceT2.loc['DPI_PQ'][~((Df_IndiceT2.loc['DPI_PQ']['UV DPI']==0) & (Df_IndiceT2.loc['DPI_PQ']['OV DPI']>0))].sort_values('UV DPI', ascending=False)[['UV condition', 'UV DPI']]
                    Df_PV_OV = Df_IndiceT2.loc['DPI_PV'][~((Df_IndiceT2.loc['DPI_PV']['OV DPI']==0) & (Df_IndiceT2.loc['DPI_PV']['UV DPI']>0))].sort_values('OV DPI', ascending=False)[['OV condition', 'OV DPI']]
                    Df_PV_UV = Df_IndiceT2.loc['DPI_PV'][~((Df_IndiceT2.loc['DPI_PV']['UV DPI']==0) & (Df_IndiceT2.loc['DPI_PV']['OV DPI']>0))].sort_values('UV DPI', ascending=False)[['UV condition', 'UV DPI']]

                    Df_IndiceT2.to_csv(self.cenario + '/Data/Indice/Df_DPI_S2.csv')
                    path_script_org = self.cenario + "/Data/Indice/RelatorioIndice.txt"
                    numeroPO = len(set(Df_IndiceT2.index.to_frame()[['Dia','Hora']].apply(tuple, axis=1).values))
                    with open(path_script_org, 'w') as f:
                        f.write('O numero de pontos de operação analisados são: ' + str(numeroPO) + '\n')
                        f.write('=============================\n Informação Barras PQ:\n=============================\n')
                        regions = Df_PQ_OV.reset_index('REG')['REG'].unique()
                        for reg in regions:
                            f.write('- Sobretensão ' + reg +'\n')
                            try:
                                df_reg_sob= Df_PQ_OV.loc[:,:,reg]
                                f.write('numero de casos Inseguros: '+ str(df_reg_sob[df_reg_sob['OV condition']=='Inseguro'].shape[0])+'\n')
                                f.write('numero de casos Alarme: '+ str(df_reg_sob[df_reg_sob['OV condition']=='Alarme'].shape[0])+'\n')
                                f.write('numero de casos Seguros: '+ str(df_reg_sob[df_reg_sob['OV condition']=='Seguro'].shape[0])+'\n')
                                f.write('- Subtensão '+ reg +'\n')
                                df_reg_sub = Df_PQ_UV.loc[:,:,reg]
                                f.write('numero de casos Inseguros: ' + str(df_reg_sub[df_reg_sub['UV condition']=='Inseguro'].shape[0])+'\n')
                                f.write('numero de casos Alarme: ' + str(df_reg_sub[df_reg_sub['UV condition']=='Alarme'].shape[0])+'\n')
                                f.write('numero de casos Seguros: ' + str(df_reg_sub[df_reg_sub['UV condition']=='Seguro'].shape[0])+'\n')
                                f.write('--------------------------\n')
                            except:
                                pass
                        f.write('=============================\n Informação Barras PV:\n=============================\n')
                        for reg in regions:
                            f.write('- Sobretensão ' + reg +'\n')
                            try:
                                df_reg_sob= Df_PV_OV.loc[:,:,reg]
                                f.write('numero de casos Inseguros: '+ str(df_reg_sob[df_reg_sob['OV condition']=='Inseguro'].shape[0])+'\n')
                                f.write('numero de casos Alarme: '+ str(df_reg_sob[df_reg_sob['OV condition']=='Alarme'].shape[0])+'\n')
                                f.write('numero de casos Seguros: '+ str(df_reg_sob[df_reg_sob['OV condition']=='Seguro'].shape[0])+'\n')
                                f.write('- Subtensão '+ reg +'\n')
                                df_reg_sub = Df_PV_UV.loc[:,:,reg]
                                f.write('numero de casos Inseguros: ' + str(df_reg_sub[df_reg_sub['UV condition']=='Inseguro'].shape[0])+'\n')
                                f.write('numero de casos Alarme: ' + str(df_reg_sub[df_reg_sub['UV condition']=='Alarme'].shape[0])+'\n')
                                f.write('numero de casos Seguros: ' + str(df_reg_sub[df_reg_sub['UV condition']=='Seguro'].shape[0])+'\n')
                                f.write('--------------------------\n')
                            except:
                                pass

                    grouped_pv_nunique = self.df_busPV_mod[(self.df_busPV_mod['IndiceInf'] > 0)].groupby(by=['REG'])['BUS_ID'].nunique()
                    grouped_pq_nunique = self.df_busPQ_mod[(self.df_busPQ_mod['IndiceInf'] > 0)].groupby(by=['REG'])['BUS_ID'].nunique()
                    grouped_pv_unique = self.df_busPV_mod[(self.df_busPV_mod['IndiceInf'] > 0)].groupby(by=['REG'])['BUS_ID'].unique()
                    grouped_pq_unique = self.df_busPQ_mod[(self.df_busPQ_mod['IndiceInf'] > 0)].groupby(by=['REG'])['BUS_ID'].unique()
                    # Separator string
                    separator1 = '\n' + '=' * 20 + '\n' + '=' * 20 + '\n'
                    separator2 = '\n' + '=' * 20 + '\n' + '=' * 20 + '\n'
                    # 2. Combine with separators and Format
                    with open(self.cenario + '/Data/Indice/DPI_Critical_Buses.txt', 'w') as f:
                        f.write(grouped_pv_nunique.rename('Critical_infPVbuses').to_string() + separator1)
                        f.write(grouped_pq_nunique.rename('Critical_infPQbuses').to_string() + separator2)
                        f.write(grouped_pv_unique.rename('Critical_infPVbuses_bus').to_string() + separator1)
                        f.write(grouped_pq_unique.rename('Critical_infPQbuses_bus').to_string())  

                    self.df_DPI_PO[self.df_DPI_PO['DPI_PO_final'] > 1].index.to_frame()[['Dia', 'Hora']].apply(tuple, axis=1).to_csv(f"{self.cenario}/PO_Inseguros.txt", index=None)
                    



