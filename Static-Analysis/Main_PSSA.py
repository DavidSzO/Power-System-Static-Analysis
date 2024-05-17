import os
import sys
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from computeDPI import *
from Maps import *
from Handle_Plots_Static import *
from Read_Process_Cases import *

start_time = time.time()
# ************************************************************************************************
#                                       OPÇÕES DE EJECUÇÃO
# ************************************************************************************************

Options_ReadProcess= {
                        'gen_script4lines' : False,
                        'extract_fromcsv' : True,
                        'ConvergenceAnalise' : True,
                        'busdata' : True,
                    }

LinhaAnalise = True
HVDCAnalise = True
ReservaAnalise = True
IntercambiosAnalise = True
linhascsv = True
reservacsv =True
HVDCcsv = True
ComputeDPI = True
resumoIndice = True

PlotGeralPotencia = True
MapasPlots = True
Plot_Tensao_Geral = True  #PERMITE SALVAR OS DATAFRAMES DE df_Final_ger e df_Final_nt
plotDPI =  True
PlotDPICriticosPot = True
Plot_Boxplot_DPI = True
PlotIntercambios = True

# ************************************************************************************************
#                                              PATHS
# ************************************************************************************************

# path_folder = 'D:/MPV_(FNS Lim)_RC/'
path_folder = 'D:/MPV_(FNS Lim)_RC_test/'

# ============================= CASOS 2026 V1A1F_===========================================
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F2_RESP_FNS_lim_rev1_2026/'
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F3_RESP_FNS_lim_rev1_2026/'
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F4_RESP_FNS_lim_rev1_2026/'
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F5_RESP_FNS_lim_rev1_2026/'

# ************************************************************************************************
#                                PATH for SAVE THE PLOTS AND OTHER
# ************************************************************************************************

user_specified_dir = input("Please enter the directory path where you want to save the files: ")
# user_specified_dir = "C:/Users/david/OneDrive/Documents/FERV_documentos/0_Repositorio_Resultados"
notebook_dir = os.path.abspath(user_specified_dir)
folder_path = os.path.join(notebook_dir, os.path.basename(os.path.normpath(path_folder)))
subfolders = ['Mapas', 'Intercambios', 'Indice', 'Potencia', 'Boxplot', 'CriticalBuses']
for subfolder in subfolders:
    os.makedirs(os.path.join(folder_path, subfolder), exist_ok=True)
print(f"The directories have been created in: {folder_path}")

# ************************************************************************************************
#                                             EXTRACTION AND PROCESS
# ************************************************************************************************

gen_script4lines = Options_ReadProcess['gen_script4lines']
extract_fromcsv = Options_ReadProcess['extract_fromcsv']
ConvergenceAnalise = Options_ReadProcess['ConvergenceAnalise']
busdata = Options_ReadProcess['busdata']

if extract_fromcsv:
    pathcsv1 = os.path.join(path_folder, 'ProcessedDataBase.csv')
    pathcsv2 = None
else:
    pathcsv1 = None
    pathcsv2 = os.path.join(path_folder, 'ProcessedDataBase.csv')

if Options_ReadProcess['gen_script4lines']:
    Read_Scenarios(path_folder, folder_path, pathcsv=pathcsv1, genscript=gen_script4lines)
    sys.exit()
else:
    print('******************** EXTRAÇÃO DE DADOS ********************')
    cases = Read_Scenarios(path_folder, folder_path, pathcsv=pathcsv1, genscript=gen_script4lines)
    # return cases
    cenario = folder_path
    print('******************** PROCESSAMENTO DE DADOS ********************')
    processdata = ProcessData(df= cases.Df_Cases, cenario = cenario, pathcsv = pathcsv2, extract_fromcsv = extract_fromcsv, busdata = busdata)


bool_PWF_NConv = cases.PWF_NC[['Dia', 'Hora']].apply(tuple, axis=1)
cenario = cases.cenario
df_Final_ger = processdata.df_Final_ger
df_Final_nt = processdata.df_Final_nt
dff_Ger_map = processdata.dff_Ger_map
dff_Ger_map.loc[dff_Ger_map['Gen_Type']=='UNE','Gen_Type'] = 'UTE' # cambia de designación de usinas nucleares a termicas para ser plotadas en el mapa
dff_NT_map = processdata.dff_NT_map
DF_REGIONAL_GER = processdata.DF_REGIONAL_GER
DF_REGIONAL_PQ = processdata.DF_REGIONAL_PQ

plots_static = Plots_Static(cenario, svg=False)

# =============================================================================================================================
#                                                LEITURA LINHAS E RESERVA
# =============================================================================================================================

## ***************** (Este código obtem as informações das linhas AC e DC e reserva por maquina) *****************
if linhascsv and LinhaAnalise:
    PWF16_concatenados = dd.read_csv(path_folder + '/LinhasInfo.csv', sep=',').compute()
    PWF16_concatenados['Dia'] = PWF16_concatenados['Dia'].astype(str).str.zfill(2)
    cases.get_Intercambios(df=PWF16_concatenados)
    DF_Intercambios = cases.DF_Intercambios

if HVDCcsv and HVDCAnalise:
    DCLinks_concatenados = dd.read_csv(path_folder + '/HVDCInfo.csv', sep=',').compute()
    DCLinks_concatenados['Dia'] = DCLinks_concatenados['Dia'].astype(str).str.zfill(2)

if reservacsv and ReservaAnalise:
    SGN01_concatenados = dd.read_csv(path_folder + '/ReservaInfo.csv', sep=',').compute()
    SGN01_concatenados['Dia'] = SGN01_concatenados['Dia'].astype(str).str.zfill(2)

if not (linhascsv and reservacsv and HVDCcsv):
    cases.get_Networkinfo(linhas=not linhascsv, Reserva=not reservacsv, Intercambios=not HVDCcsv)

    if not linhascsv and LinhaAnalise:
        PWF16_concatenados = cases.linesInfo
        DF_Intercambios = cases.DF_Intercambios

    if not reservacsv and ReservaAnalise:
        SGN01_concatenados = cases.ReserveInfo

    if not HVDCcsv and HVDCAnalise:
        DCLinks_concatenados = cases.HVDCInfo
        # DF_Intercambios = cases.DF_Intercambios

if LinhaAnalise:

    def addUF_linha(from_bus, to_bus, bus_info_map, vbasekv_map):

        reg1 = bus_info_map.get(from_bus, np.nan)
        Vbase1 = vbasekv_map.get(from_bus, np.nan)
        reg2 = bus_info_map.get(to_bus, np.nan)

        if reg1 == reg2 and reg1 is not np.nan:
            return reg1, Vbase1
        else:
            return np.nan, np.nan

    def Main_linha_addREG(PWF16_concatenados):

        Df_VF_SF = processdata.Df_VF_SF
        InfoBarras = Df_VF_SF[(Df_VF_SF['Dia']=='01') & (Df_VF_SF['Hora']=='00-00')][['BUS_ID','BUS_NAME', 'ARE','VBASEKV','REG', 'U_FED', 'Gen_Type','Latitude', 'Longitude']]
        PWF16_concatenados_d1 = PWF16_concatenados[(PWF16_concatenados['Dia'] == '01') & (PWF16_concatenados['Hora'] == '00-00')].groupby(by=['From#','To#']).first().reset_index().copy()

        # Create a dictionary to map 'BUS_ID' to 'REG' and 'VBASEKV'
        bus_info_map = dict(zip(InfoBarras['BUS_ID'], InfoBarras['REG']))
        vbasekv_map = dict(zip(InfoBarras['BUS_ID'], InfoBarras['VBASEKV']))

        # Use the apply function to create 'REG' and 'VBASEKV' columns in PWF16_concatenados_d1
        PWF16_concatenados_d1['REG'], PWF16_concatenados_d1['VBASEKV'] = zip(*PWF16_concatenados_d1.apply(lambda row: addUF_linha(row['From#'], row['To#'], bus_info_map, vbasekv_map), axis=1))

        PWF16_concatenados_R = PWF16_concatenados.merge(PWF16_concatenados_d1[['From#','To#','REG','VBASEKV']], on=['From#','To#'], how='left')
        PWF16_Filt_linhas = PWF16_concatenados_R[(PWF16_concatenados_R['Type'] == ' TL') & ~(PWF16_concatenados_R['REG'].isna())]
        PWF16_Filt_TRAFO = PWF16_concatenados_R[(PWF16_concatenados_R['Type'] == ' TRAFO') & ~(PWF16_concatenados_R['REG'].isna())]

        return PWF16_Filt_linhas, PWF16_Filt_TRAFO

    PWF16_Filt_linhas, PWF16_Filt_TRAFO = Main_linha_addREG(PWF16_concatenados)

    PWF16_Filt_linhas[['From#','To#','From Name','To Name','% L1', 'L1(MVA)', 'Mvar:Losses','Dia', 'Hora','REG', 'VBASEKV','MVA', 'MW:From-To', 'MW:To-From','Power Factor:From-To','Power Factor:To-From']].to_csv(cenario+'/Linhas.csv', index=None)

    PWF16_Filt_TRAFO[['From#','To#','From Name','To Name','% L1', 'L1(MVA)', 'Mvar:Losses','Dia', 'Hora','REG', 'VBASEKV','MVA', 'MW:From-To', 'MW:To-From','Power Factor:From-To','Power Factor:To-From']].to_csv(cenario+'/Trafo.csv', index=None)

    PWF16_Filt_grouped = PWF16_Filt_linhas[PWF16_Filt_linhas['VBASEKV'].isin([230, 345, 440, 500, 525, 765])].groupby(by = ['Dia','Hora','REG']).agg({'% L1':'mean', 'Mvar:Losses':'sum'}) 

    if IntercambiosAnalise:
        ## ========================================== ELOS SEPARADOS POR BIPOLOS:
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

        ## ========================================== ELOS HVDC SEM SEPARAÇÃO POR POLOS:

        dfelo1 = DCLinks_concatenados[DCLinks_concatenados['Bus #'] == 85].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
        dfelo1['Nome Elo'] = 'Elo_FOZ-IBIUNA'
        dfelo2 = DCLinks_concatenados[DCLinks_concatenados['Bus #'] == 7055].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
        dfelo2['Nome Elo'] = 'Elo_PVEL-ARARQ'
        dfelo3 = DCLinks_concatenados[DCLinks_concatenados['Bus #'] == 7059].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
        dfelo3['Nome Elo'] = 'Elo_CPVBTB-PVEL'
        dfelo4 = DCLinks_concatenados[(DCLinks_concatenados['Bus #'] == 8100)].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
        dfelo4['Nome Elo'] = 'Elo_XINGU-SE'
        dfelo4 = DCLinks_concatenados[(DCLinks_concatenados['Bus #'] == 8100) & (DCLinks_concatenados[' Pole #'].isin([1,2]))].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
        # dfelo4['Nome Elo'] = 'Elo_XINGU-ESTREI'
        # dfelo5 = DCLinks_concatenados[(DCLinks_concatenados['Bus #'] == 8100) & (DCLinks_concatenados[' Pole #'].isin([3,4]))].groupby(by=['Dia', 'Hora']).agg({' P(MW)': 'sum', ' Q(Mvar)': 'sum'})
        # dfelo5['Nome Elo'] = 'Elo_XINGU-T.RIO'
        # Merge all dataframes
        # df_HVDC = pd.concat([dfelo1, dfelo2, dfelo3, dfelo4, dfelo5], axis=0, keys=['Elo_FOZ-IBIUNA', 'Elo_PVEL-ARARQ', 'Elo_CPVBTB-PVEL' ,'Elo_XINGU-ESTREI', 'Elo_XINGU-T.RIO'])
        df_HVDC = pd.concat([dfelo1, dfelo2, dfelo3, dfelo4], axis=0, keys=['Elo_FOZ-IBIUNA', 'Elo_PVEL-ARARQ', 'Elo_CPVBTB-PVEL' ,'Elo_XINGU-SE'])
        df_HVDC.to_csv(cenario+'/DF_HVDC.csv')

        if PlotIntercambios == True:
            plots_static.plot_Intercambio (DF_Intercambios, df_HVDC , '(MW)', 'Exportação (N-S, NE-SE) e  Elo Xingu-SE', ['Fluxo_N-S', 'Fluxo_NE-SE'], ['Elo_XINGU-SE'], )
            plots_static.plot_Intercambio (DF_Intercambios, df_HVDC , '(MW)', 'Exportação N-S e  Elo Xingu-SE', ['Fluxo_N-S'], ['Elo_XINGU-SE'], )
            plots_static.plot_Intercambio (DF_Intercambios, df_HVDC , '(MW)', 'Comparativo Exportação NE-N e Elo Xingu-SE', ['Fluxo_NE-N'], ['Elo_XINGU-SE'], Xlimites=None)
            plots_static.plot_Intercambio (DF_Intercambios, df_HVDC , '(MW)', 'Comparativo Exportação NE-SE e Elo FOZ-IBIUNA', ['Fluxo_NE-SE'], ['Elo_FOZ-IBIUNA'], Xlimites=None)
            plots_static.plot_Intercambio (DF_Intercambios, df_HVDC , '(MW)', 'Comparativo Exportação SUL-SECO e Elo FOZ-IBIUNA', ['Fluxo_SUL-SECO'], ['Elo_FOZ-IBIUNA'], Xlimites=None)

#=============================================================================================================================
#                                                    PRESERVA REGIONAL
#=============================================================================================================================

if ReservaAnalise == True:
    if (SGN01_concatenados.empty == False):
        dia = df_Final_ger['Dia'].iloc[0]
        hora = df_Final_ger['Hora'].iloc[0]
        df_Final_ger_mod = df_Final_ger[(df_Final_ger['Dia'] == dia) & (df_Final_ger['Hora'] == hora)][['BUS_ID', 'Gen_Type', 'U_FED', 'REG']]
        SGN01_concatenados.rename(columns={'Bus':'BUS_ID', }, inplace=True)
        Df_Reserva = SGN01_concatenados.merge(df_Final_ger_mod, how = 'left', on='BUS_ID')

        REG_groupReserve = Df_Reserva.groupby(by = ['Dia','Hora', 'REG']).agg({' Reserve': 'sum'})
        GroupReserve = Df_Reserva.groupby(by = ['Dia','Hora']).agg({' Reserve': 'sum'})
        GroupReserve[' Reserve'].to_csv(cenario + '/Reserva_MW_PO.csv', header=True, index=True)
        REG_groupReserve[' Reserve'].to_csv(cenario + '/Reserva_MW_PO_REG.csv', header=True, index=True)

        plots_static.plot_Potencia(GroupReserve[' Reserve'], '(MW)', 'RESERVA (MW) - SIN', limites=None)
        plots_static.plot_reserva_reg (REG_groupReserve, '(MW)', 'Reserva por Região', 'RESERVA POR REGIÃO', ' Reserve', xlimites=None,ylimites=None, order = False)

    # ======================ESSE DATAFRAME É SÓ DA RESERVA DAS MAQUINAS COM MODELO DO GERADOR PARA O CONTROLE DE FREQ
    # ===========================================================================================================================
    dff_reserva = Df_Reserva.merge(df_Final_ger[['BUS_ID','Dia', 'Hora', 'QMX_MVAR', 'QMN_MVAR', 'Ger_Units','QG_MVAR']], on=['BUS_ID','Dia', 'Hora'], how='left')
    dff_reserva['Qmin'] = (dff_reserva['QMN_MVAR'] / dff_reserva['Ger_Units']) * dff_reserva[' Units']
    dff_reserva['Qmax'] = (dff_reserva['QMX_MVAR'] / dff_reserva['Ger_Units']) * dff_reserva[' Units']
    dff_reserva['ReservaIND'] = np.where(dff_reserva['QG_MVAR'] < 0, dff_reserva['Qmin'] - dff_reserva['QG_MVAR'], dff_reserva['Qmin'])
    dff_reserva['ReservaCAP'] = np.where(dff_reserva['QG_MVAR'] > 0, dff_reserva['Qmax'] - dff_reserva['QG_MVAR'], dff_reserva['Qmax'])
    # ============================================================================================================================
    dffreservaPO = df_Final_ger.groupby(['Dia', 'Hora']).agg({'QG_MVAR': 'sum', 'ReservaIND':'sum', 'ReservaCAP':'sum'})
    dffreservaPO_REG = df_Final_ger.groupby(['Dia', 'Hora', 'REG']).agg({'QG_MVAR': 'sum', 'ReservaIND':'sum', 'ReservaCAP':'sum'})
    # Salvando dataframe de reserva mvar ============================================
    dffreservaPO.to_csv(cenario + '/ReservaMVAR_PO.csv', header=True, index=True)
    dffreservaPO_REG.to_csv(cenario + '/ReservaMVAR_PO_REG.csv', header=True, index=True)

    #=============================================================================================================================
    #                                                                   PLOTS RESERVA MVAR
    #=============================================================================================================================

    plots_static.plot_reserva_reg (dffreservaPO_REG, '(MVAR)', 'Reserva Capacitiva por Região MVAR', 'RESERVA CAPACITIVA POR REGIÃO MVAR', 'ReservaCAP', xlimites=None,ylimites=None, order = False)
    plots_static.plot_reserva_reg (dffreservaPO_REG, '(MVAR)', 'Reserva Indutiva por Região MVAR', 'RESERVA INDUTIVA POR REGIÃO MVAR', 'ReservaIND', xlimites=None,ylimites=None, order = False)

    fig, ax = plt.subplots(figsize=(20,10))
    dffreservaPO['ReservaCAP'].plot(figsize=(20,10), grid=True, title='RESERVA CAPACITIVA (Mvar)',legend='RESERVA')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('PO',fontsize = 15)
    ax.set_ylabel('(MVAR)',fontsize = 15)
    ax.set_title('RESERVA CAPACITIVA (Mvar)', fontsize = 20)
    ax.legend(fontsize = 15)
    nome = cenario + '/Potencia/Reserva_cap_mvar.png'
    plt.savefig(nome, bbox_inches = 'tight')

    fig, ax = plt.subplots(figsize=(20,10))
    dffreservaPO['ReservaIND'].plot(figsize=(20,10), grid=True, title='RESERVA INDUTIVA (Mvar)',legend='RESERVA')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('PO',fontsize = 15)
    ax.set_ylabel('(MVAR)',fontsize = 15)
    ax.set_title('RESERVA INDUTIVA (Mvar)', fontsize = 20)
    ax.legend(fontsize = 15)
    nome = cenario + '/Potencia/Reserva_ind_mvar.png'
    plt.savefig(nome, bbox_inches = 'tight')

#=============================================================================================================================
#                                                POTENCIA ATIVA E REATIVA
#=============================================================================================================================

regioes = DF_REGIONAL_GER.reset_index()['REG'].unique()
df_pg = DF_REGIONAL_GER.reset_index(level=['Dia','Hora', 'REG'])[['PG_MW','PL_MW','PG_EOL','PG_SOL', 'Dia', 'Hora', 'MODV_PU', 'QG_MVAR']]
df_pg['PG_FERV'] =  (df_pg['PG_EOL'] + df_pg['PG_SOL'])/df_pg['PL_MW']
df_grouped = df_pg.groupby(by = ['Dia', 'Hora'])[['PG_FERV', 'MODV_PU', 'QG_MVAR','PG_MW', 'PL_MW' ]].sum()
# Salvando dataframe de POTENCIA MW, RESERVA E MVAR===========================================
df_grouped['PG_MW'].to_csv(cenario + '/MW_PO.csv', header=True, index=True)
df_grouped['QG_MVAR'].to_csv(cenario + '/MVAR_PO.csv', header=True, index=True)

if PlotGeralPotencia:
    plots_static.plot_Potencia(df_grouped['QG_MVAR'], '(MVAR)', 'POTENCIA REATIVA GERADA (MVAR) - SIN', limites=None)
    plots_static.plot_Potencia(df_grouped['PG_MW'], '(MW)', 'POTENCIA ATIVA GERADA (MW) - SIN', limites=None)
    typeGenDic = {'QG_UHE':'Num_Usinas_UHE', 'QG_UTE':'Num_Usinas_UTE', 'QG_EOL':'Num_Usinas_EOL','QG_SOL':'Num_Usinas_SOL', 'QG_BIO':'Num_Usinas_BIO'}
    typeGenRegDic = {'Norte':['QG_UHE','QG_EOL','QG_SOL','QG_UTE'],'Nordeste':['QG_UHE','QG_EOL','QG_SOL','QG_UTE'],'Sudeste-Centro-Oeste':['QG_UHE','QG_EOL','QG_SOL','QG_UTE','QG_BIO'],'Sul':['QG_UHE','QG_EOL','QG_UTE','QG_BIO'], 'AC-RO':['QG_UHE','QG_UTE']}
    typeGenRegDic_MW = {'Norte':['PG_UHE','PG_EOL','PG_SOL','PG_UTE'],'Nordeste':['PG_UHE','PG_EOL','PG_SOL','PG_UTE'],'Sudeste-Centro-Oeste':['PG_UHE','PG_EOL','PG_SOL','PG_UTE','PG_BIO'],'Sul':['PG_UHE','PG_EOL','PG_UTE','PG_BIO'], 'AC-RO':['PG_UHE','PG_UTE']}
    typeGenDic_MW = {'PG_UHE':'Num_Usinas_UHE', 'PG_UTE':'Num_Usinas_UTE', 'PG_EOL':'Num_Usinas_EOL','PG_SOL':'Num_Usinas_SOL', 'PG_BIO':'Num_Usinas_BIO'}

    for reg in regioes:
        plots_static.plot_Potencia(DF_REGIONAL_GER.loc[:,:,reg]['QG_MVAR'], '(MVAR)', 'POTENCIA REATIVA GERADA (MVAR) - ' + reg, limites=None)
        plots_static.plot_Potencia(DF_REGIONAL_GER.loc[:,:,reg]['PG_MW'], '(MW)', 'POTENCIA ATIVA GERADA (MW) - ' + reg, limites=None)
        for tog in typeGenRegDic[reg]:
            numUsinas = DF_REGIONAL_GER.loc[:,:,reg][typeGenDic[tog]].iloc[0]
            nome = str('MVAR ' + reg.replace('-',' ')  + ' (' + tog.replace('_','-') + ') - Numero de Usinas ' + str(int(numUsinas)))
            plots_static.plot_Potencia(DF_REGIONAL_GER.loc[:,:,reg][tog], '(MVAR)', nome , limites=None)

        for tog in typeGenRegDic_MW[reg]:
            numUsinas = DF_REGIONAL_GER.loc[:,:,reg][typeGenDic_MW[tog]].iloc[0]
            nome = str('MW ' + reg.replace('-',' ')  + ' (' + tog.replace('_','-') + ') - Numero de Usinas ' + str(int(numUsinas)))
            plots_static.plot_Potencia(DF_REGIONAL_GER.loc[:,:,reg][tog], '(MW)', nome , limites=None)

#=============================================================================================================================
#                                                          TENSÃO
#=============================================================================================================================

if Options_ReadProcess['ConvergenceAnalise']:
    if Plot_Tensao_Geral:

        def boxplot_barrasGeracao(Df_VF):
            
            Df_groupbyUF = Df_VF.groupby(by = ['Gen_Type']).agg({'BUS_ID': 'unique', 'MODV_PU': list,})
            Df_groupbyReg = Df_VF.groupby(by = ['REG']).agg({'BUS_ID': 'unique', 'MODV_PU': list,})
            #***************************************************************************************************
            data =  Df_groupbyUF['MODV_PU'][['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO']] 
            Nbarras = Df_groupbyUF['BUS_ID'][['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO']]
            labels = ['Hydro','Thermal','SHP', 'Wind', 'Solar', 'Bio']  # Tus etiquetas de datos
            title = 'Bus Voltage Distribution by Type of Generation'  # Tu título
            xlabel = 'Type of Generation'  # Tu etiqueta del eje x
            ylabel = 'Voltage (pu)'  # Tu etiqueta del eje y
            # plots_static.plot_boxplot(data, labels, title, xlabel, ylabel,text= True, nbarra = Nbarras)#limites = [0.88,1.125]
            #***************************************************************************************************
            Nbarras = Df_groupbyReg['BUS_ID'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
            data = Df_groupbyReg['MODV_PU'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
            labels = ['North', 'Northeast', 'SE-CW', 'South','AC-RO']  # Tus etiquetas de datos
            title = 'Bus Voltage Distribution by Region for Voltage-Controled Buses'  # Tu título
            xlabel = 'Region'  # Tu etiqueta del eje x
            ylabel = 'Voltage (pu)'  # Tu etiqueta del eje y
            # plots_static.plot_boxplot(data, labels, title, xlabel, ylabel,text= True, nbarra = Nbarras)#, limites = [0.82,1.18]

        def boxplot_barrasCarga(Df_Vfpt):

            #***************************************************************************************************
            V_NT = Df_Vfpt.groupby(by = ['VBASEKV']).agg({'BUS_ID': 'unique', 'MODV_PU': list,})
            n_bar = V_NT['BUS_ID'][[138,230,345,440,500,525,765]]
            numbarra = list(n_bar)
            data = V_NT['MODV_PU'][[138,230,345,440,500,525,765]]
            labels = ['138','230','345','440','500','525','765']  # Tus etiquetas de datos
            title = 'Bus Voltage Distribution by Voltage Level'  # Tu título
            xlabel = 'Voltage Level (kV)'  # Tu etiqueta del eje x
            ylabel = 'Voltage (pu)'  # Tu etiqueta del eje y
            # plots_static.plot_boxplot(data, labels, title, xlabel, ylabel, text=True, nbarra = numbarra)#limites = [0.95, 1.105]
            #***************************************************************************************************
            Df_groupbyReg = Df_Vfpt.groupby(by = ['REG']).agg({'BUS_ID': 'unique', 'MODV_PU': list,})
            Nbarras = Df_groupbyReg['BUS_ID'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
            data = Df_groupbyReg['MODV_PU'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
            labels =  ['North', 'Northeast', 'SE-CW', 'South','AC-RO']  # Tus etiquetas de datos
            title = 'Bus Voltage Distribution by Region for Load Buses'  # Tu título
            xlabel = 'Region'  # Tu etiqueta del eje x
            ylabel = 'Voltage (pu)'  # Tu etiqueta del eje y
            # plots_static.plot_boxplot(data, labels, title, xlabel, ylabel,text= True, nbarra = Nbarras)#, limites = [0.82,1.18]

        def plottensaoG():
            Df_VF = processdata.Df_VF
            DFF_Geral = Df_VF[(Df_VF['VBASEKV'].isin([138,230,345,440,500,525,765])) | (Df_VF['Gen_Type'].isin(['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO']))] 
            filtro1 = (DFF_Geral[['Dia', 'Hora']].apply(tuple, axis=1).isin(bool_PWF_NConv))
            DFF_Geral_PWFC = DFF_Geral[~filtro1]
            data = [DFF_Geral_PWFC['MODV_PU'].values]  # Tu conjunto de datos
            labels = ['G. Sincrona']  # Tus etiquetas de datos
            title = 'Bus Voltage Distribution of the System'  # Tu título
            xlabel = 'Voltage (pu)'  # Tu etiqueta del eje x
            ylabel = 'Bus Voltages'  # Tu etiqueta del eje y

            # plots_static.plot_boxplot(data, labels, title, xlabel, ylabel, vert = False, text=True, rotation = 0)

            # Q1 = np.percentile(data, 25)
            # Q3 = np.percentile(data, 75)
            # print(f'Primer cuartil (Q1): {Q1}')
            # print(f'Tercer cuartil (Q3): {Q3}')
            # IQR = Q3 - Q1
            # limite_inferior = Q1 - 1.5 * IQR
            # limite_superior = Q3 + 1.5 * IQR
            # print(f'El límite inferior es: {limite_inferior}')
            # print(f'El límite superior es: {limite_superior}')

        def plottensaoPR():
            df_ger = df_Final_ger[df_Final_ger['Gen_Type'].isin(['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO'])]
            filtro1 = (df_ger[['Dia', 'Hora']].apply(tuple, axis=1).isin(bool_PWF_NConv))
            df_Final_ger_PWFC = df_ger[~filtro1]
            df_Final_ger_PWFC[['BUS_ID','ARE', 'MODV_PU', 'ANGV_DEG','PG_MW', 'QG_MVAR','Dia', 'Hora', 'U_FED', 'Gen_Type', 'REG', 'B0_MVAR', 'ST', 'SHUNT_INST_IND', 'SHUNT_INST_CAP', 'ReservaIND', 'ReservaCAP']].to_csv(cenario+'/Df_ger.csv', index=None)
            df_nt = df_Final_nt[df_Final_nt['VBASEKV'].isin([138,230,345,440,500,525,765])]
            filtro2 = (df_nt[['Dia', 'Hora']].apply(tuple, axis=1).isin(bool_PWF_NConv))
            df_Final_nt_PWFC = df_nt[~filtro2]
            df_Final_nt_PWFC[['BUS_ID','ARE', 'MODV_PU', 'ANGV_DEG','VBASEKV', 'PL_MW', 'QL_MVAR','Dia', 'Hora', 'U_FED', 'REG', 'B0_MVAR', 'ST', 'SHUNT_INST_IND', 'SHUNT_INST_CAP','ReservaINDshunt', 'ReservaCAPshunt']].to_csv(cenario+'/Df_nt.csv', index=None)
            
            # plots_static.boxplot_barrasCarga(df_Final_nt_PWFC)
            # plots_static.boxplot_barrasGeracao(df_Final_ger_PWFC)

        plottensaoG()
        plottensaoPR()

#=============================================================================================================================
#                                                           MAPAS
#=============================================================================================================================

if MapasPlots:
    Df_VF = processdata.Df_VF
    options = {'Limit Violations All': True, 'Mean and Variance': True, 'Limit Violations by Group': True, 'HeatMap by state 1': True, 'Limit Violations PO': False}
    Maps(Df_VF, dff_NT_map, dff_Ger_map, cenario, options)

#=============================================================================================================================
#                                             DPI (DECOMPOSED PERFORMANCE INDEX)
#=============================================================================================================================

if ComputeDPI:

    print('CALCULO DO DPI para todos os Cenários:')
    ts = 0.8
    tb = 1
    n=2
    VVI = computeDPI(df_Final_nt, df_Final_ger, ts, tb, p_norm = n, p_inf = False, NBcv = True)
    dfPQ_CSI = VVI.dfPQ_CSI
    dfPV_CSI = VVI.dfPV_CSI
    df_PQ_reg = VVI.df_PQ_reg
    df_PV_reg = VVI.df_PV_reg
    df_busPQ = VVI.df_busPQ
    df_busPV = VVI.df_busPV
    n_maior = VVI.n_maior

    dfPQ_CSI = dfPQ_CSI.groupby(['Dia' , 'Hora', 'REG']).first()
    dfPV_CSI = dfPV_CSI.groupby(['Dia' , 'Hora', 'REG']).first()
    dffPQgb = df_PQ_reg.groupby(by=['Dia','Hora','REG','VBASEKV']).agg({'CSI_INF':'first','CSI_SUP':'first'})
    dffPVgb = df_PV_reg.groupby(by=['Dia','Hora','REG','Gen_Type']).agg({'CSI_INF':'first','CSI_SUP':'first'})

    if Options_ReadProcess['ConvergenceAnalise']:
        for index in bool_PWF_NConv:
            dfPQ_CSI.drop((index[0], index[1]), inplace=True)
            dfPV_CSI.drop((index[0], index[1]), inplace=True)
            dffPQgb.drop((index[0], index[1]), inplace=True)
            dffPVgb.drop((index[0], index[1]), inplace=True)
        filtro1 = (df_busPQ[['Dia', 'Hora']].apply(tuple, axis=1).isin(bool_PWF_NConv))
        df_busPQ_mod = df_busPQ[~filtro1].copy()
        filtro2 = (df_busPV[['Dia', 'Hora']].apply(tuple, axis=1).isin(bool_PWF_NConv))
        df_busPV_mod = df_busPV[~filtro2].copy()
    else:
        df_busPQ_mod = df_busPQ
        df_busPV_mod = df_busPV
        
    dffPQgb.to_csv(cenario+'/Indice_DecompPQ.csv',index = True)
    dffPVgb.to_csv(cenario+'/Indice_DecompPV.csv',index = True)

    dfPQ_CSI['DPI_2N_INF'] = dfPQ_CSI['CSI_INF_FINAL'].pow(2*n) 
    dfPQ_CSI['DPI_2N_SUP'] = dfPQ_CSI['CSI_SUP_FINAL'].pow(2*n) 
    dfPV_CSI['DPI_2N_INF'] = dfPV_CSI['CSI_INF_FINAL'].pow(2*n) 
    dfPV_CSI['DPI_2N_SUP'] = dfPV_CSI['CSI_SUP_FINAL'].pow(2*n) 
    ddf_pq = dfPQ_CSI.reset_index().groupby(by = ['Dia','Hora']).agg({'DPI_2N_INF':  'sum', 'DPI_2N_SUP':  'sum'})
    ddf_pq['PQ_lower'] = ddf_pq['DPI_2N_INF'].pow(1/(2*n))
    ddf_pq['PQ_upper'] = ddf_pq['DPI_2N_SUP'].pow(1/(2*n))
    ddf_pv = dfPV_CSI.reset_index().groupby(by = ['Dia','Hora']).agg({'DPI_2N_INF':  'sum', 'DPI_2N_SUP':  'sum'})
    ddf_pv['PV_lower'] = ddf_pv['DPI_2N_INF'].pow(1/(2*n))
    ddf_pv['PV_upper'] = ddf_pv['DPI_2N_SUP'].pow(1/(2*n))
    DF_DPI_pq_pv_ul = pd.concat([ddf_pv[['PV_lower','PV_upper']],ddf_pq[['PQ_lower','PQ_upper']]], axis=1)
    DF_DPI_pq_pv_ul.to_csv(cenario+'/Indice_Modif.csv')

    # ==========================================================================================================================
    dfPQ_CSI['DPI_2N_INF'] = dfPQ_CSI['CSI_INF_FINAL'].pow(2*n) 
    dfPQ_CSI['DPI_2N_SUP'] = dfPQ_CSI['CSI_SUP_FINAL'].pow(2*n) 
    dfPV_CSI['DPI_2N_INF'] = dfPV_CSI['CSI_INF_FINAL'].pow(2*n) 
    dfPV_CSI['DPI_2N_SUP'] = dfPV_CSI['CSI_SUP_FINAL'].pow(2*n) 

    df_DPI_PO_ = dfPQ_CSI['DPI_2N_INF'] + dfPQ_CSI['DPI_2N_SUP'] + dfPV_CSI['DPI_2N_INF'] +  dfPV_CSI['DPI_2N_SUP'] 
    df_DPI_PO = pd.DataFrame(df_DPI_PO_)
    df_DPI_PO = df_DPI_PO.reset_index().groupby(by = ['Dia','Hora']).agg(DPI_PO =(0, 'sum'))
    df_DPI_PO['DPI_PO_final']= df_DPI_PO['DPI_PO'].pow(1/(2*n))
    df_DPI_PO['DPI_PO_final'].to_csv(cenario+'/Indice_PO.csv')

    plots_static.plot_indice_0 (df_DPI_PO, r'$\mathrm{DPI}$', 'DPI_PO_final','','DPI_PO_final', order=True, ylimites=[-0.05, 1.5] )
    (df_DPI_PO[df_DPI_PO['DPI_PO_final']>1].index.to_frame()[['Dia', 'Hora']].apply(tuple, axis=1)).to_csv(cenario + '/PO_Inseguros.txt', index = None)

    if plotDPI:

        plots_static.plot_indice (dfPQ_CSI, r'$\mathrm{DPI}_\mathrm{PQ}^\mathrm{u}$', 'DPI_(u)_PQ','','CSI_SUP_FINAL', order=True, ylimites=[0, 1])
        plots_static.plot_indice (dfPQ_CSI, r'$\mathrm{DPI}_\mathrm{PQ}^\mathrm{l}$', 'DPI_(l)_PQ','', 'CSI_INF_FINAL', order=True, ylimites=[0, 1])
        plots_static.plot_indice (dfPV_CSI, r'$\mathrm{DPI}_\mathrm{PV}^\mathrm{u}$', 'DPI_(u)_PV' ,'','CSI_SUP_FINAL', order=True, ylimites=[0, 1])
        plots_static.plot_indice (dfPV_CSI, r'$\mathrm{DPI}_\mathrm{PV}^\mathrm{l}$', 'DPI_(l)_PV', '','CSI_INF_FINAL', order=True, ylimites=[0, 1])

        plots_static.plot_indice_1 (dfPV_CSI, dfPQ_CSI, 'DPI', 'North','Norte',order=False)#limites=[0,1.3]
        plots_static.plot_indice_1 (dfPV_CSI, dfPQ_CSI, 'DPI', 'Northeast','Nordeste',order=False )#limites=[0,1.3], limites=[-0.1,2]
        plots_static.plot_indice_1 (dfPV_CSI, dfPQ_CSI, 'DPI', 'Southeast Center West','Sudeste-Centro-Oeste',order=False)#limites=[0,1.3], limites=[-0.1,9]
        plots_static.plot_indice_1 (dfPV_CSI, dfPQ_CSI, 'DPI', 'AC-RO','AC-RO',order=False)#limites=[0,1.3]
        plots_static.plot_indice_1 (dfPV_CSI, dfPQ_CSI, 'DPI', 'South','Sul',order=False)#limites=[0,1.3], limites=[-0.1,1.5]


        def main_plot_indice_2(dffPQgb, dffPVgb):
            regioes = df_PQ_reg['REG'].unique()
            region_map = {'Nordeste':'Northeast', 'Norte':'North', 'Sudeste-Centro-Oeste':'SE-CW', 'Sul':'South','AC-RO':'AC-RO'}
            for i in regioes:
                Indice = 'CSI_INF'
                plots_static.plot_indice_2 (dffPQgb, r'$\mathrm{DPI}_\mathrm{PQ}^\mathrm{l}$', 'DPI_(l)_PQ_' + region_map[i], region_map[i] ,i, Indice, 'VBASEKV',limites=[0,2.5])
                plots_static.plot_indice_2 (dffPVgb, r'$\mathrm{DPI}_\mathrm{PV}^\mathrm{l}$', 'DPI_(l)_PV_' + region_map[i], region_map[i] ,i, Indice, 'Gen_Type', limites=[0,2.5])
                Indice = 'CSI_SUP'
                plots_static.plot_indice_2 (dffPQgb, r'$\mathrm{DPI}_\mathrm{PQ}^\mathrm{u}$', 'DPI_(u)_PQ_' + region_map[i], region_map[i] ,i, Indice, 'VBASEKV',limites=[0,2.5])
                plots_static.plot_indice_2 (dffPVgb, r'$\mathrm{DPI}_\mathrm{PV}^\mathrm{u}$', 'DPI_(u)_PV_' + region_map[i], region_map[i] ,i, Indice, 'Gen_Type', limites=[0,2.5])

        main_plot_indice_2(dffPQgb, dffPVgb)

    if Plot_Boxplot_DPI:
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
                        plots_static.plot_boxplot(df_boxplot['MODV_PU'], df_boxplot.index, numbuses , 'BUSES', 'VOLTAGE (pu)', text = True, rotation=45, limites=[minimo,maximo])
                        
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

    if resumoIndice:

        df_busPQ_mod['BUS_ID'] = df_busPQ_mod['BUS_ID'].astype(int)
        df_busPV_mod['BUS_ID'] = df_busPV_mod['BUS_ID'].astype(int)
        df_busPV_mod[(df_busPV_mod['IndiceInf']>0.1)].groupby(by = ['REG'])['BUS_ID'].nunique().to_csv(cenario + '/Critical_infPVbuses.txt', header=True, index=True)
        df_busPQ_mod[(df_busPQ_mod['IndiceInf']>0.1)].groupby(by = ['REG'])['BUS_ID'].nunique().to_csv(cenario + '/Critical_infPQbuses.txt', header=True, index=True)
        df_busPV_mod[(df_busPV_mod['IndiceInf']>0.1)].groupby(by = ['REG'])['BUS_ID'].unique().to_csv(cenario + '/Critical_infPVbuses_bus.txt', header=True, index=True)
        df_busPQ_mod[(df_busPQ_mod['IndiceInf']>0.1)].groupby(by = ['REG'])['BUS_ID'].unique().to_csv(cenario + '/Critical_infPQbuses_bus.txt', header=True, index=True)

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

        Df_seguros_PQ = Df_IndiceT2.loc['DPI_PQ'][((Df_IndiceT2.loc['DPI_PQ']['OV DPI']==0) & (Df_IndiceT2.loc['DPI_PQ']['UV DPI']==0))][['OV condition', 'OV DPI']]
        Df_seguros_PV = Df_IndiceT2.loc['DPI_PV'][((Df_IndiceT2.loc['DPI_PV']['OV DPI']==0) & (Df_IndiceT2.loc['DPI_PV']['UV DPI']==0))][['OV condition', 'OV DPI']]
        Df_PQ_OV = Df_IndiceT2.loc['DPI_PQ'][~((Df_IndiceT2.loc['DPI_PQ']['OV DPI']==0) & (Df_IndiceT2.loc['DPI_PQ']['UV DPI']>0))].sort_values('OV DPI', ascending=False)[['OV condition', 'OV DPI']]
        Df_PQ_UV = Df_IndiceT2.loc['DPI_PQ'][~((Df_IndiceT2.loc['DPI_PQ']['UV DPI']==0) & (Df_IndiceT2.loc['DPI_PQ']['OV DPI']>0))].sort_values('UV DPI', ascending=False)[['UV condition', 'UV DPI']]
        Df_PV_OV = Df_IndiceT2.loc['DPI_PV'][~((Df_IndiceT2.loc['DPI_PV']['OV DPI']==0) & (Df_IndiceT2.loc['DPI_PV']['UV DPI']>0))].sort_values('OV DPI', ascending=False)[['OV condition', 'OV DPI']]
        Df_PV_UV = Df_IndiceT2.loc['DPI_PV'][~((Df_IndiceT2.loc['DPI_PV']['UV DPI']==0) & (Df_IndiceT2.loc['DPI_PV']['OV DPI']>0))].sort_values('UV DPI', ascending=False)[['UV condition', 'UV DPI']]
        Df_IndiceT2.to_csv(cenario+'/Indice.csv')
        Df_PV_OV.to_csv(cenario+'/IndicePV_OV.csv')
        Df_PV_UV.to_csv(cenario+'/IndicePV_UV.csv')
        Df_PQ_OV.to_csv(cenario+'/IndicePQ_OV.csv')
        Df_PQ_UV.to_csv(cenario+'/IndicePQ_UV.csv')

        path_script_org = cenario + "/RelatorioIndice.txt"
        numeroPO = len(set(Df_IndiceT2.index.to_frame()[['Dia','Hora']].apply(tuple, axis=1).values))

        with open(path_script_org, 'w') as f:
            f.write('O numero de pontos de operação analisados são: ' + str(numeroPO) + '\n')
            f.write('=============================\n Informação Barras PQ:\n=============================\n')
            regions = Df_PQ_OV.reset_index('REG')['REG'].unique()
            for reg in regions:
                f.write('- Sobretensão ' + reg +'\n')
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
            f.write('=============================\n Informação Barras PV:\n=============================\n')
            for reg in regions:
                f.write('- Sobretensão ' + reg +'\n')
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


# Guarda el tiempo de finalización
end_time = time.time()
# Calcula la diferencia de tiempo
execution_time = end_time - start_time
print("Tiempo de ejecución:", execution_time, "segundos")