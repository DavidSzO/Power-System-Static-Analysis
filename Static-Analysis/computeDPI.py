import numpy as np
import pandas as pd

class computeDPI():

    def __init__(self, df_Final_nt, df_Final_ger, ts, tb, p_norm = None, p_inf = False, NBcv = False):

        # Limits of ALERT and Insecurity
        self.limitAlertaPQ = {138: [0.95,1.05],230: [0.95,1.05],345: [0.95,1.05], 440: [0.95,1.046], 500: [1,1.10],525: [0.95,1.05],765: [0.90,1.046],}
        self.limitSecurityPQ = {138: [0.9,1.1],230: [0.9,1.1],345: [0.9,1.1], 440: [0.9,1.1], 500: [0.95,1.15],525: [0.9,1.1],765: [0.89,1.1],}
        self.limitAlertaPV = {'UHE': [0.95,1.05],'PCH': [0.95,1.05], 'UTE': [0.95,1.05], 'EOL': [0.95,1.05], 'UFV': [0.95,1.05], 'BIO': [0.95,1.05], 'SIN':[0.95,1.05]}
        self.limitSecurityPV = {'UHE': [0.9,1.1],'PCH': [0.9,1.1], 'UTE': [0.9,1.1], 'EOL': [0.9,1.1], 'UFV': [0.9,1.1], 'BIO': [0.9,1.1], 'SIN':[0.9,1.1]}

        # self.limitAlertaPQ = {138: [0.90,1.1],230: [0.90,1.1],345: [0.90,1.1], 440: [0.90,1.1], 500: [0.90,1.1],525: [0.90,1.1],765: [0.90,1.1],}
        # self.limitSecurityPQ = {138: [0.8,1.2],230: [0.8,1.2],345: [0.8,1.2], 440: [0.8,1.2], 500: [0.8,1.2],525: [0.8,1.2],765: [0.8,1.2],}
        # self.limitAlertaPV = {'UHE': [0.90,1.1],'PCH': [0.90,1.1], 'UTE': [0.90,1.1], 'EOL': [0.90,1.1], 'UFV': [0.90,1.1], 'BIO': [0.90,1.1]}
        # self.limitSecurityPV = {'UHE': [0.8,1.2],'PCH': [0.8,1.2], 'UTE': [0.8,1.2], 'EOL': [0.8,1.2], 'UFV': [0.8,1.2], 'BIO': [0.8,1.2]}

         # List of regions and states
        self.regioes = df_Final_nt['REG'].unique()
        self.estados = ['AC', 'RO', 'AM', 'AP', 'PA', 'TO', 'MA', 'AL', 'BA', 'CE', 'PB', 'PE', 'PI', 'RN', 'SE', 'DF', 'GO', 'MT', 'MS', 'ES', 'MG', 'RJ', 'SP', 'PR', 'RS', 'SC']

        self.get_computeindex(df_Final_nt, df_Final_ger,ts,tb, p_norm, p_inf, NBcv)

    def get_computeindex(self, df_Final_nt, df_Final_ger,ts,tb, p_norm, p_inf, NBcv):
        
        def compute_index_CSI(valor,securelimit_l,securelimit_u,alarm_l,alarm_u):
             
            d_u = ((valor - alarm_u) if valor > alarm_u else 0)
            d_l = ((alarm_l - valor)  if valor < alarm_l else 0)
            g_u = (securelimit_u - alarm_u)
            g_l =  (alarm_l - securelimit_l)
            dl_gl = np.divide(d_l,g_l)
            du_gu = np.divide(d_u,g_u)
            return dl_gl, du_gu

        def Composite_sec_index_REG(data1, data2):
            # Iterate over unique regions and NT values
            df_1 = []
            df_2 = []
            dff1 = []
            dff2 = []
            n_maior = 1
            for regiao in data1['REG'].unique():
                try:
                    self.PQ_numbarras_REG[regiao]
                    for lista in self.PQ_numbarras_REG[regiao]:
                            if p_norm == None:
                                n = round(np.log(np.divide(1,lista[1]*tb))/(2*np.log(ts)))
                            else:
                                n=p_norm

                            if n>n_maior:
                                n_maior =n
                            if n<2:
                                n=2
                            elif n>20:
                                n=20

                            NT  = lista[0]
                            securelimit_l = self.limitSecurityPQ[NT][0]
                            securelimit_u = self.limitSecurityPQ[NT][1]
                            alarm_l = self.limitAlertaPQ[NT][0]
                            alarm_u = self.limitAlertaPQ[NT][1]

                            condition1_pq = (data1['REG'] == regiao) & (data1['VBASEKV'] == NT)
                            data1.loc[condition1_pq,'IndiceInf'], data1.loc[condition1_pq,'IndiceSup'] = zip(*data1.loc[condition1_pq,'MODV_PU'].apply(lambda x: compute_index_CSI(x, securelimit_l,securelimit_u,alarm_l,alarm_u)))

                            data1.loc[condition1_pq, 'IndiceInf_2n'] = data1.loc[condition1_pq,'IndiceInf'].pow(2*n) 
                            data1.loc[condition1_pq, 'IndiceSup_2n'] = data1.loc[condition1_pq,'IndiceSup'].pow(2*n) 
                            # dff1.append(data1.loc[condition1_pq, ['BUS_ID','BUS_NAME','ARE','VBASEKV','MODV_PU','ANGV_DEG','IndiceInf','IndiceSup','Dia','Hora', 'Gen_Type','REG', 'U_FED']])
                            dff1.append(data1.loc[condition1_pq, :])
                            
                            data1_ = data1.loc[condition1_pq, :]  # filtara el dataframe para una region y un nivel de tension en especifico
                            num_nonzeros_inf = data1_[data1_['IndiceInf']>0].groupby(['Dia' , 'Hora']).agg({'BUS_ID':'count'})
                            num_nonzeros_sup = data1_[data1_['IndiceSup']>0].groupby(['Dia' , 'Hora']).agg({'BUS_ID':'count'})

                            if p_inf == False:
                                grouped1 = data1.loc[condition1_pq, :].groupby(['Dia' , 'Hora']).agg({'REG': 'first', 'VBASEKV':'first', 'IndiceSup_2n':'sum', 'IndiceInf_2n': 'sum', 'BUS_ID':'count'})
                                grouped1.loc[:, 'CSI_INF'] = np.power((grouped1['IndiceInf_2n']), 1/(2*n))
                                grouped1.loc[:, 'CSI_SUP'] = np.power((grouped1['IndiceSup_2n']), 1/(2*n))
                            else:
                                grouped1 = data1.loc[condition1_pq, :].groupby(['Dia' , 'Hora']).agg({'REG': 'first', 'VBASEKV':'first', 'IndiceSup':'max', 'IndiceInf': 'max', 'BUS_ID':'count'})
                                grouped1.loc[:, 'CSI_INF'] = grouped1.loc[:, 'IndiceInf'] 
                                grouped1.loc[:, 'CSI_SUP'] = grouped1.loc[:, 'IndiceSup'] 
                            

                            if NBcv == False:
                                grouped1.loc[:, 'CSI_INF_POND'] = grouped1['CSI_INF']*grouped1['BUS_ID']*(NT/500)
                                grouped1.loc[:, 'CSI_SUP_POND'] = grouped1['CSI_SUP']*grouped1['BUS_ID']*(NT/500)
                            else:
                                grouped1.loc[:, 'CSI_INF_POND'] = grouped1['CSI_INF']*num_nonzeros_inf['BUS_ID']*(NT/500)
                                grouped1.loc[:, 'CSI_SUP_POND'] = grouped1['CSI_SUP']*num_nonzeros_sup['BUS_ID']*(NT/500)

                            df_1.append(grouped1.reset_index())
                except KeyError:
                    pass 
                
                try:
                    self.PV_numbarras_REG[regiao]
                    for lista in self.PV_numbarras_REG[regiao]:
                            if p_norm == None:
                                n = round(np.log(np.divide(1,lista[1]*tb))/(2*np.log(ts)))
                            else:
                                n=p_norm

                            if n>n_maior:
                                n_maior =n
                            if n<2:
                                n=2
                            elif n>20:
                                n=20
                                
                            TG  = lista[0]
                            securelimit_l = self.limitSecurityPV[TG][0]
                            securelimit_u = self.limitSecurityPV[TG][1]
                            alarm_l = self.limitAlertaPV[TG][0]
                            alarm_u = self.limitAlertaPV[TG][1]

                            condition1_pv  = (data2['REG'] == regiao) & (data2['Gen_Type'] == TG) #barras PV
                            data2.loc[condition1_pv,'IndiceInf'], data2.loc[condition1_pv,'IndiceSup'] = zip(*data2.loc[condition1_pv,'MODV_PU'].apply(lambda x: compute_index_CSI(x, securelimit_l,securelimit_u,alarm_l,alarm_u)))

                            data2.loc[condition1_pv, 'IndiceInf_2n'] = data2.loc[condition1_pv,'IndiceInf'].pow(2*n)
                            data2.loc[condition1_pv, 'IndiceSup_2n'] = data2.loc[condition1_pv,'IndiceSup'].pow(2*n) 
                            # dff2.append(data2.loc[condition1_pv, ['BUS_ID','BUS_NAME','ARE','VBASEKV','MODV_PU','ANGV_DEG','IndiceInf','IndiceSup','Dia','Hora', 'Gen_Type','REG', 'U_FED']])
                            dff2.append(data2.loc[condition1_pv,:])

                            data2_ = data2.loc[condition1_pv, :]  # filtara el dataframe para una region y un nivel de tension en especifico
                            num_nonzeros_inf = data2_[data2_['IndiceInf']>0].groupby(['Dia' , 'Hora']).agg({'BUS_ID':'count'})
                            num_nonzeros_sup = data2_[data2_['IndiceSup']>0].groupby(['Dia' , 'Hora']).agg({'BUS_ID':'count'})


                            if p_inf == False:
                                grouped2 = data2.loc[condition1_pv, :].groupby(['Dia' , 'Hora']).agg({'REG': 'first', 'Gen_Type':'first', 'IndiceSup_2n':'sum', 'IndiceInf_2n': 'sum', 'BUS_ID':'count'})
                                grouped2.loc[:, 'CSI_INF'] = np.power((grouped2['IndiceInf_2n']), 1/(2*n))
                                grouped2.loc[:, 'CSI_SUP'] = np.power((grouped2['IndiceSup_2n']), 1/(2*n))
                            else:
                                grouped2 = data2.loc[condition1_pv, :].groupby(['Dia' , 'Hora']).agg({'REG': 'first', 'Gen_Type':'first', 'IndiceSup':'max', 'IndiceInf': 'max', 'BUS_ID':'count'})
                                grouped2.loc[:, 'CSI_INF'] = grouped2.loc[:, 'IndiceInf'] 
                                grouped2.loc[:, 'CSI_SUP'] = grouped2.loc[:, 'IndiceSup'] 

                            if NBcv == False:
                                grouped2.loc[:, 'CSI_INF_POND'] = grouped2['CSI_INF']*grouped2['BUS_ID']
                                grouped2.loc[:, 'CSI_SUP_POND'] = grouped2['CSI_SUP']*grouped2['BUS_ID']
                            else:
                                grouped2.loc[:, 'CSI_INF_POND'] = grouped2['CSI_INF']*num_nonzeros_inf['BUS_ID']
                                grouped2.loc[:, 'CSI_SUP_POND'] = grouped2['CSI_SUP']*num_nonzeros_sup['BUS_ID']

                            df_2.append(grouped2.reset_index())
                except KeyError:
                    pass
                 
            self.n = n 
            df_C1 = pd.concat(df_1, axis=0, sort=False) #BARRAS PQ
            df_C2 = pd.concat(df_2, axis=0, sort=False) #BARRAS PV
            df_bus1 = pd.concat(dff1, axis=0, sort=False) #BARRAS PQ
            df_bus2 = pd.concat(dff2, axis=0, sort=False) #BARRAS PV       
            self.n_maior = n_maior

            return df_C1, df_C2, df_bus1, df_bus2


        def get_number_of_buses(grouped_data, keys):
            data_dict = {}
            for k in keys:
                lista = []
                try:
                    index = grouped_data.loc[GROUPEDntREG.index[0][:2]].loc[k].index.unique()
                    for i in index.values:
                        lista.append([i, grouped_data.loc[grouped_data.index[0][:2]].loc[k,i]['BUS_ID'].size])
                    data_dict[k] = lista
                except KeyError:
                    print('error', k)
                    pass
            return data_dict
        

        # Filter data
        df_ntbarNTF = df_Final_nt[df_Final_nt['VBASEKV'].isin([230, 345, 440, 500, 525, 765])]
        # df_ntbarNTF = df_Final_nt[df_Final_nt['VBASEKV']==230]
        df_Final_ger.loc[df_Final_ger['Gen_Type']=='UNE','Gen_Type'] = 'UTE'
        # df_gerbarNGF = df_Final_ger[df_Final_ger['Gen_Type'].isin(['UHE', 'PCH', 'UTE', 'EOL', 'UFV', 'BIO'])]
        df_gerbarNGF = df_Final_ger[df_Final_ger['Gen_Type'].isin(['UHE', 'PCH', 'UTE', 'EOL', 'UFV', 'BIO', 'SIN'])]
        # df_gerbarNGF = df_Final_ger[df_Final_ger['Gen_Type']=='UHE']
        # Group data
        GROUPEDntREG = df_ntbarNTF.groupby(by=['Dia', 'Hora', 'REG', 'VBASEKV']).agg({'BUS_ID': 'unique'})
        GROUPEDgerREG = df_gerbarNGF.groupby(by=['Dia', 'Hora', 'REG', 'Gen_Type']).agg({'BUS_ID': 'unique'})   
        # Create dictionaries
        PQ_numbarras_REG = get_number_of_buses(GROUPEDntREG, self.regioes)
        PV_numbarras_REG = get_number_of_buses(GROUPEDgerREG, self.regioes)

        self.PQ_numbarras_REG = PQ_numbarras_REG
        self.PV_numbarras_REG = PV_numbarras_REG
        self.n_maior = 1

        df_PQ_reg, df_PV_reg, df_busPQ, df_busPV = Composite_sec_index_REG(df_Final_nt, df_Final_ger)

        self.df_busPQ = df_busPQ
        self.df_busPV = df_busPV
        self.df_PQ_reg = df_PQ_reg
        self.df_PV_reg = df_PV_reg

        dfPQ_CSI = df_PQ_reg.groupby(['Dia' , 'Hora', 'REG']).agg({'BUS_ID':'sum', 'CSI_INF_POND': 'sum', 'CSI_SUP_POND': 'sum'})
        dfPQ_CSI.loc[:, 'CSI_INF_FINAL'] = dfPQ_CSI['CSI_INF_POND']/dfPQ_CSI['BUS_ID']
        dfPQ_CSI.loc[:, 'CSI_SUP_FINAL'] = dfPQ_CSI['CSI_SUP_POND']/dfPQ_CSI['BUS_ID']
        self.dfPQ_CSI = dfPQ_CSI

        dfPV_CSI = df_PV_reg.groupby(['Dia' , 'Hora', 'REG']).agg({'BUS_ID':'sum', 'CSI_INF_POND': 'sum', 'CSI_SUP_POND': 'sum'})
        dfPV_CSI.loc[:, 'CSI_INF_FINAL'] = dfPV_CSI['CSI_INF_POND']/dfPV_CSI['BUS_ID']
        dfPV_CSI.loc[:, 'CSI_SUP_FINAL'] = dfPV_CSI['CSI_SUP_POND']/dfPV_CSI['BUS_ID']
        self.dfPV_CSI = dfPV_CSI
                


# df_Final_nt = pd.read_csv('V1A1F3_RESP_FNS_lim_rev1_2026/Df_nt.csv')
# df_Final_ger = pd.read_csv('V1A1F3_RESP_FNS_lim_rev1_2026/Df_ger.csv')
# # =====================================================================

# df_Final_nt = df_Final_nt[(df_Final_nt['REG']=='Sudeste-Centro-Oeste') & (df_Final_nt['VBASEKV']==500) & (df_Final_nt['Dia']==1) & (df_Final_nt['Hora']=='00-00')].reset_index(drop=True)
# df_Final_ger = df_Final_ger[(df_Final_ger['REG']=='Sudeste-Centro-Oeste') & (df_Final_ger['Gen_Type']=='UHE') & (df_Final_ger['Dia']==1) & (df_Final_ger['Hora']=='00-00')].reset_index(drop=True)

# df_Final_nt['MODV_PU'] = 1.05
# cantidad_inseguro = 20
# df_Final_nt.loc[:cantidad_inseguro-1, 'MODV_PU'] = 1.124

# print('CALCULO DO DPI para todos os Cenários:')
# ts = 0.8
# tb = 1
# VVI = computeDPI(df_Final_nt, df_Final_ger, ts, tb, p_norm = 2, p_inf = False, NBcv = True)
# dfPQ_CSI = VVI.dfPQ_CSI
# df_PQ_reg = VVI.df_PQ_reg
# df_busPQ = VVI.df_busPQ
# n_maior = VVI.n_maior

# dfPQ_CSI = dfPQ_CSI.groupby(['Dia' , 'Hora', 'REG']).first()
# dffPQgb = df_PQ_reg.groupby(by=['Dia','Hora','REG','VBASEKV']).agg({'CSI_INF':'first','CSI_SUP':'first'})

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.rcParams["font.family"] = "Times New Roman"
# paletadcolor = sns.color_palette()

# def plot_indice_2 (df, eje_y, name ,title, regiao, INDICE, GB, limites=None, order = True):

#     fig, axs = plt.subplots(nrows=1, figsize=(10, 6))
#     labelG = {'BIO': 'Bio', 'EOL': 'Wind', 'PCH': 'SHP','UFV': 'Solar', 'UHE': 'Hydro','UTE': 'Thermal', 'SIN': 'Synchronous C.'}
#     if GB=='Gen_Type':
#         colores = {'BIO': paletadcolor[4], 'EOL': paletadcolor[0], 'PCH': paletadcolor[3],'UFV': paletadcolor[2], 'UHE': paletadcolor[5],'UTE': paletadcolor[1], 'SIN': paletadcolor[6]}
#     else:
#         colores = {230: paletadcolor[4], 345: paletadcolor[0], 440: paletadcolor[3],500: paletadcolor[2], 525: paletadcolor[5], 765: paletadcolor[1]}
#     data = df.loc[:, :, regiao]
#     Busgroup = np.array(data.reset_index(GB)[GB].unique())
#     print(Busgroup)
#     for idx, G_bus in enumerate(Busgroup):
#         if order:
#             data_ = df.loc[:, :, regiao, G_bus][INDICE]
#         else:
#             data_ = df.loc[:, :, regiao, G_bus].sort_values(INDICE, ascending=False)[INDICE]
#         if GB=='Gen_Type':
#             label = labelG[G_bus]
#         else:
#             label = G_bus
#         # axs.plot(data_.values, color=colores[G_bus], label= label,lw=2)
#         axs.bar(str(G_bus), data_.values, color=colores[G_bus], label=label, linewidth=2)

#     axs.legend(loc='best', fontsize=14)
    
#     axs.set_xlabel('Operating points', fontsize=23)
#     axs.set_ylabel(eje_y, fontsize=20)
#     axs.set_title(title, fontsize=25)
#     if limites != None:
#         axs.set_ylim(limites)
#     axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
#     plt.show()

# def main_plot_indice_2(dffPQgb):
#     regioes = df_PQ_reg['REG'].unique()
#     region_map = {'Nordeste':'Northeast', 'Norte':'North', 'Sudeste-Centro-Oeste':'SE-CW', 'Sul':'South','AC-RO':'AC-RO'}
#     for i in regioes:
#         Indice = 'CSI_INF'
#         plot_indice_2 (dffPQgb, r'$\mathrm{DPI}_\mathrm{PQ}^\mathrm{l}$', 'DPI_(l)_PQ_' + region_map[i], region_map[i] ,i, Indice, 'VBASEKV', limites=[0,1.7])
#         Indice = 'CSI_SUP'
#         plot_indice_2 (dffPQgb, r'$\mathrm{DPI}_\mathrm{PQ}^\mathrm{u}$', 'DPI_(u)_PQ_' + region_map[i], region_map[i] ,i, Indice, 'VBASEKV',)

# main_plot_indice_2(dffPQgb)
