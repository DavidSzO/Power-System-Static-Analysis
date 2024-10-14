import numpy as np
import pandas as pd

class ComputeDPI():

    def __init__(self, df_Final_nt, df_Final_ger, ts, tb, p_norm = None, p_inf = False, NBcv = False):

        # Limits of ALERT and Insecurity
        self.limitAlertaPQ = {138: [0.95,1.05],230: [0.95,1.05],345: [0.95,1.05], 440: [0.95,1.046], 500: [1,1.10],525: [0.95,1.05],765: [0.90,1.046],}
        self.limitSecurityPQ = {138: [0.9,1.1],230: [0.9,1.1],345: [0.9,1.1], 440: [0.9,1.1], 500: [0.95,1.15],525: [0.9,1.1],765: [0.89,1.1],}
        self.limitAlertaPV = {'UHE': [0.95,1.05],'PCH': [0.95,1.05], 'UTE': [0.95,1.05], 'EOL': [0.95,1.05], 'UFV': [0.95,1.05], 'BIO': [0.95,1.05], 'SIN':[0.95,1.05]}
        self.limitSecurityPV = {'UHE': [0.9,1.1],'PCH': [0.9,1.1], 'UTE': [0.9,1.1], 'EOL': [0.9,1.1], 'UFV': [0.9,1.1], 'BIO': [0.9,1.1], 'SIN':[0.9,1.1]}

         # List of regions and states
        self.regioes = df_Final_nt['REG'].unique()
        self.estados = ['AC', 'RO', 'AM', 'AP', 'PA', 'TO', 'MA', 'AL', 'BA', 'CE', 'PB', 'PE', 'PI', 'RN', 'SE', 'DF', 'GO', 'MT', 'MS', 'ES', 'MG', 'RJ', 'SP', 'PR', 'RS', 'SC']
        # self.estados = ['AC', 'RO', 'AM', 'AP', 'PA', 'TO', 'MA', 'AL', 'BA', 'CE', 'PB', 'PI', 'DF', 'GO', 'MT', 'MS', 'ES', 'MG', 'RJ', 'SP', 'PR', 'RS', 'SC']

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
                                # grouped1.loc[:, 'CSI_INF_POND'] = grouped1['CSI_INF']*grouped1['BUS_ID']
                                # grouped1.loc[:, 'CSI_SUP_POND'] = grouped1['CSI_SUP']*grouped1['BUS_ID']
                            else:
                                grouped1.loc[:, 'CSI_INF_POND'] = grouped1['CSI_INF']*num_nonzeros_inf['BUS_ID']*(NT/500)
                                grouped1.loc[:, 'CSI_SUP_POND'] = grouped1['CSI_SUP']*num_nonzeros_sup['BUS_ID']*(NT/500)
                                # grouped1.loc[:, 'CSI_INF_POND'] = grouped1['CSI_INF']*num_nonzeros_inf['BUS_ID']
                                # grouped1.loc[:, 'CSI_SUP_POND'] = grouped1['CSI_SUP']*num_nonzeros_sup['BUS_ID']
                            
                            # grouped1.loc[:, 'CSI_INF_POND'] = grouped1['CSI_INF']*(NT/500)
                            # grouped1.loc[:, 'CSI_SUP_POND'] = grouped1['CSI_SUP']*(NT/500)

                            df_1.append(grouped1.reset_index())
                except KeyError:
                    print('ERROR: No existe la region')
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

                            # grouped2.loc[:, 'CSI_INF_POND'] = grouped2['CSI_INF']
                            # grouped2.loc[:, 'CSI_SUP_POND'] = grouped2['CSI_SUP']

                            df_2.append(grouped2.reset_index())
                except KeyError:
                    print('ERROR: No existe la region')
                    pass
                 
            self.n =n 
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
        df_Final_ger.loc[df_Final_ger['Gen_Type']=='UNE','Gen_Type'] = 'UTE'
        df_gerbarNGF = df_Final_ger[df_Final_ger['Gen_Type'].isin(['UHE', 'PCH', 'UTE', 'EOL', 'UFV', 'BIO', 'SIN'])]
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
        # dfPQ_CSI = df_PQ_reg.groupby(['Dia' , 'Hora', 'REG']).agg({'BUS_ID':'count', 'CSI_INF_POND': 'sum', 'CSI_SUP_POND': 'sum'}) # solo para teste
        dfPQ_CSI.loc[:, 'CSI_INF_FINAL'] = dfPQ_CSI['CSI_INF_POND']/dfPQ_CSI['BUS_ID']
        dfPQ_CSI.loc[:, 'CSI_SUP_FINAL'] = dfPQ_CSI['CSI_SUP_POND']/dfPQ_CSI['BUS_ID']
        self.dfPQ_CSI = dfPQ_CSI

        dfPV_CSI = df_PV_reg.groupby(['Dia' , 'Hora', 'REG']).agg({'BUS_ID':'sum', 'CSI_INF_POND': 'sum', 'CSI_SUP_POND': 'sum'})
        # dfPV_CSI = df_PV_reg.groupby(['Dia' , 'Hora', 'REG']).agg({'BUS_ID':'count', 'CSI_INF_POND': 'sum', 'CSI_SUP_POND': 'sum'}) # solo para teste
        dfPV_CSI.loc[:, 'CSI_INF_FINAL'] = dfPV_CSI['CSI_INF_POND']/dfPV_CSI['BUS_ID']
        dfPV_CSI.loc[:, 'CSI_SUP_FINAL'] = dfPV_CSI['CSI_SUP_POND']/dfPV_CSI['BUS_ID']
        self.dfPV_CSI = dfPV_CSI
                
