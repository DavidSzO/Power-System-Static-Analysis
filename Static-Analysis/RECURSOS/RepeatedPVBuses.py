import pandas as pd

class RepeatedPVBuses():
    def __init__(self,  df):
        
        self.df  = df
        self.getUniqueBus()

    def getUniqueBus(self): #Recupera los datos de archivos .csv previamente guardados

        # [['BUS_ID', 'BUS_NAME', 'VBASEKV', 'TP', 'ARE', 'MODV_PU','ANGV_DEG']]
        df_modified_BUS = self.df[0].groupby(['BUS_ID']).agg({
        'BUS_NAME': 'first',
        'VBASEKV': 'first',
        'TP': 'first',
        'ARE': 'first',
        'MODV_PU': 'first',
        'ANGV_DEG': 'first',
        }).reset_index()

        # [['BUS_ID', 'PG_MW', 'QG_MVAR','QMX_MVAR', 'QMN_MVAR','PMAX_MW', 'PMIN_MW']]
        df_modified_GEN = self.df[1].groupby(['BUS_ID']).agg({
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
        df_GEN = self.df[1].groupby(['BUS_ID']).agg({'PG_MW': 'count'}).reset_index().rename(columns={'PG_MW': 'Ger_Units'})
        df_modified_GEN = pd.merge(df_modified_GEN, df_GEN,on='BUS_ID', how='left')
        

        # [['BUS_ID','PL_MW', 'QL_MVAR']]
        df_modified_LOAD = self.df[2].groupby(['BUS_ID']).agg({
        'PL_MW': 'sum',
        'QL_MVAR': 'sum',
        }).reset_index()

        self.DF_grouped_BUS = df_modified_BUS
        self.DF_grouped_GEN = df_modified_GEN
        self.DF_grouped_LOAD = df_modified_LOAD
        
        

