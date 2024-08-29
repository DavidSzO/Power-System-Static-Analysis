import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import colorcet as cc

class PlotterMainStatic():

    def __init__(self, path, selectplot, variable_y, variable_x = None, variable_z = None ,df=None):
        
        self.pathtosave = ''
        self.path = path #Toma como parametro o path do arquivo
        self.selectplot = selectplot #Toma como parametro o tipo de plot como string: "Lineplot" e "Boxplot"
        self.variable_y = variable_y
        self.variable_x = variable_x
        self.variable_z = variable_z

        plt.rcParams["font.family"] = "Times New Roman"
        self.paletadcolor = sns.color_palette()
        self.intense_palette = sns.color_palette(cc.glasbey, n_colors=12)  # You can adjust the number of colors as

        self.readData()
        self.handeldataplot()

    def readData (self):
        self.df = pd.read_csv(self.path)
        self.selectdata = self.path.split('/')[-1].replace('.csv', '')

    def handeldataplot (self):

        if 'Df_ger' or 'Df_nt' in self.selectdata:

            def boxplot_barrasGeracao(Df_VF, groupflag):

                if groupflag:
                    grouped_Reg = Df_VF.groupby('REG').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                    data_Reg = [grouped_Reg.at[region, 'MODV_PU'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                    Nbarras_Reg = [grouped_Reg.at[region, 'BUS_ID'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                    labels_Reg = ['North', 'Northeast', 'SE-CW', 'South', 'AC-RO']
                    self.plot_boxplot(data_Reg, labels_Reg, 'Bus Voltage Distribution by Region for PV Buses', 'Region', 'Voltage (pu)', text=True, nbarra=Nbarras_Reg, pathtosave=None)
                else:
                    grouped_UF = Df_VF.groupby('Gen_Type').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                    data_UF = [grouped_UF.at[gen_type, 'MODV_PU'] for gen_type in ['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO','SIN']]
                    Nbarras_UF = [grouped_UF.at[gen_type, 'BUS_ID'] for gen_type in ['UHE', 'UTE', 'PCH', 'EOL', 'UFV', 'BIO','SIN']]
                    labels_UF = ['Hydro', 'Thermal', 'SHP', 'Wind', 'Solar', 'Bio','Sync. Comp']
                    self.plot_boxplot(data_UF, labels_UF, 'Bus Voltage Distribution by Type of Generation','Type of Generation', 'Voltage (pu)', text=True, nbarra=Nbarras_UF,pathtosave=None)

            def boxplot_barrasCarga(Df_Vfpt, groupflag):

                if groupflag:
                    grouped_Reg = Df_Vfpt.groupby('REG').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                    data_Reg = [grouped_Reg.at[region, 'MODV_PU'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                    Nbarras_Reg = [grouped_Reg.at[region, 'BUS_ID'] for region in ['Norte', 'Nordeste', 'Sudeste-Centro-Oeste', 'Sul', 'AC-RO']]
                    labels_Reg = ['North', 'Northeast', 'SE-CW', 'South', 'AC-RO']
                    self.plot_boxplot(data_Reg, labels_Reg, 'Bus Voltage Distribution by Region for PQ Buses', 'Region', 'Voltage (pu)', text=True, nbarra=Nbarras_Reg, pathtosave=None)
                else:
                    grouped_VBASEKV = Df_Vfpt.groupby('VBASEKV').agg({'BUS_ID': 'unique', 'MODV_PU': list})
                    voltage_levels = [230, 345, 440, 500, 525, 765]
                    data_VBASEKV = [grouped_VBASEKV.at[level, 'MODV_PU'] for level in voltage_levels]
                    Nbarras_VBASEKV = [grouped_VBASEKV.at[level, 'BUS_ID'] for level in voltage_levels]
                    labels_VBASEKV = ['230', '345', '440', '500', '525', '765']
                    self.plot_boxplot(data_VBASEKV, labels_VBASEKV, 'Bus Voltage Distribution by Voltage Level', 'Voltage Level (kV)', 'Voltage (pu)', text=True, nbarra=Nbarras_VBASEKV, pathtosave=None)

            # PARAMETROS A SER ESCOLHIDOS NA INTERFAZ ANTES DE PLOTAS ***
            self.groupbyreg = False  # parametro que agrupa por região se for TRUE, se for FALSE agrupa por tipo de barra
            if self.selectplot == 'Boxplot':
                if self.selectdata == 'Df_ger':
                    boxplot_barrasGeracao(self.df, self.groupbyreg) 
                else: # Caso a data venha de Df_nt
                    boxplot_barrasCarga(self.df, self.groupbyreg)               

        elif 'Df_PQ_DPI_S1' or 'Df_PV_DPI_S1' in self.selectdata:

            self.all_sameplot  = True
            self.REG = 'Norte'
            self.busgroup = 230

            if self.selectdata == 'Df_PQ_DPI_S1':
                BusGroup = 'VBASEKV'
            else:
                BusGroup = 'Gen_Type'

            y_var = self.variable_y
            df = self.df.set_index(['Dia','Hora','REG',BusGroup])

            if self.selectplot == 'Lineplot':
                if self.all_sameplot:
                    self.plot_line(df.loc[:,:, self.REG, :], y_var, 'Pontos de Operação', '(p.u)', f'{y_var} para {self.REG}', '', byClass = True, clase=BusGroup, order=False, month = False)
                else:
                    self.plot_line(df.loc[:,:,self.REG, self.busgroup], y_var, 'Pontos de Operação', '(p.u)', f'{y_var} por Região - {self.REG}/{str(self.busgroup)}', '', order=False, month = False)

            elif self.selectplot == 'Persistence Curve':
                if self.all_sameplot:
                    self.plot_line(df.loc[:,:, self.REG, :], y_var, 'Porcentagem de meias horas em um mês (%)', '(p.u)', f'{y_var} para {self.REG}', '',  byClass = True, clase=BusGroup, order=True, month = False)
                else:
                    self.plot_line(df.loc[:,:,self.REG, self.busgroup], y_var, 'Porcentagem de meias horas em um mês (%)', '(p.u)', f'{y_var} por Região - {self.REG}/{str(self.busgroup)}', '', order=True, month = False)

        elif 'Df_DPI_S2' in self.selectdata:

            self.all_sameplot  = False
            self.REG = 'Norte'
            self.BusType = 'PQ'
            y_var = self.variable_y

            df = self.df.set_index(['level_0','Dia','Hora','REG'])

            if self.selectplot == 'Lineplot':
                if self.all_sameplot:
                    self.plot_line(df.loc[f'DPI_{self.BusType}',:,:,:], y_var, 'Pontos de Operação', '(p.u)', f'{y_var} por Região', '', byClass = True, clase='REG', order=False, month = False)
                else:
                    self.plot_line(df.loc[f'DPI_{self.BusType}',:,:,self.REG], y_var, 'Pontos de Operação', '(p.u)', f'{y_var} por Região - {self.REG}', '', order=False, month = False)

            elif self.selectplot == 'Persistence Curve':
                if self.all_sameplot:
                    self.plot_line(df.loc[f'DPI_{self.BusType}',:,:,:], y_var, 'Porcentagem de meias horas em um mês (%)', '(p.u)', f'{y_var} por Região', '',  byClass = True, clase='REG', order=True, month = False)
                else:
                    self.plot_line(df.loc[f'DPI_{self.BusType}',:,:,self.REG], y_var, 'Porcentagem de meias horas em um mês (%)', '(p.u)', f'{y_var} por Região - {self.REG}', '', order=True, month = False)

        elif 'Df_DPI_S3' or 'Df_DPI_S4' in self.selectdata:
        
            y_var = self.variable_y
            df = self.df

            if self.selectplot == 'Lineplot':
                self.plot_line(df, y_var, 'Pontos de Operação', '(p.u)', f'{y_var} por Ponto de Operação', '', order=False, month = False)
            elif self.selectplot == 'Persistence Curve':
                self.plot_line(df, y_var, 'Porcentagem de meias horas em um mês (%)', '(p.u)', f'{y_var}', '', order=True, month = False)
                 
        elif '_REG_' or '_Reg' in self.selectdata:

            self.all_sameplot  = True
            self.REG = 'Norte'
            y_var = self.variable_y

            df = self.df.set_index(['Dia','Hora','REG'])

            if self.selectplot == 'Lineplot':
                if self.all_sameplot:
                    self.plot_line(df, y_var, 'Pontos de Operação', '(MW)', f'{y_var} por Região', '', byClass = True, clase='REG', order=False, month = False)
                else:
                    self.plot_line(df.loc[:,:,self.REG], y_var, 'Pontos de Operação', '(MW)', f'{y_var} por Região - {self.REG}', '', order=False, month = False)

            elif self.selectplot == 'Persistence Curve':
                if self.all_sameplot:
                    self.plot_line(df, y_var, 'Porcentagem de meias horas em um mês (%)', '(MW)', f'{y_var} por Região', '', byClass = True, clase='REG', order=True, month = False)
                else:
                    self.plot_line(df.loc[:,:,self.REG], y_var, 'Porcentagem de meias horas em um mês (%)', '(MW)', f'{y_var} por Região - {self.REG}', '', order=True, month = False)

        elif '_PO' in self.selectdata:

            y_var = self.variable_y
            df = self.df
            
            if self.selectplot == 'Lineplot':
                self.plot_line(df, y_var, 'Pontos de Operação', '(MW)', f'{y_var} por Ponto de Operação', '', order=False, month = False)
            elif self.selectplot == 'Persistence Curve':
                self.plot_line(df, y_var, 'Porcentagem de meias horas em um mês (%)', '(MW)', f'{y_var}', '', order=True, month = False)

    def plot_line (self, df_data, column, xlbl, ylbl, title, pathsave = None, clase=None, xlimites=None, ylimites=None, order = False, byClass = False, month = False):

        fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=False)

        if byClass:
            for idx, clase in enumerate(df_data.index.get_level_values(clase).unique().values):
                if order:
                    data = df_data.loc[:, :, clase].sort_values(column, ascending=False)
                    axs.plot(data[column].values, color=self.intense_palette[idx], label=clase, lw=2.5, linestyle='-', alpha=1)
                    num = data.shape[0]
                    data_points_xaxis = 10
                    num_days = num*data_points_xaxis/100
                    axs.set_xticks([round(i * num_days) for i in range(data_points_xaxis+1)])
                    axs.set_xticklabels([f'{i*10}' for i in range(data_points_xaxis+1)], fontsize=12, rotation=0, ha='center')
                else:
                    data = df_data.loc[:, :, clase]
                    axs.plot(data['key'], data[column], color=self.intense_palette[idx], label=clase, lw=2.5, linestyle='-', alpha=1)
                    axs.xaxis.set_major_locator(plt.MaxNLocator(28))
                axs.legend(loc='upper right', fontsize=18)
        else:
            if order:
                data = df_data.sort_values(column, ascending=False)
                axs.plot(data[column].values, color=self.intense_palette[0], lw=2.5, linestyle='-', alpha=1)
                num = data.shape[0]
                data_points_xaxis = 10
                num_days = num*data_points_xaxis/100
                axs.set_xticks([round(i * num_days) for i in range(data_points_xaxis+1)])
                axs.set_xticklabels([f'{i*10}' for i in range(data_points_xaxis+1)], fontsize=12, rotation=0, ha='center')
            else:
                axs.plot(df_data['key'], df_data[column], color=self.intense_palette[0], lw=2.5, linestyle='-', alpha=1)
                axs.xaxis.set_major_locator(plt.MaxNLocator(28))

        axs.tick_params(axis='y', labelsize=24)
        axs.tick_params(axis='x', labelsize=15, rotation = 45)
        axs.set_xlabel(xlbl, fontsize=18)
        axs.set_ylabel(ylbl, fontsize=22)
        axs.set_title(title, fontsize=22)
        
        if month:
            days = ['Oct {}'.format(i+1) for i in range(28)]
            plt.xticks(ticks=[i*48 for i in range(28)], labels=days, rotation=45)
            plt.xlabel('',fontsize=10)

        if xlimites != None:
            axs.set_xlim(xlimites)
        if ylimites != None:
            axs.set_ylim(ylimites)
            
        axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
        plt.tight_layout()
        plt.show()
        # nome = pathsave
        # plt.savefig(nome, bbox_inches = 'tight')
        # plt.close()

    def plot_boxplot(self, data, labels, title, xlabel, ylabel, pathtosave = None, limites=None, vert = True, rotation = 0, text = True, nbarra = None):
        
        markerfacecolor = dict(markerfacecolor='gray', marker='o')  # El diccionario que define el color y marcador

        fig1, axs = plt.subplots(figsize=(25, 20))
        if vert:
            axs.boxplot(data, flierprops=markerfacecolor)
            axs.tick_params(bottom=False, left=False)
            axs.set_axisbelow(True)

            if text:
                for i, data_item in enumerate(data):
                    num_muestras = len(data_item)
                    if num_muestras > 0:

                        Q1 = np.quantile(data_item, q=0.25) 
                        Q3 = np.quantile(data_item, q=0.75)
                        max_value = np.quantile(data_item, q=1)
                        min_value = np.quantile(data_item, q=0)
                        IQR = Q3 - Q1
                        lower_whisker = max(min_value, Q1 - 1.5 * IQR)
                        upper_whisker = min(max_value, Q3 + 1.5 * IQR)

                        try:
                            axs.text(i+1, np.max(data_item)+0.005, f'Buses = {nbarra[i].size} ', ha='center', va='bottom',size=15)
                            axs.text(i+1, np.median(data_item), f' {np.mean(data_item):.3f} ', ha='center', va='bottom',size=15)
                            axs.text(i+1.35, lower_whisker, f' {lower_whisker:.3f} ', ha='center', va='bottom',size=15)
                            axs.text(i+1.35, upper_whisker, f' {upper_whisker:.3f} ', ha='center', va='bottom',size=15)
                        except:
                            axs.text(i+1.35, np.mean(data_item), f'{num_muestras}', ha='center', va='bottom',size=25)
            if limites != None:
                axs.set_ylim(limites)
            plt.xticks(range(1, len(labels)+1), labels, fontsize=15)
            
        else:
            axs.boxplot(data, vert=False, flierprops=markerfacecolor)
            if limites != None:
                axs.set_xlim(limites)
                axs.set_xticks(np.linspace(limites[0],limites[1],20))
            if text:
                    axs.text(np.median(data), 1.1 , f' {np.mean(data):.3f} ', ha='center', va='bottom',size=18)

        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel(ylabel, fontsize=22)
        plt.title(title, fontsize=25)
        plt.xticks(fontsize=20, rotation = rotation)
        plt.yticks(fontsize=25)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        axs.xaxis.grid(False)
        plt.show()
        # nome = pathtosave + title + '.png'
        # plt.savefig(nome, bbox_inches = 'tight')
        # if self.svg:
        #     nome = pathtosave + title + '.svg'
        #     plt.savefig(nome)
        # plt.close()



PlotterMainStatic('C:/Users/David/OneDrive/Documents/FERV_documentos/RESULTS/V1A1F2_rev5/StaticAnalysis/Data/Geral/Df_ger.csv','Boxplot', 'CSI_INF')