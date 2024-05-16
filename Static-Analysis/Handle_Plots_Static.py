import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd


class Plots_Static():

    def __init__(self, cenario, svg = False) -> None:
        
        self.cenario = cenario
        self.svg = svg
        plt.rcParams["font.family"] = "Times New Roman"
        self.paletadcolor = sns.color_palette()

    ## Definição de Funções Gerais =========================================================================================

    def plot_boxplot(self, data, labels, title, xlabel, ylabel, limites=None, vert = True, rotation = 0, text = True, nbarra = None):
        
        markerfacecolor = dict(markerfacecolor='gray', marker='o')  # El diccionario que define el color y marcador
        fig1, axs = plt.subplots(figsize=(25, 20))
        if vert:
            axs.boxplot(data, flierprops=markerfacecolor)
            
            # axs.spines['top'].set_visible(False)
            # axs.spines['right'].set_visible(False)
            # axs.spines['left'].set_visible(False)
            # axs.spines['bottom'].set_color('#DDDDDD')

            axs.tick_params(bottom=False, left=False)
            axs.set_axisbelow(True)

            if text:
                for i, data_item in enumerate(data):
                    num_muestras = len(data_item)
                    if num_muestras > 0: 

                        try:
                            axs.text(i+1, np.max(data_item)+0.005, f'Buses = {nbarra[i].size} ', ha='center', va='bottom',size=18)
                            axs.text(i+1, np.median(data_item), f' {np.mean(data_item):.3f} ', ha='center', va='bottom',size=15)
                            axs.text(i+1.4, np.quantile(data_item, q=0.25), f' {np.quantile(data_item, q=0.25):.3f} ', ha='center', va='bottom',size=15)
                            axs.text(i+1.4, np.quantile(data_item, q=0.75), f' {np.quantile(data_item, q=0.75):.3f} ', ha='center', va='bottom',size=15)
                        except:
                            axs.text(i+1.4, np.mean(data_item), f'{num_muestras}', ha='center', va='bottom',size=25)
            
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
        nome = self.cenario + '/BoxPlot/' + title + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = self.cenario + '/BoxPlot/' + title + '.svg'
            plt.savefig(nome)
        plt.close('all')

    def plot_Potencia(self, df_data, eje_y, title, limites=None):
        
        fig, axs = plt.subplots(nrows=1, figsize=(20, 10), sharex=False)

        axs.plot(df_data.values, color=sns.color_palette("Paired")[1])
            
        # axs.legend(loc='best', fontsize=12)
        # Calculate the number of data points in a day (assuming each day has 48 data points)
        data_points_per_day = 48
        # Calculate the number of days based on the length of the data
        num_days = len(df_data) // data_points_per_day
        # Set x-axis ticks and labels for each day
        axs.set_xticks([i * data_points_per_day for i in range(num_days)])
        axs.set_xticklabels([f'Day {i+1}' for i in range(num_days)], fontsize=18, rotation=45, ha='right')
        axs.tick_params(axis='y', labelsize=18)
        axs.set_xlabel('Days', fontsize=18)
        axs.set_ylabel(eje_y, fontsize=18)
        axs.set_title(title, fontsize=20)
        if limites != None:
            axs.set_xlim(limites)
        axs.grid()
        plt.tight_layout()
        nome = self.cenario + '/Potencia/' + title + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        plt.close('all')

    def plot_reserva_reg (self, df_data, eje_y, name, title, INDICE, xlimites=None,ylimites=None, order = False):

        fig, axs = plt.subplots(nrows=1, figsize=(20, 10), sharex=False)
        colores = [sns.color_palette("Paired")[1], sns.color_palette("Paired")[3], sns.color_palette("Paired")[4],sns.color_palette("Paired")[7],sns.color_palette("Paired")[9]]
        region_map = {'Nordeste':'Northeast', 'Norte':'North', 'Sudeste-Centro-Oeste':'SE-CW', 'Sul':'South','AC-RO':'AC-RO'}
        for idx, regiao in enumerate(['Norte','Nordeste','Sudeste-Centro-Oeste', 'Sul', 'AC-RO']):
            
            if order:
                data = df_data.loc[:, :, regiao].sort_values(INDICE, ascending=False)[INDICE]
            else:
                data = df_data.loc[:, :, regiao][INDICE]
            
            axs.plot(data.values, color=colores[idx], label=region_map[regiao], lw=2, linestyle='-')
            
        axs.legend(loc='upper right', fontsize=18)
        axs.tick_params(axis='y', labelsize=24)
        axs.tick_params(axis='x', labelsize=24)
        axs.set_xlabel('Pontos de Operação', fontsize=23)
        axs.set_ylabel(eje_y, fontsize=22)
        axs.set_title(title, fontsize=22)
        if xlimites != None:
            axs.set_xlim(xlimites)
        if ylimites != None:
            axs.set_ylim(ylimites)
        axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
        plt.tight_layout()
        nome = self.cenario + '/Potencia/' + name + '.png'
        plt.savefig(nome)
        plt.close('all')

    def plot_Intercambio (self, df_AC, df_DC , eje_y, title, COL_AC, COL_DC, Ylimites=None, Xlimites=  None):

        fig, axs = plt.subplots(nrows=1, figsize=(16, 7))
        colores1 = [self.paletadcolor[0], self.paletadcolor[1], self.paletadcolor[2],self.paletadcolor[3],self.paletadcolor[4], self.paletadcolor[5]]
        colores2 = [self.paletadcolor[5], self.paletadcolor[4], self.paletadcolor[3],self.paletadcolor[2],self.paletadcolor[1], self.paletadcolor[0]]

        # DF_REGIONAL_GER.loc[:,:,'Nordeste']['PG_EOL'].plot(figsize=(10, 6),lw=1.5, color = paletadcolor[5])

        for idx, fluxo in enumerate(COL_AC): 
            data_ = df_AC.loc[fluxo]['MW:From-To']
            axs.plot(data_.values, color=colores1[idx], label= fluxo.replace('_',' '), lw=1.4, linestyle='-')
        for idx, fluxo in enumerate(COL_DC): 
            data_ = df_DC.loc[fluxo][' P(MW)']
            axs.plot(data_.values, color=colores2[idx+1], label= fluxo.replace('_',' '), lw=2.2, linestyle='-')

        axs.xaxis.set_major_locator(plt.MaxNLocator(12))
        axs.legend(loc='best', fontsize=14)
        axs.tick_params(axis='y', labelsize=15)
        axs.tick_params(axis='x', labelsize=15)
        axs.set_xlabel('Semihoras', fontsize=15)
        axs.set_ylabel(eje_y, fontsize=15)
        axs.set_title(title, fontsize=20)
        if Ylimites != None:
            axs.set_ylim(Ylimites)
        if Xlimites != None:
            axs.set_xlim(Xlimites)
        axs.grid(True, linestyle='--', linewidth=1, alpha=0.2)
        plt.tight_layout()
        nome = self.cenario + '/Intercambios/' + title + '.png'
        plt.savefig(nome)
        if self.svg:
            nome = self.cenario + '/Intercambios/' + title + '.svg'
            plt.savefig(nome)
        plt.close('all')

    def plot_indice_0 (self, df_data, eje_y, name, title, INDICE, xlimites=None,ylimites=None, order = False, ax=None):
        
        if ax is None:
            fig, axs = plt.subplots(nrows=1, figsize=(15, 6), sharex=False)
        else:
            axs = ax

        colores = [sns.color_palette("Paired")[5], sns.color_palette("Paired")[1], sns.color_palette("Paired")[3]]
        if order:
            data = df_data.sort_values(INDICE, ascending=False)[INDICE]
            data_points_per_day = 10
            num_days = (len(df_data))*data_points_per_day / 100
            axs.set_xticks([round(i * num_days) for i in range(data_points_per_day+1)])
            axs.set_xticklabels([f'{i*10}' for i in range(data_points_per_day+1)], fontsize=12, rotation=0, ha='center')
            axs.set_xlabel('Percentage of half hours in a month (%)', fontsize=23)
        else:
            data = df_data[INDICE]
            axs.set_xlabel('Operating Points', fontsize=23)
        
        area_trapezoidal = np.trapz(data.values)/len(data)
        media = np.mean(data.values)
        axs.plot(data.values, color=colores[1], label='Todos os Cenarios', lw=2, linestyle='-')
            
        axs.legend(loc='upper right', fontsize=18)
        axs.tick_params(axis='y', labelsize=24)
        axs.tick_params(axis='x', labelsize=24)
        axs.set_ylabel(eje_y, fontsize=22)
        axs.set_title(f'{title} normalized area/mean: {area_trapezoidal, media}', fontsize=15)
        if xlimites is not None:
            axs.set_xlim(xlimites)
        if ylimites is not None:
            axs.set_ylim(ylimites)
        axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
        plt.tight_layout()
        if ax is None:
            nome = self.cenario + '/Indice/' + name + '.png'
            plt.savefig(nome, bbox_inches = 'tight')
            if self.svg:
                nome = self.cenario + '/Indice/' + name + '.svg'
                plt.savefig(nome, bbox_inches = 'tight')
            plt.close('all')

        return area_trapezoidal

    def plot_indice (self, df_data, eje_y, name, title, INDICE, xlimites=None,ylimites=None,  order = False):

        fig, axs = plt.subplots(nrows=1, figsize=(10, 6), sharex=False)
        colores = [sns.color_palette("Paired")[1], sns.color_palette("Paired")[3], sns.color_palette("Paired")[5],sns.color_palette("Paired")[7],sns.color_palette("Paired")[9]]
        region_map = {'Nordeste':'Northeast', 'Norte':'North', 'Sudeste-Centro-Oeste':'SE-CW', 'Sul':'South','AC-RO':'AC-RO'}
        for idx, regiao in enumerate(['Norte','Nordeste','Sudeste-Centro-Oeste', 'Sul', 'AC-RO']):
            
            if order:
                data = df_data.loc[:, :, regiao].sort_values(INDICE, ascending=False)[INDICE]
                data_points_per_day = 10
                num_days = 1344*data_points_per_day / 100
                axs.set_xticks([round(i * num_days) for i in range(data_points_per_day+1)])
                axs.set_xticklabels([f'{i*10}' for i in range(data_points_per_day+1)], fontsize=12, rotation=0, ha='center')
            else:
                data = df_data.loc[:, :, regiao][INDICE]
                data_points_per_day = 48
                num_days = len(data) // data_points_per_day
                axs.set_xticks([i * data_points_per_day for i in range(num_days)])
                axs.set_xticklabels([f'{i+1}' for i in range(num_days)], fontsize=18, rotation=0, ha='center')
            
            axs.plot(data.values, color=colores[idx], label=region_map[regiao], lw=2, linestyle='-')
            
        axs.legend(loc='upper right', fontsize=18)
        axs.tick_params(axis='y', labelsize=24)
        axs.tick_params(axis='x', labelsize=18)
        axs.set_xlabel('Percentage of half hours in a month (%)', fontsize=23)
        axs.set_ylabel(eje_y, fontsize=22)
        axs.set_title(title, fontsize=22)
        if xlimites != None:
            axs.set_xlim(xlimites)
        if ylimites != None:
            axs.set_ylim(ylimites)
        axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
        plt.tight_layout()
        nome = self.cenario + '/Indice/' + name + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = self.cenario + '/Indice/' + name + '.svg'
            plt.savefig(nome)
        plt.close('all')     

    def plot_indice_1 (self, df_pv, df_pq, eje_y, title, regiao, limites=None, order = True):

        fig, axs = plt.subplots(nrows=1, figsize=(10, 6))
        colores = [self.paletadcolor[4], self.paletadcolor[0], self.paletadcolor[3],self.paletadcolor[2],self.paletadcolor[3]]
        for idx, indice in enumerate(['CSI_SUP_FINAL', 'CSI_INF_FINAL']):
            if order:
                datapq = df_pq.loc[:, :, regiao][indice]
                datapv = df_pv.loc[:, :, regiao][indice]
            else:
                datapq = df_pq.loc[:, :, regiao].sort_values(indice, ascending=False)[indice]
                datapv = df_pv.loc[:, :, regiao].sort_values(indice, ascending=False)[indice]
                
            axs.plot(datapq.values, color=colores[idx], label='PQ_'+ indice[4:7],lw=2, linestyle='-')
            axs.plot(datapv.values, color=colores[idx+2], label='PV_'+ indice[4:7],lw=2, linestyle='-')

        axs.legend(loc='best', fontsize=18)

        # # Calculate the number of data points in a day (assuming each day has 48 data points)
        # data_points_per_day = 48
        # # Calculate the number of days based on the length of the data
        # num_days = len(datapq) // data_points_per_day
        # # Set x-axis ticks and labels for each day
        # axs.set_xticks([i * data_points_per_day for i in range(num_days)])
        # axs.set_xticklabels([f'{i+1}' for i in range(num_days)], fontsize=12, rotation=0, ha='right')

        axs.tick_params(axis='y', labelsize=18)
        axs.tick_params(axis='x', labelsize=18)
        axs.set_xlabel('Operating points', fontsize=23)
        axs.set_ylabel(eje_y, fontsize=20)
        axs.set_title(title, fontsize=25)
        if limites != None:
            axs.set_ylim(limites)
        axs.grid(True, linestyle='--', linewidth=1, alpha=0.2)
        plt.tight_layout()
        nome = self.cenario + '/Indice/' + title + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = self.cenario + '/Indice/' + title + '.svg'
            plt.savefig(nome)       
        plt.close('all')

    def plot_indice_2 (self, df, eje_y, name ,title, regiao, INDICE, GB, limites=None, order = True):

        fig, axs = plt.subplots(nrows=1, figsize=(10, 6))
        labelG = {'BIO': 'Bio', 'EOL': 'Wind', 'PCH': 'SHP','UFV': 'Solar', 'UHE': 'Hydro','UTE': 'Thermal', 'SIN': 'Synchronous C.'}
        if GB=='Gen_Type':
            colores = {'BIO': self.paletadcolor[4], 'EOL': self.paletadcolor[0], 'PCH': self.paletadcolor[3],'UFV': self.paletadcolor[2], 'UHE': self.paletadcolor[5],'UTE': self.paletadcolor[1], 'SIN': self.paletadcolor[6]}
        else:
            colores = {230: self.paletadcolor[4], 345: self.paletadcolor[0], 440: self.paletadcolor[3],500: self.paletadcolor[2], 525: self.paletadcolor[5], 765: self.paletadcolor[1]}
        data = df.loc[:, :, regiao]
        Busgroup = np.array(data.reset_index(GB)[GB].unique())
        for idx, G_bus in enumerate(Busgroup):
            if order:
                data_ = df.loc[:, :, regiao, G_bus][INDICE]
            else:
                data_ = df.loc[:, :, regiao, G_bus].sort_values(INDICE, ascending=False)[INDICE]
            if GB=='Gen_Type':
                label = labelG[G_bus]
            else:
                label = G_bus
            axs.plot(data_.values, color=colores[G_bus], label= label,lw=2)

        axs.legend(loc='best', fontsize=14)
        # Calculate the number of data points in a day (assuming each day has 48 data points)
        data_points_per_day = 48
        # Calculate the number of days based on the length of the data
        num_days = len(data_) // data_points_per_day
        # Set x-axis ticks and labels for each day
        axs.set_xticks([i * data_points_per_day for i in range(num_days)])
        axs.set_xticklabels([f'{i+1}' for i in range(num_days)], fontsize=18, rotation=0, ha='center')
        axs.tick_params(axis='y', labelsize=18)
        axs.set_xlabel('Days', fontsize=23)
        axs.set_ylabel(eje_y, fontsize=20)
        axs.set_title(title, fontsize=25)
        if limites != None:
            axs.set_ylim(limites)
        axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
        plt.tight_layout()
        nome = self.cenario + '/Indice/' + name + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = self.cenario + '/Indice/' + name + '.svg'
            plt.savefig(nome)
        plt.close('all')