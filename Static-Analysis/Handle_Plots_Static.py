import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Plots_Static():

    def __init__(self, cenario, svg = False, PO = False) -> None:
        
        self.cenario = cenario
        self.svg = svg
        self.PO = PO
        plt.rcParams["font.family"] = "Times New Roman"
        self.paletadcolor = sns.color_palette()

    ## Definição de Funções Gerais =========================================================================================

    def plot_boxplot(self, data, labels, title, xlabel, ylabel, pathtosave, limites=None, vert = True, rotation = 0, text = True, nbarra = None):
        
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
        nome = pathtosave + title + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = pathtosave + title + '.svg'
            plt.savefig(nome)
        plt.close()

    def plot_Potencia(self, df_data, eje_y, title, limites=None):
        
        fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharex=False)

        axs.plot(df_data.values, color=sns.color_palette("Paired")[1], lw=2.5)
            
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
        if title.strip().split()[0]=='MW':
            nome = self.cenario + '/Plots/Perfil Potência Ativa/' + title + '.png'
        elif title.strip().split()[0]=='MVAR':
            nome = self.cenario + '/Plots/Perfil Potência Reativa/' + title + '.png'
        else:
            nome = self.cenario + '/Plots/Reserva/' + title + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        plt.close()

        # plt.plot([1, 2, 3, 4])
        # plt.ylabel('some numbers')
        # plt.show()

    def plot_reserva_reg (self, df_data, eje_y, name, title, INDICE, xlimites=None,ylimites=None, order = False):

        fig, axs = plt.subplots(nrows=1, figsize=(20, 10), sharex=False)
        colores = [sns.color_palette("Paired")[1], sns.color_palette("Paired")[3], sns.color_palette("Paired")[4],sns.color_palette("Paired")[7],sns.color_palette("Paired")[9]]
        region_map = {'Nordeste':'Northeast', 'Norte':'North', 'Sudeste-Centro-Oeste':'SE-CW', 'Sul':'South','AC-RO':'AC-RO'}
        for idx, regiao in enumerate(df_data.index.get_level_values('REG').unique().values):
            
            if order:
                data = df_data.loc[:, :, regiao].sort_values(INDICE, ascending=False)[INDICE]
            else:
                data = df_data.loc[:, :, regiao][INDICE]
            
            axs.plot(data.values, color=colores[idx], label=region_map[regiao], lw=2.5, linestyle='-')
            
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
        nome = self.cenario + '/Plots/Reserva/' + name + '.png'
        plt.savefig(nome)
        plt.close()

    def plot_Intercambio (self, df_AC, df_DC , eje_y, title, COL_AC, COL_DC, Ylimites=None, Xlimites=  None):

        fig, axs = plt.subplots(1, 1, figsize=(16, 7))
        colores1 = self.paletadcolor[:len(COL_AC)]
        colores2 = self.paletadcolor[:len(COL_DC)][::-1]

        for idx, fluxo in enumerate(COL_AC):
            data_ = df_AC.loc[fluxo, 'MW:From-To']
            axs.plot(data_.values, color=colores1[idx], label=fluxo.replace('_', ' '), lw=1.4, linestyle='-')
        for idx, fluxo in enumerate(COL_DC):
            data_ = df_DC.loc[fluxo, 'P(MW)']
            axs.plot(data_.values, color=colores2[idx], label=fluxo.replace('_', ' '), lw=2.2, linestyle='-')

        axs.xaxis.set_major_locator(plt.MaxNLocator(12))
        axs.legend(loc='best', fontsize=14)
        axs.tick_params(axis='y', labelsize=15)
        axs.tick_params(axis='x', labelsize=15)
        axs.set_xlabel('Semihoras', fontsize=15)
        axs.set_ylabel(eje_y, fontsize=15)
        axs.set_title(title, fontsize=20)

        if Ylimites is not None:
            axs.set_ylim(Ylimites)
        if Xlimites is not None:
            axs.set_xlim(Xlimites)

        axs.grid(True, linestyle='--', linewidth=1, alpha=0.2)
        plt.tight_layout()

        plot_dir = f"{self.cenario}/Plots/Intercambios AC-DC/"
        plt.savefig(f"{plot_dir}{title}.png")
        if self.svg:
            plt.savefig(f"{plot_dir}{title}.svg")

        plt.close()

    def plot_indice_0 (self, df_data, eje_y, name, title, INDICE, pathtosave, xlimites=None,ylimites=None, order = False, ax=None):
        
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
            nome = pathtosave + name + '.png'
            plt.savefig(nome, bbox_inches = 'tight')
            if self.svg:
                nome = pathtosave + name + '.svg'
                plt.savefig(nome, bbox_inches = 'tight')
            plt.close()

        return area_trapezoidal

    def plot_indice (self, df_data, eje_y, name, title, INDICE, pathtosave, xlimites=None, ylimites=None, order = False):

        fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharex=False)
        colores = [sns.color_palette("Paired")[1], sns.color_palette("Paired")[3], sns.color_palette("Paired")[5],sns.color_palette("Paired")[7],sns.color_palette("Paired")[9]]
        region_map = {'Nordeste':'Northeast', 'Norte':'North', 'Sudeste-Centro-Oeste':'SE-CW', 'Sul':'South','AC-RO':'AC-RO'}
        for idx, regiao in enumerate(['Norte','Nordeste','Sudeste-Centro-Oeste', 'Sul', 'AC-RO']):

            if not self.PO:         
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

            else:
                if order:
                    data = df_data.loc[:, :, regiao].sort_values(INDICE, ascending=False)[INDICE]
                else:
                    data = df_data.loc[:, :, regiao][INDICE]
            
                axs.bar(region_map[regiao], data, color=colores[idx], label=region_map[regiao], linewidth=2, linestyle='-')
            
        axs.legend(loc='upper right', fontsize=18)
        axs.tick_params(axis='y', labelsize=24)
        axs.tick_params(axis='x', labelsize=18)
        if not self.PO:
            axs.set_xlabel('Percentage of half hours in a month (%)', fontsize=23)
        else:
            axs.set_xlabel('Region', fontsize=23)
        axs.set_ylabel(eje_y, fontsize=22)
        axs.set_title(title, fontsize=22)
        if xlimites != None:
            axs.set_xlim(xlimites)
        if ylimites != None:
            axs.set_ylim(ylimites)
        axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
        plt.tight_layout()
        nome = pathtosave + name + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = pathtosave + name + '.svg'
            plt.savefig(nome)
        plt.close()     

    def plot_indice_1 (self, df_pv, df_pq, eje_y, title, regiao, pathtosave, limites=None, order = True):

        fig, axs = plt.subplots(nrows=1, figsize=(10, 6))
        colores = [self.paletadcolor[4], self.paletadcolor[0], self.paletadcolor[3],self.paletadcolor[2],self.paletadcolor[3]]
        for idx, indice in enumerate(['CSI_SUP_FINAL', 'CSI_INF_FINAL']):
            if order:
                datapq = df_pq.loc[:, :, regiao][indice]
                datapv = df_pv.loc[:, :, regiao][indice]
            else:
                datapq = df_pq.loc[:, :, regiao].sort_values(indice, ascending=False)[indice]
                datapv = df_pv.loc[:, :, regiao].sort_values(indice, ascending=False)[indice]
            if not self.PO:     
                axs.plot(datapq.values, color=colores[idx], label='PQ_'+ indice[4:7],lw=2, linestyle='-')
                axs.plot(datapv.values, color=colores[idx+2], label='PV_'+ indice[4:7],lw=2, linestyle='-')
            else:
                axs.bar(indice, datapq.values, color=colores[idx], label='PQ_'+ indice[4:7], linewidth=2, linestyle='-')
                axs.bar(indice, datapv.values, color=colores[idx + 2 ], label='PV_'+ indice[4:7], linewidth=2, linestyle='-')

        axs.legend(loc='best', fontsize=18)
        axs.tick_params(axis='y', labelsize=18)
        axs.tick_params(axis='x', labelsize=18)
        if not self.PO: 
            axs.set_xlabel('Operating points', fontsize=23)
        else:
            axs.set_xlabel('Index', fontsize=23)

        axs.set_ylabel(eje_y, fontsize=20)
        axs.set_title(title, fontsize=25)
        if limites != None:
            axs.set_ylim(limites)
        axs.grid(True, linestyle='--', linewidth=1, alpha=0.2)
        plt.tight_layout()
        nome = pathtosave + title + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = pathtosave + title + '.svg'
            plt.savefig(nome)       
        plt.close()

    def plot_indice_2 (self, df, eje_y, name ,title, regiao, INDICE, GB, pathtosave, limites=None, order = True):

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
            if not self.PO:
                axs.plot(data_.values, color=colores[G_bus], label= label,lw=2)
            else:
                axs.bar(str(G_bus), data_.values, color=colores[G_bus], label=label, linewidth=2)

        axs.legend(loc='best', fontsize=14)
        data_points_per_day = 48
        num_days = len(data_) // data_points_per_day
        axs.set_xticks([i * data_points_per_day for i in range(num_days)])
        axs.set_xticklabels([f'{i+1}' for i in range(num_days)], fontsize=18, rotation=0, ha='center')
        axs.tick_params(axis='y', labelsize=18)
        if not self.PO:
            axs.set_xlabel('Days', fontsize=23)
        else:
            axs.set_xlabel('Bus Group', fontsize=23)
            
        axs.set_ylabel(eje_y, fontsize=20)
        axs.set_title(title, fontsize=25)
        if limites != None:
            axs.set_ylim(limites)
        axs.grid(True, linestyle='-', linewidth=1.2, alpha=0.4)
        plt.tight_layout()
        nome = pathtosave + name + '.png'
        plt.savefig(nome, bbox_inches = 'tight')
        if self.svg:
            nome = pathtosave + name + '.svg'
            plt.savefig(nome)
        plt.close()

    def analise_regiao_plot(self, data_GER, nome):
        
        # Data for the first plot
        sns.set_context('talk')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)  # 2 rows, 1 column
        bar_width = 0.4
        data_pg = data_GER[['PG_MW', 'PL_MW']].loc[['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        x = np.arange(len(data_pg.index))
        b1 = ax1.bar(x, data_pg['PG_MW'], width=bar_width, label='PG_MW')
        b2 = ax1.bar(x + bar_width, data_pg['PL_MW'], width=bar_width, label='PL_MW')
        # Fix the x-axes.
        ax1.set_xticks(x + bar_width / 2)
        ax1.set_xticklabels(data_pg.index.unique(), rotation=0, ha='center') 
        # Add legend.
        ax1.legend()
        # Axis styling.
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_color('#DDDDDD')
        ax1.tick_params(bottom=False, left=False)
        ax1.set_axisbelow(True)
        ax1.yaxis.grid(True, color='#EEEEEE')
        ax1.xaxis.grid(False)
        # Add a legend for the second subplot
        ax1.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Add axis and chart labels.
        ax1.set_ylabel('MW', labelpad=15)
        ax1.set_title('Potencia Gerada y Demanda Liquida por Região - MW')
        # For each bar in the chart, add a text label.
        for bar in b1 + b2:
                bar_value = bar.get_height()
                text = f'{bar_value:,.2f}'
                text_x = bar.get_x() + bar.get_width() / 2
                text_y = bar.get_y() + bar_value
                bar_color = bar.get_facecolor()
                ax1.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=12)

        # Data for the second plot
        df = data_GER[['PG_UHE','PG_UTE','PG_EOL','PG_SOL']].loc[['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        # Plotting on the second subplot
        df.plot.bar(stacked=True, ax=ax2, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        # Customize label size and rotation for the second subplot
        ax2.tick_params(axis='y', labelsize=15, rotation=0)
        # Axis styling.
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_color('#DDDDDD')
        ax2.tick_params(bottom=False, left=False)
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(True, color='#EEEEEE')
        ax2.xaxis.grid(False)
        # Add a legend for the second subplot
        ax2.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xticklabels(data_pg.index.unique(), rotation=0, ha='center') 
        ax2.set_ylabel('MW', labelpad=15)
        ax2.set_title('Potencia Gerada por tipo de geração - MW')
        plt.tight_layout()
        nomesave = self.cenario + '/Plots/Potencia/MW_' + nome + '.png'
        plt.savefig(nomesave)
        # plt.show()

        # Data for the second plot ================================================================

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)  # 2 rows, 1 column
        # First plot
        bar_width = 0.4
        x = np.arange(len(data_GER.index))
        data_qg = data_GER['QG_MVAR'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        b1 = ax1.bar(x, data_qg, width=bar_width, label='QG_MVAR')
        data_ql = data_GER['QL_MVAR'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        b2 = ax1.bar(x + bar_width, data_ql ,width=bar_width, label='QL_MVAR')
        # Fix the x-axes.
        ax1.set_xticks(x + bar_width / 2)
        ax1.set_xticklabels(data_qg.index.unique(), rotation=45, ha='right')  # Rotate the x-axis labels
        # Add legend.
        ax1.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Axis styling.
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_color('#DDDDDD')
        ax1.tick_params(bottom=False, left=False)
        ax1.set_axisbelow(True)
        ax1.yaxis.grid(True, color='#EEEEEE')
        ax1.xaxis.grid(False)
        # Add axis and chart labels.
        ax1.set_xlabel('Região', labelpad=15)
        ax1.set_ylabel('MVAR', labelpad=15)
        ax1.set_title('Potencia Gerada y Demanda Liquida por Região - MVAR', pad=20)
        # For each bar in the chart, add a text label.
        for bar in b1 + b2:
                bar_value = bar.get_height()
                text = f'{bar_value:,.2f}'
                text_x = bar.get_x() + bar.get_width() / 2
                text_y = bar.get_y() + bar_value
                bar_color = bar.get_facecolor()
                ax1.text(text_x, text_y, text, ha='center', va='bottom', color='black', size=12)
        # Second plot
        sns.set_context('talk')  # Use Seaborn's context settings to make fonts larger.
        bar_width = 0.4
        data_shunt = data_GER['Shunt_Ind'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        b1 = ax2.bar(x, data_shunt, width=bar_width, color=sns.color_palette("Paired")[1], label='Shunt_Ind')
        data_shunt_inst = data_GER['SHUNT_INST_IND'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        b2 = ax2.bar(x + bar_width, data_shunt_inst ,width=bar_width, color=sns.color_palette("Paired")[0], label='Shunt_Ind_Inst')
        data_shunt_cap = data_GER['Shunt_Cap'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        b3 = ax2.bar(x, data_shunt_cap, width=bar_width, color=sns.color_palette("Paired")[3], label='Shunt_Cap')
        data_shunt_cap_inst = data_GER['SHUNT_INST_CAP'][['Norte','Nordeste','Sudeste-Centro-Oeste','Sul','AC-RO']]
        b4 = ax2.bar(x + bar_width, data_shunt_cap_inst ,width=bar_width, color=sns.color_palette("Paired")[2], label='Shunt_Cap_Inst')
        # Fix the x-axes.
        ax2.set_xticks(x + bar_width / 2)
        ax2.set_xticklabels(data_shunt.index.unique(), rotation=45, ha='right')  # Rotate the x-axis labels
        ax2.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Axis styling.
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_color('#DDDDDD')
        ax2.tick_params(bottom=False, left=False)
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(True, color='#EEEEEE')
        ax2.xaxis.grid(False)
        # Add axis and chart labels.
        ax2.set_xlabel('Região', labelpad=15)
        ax2.set_ylabel('MVAR', labelpad=15)
        ax2.set_title('Shunt Alocado e Shunt Instalado por Região', pad=15)
        # For each bar in the chart, add a text label.
        for bar in b1 + b2 + b3 + b4:
                bar_value = bar.get_height()
                text = f'{bar_value:,.2f}'
                text_x = bar.get_x() + bar.get_width() / 2
                text_y = bar.get_y() + bar_value
                bar_color = bar.get_facecolor()
                ax2.text(text_x, text_y, text, ha='center', va='bottom', color='black', size=12)
                
        ax2.set_xticklabels(data_pg.index.unique(), rotation=0, ha='center') 
        plt.tight_layout()
        nomesave = self.cenario + '/Plots/Potencia/MW_' + nome + '.png'
        plt.savefig(nomesave)
        plt.close()



if __name__ == "__main__":

    path = 'C:/Users/David/OneDrive/Documents/GitHub/Power-System-Static-Analysis/RESULTS/V1A1F2_RESP_FNS_lim_rev1_2026/StaticAnalysis/Data/Potencia/Df_Reserva_PO_MW.csv'
    data = pd.read_csv(path, index_col=0)
    cenario = 'C:/Users/David/OneDrive/Documents/GitHub/Power-System-Static-Analysis/RESULTS/V1A1F2_RESP_FNS_lim_rev1_2026/StaticAnalysis/'
    plot = Plots_Static(cenario)
    plot.plot_Potencia(data[' Reserve'], 'MW','FIGURA')


