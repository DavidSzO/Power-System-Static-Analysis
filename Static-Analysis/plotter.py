import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go 
from matplotlib.colors import TwoSlopeNorm

import warnings
warnings.filterwarnings('ignore')

class CreatePlots:
    def __init__(self):
        pass

    def persistency_curve(self, dataset, col, path=None, ax_fontsize=None):
        df_pwf_elos   = dataset['Nome Elo'].unique()

        for k in range(len(df_pwf_elos)):
            df_elos = dataset

            if k == 0:
                frm = df_pwf_elos[k][-2:]
                to = 'T'
            elif k == 5:
                frm = df_pwf_elos[k].split('_')[1]
                to  = 'T'
            else:
                splt1 = df_pwf_elos[k].split('_')
                splt2 = splt1[1].split('-')
                frm = splt2[0]
                to = splt2[1]

            
            df_elos = pd.DataFrame(df_elos[df_elos['Nome Elo'] == df_pwf_elos[k]][col])

            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.sort(np.array(df_elos[col]))
            N = len(x)
            y = np.linspace(1, N, N)

            import_values = len(df_elos[df_elos[col] < 0])
            export_values = len(df_elos[df_elos[col] > 0])
            
            total_import = round((import_values / N) * 100, 2)
            total_export = round((export_values / N) * 100, 2)


            plt.plot(y[:import_values], x[:import_values], lw=3, c='red', label=f'Import: {total_import}% (-)')
            plt.plot(y[import_values:export_values], x[import_values:export_values], lw=3, c='blue', label=f'Export: {total_export}% (+)')
            ax.set_ylabel(f'{col}', weight='bold', fontsize=14) 
            ax.set_title(fr'Persistency Curve {frm}$\rightarrow${to}', weight='bold', fontsize=20)
            ax.set_xlabel('# Operation Points', weight='bold', fontsize=14)
    
            ax.axhline(y=0, color='k', ls='--', alpha=0.3)
            ax.tick_params(color='black', labelcolor='black')
            ax.grid(visible=True, alpha=0.3)

            if ax_fontsize != None:
                ax.tick_params(axis='both', which='major', labelsize=ax_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=ax_fontsize)

            ax.legend(bbox_to_anchor=(0.01, 1.00), fontsize=13, 
            loc='upper left', fancybox=True, shadow=True)
            
            plt.tight_layout()

            if path != None:
                plt.savefig(f'{path}/PC_{df_pwf_elos[k]}')

            plt.show()

    def box_plots(self, dataset, col, split_flows=None, 
                  path=None, ax_fontsize=None, scenario=None):
        
        if split_flows == False:
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            ax = sns.boxplot(
                data=dataset, 
                x="Nome Elo", 
                y=col, 
                flierprops={"marker": "x"}, 
                hue=dataset['Nome Elo'],
                palette=sns.color_palette("hls", 8)
            )

            if scenario != None:
                ax.set_title(f'{scenario}')

            ax.set_ylabel(f'{col}', weight='bold', fontsize=13)
            ax.set_xlabel('Nome Elo', weight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels(dataset['Nome Elo'].unique(), rotation=30)

            if scenario != None:
                ax.set_title(f'{scenario}', weight='bold', fontsize=18)

            if ax_fontsize != None:
                ax.tick_params(axis='both', which='major', labelsize=ax_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=ax_fontsize)

            plt.tight_layout()

            if path != None:
                plt.savefig(f'{path}/BoxPlot_{col}_NSplit')

            plt.show()
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            dataset1 = dataset.iloc[:int(dataset.shape[0]/2+1344), :]
            sns.boxplot(
                data=dataset1, 
                x="Nome Elo", 
                y=col, 
                flierprops={"marker": "x"}, 
                hue=dataset1['Nome Elo'],
                ax=axes[0]
            )
            dataset2 = dataset.iloc[int(dataset.shape[0]/2)+1344:, :]
            sns.boxplot(
                data=dataset2, 
                x="Nome Elo", 
                y=col, 
                flierprops={"marker": "x"}, 
                hue=dataset2['Nome Elo'],
                palette=sns.color_palette("hls", 4),
                ax=axes[1]
            )
            for i in range(0, 2):
                axes[i].set_ylabel(f'{col}', weight='bold', fontsize=13)
                axes[i].set_xlabel('Nome Elo', weight='bold', fontsize=13)
                axes[i].grid(True, alpha=0.5)
                axes[i].tick_params(axis='both', which='major', labelsize=ax_fontsize)
                axes[i].tick_params(axis='both', which='minor', labelsize=ax_fontsize)
            
            if scenario != None:
                plt.suptitle(f'{scenario}', weight='bold', fontsize=20)

            plt.tight_layout()

            if path != None:
                plt.savefig(f'{path}/BoxPlot_{col}_Split') 
            plt.show()

    def violin_plots(self, dataset, col, split_flows=None, 
                  path=None, ax_fontsize=None, scenario=None):
        if split_flows == False:
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            ax = sns.violinplot(
                data=dataset, 
                x="Nome Elo", 
                y=col, 
                hue=dataset['Nome Elo'],
                palette=sns.color_palette("hls", 8)
            )

            if scenario != None:
                ax.set_title(f'{scenario}')

            ax.set_ylabel(f'{col}', weight='bold', fontsize=13)
            ax.set_xlabel('Nome Elo', weight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels(dataset['Nome Elo'].unique(), rotation=30)

            if scenario != None:
                ax.set_title(f'{scenario}', weight='bold', fontsize=18)

            if ax_fontsize != None:
                ax.tick_params(axis='both', which='major', labelsize=ax_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=ax_fontsize)

            plt.tight_layout()

            if path != None:
                plt.savefig(f'{path}/ViolinPlot_{col}_NSplit')
            plt.show()
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            dataset1 = dataset.iloc[:int(dataset.shape[0]/2+1344), :]
            sns.violinplot(
                data=dataset1, 
                x="Nome Elo", 
                y=col, 
                hue=dataset1['Nome Elo'],
                ax=axes[0]
            )
            dataset2 = dataset.iloc[int(dataset.shape[0]/2)+1344:, :]
            sns.violinplot(
                data=dataset2, 
                x="Nome Elo", 
                y=col, 
                hue=dataset2['Nome Elo'],
                palette=sns.color_palette("hls", 4),
                ax=axes[1]
            )
            for i in range(0, 2):
                axes[i].set_ylabel(f'{col}', weight='bold', fontsize=13)
                axes[i].set_xlabel('Nome Elo', weight='bold', fontsize=13)
                axes[i].grid(True, alpha=0.5)

                if ax_fontsize != None:
                    axes[i].tick_params(axis='both', which='major', labelsize=ax_fontsize)
                    axes[i].tick_params(axis='both', which='minor', labelsize=ax_fontsize)

            if scenario != None:
                plt.suptitle(f'{scenario}', weight='bold', fontsize=20)

            plt.tight_layout()

            if path != None:
                plt.savefig(f'{path}/ViolinPlot_{col}_Split') 
            plt.show()
    
    def create_heatmap(self, dataset, col, path=None, ax_fontsize=None):
        df_pwf_elos   = dataset['Nome Elo'].unique()

        for k in range(len(df_pwf_elos)):
            df_elos = dataset

            if k == 0:
                frm = df_pwf_elos[k][-2:]
                to = 'T'
            elif k == 5:
                frm = df_pwf_elos[k].split('_')[1]
                to  = 'T'
            else:
                splt1 = df_pwf_elos[k].split('_')
                splt2 = splt1[1].split('-')
                frm = splt2[0]
                to = splt2[1]

            max_day = max(sorted(list(set(df_elos['Dia']))))
            min_day = min(sorted(list(set(df_elos['Dia']))))
            n_days = len(sorted(list(set(df_elos['Dia']))))
            half_hour_list = sorted(list(set(df_elos['Hora'])))
            n_half_hour = len(half_hour_list)
            half_hour_periods = pd.date_range(start='2023-10-17', 
                                            periods=n_half_hour, freq='30T').strftime('%H:%M')

            elo_dataset = df_elos[df_elos['Nome Elo'] == df_pwf_elos[k]]

            active_power = np.array(elo_dataset[col]).reshape(n_days, -1)
            heatmap_data = pd.DataFrame(active_power[::-1], 
                                        index=range(max_day, min_day-1, -1), 
                                        columns=half_hour_periods) 

            vmax = elo_dataset[col].max()
            vmin = elo_dataset[col].min()

            neg_values = len(elo_dataset[elo_dataset[col] < 0])

            if vmin >= 0 or (vmin <= 0 and vmax <= 0) or neg_values < 30:
                cmap = sns.diverging_palette(230, 10, as_cmap=False)
            else:
                cmap = 'seismic'

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            ax = sns.heatmap(heatmap_data, cmap=cmap, center=0, 
                        xticklabels=2, cbar_kws={'label': col}, 
                        linewidths=0.1, linecolor="white", mask=heatmap_data == 0.0)  
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=11)
            cbar.ax.set_ylabel(col, size=12, weight='bold')

            plt.xticks(rotation=45)

            ax.set_xlabel('Half Hour Period', weight='bold', fontsize=12)
            ax.set_ylabel('Day of the Month', weight='bold', fontsize=12)
            ax.set_title(fr'Fluxo {frm}$\rightarrow${to}', weight='bold', fontsize=18)

            if ax_fontsize != None:
                ax.tick_params(axis='both', which='major', labelsize=ax_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=ax_fontsize)

            plt.tight_layout()
            if path != None:
                plt.savefig(f'{path}/Heatmap_{frm}_{to}')
            plt.show()

    def flow_profiles(self, dataset, col, scenario, path=None):
        df_pwf_elos = dataset['Nome Elo'].unique()
        fig = go.Figure(
            data=[
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[0]][col]/1000, 
                           name=df_pwf_elos[0], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[1]][col]/1000, 
                           name=df_pwf_elos[1], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[2]][col]/1000, 
                           name=df_pwf_elos[2], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[3]][col]/1000, 
                           name=df_pwf_elos[3], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[4]][col]/1000, 
                           name=df_pwf_elos[4], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[5]][col]/1000, 
                           name=df_pwf_elos[5], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[6]][col]/1000, 
                           name=df_pwf_elos[6], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[7]][col]/1000, 
                           name=df_pwf_elos[7], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[8]][col]/1000, 
                           name=df_pwf_elos[8], legend="legend"),
                go.Scatter(x=dataset.index, 
                           y=dataset[dataset['Nome Elo'] == df_pwf_elos[9]][col]/1000, 
                           name=df_pwf_elos[9], legend="legend"),
            ],   
            layout=dict(
                legend={
                    "title": f"<b>{scenario}</b>",
                    "xref": "container",
                    "yref": "container",
                    "x": 1.2,
                    "y": 0.8,
                    "orientation": "v",
                },
                
            ),
        )

        fig.update_layout(legend=dict(font=dict(family="Courier", size=20, color="black")),
                        legend_title=dict(font=dict(family="Courier", size=25, color="black")))
        fig.update_traces(visible="legendonly")
        fig.for_each_trace(lambda t: t.update(name='<b>'+ t.name +'</b>'))

        fig.update_layout(
            title_x=0.5,
            autosize=False,
            width=1800,
            height=800,
            font_family="Courier",
            xaxis_title=dict(text='<b># Operation Points</b>', font=dict(size=20, color='black')),
            yaxis_title=dict(text=f'<b>{col[0]}(GW)</b>', font=dict(size=20, color='black')),
            font_color="black",
            title=dict(text=f"<b>Flow Profiles</b>", font=dict(size=45), yref='paper')
        )

        fig.update_xaxes(ticks="outside", 
                    tickwidth=2, 
                    tickcolor='black', 
                    ticklen=5,
                    tickfont_family="Arial Black"
                    )

        fig.update_yaxes(ticks="outside", 
                    tickwidth=2, 
                    tickcolor='black', 
                    ticklen=5, 
                    tickfont_family="Arial Black"
                    )
        
        if path != None:
            fig.write_html(f"{path}/plot_flow_profiles.html")
            
        fig.show()

    def create_contourplot(self, dataset, col, path=None, ax_fontsize=12):
        df_pwf_elos   = dataset['Nome Elo'].unique()

        Dia           = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[0]]['Dia'])
        Hora          = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[0]]['Hora'])
        Fluxo_NE      = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[0]][col])
        Fluxo_NESE    = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[1]][col])
        Fluxo_NS      = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[2]][col])
        Fluxo_SULSECO = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[3]][col])
        Fluxo_NEN     = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[4]][col])
        Fluxo_RSUL    = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[5]][col])
        Elo_FOZIBIUNA = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[6]][col])
        Elo_PVELARARQ = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[7]][col])
        Elo_CPVBPVEL  = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[8]][col])
        Elo_XINGUSE   = np.array(dataset[dataset['Nome Elo'] == df_pwf_elos[9]][col])

        df_dict = {'Dia': Dia, 'Hora': Hora, df_pwf_elos[0]: Fluxo_NE, df_pwf_elos[1]: Fluxo_NESE, 
                df_pwf_elos[2]: Fluxo_NS, df_pwf_elos[3]: Fluxo_SULSECO, df_pwf_elos[4]: Fluxo_NEN,
                df_pwf_elos[5]: Fluxo_RSUL, df_pwf_elos[6]: Elo_FOZIBIUNA, df_pwf_elos[7]: Elo_PVELARARQ,
                df_pwf_elos[8]: Elo_CPVBPVEL, df_pwf_elos[9]: Elo_XINGUSE}
        df_pwf_total = pd.DataFrame(df_dict)
        df_pwf_hour = df_pwf_total[::2].reset_index().drop('index', axis=1)

        dia = np.unique(df_pwf_hour['Dia'])
        hora = np.unique(df_pwf_hour['Hora'])

        z_flows = []
        vals = []
        flow_cols = list(df_pwf_hour.columns[2:])

        for j in range(0, len(flow_cols)):
            lst = []
            for i in range(0, 673):
                if i % 24 == 0 and i != 0:
                    lst.append(vals)
                    vals = []
                
                if i == 672:
                    break
                vals.append(df_pwf_hour[flow_cols[j]][i])
            z_flows.append(lst)


        for k in range(len(df_pwf_elos)):

            if k == 0:
                frm = df_pwf_elos[k][-2:]
                to = 'T'
            elif k == 5:
                frm = df_pwf_elos[k].split('_')[1]
                to  = 'T'
            else:
                splt1 = df_pwf_elos[k].split('_')
                splt2 = splt1[1].split('-')
                frm = splt2[0]
                to = splt2[1]

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            norm   = TwoSlopeNorm(0)
            contour = ax.contourf(hora, dia[::-1], np.array(z_flows[k][::-1]), 20, 
                                cmap='RdBu_r', 
                                extend='max', 
                                norm=norm)

            cbar1 = plt.colorbar(contour, ax=ax)
            cbar1.formatter.set_powerlimits((3, 3))
            cbar1.formatter.set_useMathText(True)
            cbar1.ax.set_ylabel(f'{col}', rotation=90, weight='bold', size=12)

            ax.set_xlabel("Hour", weight='bold', fontsize=ax_fontsize)
            ax.set_ylabel("Day", weight='bold', fontsize=ax_fontsize)
            ax.set_yticks(np.arange(2, 30))
            ax.set_xticks(np.arange(0, 24))
            ax.set_xticklabels(hora, rotation=45)
            ax.set_title(fr'Fluxo {frm}$\rightarrow${to}', weight='bold', fontsize=18)

            if ax_fontsize != None:
                ax.tick_params(axis='both', which='major', labelsize=ax_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=ax_fontsize)

            plt.tight_layout()

            if path != None:
                plt.savefig(f'{path}/ContourPlot_{frm}_{to}')
            plt.show()  