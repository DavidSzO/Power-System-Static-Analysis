from plotter import CreatePlots
import pandas as pd
import os 

# trocar só o cenário para o analise correspondente
scenario = 'MPV_(FNS Lim)_RC'
main_path = f'../RESULTS/{scenario}/StaticAnalysis'

folder_plot = 'Intercambios AC-DC' # mudar para guardar os plots

os.makedirs(main_path + f'/Plots/{folder_plot}', exist_ok=True)

df_pwf16 = pd.read_csv(main_path+'/Data/Fluxo em Ramos/DF_Intercambios.csv')
df_pwf25 = pd.read_csv(main_path+'/Data/Fluxo em Ramos/DF_HVDC.csv').drop('Nome Elo', axis=1)
df_pwf16.columns = ['Nome Elo', 'Dia', 'Hora', 'P(MW)', 'Q(Mvar)']
df_pwf25.columns = ['Nome Elo', 'Dia', 'Hora', 'P(MW)', 'Q(Mvar)']

df_pwf = pd.concat([df_pwf16, df_pwf25], axis=0).reset_index().drop('index', axis=1)
# ['EXP_NE', 'Fluxo_NE-SE', 'Fluxo_N-S', 'Fluxo_SUL-SECO', 'Fluxo_NE-N', 'Fluxo_RSUL', 
# 'Elo_FOZ-IBIUNA', 'Elo_PVEL-ARARQ', 'Elo_CPVBTB-PVEL', 'Elo_XINGU-SE']

if __name__ == '__main__':
    path =  main_path + f'/Plots/{folder_plot}/'
    plotter = CreatePlots()
    # plotter.persistency_curve(dataset=df_pwf, 
    #                           col='P(MW)', 
    #                           path=path, 
    #                           ax_fontsize=13)
    # plotter.box_plots(dataset=df_pwf, 
    #                   col='P(MW)', 
    #                   split_flows=True,
    #                   path=path,
    #                   ax_fontsize=11,
    #                   scenario='V1A1F2')
    # plotter.violin_plots(dataset=df_pwf, 
    #                      col='P(MW)', 
    #                      split_flows=False,
    #                      path=path,
    #                      ax_fontsize=11,
    #                      scenario='V1A1F2')
    plotter.create_heatmap(dataset=df_pwf, 
                           col='P(MW)',
                           path=path, 
                           ax_fontsize=11)
    # plotter.flow_profiles(dataset=df_pwf, 
    #                       col='P(MW)', 
    #                       scenario="V2A2F2",
    #                       path=path)
    # plotter.create_contourplot(dataset=df_pwf, 
    #                            col='P(MW)', 
    #                            path=path, 
    #                            ax_fontsize=11)