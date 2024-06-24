from plotter import CreatePlots
import pandas as pd
import os 

# trocar só o cenário para o analise correspondente
scenario = 'V2A2F2_REV02_2026'
main_path = f'C:/Users/David/OneDrive/Documents/FERV_documentos/RESULTS/{scenario}/StaticAnalysis'
# scenario = 'MPV_(FNS Lim)_RC'
# main_path = f'D:/Mestrado/ENGIE/Power-System-Static-Analysis/RESULTS/{scenario}/StaticAnalysis'

folder_plot = 'Intercambios AC-DC/Intercambios' # mudar para guardar os plots

os.makedirs(main_path + f'/Plots/{folder_plot}', exist_ok=True)

df_pwf16 = pd.read_csv(main_path+'/Data/Fluxo em Ramos/DF_Intercambios.csv').drop('key', axis=1)
df_pwf16.columns = ['Nome Elo', 'Dia', 'Hora', 'P(MW)', 'Q(MVAr)']

df_pwf25 = pd.read_csv(main_path+'/Data/Fluxo em Ramos/DF_HVDC.csv').drop('key', axis=1)
df_pwf25_cols = df_pwf25.columns.tolist()
df_pwf25_cols = df_pwf25_cols[-1:] + df_pwf25_cols[:-1]
df_pwf25 = df_pwf25[df_pwf25_cols]
df_pwf25.columns = ['Nome Elo', 'Dia', 'Hora', 'P(MW)', 'Q(MVAr)']

df_pwf = pd.concat([df_pwf16, df_pwf25], axis=0).reset_index().drop('index', axis=1)

print(df_pwf)

if __name__ == '__main__':
    path =  main_path + f'/Plots/{folder_plot}/'
    plotter = CreatePlots()
    plotter.persistency_curve(dataset=df_pwf, 
                              col='P(MW)', 
                              path=path, 
                              ax_fontsize=13)
    plotter.box_plots(dataset=df_pwf, 
                      col='P(MW)', 
                      split_flows=False,
                      path=path,
                      ax_fontsize=11,
                      scenario=f'{scenario}')
    plotter.violin_plots(dataset=df_pwf, 
                         col='P(MW)', 
                         split_flows=True,
                         path=path,
                         ax_fontsize=11,
                         scenario=f'{scenario}')
    plotter.create_heatmap(dataset=df_pwf, 
                           col='P(MW)',
                           path=path, 
                           ax_fontsize=11)
    plotter.flow_profiles(dataset=df_pwf, 
                          col='P(MW)', 
                          scenario=f'{scenario}',
                          path=path)
    plotter.create_contourplot(dataset=df_pwf, 
                               col='P(MW)', 
                               path=path, 
                               ax_fontsize=11)