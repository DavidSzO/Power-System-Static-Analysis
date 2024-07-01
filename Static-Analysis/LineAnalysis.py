#Import a serem realizados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import shutil


class Analise_Linhas:
    def __init__(self, PWF16):

        PWF16_concatenados = PWF16
        # PWF16_Filt = PWF16_concatenados[(PWF16_concatenados['Type'] == ' TL') & ~(PWF16_concatenados['REG'].isna())]
        PWF16_Filt = PWF16_concatenados[~(PWF16_concatenados['REG'].isna())]
        self.PWF16_Filt_NEW = PWF16_Filt[(PWF16_Filt['VBASEKV'] > 138) & (PWF16_Filt['VBASEKV'] != 161)]
        self.VBA = self.PWF16_Filt_NEW[self.PWF16_Filt_NEW['VBASEKV'] >= 138]['VBASEKV'].unique()

        #usar para  Remover_e_salvar_L1MVA9999
        self.PW9999 = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['L1(MVA)'] == 9999)]
        #usar análise dia de semana
        #self.PW16_OLD =  self.PWF16_Filt_NEW.copy()

        self.PWF16_Filt_NEW = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['L1(MVA)']!=9999)]

        self.Grafico_Maior_L1 = True
        self.plotar_grafico_de_calor = True
        self.boxplot_REG = True
        self.plotar_grafico_por_reg_VBA = True
        self.plotar_grafico_de_calor_1VBA = True
        self.carregamentos_baixos_e_altos = True
        self.maior_carregamento_para_menor_dia_e_hora = True
        self.contador_linha = True
        self.contador_VKBASE = True
        self.media_geral_reg =True
        self.PVI = True
        self.PFI_Grafico = True
        self.PFI_Excel = True

    def Remover_e_salvar_L1MVA9999(self,Pasta):
        
        grouped_df = self.PW9999.groupby(['From#', 'To#']).first().reset_index()

        # Define o nome da pasta e o caminho completo
        
        caminho_completo = os.path.join(Pasta, 'linhas_retiradas.csv')

        # Salvando o DataFrame em CSV na pasta especificada
        grouped_df.to_csv(caminho_completo, index=False)


    def Removedor_n_convergiu(self, csv):
        #linhas a serem removidas
        self.df_csv = pd.read_csv(csv)

        
        # Converter os valores das colunas 'Dia' e 'Hora' em strings

        # Definir os valores de 'Dia' e 'Hora' do self.df_csv como strings
        self.df_csv['Dia'] = self.df_csv['Dia'].astype(str)
        self.df_csv['Hora'] = self.df_csv['Hora'].astype(str)


        self.df_csv['Dia'] = self.df_csv['Dia'].str.zfill(2)

        for index, row in self.df_csv.iterrows():
            # Identificar as linhas de PWF16_Filt que correspondem ao 'Dia' e 'Hora' atual
            indices = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['Dia'] == row['Dia']) & (self.PWF16_Filt_NEW['Hora'] == row['Hora'])].index
            # Remover as linhas correspondentes de self.PWF16_Filt_NEW
            self.PWF16_Filt_NEW.drop(indices, inplace=True)
        

    def ativar_grafico(self, nome_atributo):
        setattr(self, nome_atributo, True)
    
    def desativar_grafico(self, nome_atributo):
        setattr(self, nome_atributo, False)

    
    def Graficos_Por_REG(self, Pasta):
        import matplotlib.pyplot as plt
        if self.Grafico_Maior_L1:
            def Grafico_Maior_L1(REG):
                sns.set_theme(style="darkgrid")
                df_sul = self.PWF16_Filt_NEW[self.PWF16_Filt_NEW['REG'] == REG]

                # Agrupando por 'From#' e 'To#', e calculando a média de '% L1'
                grouped = df_sul.groupby(['From#', 'To#'])['% L1'].mean()

                # Encontrando o índice do valor máximo na média de '% L1'
                indice_max = grouped.idxmax()

                # Obtendo os valores de 'From#' e 'To#' usando o índice encontrado
                valor_from, valor_to = indice_max


                FluxoLinhas639_560 = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['From#'] == valor_from) & (self.PWF16_Filt_NEW['To#'] == valor_to)]
                from_value = FluxoLinhas639_560['From Name'].iloc[0]  # Valor da coluna 'From1' da primeira linha
                to_value = FluxoLinhas639_560['To Name'].iloc[0]  

                resultados_agregados_hora = FluxoLinhas639_560.groupby('Hora').agg({'% L1': 'mean', 'Mvar:Losses': 'sum'})

                # Criar uma figura
                fig, ax1 = plt.subplots(figsize=(12, 6))
                plt.rcParams['axes.grid'] = True 

                # Definir a escala no lado esquerdo para '% L1'
                ax1.set_xlabel('Hora', fontsize=14)
                ax1.set_ylabel('% L1', color='tab:blue', fontsize=20)
                ax1.plot(resultados_agregados_hora.index, resultados_agregados_hora['% L1'], color='tab:blue', label='% L1')

                # ... (seu código restante)

                # Definir rótulos para os eixos X e título
                ax1.set_title(f'Linha maior carregamento de {REG},{from_value} para{to_value} ', fontsize=24)

                # Configurar os locais dos ticks no eixo x
                ax1.set_xticks(range(len(resultados_agregados_hora.index)))

                # Definir rótulos do eixo x nos locais apropriados e rotacionar em 90 graus
                ax1.set_xticklabels(resultados_agregados_hora.index, rotation=90)

                ax1.tick_params(axis='x', which='major', labelsize=16)

                ax1.legend(loc='upper left', fontsize=16)
                # Depois de criar ax1 e definir seu conteúdo...

                # Criar um eixo gêmeo no lado direito para 'Mvar:Losses'
                ax2 = ax1.twinx()
                ax2.set_ylabel('Mvar:Losses', color='tab:red', fontsize=20)
                ax2.plot(resultados_agregados_hora.index, resultados_agregados_hora['Mvar:Losses'], color='tab:red', label='Mvar:Losses')

                ax1.tick_params(axis='y', which='major', labelsize=14, colors='tab:blue')
                ax2.tick_params(axis='y', which='major', labelsize=16, colors='tab:red')

                # Depois de configurar os rótulos e títulos...

                ax1.legend(loc='upper left', fontsize=16)
                ax2.legend(loc='upper right', fontsize=16)




                nome_arquivo = f'{Pasta}Linha_Maior_L1REG_{REG}'
                plt.savefig(nome_arquivo)
                
                print(f'{Pasta}Linha_Maior_L1REG_{REG}')
            Grafico_Maior_L1('Norte')
            Grafico_Maior_L1('Nordeste')
            Grafico_Maior_L1('AC-RO')
            Grafico_Maior_L1('Sudeste-Centro-Oeste')
            Grafico_Maior_L1('Sul')

        if self.plotar_grafico_de_calor:
            def plotar_grafico_de_calor(REG):
                PWF16_Filt_Reg = self.PWF16_Filt_NEW[self.PWF16_Filt_NEW['REG'] == REG]

                # Use pivot_table para reorganizar os dados
                pivot_percent_L1 = PWF16_Filt_Reg.pivot_table(index='Dia', columns='Hora', values='% L1')
                pivot_Mvar_Losses = PWF16_Filt_Reg.pivot_table(index='Dia', columns='Hora', values='Mvar:Losses', aggfunc='sum')

                # Aumente o tamanho da figura para tornar o gráfico maior
            

                # Crie um gráfico de calor para 'Mvar:Losses'
                plt.figure(figsize=(12, 6))

                ax = sns.heatmap(pivot_Mvar_Losses, cmap='Reds', cbar=True, fmt=".2f")

                cbar = ax.collections[0].colorbar
                cbar.ax.set_ylabel('Mvar:Losses', fontsize=16)  # Aumente o tamanho do rótulo da barra de legenda
                cbar.ax.tick_params(labelsize=16)               #aumenta stiker

                ax.set_xlabel('Hora', fontsize=20)
                ax.set_ylabel('Dia', fontsize=18)
                ax.set_title(f'Gráfico de Calor para Mvar:Losses - Região: {REG}', fontsize=20)

                ax.tick_params(axis='both', which='major', labelsize=12)

                nome_arquivo = f'{Pasta}Gráfico_de_Calor_para_Mvar_Losses_-_Região_{REG}.png'
                plt.savefig(nome_arquivo)

                plt.figure(figsize=(12, 6))

                ax = sns.heatmap(pivot_percent_L1, cmap='Reds_r', cbar=True, fmt=".2f")

                cbar = ax.collections[0].colorbar
                cbar.ax.set_ylabel('% L1', fontsize=16)  # Aumente o tamanho do rótulo da barra de legenda
                cbar.ax.tick_params(labelsize=16)

                ax.set_xlabel('Hora', fontsize=20)
                ax.set_ylabel('Dia', fontsize=18)
                ax.set_title(f'Gráfico de Calor para % L1 - Região: {REG}', fontsize=20)

                ax.tick_params(axis='both', which='major', labelsize=12)

                nome_arquivo = f'{Pasta}Gráfico_de_Calor_para_porcento_L1_-_Região_{REG}.png'
                plt.savefig(nome_arquivo)

            plotar_grafico_de_calor('Norte')
            plotar_grafico_de_calor('Nordeste')
            plotar_grafico_de_calor('AC-RO')
            plotar_grafico_de_calor('Sudeste-Centro-Oeste')
            plotar_grafico_de_calor('Sul')
        
        if self.boxplot_REG:
            

            # Supondo que você tenha um DataFrame chamado PWF16_Filt_NEW com as colunas 'DIA', 'HORA', 'REG' e '% L1'

            # Lista de regiões desejadas
            regioes = ['Norte', 'Nordeste', 'AC-RO', 'Sudeste-Centro-Oeste', 'Sul']
            colors = ['b', 'g', 'r', 'c', 'orange']

            # Filtrar os dados para as regiões desejadas e com VBASEKV >= 230
            PWF16_Filt_Regioes = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['REG'].isin(regioes)) & (self.PWF16_Filt_NEW['VBASEKV'] >= 230)]

            # Calcular a média do '% L1' para cada combinação de 'Dia', 'Hora' e 'REG'
            media_por_combinacao = PWF16_Filt_Regioes.groupby(['Dia', 'Hora', 'REG'])['% L1'].mean().reset_index()

            # Criação do gráfico de boxplot com preenchimento de cores
            plt.figure(figsize=(12, 6))
            plt.title('Boxplot para Média do % L1 nas Regiões por Dia e Hora', fontsize=20)

            boxprops = dict(facecolor='white', color='black')
            bp2 = plt.boxplot([media_por_combinacao[media_por_combinacao['REG'] == reg]['% L1'] for reg in regioes], 
                            vert=True, positions=range(1, len(regioes) + 1), 
                            widths=0.6, patch_artist=True, showfliers=True, boxprops=boxprops)

            # Preencha as caixas com as cores correspondentes
            for box, color in zip(bp2['boxes'], colors):
                box.set(facecolor=color)

            # Configuração do eixo X e Y
            plt.xticks(range(1, len(regioes) + 1), regioes, fontsize=16)
            plt.xlabel('Região', fontsize=18)
            plt.ylabel('Média do % L1', fontsize=18)
            plt.tick_params(axis='y', which='major', labelsize=16)

            # Configuração das cores das etiquetas do eixo X
            for label, color in zip(plt.gca().get_xticklabels(), colors):
                label.set(color=color)
            nome_arquivo2 = f'{Pasta}BoxL1.png'
            plt.savefig(nome_arquivo2)

            
            # Supondo que você tenha um DataFrame chamado self.PWF16_Filt_NEW com as colunas 'DIA', 'HORA', 'REG' e '	Mvar:Losses'

            # Lista de regiões desejadas
            regioes = ['Norte', 'Nordeste', 'AC-RO', 'Sudeste-Centro-Oeste', 'Sul']
            colors = ['b', 'g', 'r', 'c', 'orange']

            # Filtrar os dados para as regiões desejadas e com VBASEKV >= 230
            PWF16_Filt_Regioes = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['REG'].isin(regioes)) & (self.PWF16_Filt_NEW['VBASEKV'] >= 230)]

            # Calcular a média do '	Mvar:Losses' para cada combinação de 'Dia', 'Hora' e 'REG'
            media_por_combinacao = PWF16_Filt_Regioes.groupby(['Dia', 'Hora', 'REG'])['Mvar:Losses'].sum().reset_index()

            # Criação do gráfico de boxplot com preenchimento de cores
            plt.figure(figsize=(12, 6))
            plt.title('Boxplot para Soma do Mvar:Losses nas Regiões por Dia e Hora', fontsize=20)

            boxprops = dict(facecolor='white', color='black')
            bp2 = plt.boxplot([media_por_combinacao[media_por_combinacao['REG'] == reg]['Mvar:Losses'] for reg in regioes], 
                            vert=True, positions=range(1, len(regioes) + 1), 
                            widths=0.6, patch_artist=True, showfliers=False, boxprops=boxprops)

            # Preencha as caixas com as cores correspondentes
            for box, color in zip(bp2['boxes'], colors):
                box.set(facecolor=color)

            # Configuração do eixo X e Y
            plt.xticks(range(1, len(regioes) + 1), regioes, fontsize=16)
            plt.xlabel('Região', fontsize=18)
            plt.ylabel('Soma do Mvar:Losses', fontsize=18)
            plt.tick_params(axis='y', which='major', labelsize=16)

            # Configuração das cores das etiquetas do eixo X
            for label, color in zip(plt.gca().get_xticklabels(), colors):
                label.set(color=color)

            # Salvar a imagem
            nome_arquivo2 = f'{Pasta}BoxMV.png'
            plt.savefig(nome_arquivo2)

        if self.media_geral_reg:
            import matplotlib.pyplot as plt

            def plotar_grafico_por_reg(Reg):
                PWF16_Filt_Reg = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['REG'] == Reg) & (self.PWF16_Filt_NEW['VBASEKV'] >= 230)]
                resultados_agregados_hora = PWF16_Filt_Reg.groupby('Hora').agg({'% L1': 'mean', 'Mvar:Losses': 'mean'})

                fig, ax1 = plt.subplots(figsize=(12, 6))
                plt.suptitle(f'Gráfico média %L1 e Mvar: {Reg}', fontsize=24)

                ax1.plot(resultados_agregados_hora.index, resultados_agregados_hora['% L1'], label='% L1', color='blue')
                ax1.set_xlabel('SemiHoras', fontsize=18)  # Aumente o tamanho do rótulo do eixo x
                ax1.set_ylabel('% L1', color='blue', fontsize=18)

                # Criar um segundo eixo y no mesmo gráfico
                ax2 = ax1.twinx()
                ax2.plot(resultados_agregados_hora.index, resultados_agregados_hora['Mvar:Losses'], label='Mvar:Losses', color='red')
                ax2.set_ylabel('Mvar:Losses', color='red', fontsize=18)

                # Defina as posições das marcações e os rótulos do eixo x
                ax1.set_xticks(range(len(resultados_agregados_hora.index)))
                ax1.set_xticklabels(resultados_agregados_hora.index, rotation=90, fontsize=14)

                ax1.tick_params(axis='y', which='major', labelsize=16, colors='tab:blue')
                ax2.tick_params(axis='y', which='major', labelsize=16, colors='tab:red')

                ax1.legend(loc='upper left', fontsize=16)
                ax2.legend(loc='upper right', fontsize=16)

                # Defina o nome do arquivo antes de salvá-lo
                nome_arquivo = f'{Pasta}{Reg}.png'
                plt.savefig(nome_arquivo)
                

            # Chamadas de função para plotagem
            plotar_grafico_por_reg('Norte')
            plotar_grafico_por_reg('Nordeste')
            plotar_grafico_por_reg('AC-RO')
            plotar_grafico_por_reg('Sudeste-Centro-Oeste')
            plotar_grafico_por_reg('Sul')

        
    def Graficos_Por_VBA(self, Pasta):
        sns.set_theme(style="darkgrid")
        
        if self.plotar_grafico_por_reg_VBA:
            def plotar_grafico_por_reg_VBA(REG):
                PWF16_Filt_Reg = self.PWF16_Filt_NEW[self.PWF16_Filt_NEW['REG'] == REG]
                valores_VBA = PWF16_Filt_Reg['VBASEKV'].unique()
                valores_VBA.sort()

                todos_Mvar_Losses = []
                todos_percent_L1 = []
                labels = []

                for VBA in valores_VBA:
                    Resultados_maiores_230 = PWF16_Filt_Reg[PWF16_Filt_Reg['VBASEKV'] == VBA]
                    resultados_agregados_hora = Resultados_maiores_230.groupby('Hora').agg({'% L1': 'mean', 'Mvar:Losses': 'mean'})

                    # Adicione os dados de 'Mvar:Losses' e '% L1' às listas
                    todos_Mvar_Losses.append(resultados_agregados_hora['Mvar:Losses'])
                    todos_percent_L1.append(resultados_agregados_hora['% L1'])
                    
                    # Adicione o valor de VBA como rótulo
                    labels.append(f'VBA: {VBA}')

                # Crie uma figura com dois subplots empilhados verticalmente
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                plt.suptitle(f'Gráficos kV Base para Região: {REG}', fontsize=20, y=0.95)  # Ajuste o valor de y aqui e o tamanho da fonte

                # Gráfico para 'Mvar:Losses'
                for Mvar_Losses, label in zip(todos_Mvar_Losses, labels):
                    axes[0].plot(Mvar_Losses, label=label)

                axes[0].set_ylabel('Mvar:Lossses', fontsize=20)
                axes[0].legend()

                # Gráfico para '% L1'
                for percent_L1, label in zip(todos_percent_L1, labels):
                    axes[1].plot(percent_L1, label=label)

                axes[1].set_xlabel('SemiHoras', fontsize=20)
                axes[1].set_ylabel('%L1', fontsize=20)
                axes[1].legend()
            
                # Aumente o tamanho das tick labels nos eixos x e y
                axes[1].set_xticklabels(resultados_agregados_hora.index, rotation=90)
                axes[1].tick_params(axis='x', labelsize=12)
                axes[1].tick_params(axis='y', labelsize=12)

                # Defina o nome do arquivo antes de salvá-lo
                nome_arquivo = f'{Pasta}Gráficos_VBA_Região_{REG}.png'
                plt.savefig(nome_arquivo)
            plotar_grafico_por_reg_VBA('Norte')
            plotar_grafico_por_reg_VBA('Nordeste')
            plotar_grafico_por_reg_VBA('AC-RO')
            plotar_grafico_por_reg_VBA('Sudeste-Centro-Oeste')
            plotar_grafico_por_reg_VBA('Sul')
        
        if self.plotar_grafico_de_calor_1VBA:
            def plotar_grafico_de_calor_1VBA(REG):
                PWF16_Filt_Reg = self.PWF16_Filt_NEW[self.PWF16_Filt_NEW['REG'] == REG]
                valores_VBA = PWF16_Filt_Reg['VBASEKV'].unique()
                valores_VBA.sort()

                for VBA in valores_VBA:
                    Resultados_maiores_230 = PWF16_Filt_Reg[PWF16_Filt_Reg['VBASEKV'] == VBA]
                    resultados_agregados_hora = Resultados_maiores_230.groupby(['Dia', 'Hora']).agg({'% L1': 'mean', 'Mvar:Losses': 'sum'}).reset_index()

                    # Crie um gráfico de calor para 'Mvar:Losses'
                    pivot_Mvar_Losses = resultados_agregados_hora.pivot_table(index='Dia', columns='Hora', values='Mvar:Losses')
                    plt.figure(figsize=(12, 6))
                    plt.title(f'Gráfico de Calor para Mvar:Losses - Região: {REG} - kV Base: {VBA}', fontsize=20)

                    ax = sns.heatmap(pivot_Mvar_Losses, cmap='Reds_r', cbar=True, cbar_kws={'label': 'Mvar:Losses'})
                    cbar = ax.collections[0].colorbar
                    cbar.set_label('Mvar:Losses', fontsize=20)  # Alterar o tamanho da fonte da barra de cores
                    cbar.ax.tick_params(labelsize=16)

                    ax.set_xlabel('Hora', fontsize=20)
                    ax.set_ylabel('Dia', fontsize=16)
                    ax.tick_params(axis='x', labelsize=12)
                    ax.tick_params(axis='y', labelsize=12)

                    nome_arquivo = f'{Pasta}Gráfico_de_Calor_para_MVAr_VBA_{VBA}_Região_{REG}.png'
                    plt.savefig(nome_arquivo)

                    # Crie um gráfico de calor para '% L1'
                    pivot_percent_L1 = resultados_agregados_hora.pivot_table(index='Dia', columns='Hora', values='% L1')
                    plt.figure(figsize=(12, 6))
                    plt.title(f'Gráfico de Calor para % L1 - Região: {REG} - kV Base: {VBA}', fontsize=20)

                    ax = sns.heatmap(pivot_percent_L1, cmap='Reds', cbar=True, cbar_kws={'label': '% L1'})
                    cbar = ax.collections[0].colorbar
                    cbar.set_label('% L1', fontsize=20)  # Alterar o tamanho da fonte da barra de cores
                    cbar.ax.tick_params(labelsize=16)

                    ax.set_xlabel('Hora', fontsize=20)
                    ax.set_ylabel('Dia', fontsize=16)
                    ax.tick_params(axis='x', labelsize=12)
                    ax.tick_params(axis='y', labelsize=12)

                    nome_arquivo = f'{Pasta}Gráfico_de_Calor_para_porcento_L1_VBA_{VBA}_Região_{REG}.png'
                    plt.savefig(nome_arquivo)
            plotar_grafico_de_calor_1VBA('Norte')
            plotar_grafico_de_calor_1VBA('Nordeste')
            plotar_grafico_de_calor_1VBA('AC-RO')
            plotar_grafico_de_calor_1VBA('Sudeste-Centro-Oeste')
            plotar_grafico_de_calor_1VBA('Sul')
        

    def Analises_Especificas(self, Pasta):
        if self.carregamentos_baixos_e_altos:
            output_file = os.path.join(Pasta, 'Carregamentos_Altos_e_Baixos.txt')
            os.makedirs(Pasta, exist_ok=True)
            grouped = self.PWF16_Filt_NEW.groupby(['From#', 'To#', 'Hora', 'Dia'])['% L1'].mean().reset_index()

            # Em seguida, agrupe novamente apenas por 'From#' e calcule a média da média de '% L1' para cada 'From#'
            average_for_from = grouped.groupby(['From#','To#'])['% L1'].mean().reset_index()

            # Finalmente, filtre as linhas em que a média do '% L1' para 'From#' é menor ou igual a 1
            df_menos_1 = average_for_from[average_for_from['% L1'] <= 1]
            df_maior_90 = average_for_from[average_for_from['% L1'] >= 90].sort_values(by='% L1')
            df_menos_10 = average_for_from[average_for_from['% L1'] <= 10]
            # Função para realizar o processo de junção, filtragem e contagem
        
            def contar_combinacoes_e_valores(df,df_name):
                # Realize o cálculo
                merged_df = self.PWF16_Filt_NEW.merge(df[['From#', 'To#']], on=['From#', 'To#'], how='inner')
                unique_combinations = merged_df.drop_duplicates(subset=['From#', 'To#'])
                contagem_por_reg = unique_combinations['REG'].value_counts()
                contagem_valores_vbasekv_por_reg = unique_combinations.groupby(['REG', 'VBASEKV'])['VBASEKV'].count()
                total_linhas = len(unique_combinations)

                # Capture a saída em uma variável
                output = []
                output.append(f"\nResultados para {df_name} ({total_linhas} linhas no total):")
                output.append("Contagem de linhas por 'REG':")
                output.append(contagem_por_reg.to_string())
                output.append("\nContagem de valores de 'VBASEKV' por 'REG':")
                output.append(contagem_valores_vbasekv_por_reg.to_string())

                # Escreva a saída no arquivo de texto
                with open(output_file, 'a') as file:
                    file.write('\n'.join(output) + '\n')

            # Chame as funções com o arquivo de saída especificado
            contar_combinacoes_e_valores(df_menos_1, "Linhas com carregemento menores que 1")
            contar_combinacoes_e_valores(df_menos_10, "Linhas com carregemento menores que 10")
            contar_combinacoes_e_valores(df_maior_90, "Linhas com carregemento maiores que 90")
            import pandas as pd
            def contar_combinacoes_e_valores_com_nome(df, df_name, output_file):
                merged_df = self.PWF16_Filt_NEW.merge(df[['From#', 'To#']], on=['From#', 'To#'], how='inner')
                unique_combinations = merged_df.drop_duplicates(subset=['From#', 'To#'])
                df_maior_90[['From Name', 'To Name', 'REG', 'VBASEKV']] = df_maior_90.apply(lambda row: 
                    pd.Series({
                        'From Name': self.PWF16_Filt_NEW.loc[(self.PWF16_Filt_NEW['From#'] == row['From#']) & (self.PWF16_Filt_NEW['To#'] == row['To#']), 'From Name'].values[0],
                        'To Name': self.PWF16_Filt_NEW.loc[(self.PWF16_Filt_NEW['From#'] == row['From#']) & (self.PWF16_Filt_NEW['To#'] == row['To#']), 'To Name'].values[0],
                        'REG': self.PWF16_Filt_NEW.loc[(self.PWF16_Filt_NEW['From#'] == row['From#']) & (self.PWF16_Filt_NEW['To#'] == row['To#']), 'REG'].values[0],
                        'VBASEKV': self.PWF16_Filt_NEW.loc[(self.PWF16_Filt_NEW['From#'] == row['From#']) & (self.PWF16_Filt_NEW['To#'] == row['To#']), 'VBASEKV'].values[0]
                    }), axis=1)

                with open(output_file, 'a') as file:
                    file.write(" Dados sobre as linhas com carregamento maior de 90%\n")
                    file.write(df_maior_90[['From#', 'To#', 'From Name','To Name', 'REG', 'VBASEKV','% L1']].to_string(index=False) + '\n')

            # Chame as funções com o arquivo de saída especificado
            contar_combinacoes_e_valores_com_nome(df_maior_90, "Maiores 90", output_file)



        if self.maior_carregamento_para_menor_dia_e_hora:
            grouped = self.PWF16_Filt_NEW.groupby(['From#', 'To#'])['% L1'].mean().reset_index()

            # Encontrar o maior '% L1' e obter o 'Dia' e a 'Hora' correspondentes
            max_percent_L1_row = grouped.loc[grouped['% L1'].idxmax()]
            max_percent_L1 = max_percent_L1_row['% L1']
            max_percent_L1_From = max_percent_L1_row['From#']
            max_percent_L1_To = max_percent_L1_row['To#']

            # Encontrar o 'Dia' e a 'Hora' correspondentes no DataFrame original
            result = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['From#'] == max_percent_L1_From) & (self.PWF16_Filt_NEW['To#'] == max_percent_L1_To)]
            max_percent_L1_Dia = result['Dia'].values[0]
            max_percent_L1_Hora = result['Hora'].values[0]

            filtered_df = self.PWF16_Filt_NEW[(self.PWF16_Filt_NEW['Dia'] == max_percent_L1_Dia) & (self.PWF16_Filt_NEW['Hora']==  max_percent_L1_Hora)]
            sorted_df = filtered_df.sort_values(by='% L1')
            
            desired_VBASEKV = [230.0,345.0, 440.0, 500.0, 525.0]

            # Filtrar o DataFrame para os valores de 'VBASEKV' desejados
            filtered_df = sorted_df[sorted_df['VBASEKV'].isin(desired_VBASEKV)]

            # Classificar os dados em ordem decrescente de '% L1'
            filtered_df = filtered_df.sort_values(by='% L1', ascending=False)

            # Cores para as linhas e sombras
            colors = ['b', 'g', 'r', 'c','m']

            # Criar um gráfico de linha para cada valor de 'VBASEKV'
            plt.figure(figsize=(12, 6))
            for idx, VBASEKV in enumerate(desired_VBASEKV):
                df_group = filtered_df[filtered_df['VBASEKV'] == VBASEKV]
                y = df_group['% L1']
                
                # Calcular média e desvio padrão
                mean = np.mean(y)
                std = np.std(y)
                
                # Plotar a linha
                plt.plot(range(1, len(df_group) + 1), y, label=f'VBASEKV={VBASEKV}', color=colors[idx])
                
                # Plotar a média
                plt.axhline(y=mean, color=colors[idx], linestyle='--', label=f'Mean = {mean:.2f}')
                
                # Plotar a sombra com a cor correspondente
                plt.fill_between(range(1, len(df_group) + 1), mean - std, mean + std, color=colors[idx], alpha=0.2)

            # Aumentar o tamanho dos labels dos eixos x e y
            plt.xlabel('Ordem de Maior para Menor', fontsize=18)
            plt.ylabel('% L1', fontsize=18)

            # Aumentar o tamanho dos labels de legenda (VBASEKV)
            plt.legend(title='VBASEKV', loc='upper right', fontsize=12)

            # Aumentar o tamanho dos números nos eixos (major tick labels)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.title('Gráfico de % L1 por VBASEKV (Maior para Menor)', fontsize=24)
            plt.tight_layout()
            nome_arquivo = f"{Pasta}Analise_Hora_Dia_maior_carregamento.png"
            plt.savefig(nome_arquivo)

        if self.contador_linha:
            #Total de linhas mas já encaminha pra pasta
            unique_df = self.PWF16_Filt_NEW.drop_duplicates(subset=['From#', 'To#'])

            # Contando quantas vezes cada valor de 'REG' aparece no DataFrame sem duplicatas
            reg_counts = unique_df['REG'].value_counts()

            total_linhas = unique_df.shape[0]

            # Criar e escrever no arquivo de texto
            with open('Total de linhas.txt', 'w') as file:
                file.write(f"Número total de linhas no DataFrame: {total_linhas}\n")
                file.write("Contagem de valores 'REG':\n")
                file.write(reg_counts.to_string())

            # Mover o arquivo para a pasta desejada
            import shutil

            shutil.move('Total de linhas.txt', f'{Pasta}Total de linhas.txt')

        if self.contador_VKBASE:
            unique_df = self.PWF16_Filt_NEW.drop_duplicates(subset=['From#', 'To#'])

            #para cada reg quantos Vbase e o total de vbase

            # Supondo que você já tenha carregado seu DataFrame 'unique_df'
            import pandas as pd
            # Criando um DataFrame vazio para armazenar os resultados
            result_df = pd.DataFrame(columns=['REG', 'VBASEKV', 'count'])

            # Iterando sobre cada valor único de 'REG'
            for reg in unique_df['REG'].unique():
                # Obtendo um DataFrame filtrado para o valor de 'REG' atual
                reg_df = unique_df[unique_df['REG'] == reg]
                
                # Contando quantas vezes cada valor único de 'VBASEKV' aparece
                counts = reg_df['VBASEKV'].value_counts().reset_index()
                counts.columns = ['VBASEKV', 'count']
                
                # Adicionando a coluna 'REG' ao DataFrame de contagem
                counts['REG'] = reg
                
                # Adicionando os resultados ao DataFrame final
                result_df = pd.concat([result_df, counts], ignore_index=True)

            # Ordenando o DataFrame pelo valor de 'REG'
            result_df = result_df.sort_values(by=['REG'])

            # Iterando sobre cada valor único de 'REG' e imprimindo os resultados
            for reg in result_df['REG'].unique():
                print(f'REG: {reg}')
                reg_data = result_df[result_df['REG'] == reg]
                for index, row in reg_data.iterrows():
                    print(f"VBASEKV: {row['VBASEKV']} - Quantidade: {row['count']}")
                print()

            # Supondo que você já tenha carregado seu DataFrame 'unique_df'

            

            # Criando um DataFrame vazio para armazenar as contagens por região
            result_df = pd.DataFrame(columns=['REG', 'VBASEKV', 'count'])

            # Iterando sobre cada valor único de 'REG'
            for reg in unique_df['REG'].unique():
                # Obtendo um DataFrame filtrado para o valor de 'REG' atual
                reg_df = unique_df[unique_df['REG'] == reg]
                
                # Contando quantas vezes cada valor único de 'VBASEKV' aparece para esta região
                counts = reg_df['VBASEKV'].value_counts().reset_index()
                counts.columns = ['VBASEKV', 'count']
                
                # Adicionando a coluna 'REG' ao DataFrame de contagem
                counts['REG'] = reg
                
                # Adicionando os resultados ao DataFrame final
                result_df = pd.concat([result_df, counts], ignore_index=True)

            # Agora, calculando o total para cada 'VBASEKV'
            total_counts_df = result_df.groupby('VBASEKV')['count'].sum().reset_index()
            total_counts_df.columns = ['VBASEKV', 'total_count']

            # Ordenando o DataFrame pelo valor de 'VBASEKV'
            total_counts_df = total_counts_df.sort_values(by=['VBASEKV'])

            # Imprimindo os resultados
            for index, row in total_counts_df.iterrows():
                print(f"VBASEKV: {row['VBASEKV']} - Total: {row['total_count']}")

           

            # Supondo que o código para gerar os resultados já foi executado

            # Redirecionar a saída padrão para um arquivo
            with open('Quantidade VBASE.txt', 'w') as f:
                sys.stdout = f
                # Iterar sobre cada valor único de 'REG' e imprimir os resultados
                for reg in result_df['REG'].unique():
                    print(f'REG: {reg}')
                    reg_data = result_df[result_df['REG'] == reg]
                    for index, row in reg_data.iterrows():
                        print(f"VBASEKV: {row['VBASEKV']} - Quantidade: {row['count']}")
                    print()

            # Resetar a saída padrão para o console
            sys.stdout = sys.__stdout__

            # Caminho da pasta para onde você deseja mover o arquivo

            # Movendo o arquivo para a pasta específica
            shutil.move('Quantidade VBASE.txt', Pasta)


    def Analise_PF(self,Pasta):
        import os
        def IndiceLinhas(df, n):
            
            # Filter the DataFrame once
            df_filtered = df.loc[~(df['L1(MVA)'] == 9999) & df['VBASEKV'].isin([230, 345, 440, 500, 525, 765])].copy()
            
            # Calculate MW_Flow
            df_filtered.loc[:, 'MW_Flow'] = np.where(df_filtered['MW:From-To'] >= 0, df_filtered['MW:From-To'], df_filtered['MW:To-From'])
            # Calculate PI_mva
            df_filtered.loc[:, 'PI_mva'] = (df_filtered['% L1'] / 100) ** (2 * n)

            # Group by multiple columns and aggregate
            df_r_nt = df_filtered.groupby(['Dia', 'Hora',  'REG', 'VBASEKV']).agg({'MW_Flow': 'sum', 'MVA': 'sum', 'PI_mva': 'sum'})
            df_r = df_filtered.groupby(['Dia', 'Hora',  'REG']).agg({'MW_Flow': 'sum', 'MVA': 'sum', 'PI_mva': 'sum'})
            df_po = df_filtered.groupby(['Dia', 'Hora']).agg({'MW_Flow': 'sum', 'MVA': 'sum', 'PI_mva': 'sum'})

            # Calculate PFI
            df_r_nt['PFI'] = df_r_nt['MW_Flow'] / df_r_nt['MVA']
            df_r['PFI'] = df_r['MW_Flow'] / df_r['MVA']
            df_po['PFI'] = df_po['MW_Flow'] / df_po['MVA']

            # Apply PI_mva correction
            df_r_nt['PI_mva'] = df_r_nt['PI_mva'] ** (1 / (2 * n))
            df_r['PI_mva'] = df_r['PI_mva'] ** (1 / (2 * n))
            df_po['PI_mva'] = df_po['PI_mva'] ** (1 / (2 * n))

            return df_r_nt,df_r,df_po
        df_r_nt,df_r,df_po = IndiceLinhas(self.PWF16_Filt_NEW,1)
        df_r_nt = df_r_nt.reset_index()
        df_r = df_r.reset_index()
        if self.PVI:
            
            # Verificar se o caminho da pasta existe, senão criar
            if not os.path.exists(Pasta):
                os.makedirs(Pasta)

            # Cor azul para as barras
            bar_color = 'blue'
            grouped = df_r.groupby('REG')

            for name, group in grouped:
                plt.figure(figsize=(10, 6))

                # Plotar o gráfico de barras com cor azul
                plt.plot(group.index, group['PI_mva'], color=bar_color)

                # Identificar mudanças de 'Dia' para definir os rótulos dos major ticks
                xticks = []
                xtick_labels = []
                last_day = None  # Manter o último 'Dia' registrado
                for idx, row in group.iterrows():
                    if row['Dia'] != last_day:
                        xticks.append(idx)
                        xtick_labels.append(row['Dia'])
                        last_day = row['Dia']

                plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
                plt.xlabel('Dia na Hora 00-00')
                plt.ylabel('PI_mva')
                plt.title(f'PI_mva por SemiHora/{name}')  # Adicionando título com o nome da região

                plt.tight_layout()

                # Salvar o gráfico na pasta especificada
                file_path = os.path.join(Pasta, f'PI_mva_por_SemiHora_{name}.png')
                plt.savefig(file_path)
                print(f"Gráfico salvo em: {file_path}")

                plt.close()  # Fechar a figura para liberar memória



        if self.PFI_Grafico:
                        
            # Verificar se a pasta existe, senão criar
            if not os.path.exists(Pasta):
                os.makedirs(Pasta)

            for name, group in grouped:
                plt.figure(figsize=(10, 6))
                plt.plot(group.index, group['PFI'], color=bar_color)
                xticks = []
                xtick_labels = []
                last_day = None
                for idx, row in group.iterrows():
                    if row['Dia'] != last_day:
                        xticks.append(idx)
                        xtick_labels.append(row['Dia'])
                        last_day = row['Dia']
                plt.xticks(xticks, xtick_labels, rotation=45, ha='right')
                plt.xlabel('Dia na Hora 00-00')
                plt.ylabel('PFI')
                plt.title(f'PFI por SemiHora/{name}')

                plt.tight_layout()
                
                # Definir o caminho completo para salvar o gráfico na pasta desejada
                filename = os.path.join(Pasta, f'PFI_por_SemiHora_{name}.png')
                plt.savefig(filename)
                plt.close()  # Fecha a figura após salvar para liberar memória

        if self.PFI_Excel:
            import os

        

            # Verificar se a pasta existe, senão criar
            if not os.path.exists(Pasta):
                os.makedirs(Pasta)

            # Agrupar por 'REG'
            grouped_df = df_r_nt.groupby('REG')

            # Filtrar valores abaixo de 0,8 para 'PFI' e contar as linhas para cada 'VBASEKV'
            result = {}
            for reg, group in grouped_df:
                filtered_group = group[group['PFI'] < 0.8]
                count_per_vbasekv = filtered_group['VBASEKV'].value_counts().to_dict()
                result[reg] = count_per_vbasekv

            # Criar DataFrame com os resultados
            result_df = pd.DataFrame(result).fillna(0).astype(int)

            # Definir o caminho completo para salvar o arquivo xlsx na pasta desejada
            filename = os.path.join(Pasta, 'contagem_PFI_menor_0,8.xlsx')

            # Salvar em xlsx
            result_df.to_excel(filename)


