import time
import sys
from Analyze_Save_Info import AnalyzeStaticCases

if __name__ == '__main__':

    start_time = time.time()

    # ************************************************************************************************
    #                                       OPÇÕES DE EJECUÇÃO
    # ************************************************************************************************
    Options_ReadProcess= {
        # ---------------------------------------------------
                            'generatescript' : False,
        # ---------------------------------------------------
                            'extract_fromcsv' : False,
                            'savecsv':True,
                            'ConvergenceAnalise' : True,
                            'busdata' : True,
                            'LinhaAnalise': True,
                            'HVDCAnalise':True,
                            'ReservaAnalise':True,
                            'IntercambiosAnalise':True,
        # ---------------------------------------------------
                            'linhascsv':False,
                            'reservacsv':False,
                            'HVDCcsv': False,
        # ---------------------------------------------------
                            'ComputeDPI': True,
                            'resumoIndice': True,
                            'PlotGeralPotencia': True,
                            'MapasPlots': True,
                            'Plot_Tensao_Geral': True,
                            'plotDPI': True,
                            'Plot_Boxplot_DPI': True,
                            'PlotIntercambios': True
                        }

    # ************************************************************************************************
    #                                              PATHS
    # ************************************************************************************************
    path_folder = 'D:/0 FERV/0 Dados PYTHON/Cenarios HPPA PRE-QUALI/MPV_(FNS Lim)_RC/'
    # path_folder = 'D:/MPV_(FNS Lim)_RC/'
    # path_folder = 'D:/MPV_(FNS Lim)_RC_test/'
    # path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/TESTE/'
    # path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2022_novos/V1A1F2 FNS Lim 2022/'
    # path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2022_novos/V1A1F2 FNS Lim 2022_ultimos_corrigidos/'
    # path_folder =  'D:/0 FERV/0 Dados PYTHON/Cenarios HPPA PRE-QUALI/V1A1F2_REV2_091123/'

    # ============================= CASOS 2026 V1A1F_===========================================
    # path_folder_1 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1/V1A1F2_RESP_FNS_lim_rev1_2026/'
    # path_folder_2 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1/V1A1F3_RESP_FNS_lim_rev1_2026/'
    # path_folder_3 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1/V1A1F4_RESP_FNS_lim_rev1_2026/'
    # path_folder_4 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1/V1A1F5_RESP_FNS_lim_rev1_2026/'
    # ============================= CASOS 2026 V1A1F_REV02===========================================
    # path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_2/V1A1F2_RESP_FNS_lim_rev2_2026/'
    
    # path_folders = [path_folder_1, path_folder_2, path_folder_3, path_folder_4]
    path_folders = [path_folder]

    for path_folder in path_folders:

        cenarios = AnalyzeStaticCases(path=path_folder, Options = Options_ReadProcess)
        cenarios.extraction_process()
        cenarios.LinhaAnalise()
        cenarios.ReservaAnalise()
        cenarios.ActiveReactivePower()
        cenarios.Plot_Tensao_Geral()
        cenarios.MapasPlots()
        cenarios.ComputeDPI()
        print('Saving Dataframes ...')
        cenarios.save_csv()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:", execution_time, "segundos")