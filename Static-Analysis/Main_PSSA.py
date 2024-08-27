import time
import sys
from Analyze_Save_Info import AnalyzeStaticCases
from Opf_Simulation import Opf_Simulation
from Read_Process_Cases import ReadScenarios

if __name__ == '__main__':

    start_time = time.time()

    # ************************************************************************************************
    #                                       OPÇÕES DE EJECUÇÃO
    # ************************************************************************************************
    Options_ReadProcess= {
                            'Norm': 2,          # Write None for using infinite norm in voltage analysis
                            'OneCase': 1,       # (1) for All cases or (2) for Just One Case analysis
                            'SavePath':"C:/Users/David/OneDrive/Documents/FERV_documentos/",
        # ---------------------------------------------------
                            'generatescript' : False,   # Put TRUE just for generate the script for simulation and saving the flows in Organon
                            'OPFsimulation': False,
                            'OnlyPWF_datagen': False,   # Put TRUE just for generate the data for Interconnection and Line Flow Analysis
                            'extract_fromcsv' :False,   # Put TRUE just in the first simulation, once the ProcessedDataBase.csv is generated it is not necessary
                            'savedata':True,            # To save the data of the electric variables in the folders
                            'busdata' : True,           # Let like TRUE
        # ---------------------------------------------------
                            'ConvergenceData' : True,   # To analyze just the converged cases 
                            'ReadPWF_files':True,
                            'LinhasData': True,
                            'HVDCData':True,
                            'ReservaData':True,
                            'IntercambiosData':True,
                            'ComputeDPI': True,
                            'resumoIndice': True,
        # ---------------------------------------------------
                            'linhascsv':False,          # Put TRUE once is generated the LinhasInfo file
                            'reservacsv':False,         # Put TRUE once is generated the ReserveInfo file
                            'HVDCcsv': False,           # Put TRUE once is generated the HVDCinfo file
        # ---------------------------------------------------
                            'PlotGeralPotencia': True,
                            'MapasPlots': True,
                            'Plot_Tensao_Geral': True,
                            'plotDPI': True,
                            'Plot_Boxplot_DPI': True,
                            'PlotIntercambios': False
                        }
    
    # ************************************************************************************************
    #                                        PATHS FOR ANALYSYS
    # ************************************************************************************************

    # =============================             CASOS 2022          ===========================================
    # path_folder = 'D:/MPV_(FNS Lim)_RC/'
    # path_folder = 'C:/Users/david/OneDrive/Documents/test organon/Sistema Brasileiro'
    # path_folder_1 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2022/Novos com FNS Lim/V1A1F2 FNS Lim 2022/'
    # path_folder_2 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2022/Novos com FNS Lim/V1A1F2 FNS Lim 2022_OPF/'
    # path_folder ='D:/0 FERV/0 Dados PYTHON/CASOS 2022/Antigos/MPV_(FNS Lim)_RC/'
    # path_folder_3 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2022/Novos com FNS Lim/V2A2F2 FNS Lim 2022/'
    # path_folder_4 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2022/Novos com FNS Lim/MPA FNS lim_novos/'

    # =============================         CASOS 2026 V1A1         ===========================================
    # path_folder_1 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1_02/V1A1F2_RESP_FNS_lim_rev1_2026/'
    # path_folder_1 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_2/V1A1F2_RESP_FNS_lim_rev2_2026/'
    # path_folder_2 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1_02/V1A1F3_RESP_FNS_lim_rev1_2026/'
    # path_folder_3 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1_02/V1A1F4_RESP_FNS_lim_rev1_2026/'
    # path_folder_4 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_1_02/V1A1F5_RESP_FNS_lim_rev1_2026/'

    # path_folder_0  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_5/V1A1F2_rev5/'
    # path_folder_1 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_5/V1A1F3_rev5/'
    # path_folder_2  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_5/V1A1F4_rev5/'
    # path_folder_3 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/REV_5/V1A1F5_rev5/'

    # =============================         CASOS 2026 V2A2F       =========================================== 
    # path_folder_5 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_2/V2A2F2_REV02_2026/'
    # path_folder_6 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_2/V2A2F3_REV02_2026/'
    # path_folder_7 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_2/V2A2F4_REV02_2026/'
    # path_folder_8 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_2/V2A2F5_REV02_2026/'

    # path_folder  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_5/V2A2F2_rev5/'
    # path_folder_11 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_5/V2A2F3_rev5/'
    # path_folder_2  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_5/V2A2F4_rev5/'
    # path_folder_3 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/REV_5/V2A2F5_rev5/'
    #     
    # =============================         CASOS 2026 V3A3F       ===========================================
    # path_folder_9 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/REV_1/V3A3F2_REV1_2026/'
    # path_folder_10 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/REV_1/V3A3F3_REV1_2026/'
    # path_folder_11 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/REV_1/V3A3F4_REV1_2026/'
    # path_folder_12 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/REV_1/V3A3F5_REV1_2026/'

    # ************************************************************************************************
    #                           PATHS FOR ANALYSYS SIMULATED IN ORGANON BY UNICAMP
    # ************************************************************************************************

    # path_folder_11  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/Unicamp/V1A1F2_RESP_in_FNS_lim_Unicamp/'
    # path_folder_22  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/Unicamp/V1A1F3_RESP_in_FNS_lim_Unicamp/'
    # path_folder_33  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/Unicamp/V1A1F4_RESP_in_FNS_lim_Unicamp/'
    # path_folder_44  = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/Unicamp/V1A1F5_RESP_in_FNS_lim_Unicamp/'

    path_folder_11 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/Unicamp/V2A2F2_RESP_in_FNS_lim_Unicamp/'
    path_folder_22 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/Unicamp/V2A2F3_RESP_in_FNS_lim_Unicamp/'
    # path_folder_33 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/Unicamp/V2A2F4_RESP_in_FNS_lim_Unicamp/'
    # path_folder_44 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V2A2F_/Unicamp/V2A2F5_RESP_in_FNS_lim_Unicamp/'

    # path_folder_11 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/Unicamp/V3A3F2_RESP_in_FNS_lim_Unicamp/'
    # path_folder_22 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/Unicamp/V3A3F3_RESP_in_FNS_lim_Unicamp/'
    # path_folder_33 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/Unicamp/V3A3F4_RESP_in_FNS_lim_Unicamp/'
    # path_folder_44 = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V3A3F_/Unicamp/V3A3F5_RESP_in_FNS_lim_Unicamp/'

    # ============================= List of PATHS ===========================================
    # path_folders = [path_folder_1, path_folder_2, path_folder_3, path_folder_4, path_folder_5, path_folder_6, path_folder_7, path_folder_8, path_folder_9, path_folder_10, 
    # path_folders = [path_folder_22, path_folder_33, path_folder_44]
    path_folders = [path_folder_11, path_folder_22]
    for path_folder in path_folders:

        if Options_ReadProcess['generatescript']:
            cases = ReadScenarios(path_folder, '', PO = False)
            cases.generate_script()
        elif Options_ReadProcess['OPFsimulation']:
            # =============== PARA GERAR O SCRIPT PARA RODAR O OPF E PWF DOS CASOS NO ORGANON ==================
            # ==================================================================================================
            path_folderFrom = 'Static-Analysis/RECURSOS/OPF_files/'  #Ese path deve conter as ultimas modificações dos arquivos necesarios para rodar o OPF no Organon (Ex: file.prm, file.def, file.opf, ...)
            nomefolder = ''
            OPF = Opf_Simulation(path_folderFrom, nomefolder, path_folder, copypwf=True) # Para gerar as pastas
            OPF.OPF_Process()
            OPF.HVDCchanger()
        else:
            cenarios = AnalyzeStaticCases(path=path_folder, Options = Options_ReadProcess)
            cenarios.extraction_process()
            cenarios.LinhaAnalise()
            cenarios.ReservaAnalise()
            cenarios.ActiveReactivePower()
            cenarios.Plot_Tensao_Geral()
            cenarios.MapasPlots()
            cenarios.ComputeDPI()
            cenarios.save_csv()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:", execution_time, "segundos")

