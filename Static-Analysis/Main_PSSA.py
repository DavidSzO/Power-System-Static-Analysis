import time
import sys
from Analyze_Save_Info import AnalyzeStaticCases

start_time = time.time()

# ************************************************************************************************
#                                       OPÇÕES DE EJECUÇÃO
# ************************************************************************************************

Options_ReadProcess= {
                        'gen_script4lines' : False,
                        'extract_fromcsv' : False,
                        'savecsv':True,
                        'ConvergenceAnalise' : True,
                        'busdata' : True,
                        'LinhaAnalise': True,
                        'HVDCAnalise':True,
                        'ReservaAnalise':True,
                        'IntercambiosAnalise':True,
                        'linhascsv':False,
                        'reservacsv':False,
                        'HVDCcsv': False,
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

path_folder = 'D:/MPV_(FNS Lim)_RC/'
# path_folder = 'D:/MPV_(FNS Lim)_RC_test/'
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/TESTE/'

# ============================= CASOS 2026 V1A1F_===========================================
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F2_RESP_FNS_lim_rev1_2026/'
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F3_RESP_FNS_lim_rev1_2026/'
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F4_RESP_FNS_lim_rev1_2026/'
# path_folder = 'D:/0 FERV/0 Dados PYTHON/CASOS 2026/V1A1F_/V1A1F5_RESP_FNS_lim_rev1_2026/'


cenarios = AnalyzeStaticCases(path=path_folder, Options = Options_ReadProcess)

cenarios.extraction_process()
cenarios.LinhaAnalise()
cenarios.ReservaAnalise()
cenarios.ActiveReactivePower()
cenarios.Plot_Tensao_Geral()
cenarios.MapasPlots()
cenarios.ComputeDPI()

# Guarda el tiempo de finalización
end_time = time.time()
# Calcula la diferencia de tiempo
execution_time = end_time - start_time
print("Tiempo de ejecución:", execution_time, "segundos")