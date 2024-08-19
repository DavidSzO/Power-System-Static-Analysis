import pandas as pd
import os, shutil
import re

class Opf_Simulation():

    def __init__(self, path, nome, pathtosave, copypwf):
        
        self.path_folder = path
        self.nome = nome
        self.path_main = pathtosave
        
        self.OPFnoConvergidos(copypwf)

    def OPFnoConvergidos(self, copypwf):

        path_folder = self.path_folder
        files_and_directories = os.listdir(path_folder)
        days = [nomes_archivos for nomes_archivos in files_and_directories if 'DS20' in nomes_archivos] 
        days.sort() 

        # Foldername = 'RerodadoOPF_' + self.nome 
        # folder_path = os.path.join(self.path_main, Foldername)

        folder_path = os.path.join(self.path_main)
        os.makedirs(folder_path, exist_ok=True)

        contenidos = []
        for dayfolder in days:
            
            path_to = os.path.join(folder_path, dayfolder)
            os.makedirs(path_to, exist_ok=True)
            path_to_output = path_to
            path_to = path_to + '/Output'
            os.makedirs(path_to, exist_ok=True)
            # path_to_2 = path_to + '/rodados_serialmente'
            # os.makedirs(path_to_2, exist_ok=True)

            path_from = path_folder + dayfolder + '/Output'
            arquivos_dia = os.listdir(path_from)
            nomespwf = [i for i in os.listdir(path_to_output)  if (i.startswith('PTOPER_')) & (i.endswith('.pwf')) ]

            # Copia cada arquivo ao directorio de destino
            if copypwf:
                for archivo in nomespwf:
                    try: 
                        ruta_archivo_origen = os.path.join(path_to_output, archivo)
                        ruta_archivo_destino = os.path.join(path_to, archivo)
                        shutil.copy(ruta_archivo_origen, ruta_archivo_destino)
                    except:
                        print('No se encontrou o ' + archivo + ' no directorio')

            for archivo in ['ORGANON.prm','DCShunt_Revisado.def', 'BNT1.dat', 'TENSAO_FPO.opf', 'Model.dyn', 'SEP_BMONTE_ITAIPU_CORB.sps', 'SelectedEvents.evt']: #'TENSAO_FPO.opf' 'Novo.scd',
                try: 
                    ruta_archivo_origen = os.path.join(path_from, archivo)
                    ruta_archivo_destino = os.path.join(path_to, archivo)
                    shutil.copy(ruta_archivo_origen, ruta_archivo_destino)
                    # ruta_archivo_destino_2 = os.path.join(path_to_2, archivo)
                    # shutil.copy(ruta_archivo_origen, ruta_archivo_destino_2)
                except:
                    print('No se encontrou o ' + archivo + ' no directorio')

            # *************************************************************************************
            # *************************************************************************************
            semi_hours = []
            for hour in range(24):
                semi_hours.append(f"{hour:02d}:00")
                semi_hours.append(f"{hour:02d}:30")
            semi_hours.append(f"{24:02d}:00")

            text = []
            for i in nomespwf:
                idx1 = semi_hours.index(i.strip().split('_')[2].replace('h',':'))
                idx2 = idx1 + 1
                text.append([semi_hours[idx1],semi_hours[idx2],i])

            SCNpath = path_to + "/Novo.scd" 
            with open(SCNpath, 'w') as f:
                for idx2, filespwf in enumerate(text):
                    f.write('REF ' + filespwf[0] +', '+ filespwf[1] + ', "' + filespwf[2] +'"')
                    f.write('\n')
                    if idx2 == len(text)-1:
                        f.write('END')

            path_script_org = path_to + "/Script_scenarios.txt"  
            with open(path_script_org, 'w') as f:
                
                f.write('OPEN "' + path_to + '/ORGANON.prm"')
                f.write('\n')
                f.write('OPEN "' + path_to + '/Novo.scd"')
                f.write('\n')
                f.write('OPEN "' + path_to + '/TENSAO_FPO.opf"')
                f.write('\n')
                f.write('OPEN "' + path_to + '/DCShunt_Revisado.def"')
                f.write('\n')
                f.write('SCNRUN')
                f.write('\n')
                # f.write('ALLNTWPF "' + path_to + '"')
                # f.write('\n')
                
            with open(path_script_org, 'r') as f:
                contenidos.append(f.read())

        contenido_combinado = '\n'.join(contenidos)
        nome = self.path_main + 'ScriptScenarios.txt'
        with open(nome, 'w') as f:
            f.write(contenido_combinado)

        print(' => As pastas com os arquivos para rodar o OPF e PWF no Organon foram criadas')


