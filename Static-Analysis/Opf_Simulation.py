import pandas as pd
import os, shutil
import os
import csv
import glob

class Opf_Simulation():

    def __init__(self, path, nome, pathtosave, copypwf, GerDSAFile=False):
        
        self.path_folder = path
        self.nome = nome
        self.path_main = pathtosave
        self.copypwf = copypwf
        self.GerDSAFile = GerDSAFile
    def OPF_Process(self):

        path_folder = self.path_main
        files_and_directories = os.listdir(path_folder)
        days = [nomes_archivos for nomes_archivos in files_and_directories if 'DS20' in nomes_archivos] 
        days.sort() 
        contenidos = []
        contenidosdyn = []

        for dayfolder in days:
            
            path_ = os.path.join(path_folder, dayfolder)
            path_to_output = path_ + '/Output'
            os.makedirs(path_to_output, exist_ok=True)

            path_from = self.path_folder
            nomespwf = [i for i in os.listdir(path_)  if (i.startswith('PTOPER_')) & (i.endswith('.pwf')) ]
            nomesntw = [i for i in os.listdir(path_to_output)  if (i.endswith('.ntw')) ]

            # Copia cada arquivo ao directorio de destino
            if self.copypwf:
                for archivo in nomespwf:
                    try: 
                        ruta_archivo_origen = os.path.join(path_, archivo)
                        ruta_archivo_destino = os.path.join(path_to_output, archivo)
                        shutil.copy(ruta_archivo_origen, ruta_archivo_destino)
                    except:
                        print('No se encontrou o ' + archivo + ' no directorio')

            for archivo in ['ORGANON.PRM','DCShunt_Revisado.def', 'BNT1.dat', 'TENSAO_FPO.opf', 'Model.dyn', 'SEP_BMONTE_ITAIPU_CORB.sps', 'SelectedEvents.evt']: #'TENSAO_FPO.opf' 'Novo.scd',
                try: 
                    ruta_archivo_origen = os.path.join(path_from, archivo)
                    ruta_archivo_destino = os.path.join(path_to_output, archivo)
                    shutil.copy(ruta_archivo_origen, ruta_archivo_destino)
                    # ruta_archivo_destino_2 = os.path.join(path_to_output_2, archivo)
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

            SCNpath = path_to_output + "/Novo.scd" 
            with open(SCNpath, 'w') as f:
                for idx2, filespwf in enumerate(text):
                    f.write('REF ' + filespwf[0] +', '+ filespwf[1] + ', "' + filespwf[2] +'"')
                    f.write('\n')
                    # if idx2 == len(text)-1:
                    #     f.write('END')

            path_script_org = path_to_output + "/Script_scenarios.txt"  
            with open(path_script_org, 'w') as f:
                
                f.write('OPEN "' + path_to_output + '/ORGANON.PRM"')
                f.write('\n')
                f.write('OPEN "' + path_to_output + '/Novo.scd"')
                f.write('\n')
                f.write('OPEN "' + path_to_output + '/TENSAO_FPO.opf"')
                f.write('\n')
                f.write('OPEN "' + path_to_output + '/DCShunt_Revisado.def"')
                f.write('\n')
                f.write('SCNRUN')
                f.write('\n')
                # f.write('ALLNTWPF "' + path_to_output + '"')
                # f.write('\n')
                
            with open(path_script_org, 'r') as f:
                contenidos.append(f.read())

            if self.GerDSAFile:

                for i in nomesntw:
                    path_dsa = path_to_output + f"/{i.replace('.ntw','.dsa')}"
                    with open(path_dsa, 'w') as f:
        
                        f.write(i)
                        f.write('\n')
                        f.write('Model.dyn')
                        f.write('\n')
                        f.write('DCShunt_Revisado.def')
                        f.write('\n')
                        f.write('SelectedEvents.evt')
                        f.write('\n')
                        f.write('SEP_BMONTE_ITAIPU_CORB.sps')
                        f.write('\n')


                path_dyn = path_to_output + "/Script_DynSimulation.txt"  
                with open(path_dyn, 'w') as f:
                    for i in nomesntw:
                        path_dsa = path_to_output + f"/{i.replace('.ntw','.dsa')}"
                        f.write('OPEN ' + path_dsa)
                        f.write('\n')
                        f.write('DSA DOP')
                        f.write('\n')
                        f.write('\n')

                with open(path_dyn, 'r') as f:
                    contenidosdyn.append(f.read())

        contenido_combinado = '\n'.join(contenidos)
        nome = self.path_main + 'ScriptScenarios.txt'
        with open(nome, 'w') as f:
            f.write(contenido_combinado)
        print(' => As pastas com os arquivos para rodar o OPF e PWF no Organon foram criadas')

        if self.GerDSAFile:

            contenido_combinado_dyn = '\n'.join(contenidosdyn)
            nome = self.path_main + 'ScriptDynamic.txt'
            with open(nome, 'w') as f:
                f.write(contenido_combinado_dyn)
            print(' => As pastas com os arquivos para rodar as simulações dinâmicas no Organon foram criadas')

    def HVDCchanger(self):
        # Function to fill elos_cc from CSV file
        def fill_elos_cc(elos_cc, csv_file):
            with open(csv_file, newline='', encoding='utf-8') as elos_file:
                reader = csv.reader(elos_file, delimiter=';')
                for row_idx, row in enumerate(reader):
                    elos_cc.append(row[:3])  # Taking first 3 columns as in C# code
                    # print(row_idx + 1)

        # Directory containing subdirectories with .pwf files
        read = self.path_main
        dirs = [d for d in glob.glob(os.path.join(read, '*')) if os.path.isdir(d)]

        # CSV file containing elos_cc data
        csv_file = r"Static-Analysis/RECURSOS/hvdc info.csv"
        elos_cc = []

        # Fill elos_cc data from CSV
        fill_elos_cc(elos_cc, csv_file)

        # Iterate over each directory in the main directory
        for dir_path in dirs:
            # Get all .pwf files in the directory
            output_dir = os.path.join(dir_path, "Output")
            files = glob.glob(os.path.join(output_dir, "*.pwf"))
            
            # Iterate through each file
            for file_path in files:
                row = 0
                
                # Count lines in the .pwf file
                with open(file_path, 'r') as pwf_file:
                    pwf_lines = pwf_file.readlines()

                # Initialize array to store file lines
                pwf = [""] * (len(pwf_lines) + 1)
                
                # Process each line of the .pwf file
                for row_idx, pwf_file_line in enumerate(pwf_lines):
                    pwf[row_idx] = pwf_file_line
                    line_split = pwf_file_line.split()

                    # Look for specific condition to modify the lines
                    if row_idx >= 2 and pwf[row_idx - 2].strip() == "FBAN":
                        for elos_row in elos_cc:
                            if elos_row[0].strip() == line_split[0].strip() or elos_row[2].strip() == line_split[0].strip():
                                if line_split[1].strip() == "F":
                                    pwf[row_idx] = pwf_file_line.replace("F", "D")

                # Write the output back to a new folder inside the current directory
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, os.path.basename(file_path))

                # Write the modified content to the output file
                with open(output_file, 'w', encoding='utf-8') as output:
                    for line in pwf:
                        output.write(line)

        print("Processing complete.")