import os
import csv
import glob

def HVDCchanger(path):
    # Function to fill elos_cc from CSV file
    def fill_elos_cc(elos_cc, csv_file):
        with open(csv_file, newline='', encoding='utf-8') as elos_file:
            reader = csv.reader(elos_file, delimiter=';')
            for row_idx, row in enumerate(reader):
                elos_cc.append(row[:3])  # Taking first 3 columns as in C# code
                # print(row_idx + 1)

    # Directory containing subdirectories with .pwf files
    read = path
    dirs = [d for d in glob.glob(os.path.join(read, '*')) if os.path.isdir(d)]

    # CSV file containing elos_cc data
    csv_file = r"RECURSOS/hvdc info.csv"
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
