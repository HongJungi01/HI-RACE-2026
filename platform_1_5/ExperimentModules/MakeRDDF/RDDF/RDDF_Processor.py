import csv
import os

def process_rddf(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} file not found.")
        return

    ext = os.path.splitext(input_path)[1].lower()
    is_csv = (ext == ".csv")

    # 1. Read
    with open(input_path, 'r', newline='') as f:
        if is_csv:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
        else:
            reader = csv.reader(f, delimiter='\t')
            headers = None
            rows = list(reader)

    # 2. Modify Headers (CSV only)
    if headers and len(headers) >= 5:
        headers[3] = 'Velocity'
        headers[4] = 'Evt'

    # 3. Process Data
    data_only = [row[1:] for row in rows]
    data_only.reverse()

    # 4. Write
    out_ext = os.path.splitext(output_path)[1].lower()
    with open(output_path, 'w', newline='') as f:
        if out_ext == ".csv":
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
        else:
            writer = csv.writer(f, delimiter='\t')
        for i, data in enumerate(data_only):
            new_row = [i] + data
            writer.writerow(new_row)

    print(f"Successfully created: {output_path}")

if __name__ == "__main__":
    # Path configuration
    base_dir = r"c:\Users\JungiHong\OneDrive\Hongik\HI-RACE\2026-01\platform_1_5\ExperimentModules\MakeRDDF\RDDF"
    input_file = os.path.join(base_dir, "competition.txt")
    output_file = os.path.join(base_dir, "competition_reversed.txt")
    
    process_rddf(input_file, output_file)
