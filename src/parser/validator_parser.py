import pandas as pd
import os
import re

def parse_validation_log(input_path):
    print(f"Reading raw log: {input_path}")
    data = []
    
    with open(input_path, 'r') as f:
        for line in f:
            # Only process lines that start with a numeric timestamp
            if re.match(r'^\d+\.\d+,', line.strip()):
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    data.append(parts[:7])

    if not data:
        print("Error: No data rows found. Check your .txt file format.")
        return None

    columns = ['timestamp', 'ir', 'red', 'green', 'acc_x', 'acc_y', 'acc_z']
    df = pd.DataFrame(data, columns=columns)
    
    # Convert all columns to numeric
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    output_path = input_path.replace('.txt', '_parsed.csv')
    df.to_csv(output_path, index=False)
    print(f"Success: Created {output_path}")
    return output_path

if __name__ == "__main__":
    raw_file = "/Users/johnacogan67/cursor_oralable/data/raw/Oralable_20260223_083911.txt"
    parse_validation_log(raw_file)