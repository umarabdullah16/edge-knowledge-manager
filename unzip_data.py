import os
from pathlib import Path
import zipfile

# Set your root folder here
root_folder = '/Users/umar/Downloads/nifty_50_dataset/annual_reports'  # Change this to your main folder path

for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(dirpath, filename)
            extract_dir = os.path.splitext(zip_path)[0]
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    print(f"Extracted '{zip_path}' to '{extract_dir}'")
            except zipfile.BadZipFile:
                print(f"Warning: '{zip_path}' is not a valid zip file!")
