import os
import argparse
import subprocess

def ingest_folder(folder_path):
    """
    Processes all PDF files in a given folder.

    Args:
        folder_path (str): The path to the folder containing PDF files.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The provided path '{folder_path}' is not a valid directory.")
        return

    print(f"Starting to process PDF files in: {folder_path}")

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        print(f"\n--- Processing: {pdf_file} ---")
        
        # Construct the command to run the main ingestion script
        # Using 'python3' is often more explicit on Linux systems like the Pi
        command = ["python3", "-m", "src.main", "--pdf", file_path]
        
        try:
            # Run the command and capture output
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
            print(f"--- Successfully processed {pdf_file} ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Failed to process {pdf_file} ---")
            print(f"Error: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print("Error: 'python3' command not found. Make sure Python is in your PATH.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest all PDF documents from a specified folder into Qdrant.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing PDF files.")
    
    args = parser.parse_args()
    ingest_folder(args.folder)
