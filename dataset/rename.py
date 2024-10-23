import os

def rename_images_in_folder(folder_path):
    # List all files in the folder, regardless of extension
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Sort the files (optional, ensures consistent order)
    files.sort()

    # Rename each file with sequential numbers, starting from 1
    for idx, filename in enumerate(files, start=1):
        old_file_path = os.path.join(folder_path, filename)
        # Extract the file extension (e.g., .jpg, .png)
        file_extension = os.path.splitext(filename)[1]
        new_file_name = f"{idx}{file_extension}"
        new_file_path = os.path.join(folder_path, new_file_name)
        
        try:
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {filename} to {new_file_name}")
        except Exception as e:
            print(f"Error renaming {filename}: {e}")

# Define the folder path
folder_path = "no_watermark"

rename_images_in_folder(folder_path)
