import os
import shutil
import zipfile
import PARAMETER


def extract_and_place_files(zip_path, destination=PARAMETER.BASE_PROJECT):
    """
    Extract the ZIP file and recreate the directory structure, placing .bla files in their correct positions.

    Parameters:
    - zip_path (str): Path to the ZIP file.
    - destination (str): Root directory where the ZIP content should be extracted and the directory structure recreated.
    """

    # Extract the ZIP file to the destination
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)

    # Traverse through the extracted content
    for root, dirs, files in os.walk(destination):
        for file in files:
            if file.endswith('.bla'):
                # Construct the full path of the file
                full_path = os.path.join(root, file)

                # Extract the relative path (excluding the file name)
                relative_path = os.path.relpath(full_path, destination)
                dir_path = os.path.dirname(relative_path)

                # Check if the directory structure exists in the destination
                target_dir = os.path.join(destination, dir_path)
                if not os.path.exists(target_dir):
                    print(f"Directory structure for {file} does not exist, skipping...")
                    continue

                # Move the file to its respective directory
                try:
                    print(f"Moving {full_path} to   -----> {target_dir}")
                    shutil.move(full_path, target_dir)
                except Exception as e:
                    if "already exists" in str(e):
                        pass
                        # print(f"File -{full_path} already exists, skipping")

    print("You can remove the zip now...")



if __name__ == '__main__':
    path_to_zip_pt_and_pth_files = os.path.join(PARAMETER.BASE_PROJECT, "pt_and_pth_files.zip")
    extract_and_place_files(path_to_zip_pt_and_pth_files)