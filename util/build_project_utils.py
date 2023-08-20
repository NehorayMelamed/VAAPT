import glob
import os
import shutil


def calculate_total_size_in_gb(file_paths):
    total_size_bytes = sum(os.path.getsize(file) for file in file_paths if os.path.isfile(file))
    total_size_gb = total_size_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    return total_size_gb


def find_pt_and_pth_files(directory):
    # Set the current working directory
    os.chdir(directory)

    # Use glob to find all .pt and .pth files recursively
    pt_files = glob.glob("**/*.pt", recursive=True)
    pth_files = glob.glob("**/*.pth", recursive=True)

    pt_and_pth = pth_files +pt_files
    final_list = []
    for file in pt_and_pth:
        if str(file).startswith("venv") or str(file).startswith("data"):
            continue
        else:
            final_list.append(file)

    return final_list

def create_zip_from_files(file_paths, zip_name='pt_and_pth_files.zip'):
    # Create a temporary directory
    temp_dir = 'temp_models_dir'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    # Copy files to the temporary directory
    for file_path in file_paths:
        print(f"Copying {file_path}")
        dest_path = os.path.join(temp_dir, file_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file_path, dest_path)


    print("Zipping...")
    # Zip the directory
    shutil.make_archive(zip_name.replace('.zip', ''), 'zip', temp_dir)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)


def create_directories_for_files(root_dir, file_paths):
    """
    Create the necessary directories for each path in the specified root directory.

    Parameters:
    - root_dir (str): The root directory where the directories should be created.
    - file_paths (list of str): List of paths to .pt or .pth files.
    """
    for path in file_paths:
        # Combine the root directory with the current path
        full_path = os.path.join(root_dir, path)

        # Extract the directory part of the combined path
        dir_path = os.path.dirname(full_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)



def copy_files_to_directories(root_dir, file_paths):
    """
    Create the necessary directories in the specified root directory and copy the .pt or .pth files.

    Parameters:
    - root_dir (str): The root directory where the directories should be created and files should be copied.
    - file_paths (list of str): List of paths to .pt or .pth files.
    """
    for path in file_paths:
        # Combine the root directory with the current path
        full_path = os.path.join(root_dir, path)

        # Extract the directory part of the combined path
        dir_path = os.path.dirname(full_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Copy the .pt or .pth file to the created directory
        shutil.copy(path, full_path)





if __name__ == '__main__':
    # # Replace 'path_to_project' with the path to your project directory
    # project_directory = '/home/nehoray/PycharmProjects/Shaback'
    # files = find_pt_and_pth_files(project_directory)
    #
    # files_list = []
    # # Print the found files
    # for file in files:
    #     if not ( file.startswith("data") or  file.startswith("venv") ):
    #         files_list.append(file)
    #
    # print(files_list)
    # print(calculate_total_size_in_gb(files_list))

    base_project = "/home/nehoray/PycharmProjects/Shaback"

    pt_pth_files = find_pt_and_pth_files(base_project)

    print(pt_pth_files)

    # create_zip_from_files(pt_pth_files)

    # Let's test the function with a sample directory and paths
    sample_root = "/home/nehoray/PycharmProjects/Shaback/backup"

    create_directories_for_files(sample_root, pt_pth_files)

    copy_files_to_directories(sample_root, pt_pth_files)
    # Check if directories were created
    os.listdir(sample_root)


