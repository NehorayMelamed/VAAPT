import ftplib, os, time
from pathlib import Path
from typing import List
from datetime import date


def get_config_path() -> str:
    grandparent_directory = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent)
    return str(os.path.join(grandparent_directory, "application", "config.yml"))


def get_results_dir() -> str:
    grandparent_directory = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent)
    return str(os.path.join(grandparent_directory, "data", "results") + os.path.sep)


class Uploader:
    def __init__(self, ):
        self.ftp_session = ftplib.FTP("62.0.130.5")
        self.ftp_session.login("mod", "QuickSh0t")
        self.ftp_session.cwd("Palantir/247_tool")
        self.create_ftp_dirs()
        self.ftp_head = self.ftp_session.pwd()
        print(self.ftp_head)
        self.upload_config()
        self.bgs_batch = 0
        self.bgs_frame = 0
        results_dir = get_results_dir()
        self.original_dir = os.path.join(results_dir, "ORIGINAL") + os.path.sep
        self.bgs_dir = os.path.join(results_dir, "BGS") + os.path.sep
        self.ransac_dir = os.path.join(results_dir, "RANSAC") + os.path.sep
        self.inliers_dir = os.path.join(self.ransac_dir, 'INLIERS')
        self.outliers_points_dir = os.path.join(self.ransac_dir, 'OUTLIERS')
        self.regressions_dir = os.path.join(self.ransac_dir, 'REGRESSIONS')

    def upload_config(self):
        binary_cfg = open(get_config_path(), 'rb')
        self.ftp_session.storbinary(f'STOR config.yml', binary_cfg)
        binary_cfg.close()

    def get_current_formatted_date(self):
        today = date.today()
        return today.strftime("%d_%m_%y")

    def create_results_structure(self):
        self.ftp_session.mkd("ORIGINAL")
        self.ftp_session.mkd("BGS")
        self.ftp_session.mkd("RANSAC")
        self.ftp_session.cwd("RANSAC")
        self.ftp_session.mkd("INLIERS")
        self.ftp_session.mkd("OUTLIERS")
        self.ftp_session.mkd("REGRESSIONS")
        self.ftp_session.cwd("..")
    
    def create_ftp_dirs(self):
        already_existing_dates = self.ftp_session.nlst()
        todays_date = self.get_current_formatted_date()
        if todays_date not in already_existing_dates:
            self.ftp_session.mkd(todays_date)
        self.ftp_session.cwd(todays_date)
        prior_experiments = self.ftp_session.nlst()
        experiment_no = str(len(prior_experiments))
        self.ftp_session.mkd(experiment_no)
        self.ftp_session.cwd(experiment_no)
        self.create_results_structure()

    def upload_images_list(self, images: List[str], ftp_directory: str):
        # upload images to appropriate directory
        self.ftp_session.cwd(ftp_directory)
        for img in images:
            binary_img = open(img, 'rb')
            self.ftp_session.storbinary(f'STOR {img}', binary_img)
            binary_img.close()
        self.ftp_session.cwd(self.ftp_head)

    def remove_files(self, files: List[str]):
        for file in files:
            os.remove(file)

    def upload_batch(self, directory: str, dest_directory):
        # also removes files
        try:
            self.ftp_session.pwd()  #
        except ConnectionResetError:
            self.reset_connection()
        all_files = os.listdir(directory)
        original_path = os.getcwd()
        os.chdir(directory)
        self.upload_images_list(all_files, dest_directory)
        self.remove_files(all_files)
        os.chdir(original_path)

    def reset_connection(self):
        self.ftp_session = ftplib.FTP("62.0.130.5")
        self.ftp_session.login("mod", "QuickSh0t")
        self.ftp_session.cwd(self.ftp_head)

    def perform_uploads(self):
        try:
            self.ftp_session.pwd()  #
        except ConnectionResetError:
            self.reset_connection()
        self.upload_batch(self.original_dir, "ORIGINAL")
        self.upload_batch(self.bgs_dir, "BGS")
        self.upload_batch(self.inliers_dir, "RANSAC/INLIERS")
        self.upload_batch(self.outliers_points_dir, "RANSAC/OUTLIERS")
        self.upload_batch(self.regressions_dir, "RANSAC/REGRESSIONS") # PROBLEM


if __name__ == '__main__':
    print(get_config_path())
    upload_manager = Uploader()
    while True:
        upload_manager.perform_uploads()
        print("UPLOAD COMPLETE")
        time.sleep(1)
