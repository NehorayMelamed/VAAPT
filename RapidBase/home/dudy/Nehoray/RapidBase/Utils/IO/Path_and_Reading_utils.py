
import glob
import os
import fnmatch
#from path import path
import sys
from fnmatch import filter
from functools import partial
from itertools import chain
from os import path, walk
import pathlib
from os import mkdir
import lmdb
import numpy as np 
import cv2
import torch
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import *
from RapidBase.Utils.MISCELENEOUS import get_random_start_stop_indices_for_crop
import numpngw

# Default Image: #
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import get_full_shape_torch
default_image_filename_to_load1 = r'/home/mafat/Dudy_Karl/RDND/RapidBase/Data/0846.png'
default_image_filename_to_load2 = r'/home/mafat/Dudy_Karl/RDND/RapidBase/Data/0850.png'
default_image_filename_to_load3 = r'/home/mafat/Dudy_Karl/RDND/RapidBase/Data/0886.png'
default_video_folder_path = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\GoPro_video'
outliers_full_path = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/outliers.pt'

#TODO: i need to fix this more generally, perhapse even delete some files, but gif and png files in the imagenet i got from or don't work so now TEMPORARILY i'm not including png in the IMG_EXTENTIONS
IMG_EXTENSIONS = ['.jpg',
                  '.JPG',
                  '.tif',
                  '.tiff',
                  '.jpeg',
                  '.JPEG',
                  '.png',
                  '.PNG',
                  '.ppm',
                  '.PPM',
                  '.bmp',
                  '.BMP']
IMG_EXTENSIONS_NO_PNG = ['.jpg',
                  '.JPG',
                '.tif','.tiff',
                  '.jpeg',
                  '.JPEG',
                  '.ppm',
                  '.PPM',
                  '.bmp',
                  '.BMP']
IMG_EXTENSIONS_PNG = ['.png','.PNG']

video_file_extensions = (
'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
'.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
'.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
'.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
'.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
'.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
'.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
'.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
'.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
'.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
'.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
'.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
'.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
'.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
'.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
'.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
'.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
'.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
'.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
'.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
'.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
'.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
'.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
'.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
'.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
'.zm1', '.zm2', '.zm3', '.zmv'  )

def is_video_file(filename):
    return os.path.splitext(filename)[-1] in video_file_extensions





########################################################################################################################################
####  Path Functions: ####
################################
def path_get_files_recursively(main_path, string_to_find='', flag_full_path=True):
    files = []
    for r,d,f in os.walk(main_path):
        for file in f:
            if string_to_find in file:
                if flag_full_path:
                    files.append(os.path.join(r, file))
                else:
                    files.append(file)
    return files

def path_get_files_from_folder(path, number_of_files=np.inf, flag_recursive=False, string_pattern_to_search='*', flag_full_path=True):
    count = 0
    image_filenames_list = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        if count >= number_of_files and number_of_files != np.inf:
            break
        for fname in sorted(fnames):
            if count >= number_of_files and number_of_files != np.inf:
                break

            # if string_match_pattern(fname, string_pattern_to_search):
            if string_match_pattern(os.path.join(dirpath, fname), string_pattern_to_search):
                if flag_full_path:
                    img_path = os.path.join(dirpath, fname)
                else:
                    img_path = fname

                image_filenames_list.append(img_path)
                count += 1

        if flag_recursive == False:
            break
    return image_filenames_list


def path_get_current_working_directory():
    return os.getcwd();

def path_get_mother_folder_path(path):
    return os.path.dirname(os.path.realpath(path))

def path_get_folder_name(path):
    return os.path.basename(path)

def path_get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]



def path_change_current_working_directory(path):
    return os.chdir(path)

def path_fix_path_for_linux(path):
    return path.replace('\\','/')

def path_make_directory_from_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_make_directories_from_paths(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def path_make_directory_and_rename_if_needed(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def path_make_path_if_none_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_create_path_if_none_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_is_valid_directory(path):
    return os.path.isdir(path)


def get_path_size_in_MB(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size/1e6


#########################################################################################################################################





def video_get_mat_from_figure(fig, wanted_shape=(480*2,480*2)):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.resize(data, wanted_shape)
    return data

#TODO: there is plt.savefig to save me time probably
#TODO: remember how to use this. maybe fig = plt.figure() ?
def save_mat_from_figure(fig, wanted_shape=(480*2,480*2), full_filename=''):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.resize(data, wanted_shape)
    cv2.imwrite(full_filename, data)
    return data


def video_read_video_to_numpy_tensor(input_video_path: str, frame_index_to_start, frame_index_to_end):
    if os.path.isfile(input_video_path) is False:
        raise FileNotFoundError("Failed to convert video to numpy array")
    print("Downloading target sub video to torch... it may take a few moments")

    # return skvideo.io.vread(input_video_path)
    ### Get video stream object: ###
    video_stream = cv2.VideoCapture(input_video_path)
    # video_stream.open()
    all_frames = []
    frame_index = 0
    while video_stream.isOpened():
        flag_frame_available, current_frame = video_stream.read()
        if frame_index < frame_index_to_start:
            frame_index += 1
            continue
        elif frame_index == frame_index_to_end:
            break

        if flag_frame_available:
            all_frames.append(current_frame)
            frame_index += 1
        else:
            break
    video_stream.release()
    # print("\n\n\n\npre stack")
    full_arr = np.stack(all_frames)
    # print("post stack")
    return full_arr



def video_create_movie_from_images_in_folder(images_path, frame_rate=25):
    files_list = path_get_files_recursively(images_path, '')

    image_number_list = []
    for file in files_list:
        image_number_list.append(int(os.path.split(file)[1].split('_')[0]))
    sorted_indices_list = np.argsort(image_number_list)

    ### Get sample image: ###
    image = cv2.imread(files_list[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    height, width, channels = image.shape

    ### Create video: ###
    video_name = os.path.join(images_path, 'video_out.avi')
    video_object = cv2.VideoWriter(video_name, 0, frame_rate, (width * 1, height * 1))
    for index in sorted_indices_list:
        # print(files_list[index])
        current_image = cv2.imread(files_list[index])
        video_object.write(current_image)

    cv2.destroyAllWindows()
    video_object.release()






#########################################################################################################################################
####  Reading Images: ####
##########################
def assert_image_okay_v1():
    1;

from RapidBase.Utils.MISCELENEOUS import string_match_pattern
def get_image_filenames_from_folder(path, number_of_images=np.inf, allowed_extentions=IMG_EXTENSIONS, flag_recursive=False, string_pattern_to_search='*'):
    count = 0
    image_filenames_list = []
    for dirpath, _, fnames  in sorted(os.walk(path)):
        if count >= number_of_images and number_of_images!=np.inf:
            break
        for fname in sorted(fnames):
            if count>=number_of_images and number_of_images!=np.inf:
                break
            elif is_image_file(fname, img_extentions=allowed_extentions):
                # if string_match_pattern(fname, string_pattern_to_search):
                if string_match_pattern(os.path.join(dirpath, fname), string_pattern_to_search):
                    img_path = os.path.join(dirpath, fname)
                    image_filenames_list.append(img_path)
                    count += 1
        if flag_recursive == False:
            break
    return image_filenames_list

from glob import glob

def path_get_folder_filename_and_extension(input_filename):
    filename_extension = os.path.splitext(input_filename)[-1]
    folder, filename = os.path.split(input_filename)
    filename_without_extension = filename[0:-len(filename_extension)]
    return folder, filename, filename_without_extension, filename_extension

def path_get_all_filename_parts(input_filename):
    return path_get_folder_filename_and_extension(input_filename)

def path_get_folder_names(path, flag_recursive=False):
    # path = r'/home/mafat/Insync/dudy_karl/Google Drive - Shared drives/GalGalaim/GalGalaim/Algo-Team/DataSets/Gimbaless/Experiment 09.03.2021'
    # flag_recursive = True
    folders_list = []

    if flag_recursive == False:
        filenames_in_folder = glob(path+'/*')
        for filename in filenames_in_folder:
            if os.path.isdir(filename):
                if filename != path:
                    folders_list.append(filename)
    else:
        # Look recursively at all current and children path's in given path and get all binary files
        for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
            if dirpath != path:
                folders_list.append(dirpath)
        folders_list = set(folders_list)
    return folders_list;


def get_binary_filenames_from_folder(path, flag_recursive=False):
    binary_filenames_list = []
    # Look recursively at all current and children path's in given path and get all binary files
    for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
        for fname in sorted(fnames):
            if fname.split('.')[-1] == 'bin':
                binary_file_full_filename = os.path.join(dirpath, fname)
                binary_filenames_list.append(binary_file_full_filename)
        if flag_recursive == False:
            break;
    return binary_filenames_list;


import natsort
import re
import os
#TODO: it seems that \w* is deprecated???? understand what else to do!!!!
def get_filenames_from_folder(path, flag_recursive=False, max_number_of_filenames=np.inf, flag_full_filename=True, string_pattern_to_search='*', flag_sort=False):
    count = 0
    filenames_list = []
    for dirpath, _, fnames  in sorted(os.walk(path)):
        if count >= max_number_of_filenames and max_number_of_filenames != np.inf:
            break
        for fname in sorted(fnames):
            if count >= max_number_of_filenames and max_number_of_filenames != np.inf:
                break
            elif string_match_pattern(os.path.join(dirpath, fname), string_pattern_to_search):
                full_filename = os.path.join(dirpath, fname)
                if flag_full_filename:
                    final_filename = full_filename
                else:
                    final_filename = fname
                filenames_list.append(final_filename)
                count += 1
        if flag_recursive == False:
            break

    if flag_sort:
        filenames_list = natsort.natsorted(filenames_list)

    return filenames_list


def get_filenames_from_folder_string_pattern(path, flag_recursive=False, max_number_of_filenames=np.inf, flag_full_filename=True, string_pattern_to_search='*', flag_sort=False):
    count = 0
    filenames_list = []
    for dirpath, _, fnames  in sorted(os.walk(path)):
        if count >= max_number_of_filenames and max_number_of_filenames != np.inf:
            break
        for fname in sorted(fnames):
            if count >= max_number_of_filenames and max_number_of_filenames != np.inf:
                break
            elif string_match_pattern(os.path.join(dirpath, fname), string_pattern_to_search):
                full_filename = os.path.join(dirpath, fname)
                if flag_full_filename:
                    final_filename = full_filename
                else:
                    final_filename = fname
                filenames_list.append(final_filename)
                count += 1
        if flag_recursive == False:
            break

    if flag_sort:
        filenames_list = natsort.natsorted(filenames_list)

    return filenames_list


# def get_all_filenames_from_folder(path, flag_recursive=False, flag_full_filename=True, string_pattern='\w*', flag_sort=False):
#     pattern = re.compile(string_pattern)
#
#     filenames_list = []
#     # Look recursively at all current and children path's in given path and get all binary files
#     for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
#         for fname in sorted(fnames):
#             full_filename = os.path.join(dirpath, fname)
#             if pattern.match(fname):
#                 if flag_full_filename:
#                     final_filename = full_filename
#                 else:
#                     final_filename = fname
#                 filenames_list.append(final_filename)
#         if flag_recursive == False:
#             break
#
#     if flag_sort:
#         filenames_list = natsort.natsorted(filenames_list)
#
#     return filenames_list


def path_get_all_filenames_from_folder(path, flag_recursive=False, flag_full_filename=True):
    filenames_list = []
    # Look recursively at all current and children path's in given path and get all binary files
    for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
        for fname in sorted(fnames):
            if flag_full_filename:
                file_full_filename = os.path.join(dirpath, fname)
            else:
                file_full_filename = fname
            filenames_list.append(file_full_filename)
        if flag_recursive == False:
            break;
    return filenames_list;

def get_image_filenames_from_lmdb(dataroot): #lmdb is apparently a file type which lands itself to be read by multiple workers... but there are many other options to implement!!!
    #Create an lmdb object/env with given dataroot, now i can read from the lmdb object the filenames inside that dataroot:
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')

    #if lmdb keys file exists use it, and if not Create it:
    if os.path.isfile(keys_cache_file):
        print('reading lmdb keys from cache: {}'.format(keys_cache_file))
        keys = np.pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            print('creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        np.pickle.dump(keys, open(keys_cache_file, 'wb'))

    #Get paths from lmdb file:
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths



#Should probably depricate... need to think about it.
def get_image_filenames_according_to_filenames_source_type(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = get_image_filenames_from_lmdb(dataroot)
        elif data_type == 'images':
            paths = sorted(get_image_filenames_from_folder(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def read_image_from_lmdb_object(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii') #encode and then decode? is this really efficient?
    image_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    image = image_flat.reshape(H, W, C)
    # Make Sure That We Get the Image in the correct format [H,W,C]:
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2);
    return image


### Auxiliary To Remember How To Initialize Video Writer: ###
def video_writer_initialization():
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video_writer = cv2.VideoWriter(final_movie_full_filename, fourcc, 25.0, (W, H))

from scipy.io import loadmat
from PIL import Image

def read_image_raw(path, roi_size=(100,100), flag_convert_to_rgb=1, flag_normalize_to_float=0):
    scene_infile = open(path, 'rb')
    H,W = roi_size
    scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=W * H)
    # scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
    image = Image.frombuffer("I", [W,H],
                             scene_image_array.astype('I'),
                             'raw', 'I', 0, 1)
    image = np.array(image)
    image = image_loading_post_processing(image, flag_convert_to_rgb, flag_normalize_to_float)
    return image

def read_raw_file(path, roi_size=(100,100), flag_convert_to_rgb=1, flag_normalize_to_float=0):


    ### Read Frames: ###
    Mov = np.fromfile(f, dtype=utype, count=number_of_frames_to_read * roi[0] * roi[1], offset=number_of_frames_to_skip * roi[0] * roi[1] * 2)
    Movie_len = np.int(len(Mov) / roi[0] / roi[1])
    number_of_elements = Movie_len * roi[0] * roi[1]
    Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])
    Mov = Mov[:, 2:, 2:]  # Get rid of bad frame


def read_image_general(path, flag_convert_to_rgb=1, flag_normalize_to_float=0, io_dict=None):
    if '.raw' in path:
        ### TODO: default parameters untill i insert io_dict from main script's train_dict: ###
        # 8 bit, unsigned, 2048 columns, 934 rows, little endian byte order:
        W = 2048
        H = 934

        # W = 640
        # H = 480

        scene_infile = open(path, 'rb')
        scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=W * H)
        # scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
        image = Image.frombuffer("I", [W, H],
                                       scene_image_array.astype('I'),
                                       'raw', 'I', 0, 1)
        image = np.array(image)
        image = image_loading_post_processing(image, flag_convert_to_rgb, flag_normalize_to_float)
        return image
    else:
        return read_image_cv2(path)

def read_ThirdEye(filepath, input_quantiles=None):
    W = 640
    H = 480

    scene_infile = open(path, 'rb')
    scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
    image = Image.frombuffer("I", [W, H],
                             scene_image_array.astype('I'),
                             'raw', 'I', 0, 1)
    image = np.array(image)

    ### this stretches the histogram
    if input_quantiles is None:
        image, (Q1,Q2) = scale_array_stretch_hist(image, min_max_values_to_scale_to=(0,1), quantiles=(0.01,0.99), flag_return_quantile=True)
    else:
        (Q1,Q2) = input_quantiles
        image = scale_array_from_range(image.clip(Q1, Q2),
                                               min_max_values_to_clip=(Q1, Q2),
                                               min_max_values_to_scale_to=(0, 1))

    return image, (Q1,Q2)


################
def get_ThirdEye_tensor(full_filename):
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_2.raw2'
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_3.raw2'
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_4.raw2'
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_5.raw2'

    W = 640
    H = 480

    scene_infile = open(full_filename, 'rb')
    frames_list = []

    flag_continue = True
    while flag_continue:
        scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
        if scene_image_array.size == H * W:
            image = Image.frombuffer("I", [W, H],
                                     scene_image_array.astype('I'),
                                     'raw', 'I', 0, 1)
            image = np.array(image)
            frames_list.append(torch.tensor(image).unsqueeze(0))
        else:
            flag_continue = False

    final_tensor = torch.cat(frames_list, 0).unsqueeze(1).float()

    image, (Q1, Q2) = scale_array_stretch_hist(final_tensor[0], min_max_values_to_scale_to=(0, 1), quantiles=(0.01, 0.99), flag_return_quantile=True)
    Q1 = Q1.item()
    Q2 = Q2.item()
    final_tensor = scale_array_from_range(final_tensor.clamp(Q1, Q2), (Q1, Q2))

    return final_tensor


################

def image_loading_post_processing(image, flag_convert_to_rgb=0, flag_normalize_to_float=0):
    if flag_convert_to_rgb == 1 and len(image.shape)==3 and image.shape[-1]==3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if flag_normalize_to_float == 1:
        image = image.astype(np.float32) / 255 #normalize to range [0,1] instead of [0,255]
    if image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        image = np.expand_dims(image, axis=2)
    if image.shape[2] > 4:
        # In the weird case where the image has more then 3 channels only take the first 3 channels to make it conform with 3-channels image format:
        image = image[:,:,:3]
    return image

# def read_image_cv2(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
#     # image = cv2.imread(path, cv2.IMREAD_COLOR)
#     if '.mat' in path:
#         image = loadmat(path)
#         image = image[list(image.keys())[-1]]  #.mat is basically a dictionary so i take the last key by default, maybe change that:
#     else:
#         image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#
#     image = image_loading_post_processing(image, flag_convert_to_rgb, flag_normalize_to_float)
#     return image

# from RapidBase.TrainingCore.Basic_DataSets import ImageLoaderCV

def is_image_file(filename, img_extentions=IMG_EXTENSIONS_NO_PNG):
    return any(list((filename.endswith(extension) for extension in img_extentions)))

# def is_image_file(filename, img_extentions=IMG_EXTENSIONS):
#     return True in any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_image_file_png(filename, img_extentions=IMG_EXTENSIONS_PNG):
    return True in any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def ImageLoaderCV(path):
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) == 3:  # opencv opens images as  BGR so we need to convert it to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.atleast_3d(image)  # MAKE SURE TO ALWAYS RETURN [H,W,C] FOR CONSISTENCY
    return np.float32(image)

def ImageLoaderPIL(path):
    return Image.open(path).convert('RGB')

def FolderReaderCV(folder_path, cropping_function=None):
    image_filenames = get_image_filenames_from_folder(folder_path, flag_recursive=False)
    number_of_images = len(image_filenames)

    for counter, image_filename in enumerate(image_filenames):
        ### Load Image: ###
        current_image = ImageLoaderCV(image_filename)

        ### Crop Image: ###
        if cropping_function is not None:
            current_image = cropping_function(current_image)
        if counter == 0:
            H,W,C = current_image.shape
            image_concat_array = np.zeros((H,W,C*number_of_images))

        ### Assign Current Image To Concatenated Image: ###
        image_concat_array[:, :, counter*C:(counter+1)*C] = current_image

    return image_concat_array

def get_random_number_in_range(min_num, max_num, array_size=(1)):
    return (np.random.random(array_size)*(max_num-min_num) + min_num).astype('float32')

def ImageListReaderCV(image_filenames_list, cropping_function=None):
    number_of_images = len(image_filenames_list)

    for counter, image_filename in enumerate(image_filenames_list):
        ### Load Image: ###
        current_image = ImageLoaderCV(image_filename)

        ### Crop Image: ###
        if cropping_function is not None:
            current_image = cropping_function(current_image)
        if counter == 0:
            H,W,C = current_image.shape
            image_concat_array = np.zeros((H,W,C*number_of_images))

        ### Assign Current Image To Concatenated Image: ###
        image_concat_array[:, :, counter*C : (counter+1)*C] = current_image

    return image_concat_array


def read_image_cv2(path):
    return ImageLoaderCV(path)

def read_image_torch(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
    image = cv2.imread(path, cv2.IMREAD_COLOR);
    if flag_convert_to_rgb == 1:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB);
    if flag_normalize_to_float == 1:
        image = image.astype(np.float32) / 255; #normalize to range [0,1] instead of [0,255]
    if image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        image = np.expand_dims(image, axis=2);
    if image.shape[2] > 4:
        # In the weird case where the image has more then 3 channels only take the first 3 channels to make it conform with 3-channels image format:
        image = image[:,:,:3];

    image = np.transpose(image,[2,0,1])
    image = torch.Tensor(image);
    image = image.unsqueeze(0)
    return image


def read_image_default_torch():
    return read_image_torch(default_image_filename_to_load1)/255


def read_video_default_torch(ROI=(np.inf,np.inf)):
    image_filenames = get_filenames_from_folder(default_video_folder_path, False, np.inf, True, '*.png', True)
    current_image = read_image_torch(image_filenames[0])/255
    T,C,H,W = current_image.shape
    T = len(image_filenames)
    ROI = list(ROI)
    if ROI[0]==np.inf:
        ROI[0] = current_image.shape[-2]
    if ROI[1]==np.inf:
        ROI[1] = current_image.shape[-1]
    output_tensor = torch.ones(len(image_filenames),C,int(ROI[0]),int(ROI[1]))
    for i in np.arange(len(image_filenames)):
        output_tensor[i:i+1] = crop_torch_batch(read_image_torch(image_filenames[i])/255, ROI)
    return output_tensor

def read_image_default():
    return read_image_cv2(default_image_filename_to_load1)/255

def read_outliers_default():
    return torch.load(outliers_full_path)

def read_image_stack_default_torch():
    image1 = read_image_torch(default_image_filename_to_load1)/255
    image2 = read_image_torch(default_image_filename_to_load2)/255
    image3 = read_image_torch(default_image_filename_to_load3)/255
    image1 = image1[:,:,0:400,0:400]
    image2 = image2[:,:,0:400,0:400]
    image3 = image3[:,:,0:400,0:400]
    images_total = torch.cat([image1,image2,image3],dim=0)
    return images_total

def read_image_stack_default():
    image1 = read_image_cv2(default_image_filename_to_load1)/255
    image2 = read_image_cv2(default_image_filename_to_load1)/255
    image3 = read_image_cv2(default_image_filename_to_load1)/255
    image1 = image1[0:400,0:400,:]
    image2 = image2[0:400,0:400,:]
    image3 = image3[0:400,0:400,:]
    image1 = np.expand_dims(image1, 0)
    image2 = np.expand_dims(image2, 0)
    image3 = np.expand_dims(image3, 0)
    images_total = np.concatenate((image1,image2,image3), axis=0)
    return images_total


# from skimage.color.colorlabel import label2rgb
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
def save_image_torch(folder_path=None, filename=None, torch_tensor=None, flag_convert_bgr2rgb=True,
                     flag_scale_by_255=False, flag_array_to_uint8=True, flag_imagesc=False, flag_convert_grayscale_to_heatmap=False,
                     flag_save_figure=False, flag_colorbar=False, flag_print=False):
    if flag_scale_by_255:
        scale_factor = 255
    else:
        scale_factor = 1

    if len(torch_tensor.shape) == 4:
        if flag_convert_bgr2rgb:
            saved_array = cv2.cvtColor(torch_tensor[0].cpu().data.numpy().transpose([1, 2, 0]) * scale_factor, cv2.COLOR_BGR2RGB)
        else:
            saved_array = torch_tensor[0].cpu().data.numpy().transpose([1, 2, 0]) * scale_factor
    else:
        if flag_convert_bgr2rgb:
            saved_array = cv2.cvtColor(torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor, cv2.COLOR_BGR2RGB)
        else:
            saved_array = torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor

    if flag_convert_grayscale_to_heatmap:
        if torch_tensor.shape[0]==1:
            #(1). Direct Formula ColorMap:
            saved_array = gray2color_numpy(saved_array,0)
            # #(2). Matplotlib Jet:
            # cmap = plt.cm.jet
            # norm = plt.Normalize(vmin=0, vmax=150)
            # gt_disparity = saved_array
            # gt_disparity2 = norm(gt_disparity)
            # gt_disparity3 = cmap(gt_disparity2)
            # saved_array = 255 * gt_disparity3[:,:,0:3]


    path_make_path_if_none_exists(folder_path)

    if flag_imagesc:
        new_range = (0, 255)
        new_range_delta = new_range[1]-new_range[0]
        old_range_delta = saved_array.max() - saved_array.min()
        new_min = new_range[0]
        old_min = saved_array.min()
        saved_array = ((saved_array-old_min)*new_range_delta/old_range_delta) + new_min

    if flag_array_to_uint8:
        saved_array = np.uint8(saved_array)

    if flag_convert_grayscale_to_heatmap:
        saved_array = cv2.cvtColor(saved_array, cv2.COLOR_BGR2RGB)

    if flag_save_figure:
        if len(saved_array.shape)==3:
            if saved_array.shape[2] == 1:
                imshow(saved_array.squeeze())
            else:
                imshow(saved_array)
        else:
            imshow(saved_array)
        if flag_colorbar:
            colorbar()
        plt.savefig(os.path.join(folder_path, filename))
        plt.close()
    else:
        cv2.imwrite(os.path.join(folder_path, filename), saved_array)

    if flag_print:
        # print(os.path.join(folder_path, filename))
        print(filename)

def imshow_BW(input_array):
    input_array = input_array.squeeze()
    if len(input_array.shape)==2:
        input_array_expanded = np.atleast_3d(input_array)
        input_array_stacked = np.concatenate([input_array_expanded,input_array_expanded,input_array_expanded], -1)
        imshow(input_array_stacked)
    else:
        imshow(input_array)

def save_image_numpy(folder_path=None,filename=None,numpy_array=None,flag_convert_bgr2rgb=True, flag_scale=True, flag_save_uint16=False, flag_convert_to_uint8=False):
    if flag_scale:
        scale = 255
    else:
        scale = 1
    if flag_convert_bgr2rgb:
        array_to_save = cv2.cvtColor(numpy_array * scale, cv2.COLOR_BGR2RGB)
    else:
        array_to_save = numpy_array * scale

    if flag_convert_to_uint8:
        array_to_save = array_to_save.clip(0,255).astype(np.uint8)

    if flag_save_uint16:
        numpngw.write_png(os.path.join(folder_path, filename), array_to_save.astype(np.uint16))
    else:
        cv2.imwrite(os.path.join(folder_path, filename), array_to_save)

import scipy
def save_image_mat(folder_path=None, filename=None, numpy_array=None, variable_name=None):
    scipy.io.savemat(os.path.join(folder_path, filename),
                     mdict={variable_name: numpy_array})

def save_to_matlab_mat_file(folder_path=None, filename=None, numpy_array=None, variable_name=None):
    #(*). Attention! it seems it can only save numpy array!!! it transfers lists, for instance, into numpy arrays (maybe it can handle structures?)
    scipy.io.savemat(os.path.join(folder_path, filename),
                     mdict={variable_name: numpy_array})

def load_from_matlab_mat_file(full_filename='', key=None):
    mat_file_dictionary = scipy.io.loadmat(full_filename)
    mat_file_keys = mat_file_dictionary.keys()
    if key is None:
        return mat_file_dictionary[list(mat_file_keys)[-1]]
    else:
        return mat_file_keys[key]

def load_image_mat(full_filename='', key=None):
    mat_file_dictionary = scipy.io.loadmat(full_filename)
    mat_file_keys = mat_file_dictionary.keys()
    if key is None:
        return mat_file_dictionary[list(mat_file_keys)[-1]]
    else:
        return mat_file_keys[key]



### Auxiliary function implemented: ###  #TODO: fix this
def put_image_filenames_in_txt_file(root, max_number_of_images=-1):
    max_number_of_images = 10
    txt_file_path = 'C:/Users/dkarl/PycharmProjects/dudykarl/4dudi'
    txt_filename = 'imagenet_filenames.txt'

    log = open(os.path.join(txt_file_path, txt_filename), 'a')
    counter = 0;
    for dp,dn,fn in os.walk(txt_file_path):
        for f in fn:
            counter += 1;
            if counter==max_number_of_images and max_number_of_images!=-1:
                break;
            elif os.path.getsize(os.path.join(dp,f)) > 34000 and is_image_file(f):
                log.write(os.path.join(dp,f)+'\n')


def read_images_from_filenames_list(image_filenames, flag_return_numpy_or_list='numpy', crop_size=np.inf, max_number_of_images=10, allowed_extentions=IMG_EXTENSIONS, flag_how_to_concat='C', crop_style='random', flag_return_torch=False, transform=None, flag_to_BW=False, flag_random_first_frame=False, first_frame_index=-1, flag_to_RGB=False, flag_return_position_indices=False, start_H=-1, start_W=-1):
    ### crop_style = 'random', 'random_deterministic', 'predetermined'
    number_of_images = min(len(image_filenames), max_number_of_images)
    image_concat_array = []

    if type(crop_size) is not list and type(crop_size) is not tuple:
        crop_size = [crop_size, crop_size]

    ### Decide whether to start from start or randomize initial frame returned: ###
    if flag_random_first_frame:
        start_index, stop_index = get_random_start_stop_indices_for_crop(number_of_images, len(image_filenames)) #TODO: notice everything's okey, i hopefully corrected from (number_of_images, number_of_images)
    else:
        start_index = 0
        stop_index = start_index + number_of_images

    if first_frame_index != -1:
        start_index = first_frame_index
        stop_index = start_index + number_of_images

    current_step = 0
    flag_already_set_crop_position = False
    for current_index in np.arange(start_index, stop_index, 1):
        image_filename = image_filenames[current_index]
        if is_image_file(image_filename, allowed_extentions):
            ### Read Image: ###
            current_image = read_image_general(image_filename)

            ### If Crop Size Is Legit continue on: ###
            if (current_image.shape[0] >= crop_size[0] and current_image.shape[1] >= crop_size[1]) or (crop_size == np.inf or crop_size == [np.inf, np.inf]):
                ### Get current crop: ###
                if crop_style == 'random_consistent' and flag_already_set_crop_position==False:
                    #TODO: need to be consistent, and instead of crop_x and crop_y i need to transfer to crop_H and crop_W
                    start_H, stop_H = get_random_start_stop_indices_for_crop(current_image.shape[-2], min(crop_size[0], current_image.shape[-2]))
                    start_W, stop_W = get_random_start_stop_indices_for_crop(current_image.shape[-1], min(crop_size[1], current_image.shape[-1]))
                    flag_already_set_crop_position = True #only set crop position once
                elif crop_style == 'predetermined' and start_H!=-1 and start_W!=-1:
                    stop_H = start_H + min(crop_size[0], current_image.shape[-2])
                    stop_W = start_W + min(crop_size[1], current_image.shape[-1])
                current_crop = crop_numpy_batch(current_image, crop_size_tuple_or_scalar=crop_size, crop_style=crop_style, start_H=start_H, start_W=start_W)

                ### To BW/RGB If Wanted: ###
                if flag_to_BW:
                    current_crop = RGB2BW(current_crop)
                if flag_to_RGB:
                    current_crop = BW2RGB(current_crop)

                ### Transform If Exists: ###
                if transform is not None:
                    current_crop = transform(current_crop)

                ### Permute Dimensions To Torch Convention If Wanted: ###
                H, W, C = current_crop.shape
                if flag_return_torch:
                    current_crop = np.transpose(current_crop, [2,0,1])

                if flag_return_numpy_or_list == 'list':
                    image_concat_array.append(current_crop)

                else: #'numpy'
                    ### Initialize image concat array: ###
                    if current_step == 0:
                        if flag_how_to_concat == 'T':
                            image_concat_array = np.expand_dims(current_crop, 0)
                        elif flag_how_to_concat == 'C':
                            if flag_return_torch:
                                #TODO: in case there are mixed type- add flag for type
                                image_concat_array = np.zeros((C * number_of_images, H, W)).astype(current_crop.dtype)
                                image_concat_array[0:C,:,:] = current_crop
                            else:
                                image_concat_array = np.zeros((H, W, C * number_of_images)).astype(current_crop.dtype)
                                image_concat_array[:,:,0:C] = current_crop

                    ### Assign Current Image To Concatenated Image: ###
                    else:
                        if flag_how_to_concat == 'T':
                            image_concat_array = np.concatenate((image_concat_array, np.expand_dims(current_crop, 0)), axis=0)
                        elif flag_how_to_concat == 'C':
                            if flag_return_torch:
                                image_concat_array[current_step * C:(current_step + 1) * C, :, :] = current_crop
                            else:
                                image_concat_array[:, :, current_step * C:(current_step + 1) * C] = current_crop

                ### If we're at max_number_of_images --> break: ###
                if current_step == max_number_of_images:
                    break

            ### Uptick Current Step: ###
            current_step += 1

    if flag_return_torch:
        if image_concat_array.dtype == np.uint8:
            if flag_return_position_indices:
                return torch.Tensor(image_concat_array).type(torch.uint8), start_index, start_H, start_W
            else:
                return torch.Tensor(image_concat_array).type(torch.uint8)
        elif image_concat_array.dtype == np.uint16:
            if flag_return_position_indices:
                return torch.Tensor(image_concat_array).type(torch.uint16), start_index, start_H, start_W
            else:
                return torch.Tensor(image_concat_array).type(torch.uint16)
        else:
            if flag_return_position_indices:
                return torch.Tensor(image_concat_array), start_index, start_H, start_W
            else:
                return torch.Tensor(image_concat_array)
    else:
        if flag_return_position_indices:
            return image_concat_array, start_index, start_H, start_W
        else:
            return image_concat_array



def read_images_from_folder(path, flag_recursive=False, crop_size=np.inf, max_number_of_images=np.inf, allowed_extentions=IMG_EXTENSIONS, flag_return_numpy_or_list='list', flag_how_to_concat='C', crop_style='random', flag_return_torch=False, transform=None, flag_to_BW=False, flag_random_first_frame=False, string_pattern_to_search='*'):
    image_filenames = get_image_filenames_from_folder(path, number_of_images=max_number_of_images, allowed_extentions=allowed_extentions, flag_recursive=flag_recursive, string_pattern_to_search=string_pattern_to_search)
    images = read_images_from_filenames_list(image_filenames, flag_return_numpy_or_list=flag_return_numpy_or_list, crop_size=crop_size, max_number_of_images=max_number_of_images, allowed_extentions=allowed_extentions, flag_how_to_concat=flag_how_to_concat, crop_style=crop_style, flag_return_torch=flag_return_torch, transform=transform, flag_to_BW=False)
    return images

def read_images_and_filenames_from_folder(path, flag_recursive=False, crop_size=np.inf, max_number_of_images=np.inf, allowed_extentions=IMG_EXTENSIONS, flag_return_numpy_or_list='list', flag_how_to_concat='C', crop_style='random', flag_return_torch=False, transform=None, flag_to_BW=False, flag_random_first_frame=False, string_pattern_to_search='*'):
    image_filenames = get_image_filenames_from_folder(path, number_of_images=max_number_of_images, allowed_extentions=allowed_extentions, flag_recursive=flag_recursive, string_pattern_to_search=string_pattern_to_search)
    images = read_images_from_filenames_list(image_filenames, flag_return_numpy_or_list=flag_return_numpy_or_list, crop_size=crop_size, max_number_of_images=max_number_of_images, allowed_extentions=allowed_extentions, flag_how_to_concat=flag_how_to_concat, crop_style=crop_style, flag_return_torch=flag_return_torch, transform=transform, flag_to_BW=flag_to_BW)
    return images, image_filenames[0:len(images)]

def read_image_filenames_from_folder(path, number_of_images=np.inf, allowed_extentions=IMG_EXTENSIONS, flag_recursive=False, string_pattern_to_search='*'):
    return get_image_filenames_from_folder(path, number_of_images=number_of_images, allowed_extentions=allowed_extentions, flag_recursive=flag_recursive, string_pattern_to_search=string_pattern_to_search)


def video_get_movie_file_properties(path):
    Movie_Reader = cv2.VideoCapture(path)
    # Movie_Reader.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = Movie_Reader.get(4)
    W = Movie_Reader.get(3)
    FPS = Movie_Reader.get(5)
    number_of_frames = Movie_Reader.get(7)
    return H,W,FPS,number_of_frames


from RapidBase.Utils.MISCELENEOUS import string_rjust
def video_video_to_images(video_path):
    # video_path = r'/home/mafat/DataSets/Example Videos/SWIR_video_original/SWIR_video.mp4'
    # video_path = r'/home/mafat/DataSets/DJI_DRONE1/DJI Mavic Mini 500m Altitude Test Pt.mkv'
    video_folder = os.path.split(video_path)[0]
    video_name = os.path.split(video_path)[1]
    video_stream = cv2.VideoCapture(video_path)
    # video_stream.open()
    counter = 0
    while video_stream.isOpened():
        flag_frame_available, current_frame = video_stream.read()
        if flag_frame_available:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            image_full_filename = str.replace(video_name, '.avi', '_' + string_rjust(counter, 4) + '.png')
            image_full_filename = str.replace(image_full_filename, '.mkv', '_' + string_rjust(counter, 4) + '.png')
            image_full_filename = str.replace(image_full_filename, '.mp4', '_' + string_rjust(counter, 4) + '.png')
            cv2.imwrite(
                os.path.join(video_folder, image_full_filename),
                current_frame)
            counter += 1
            print(counter)
    video_stream.release()

def video_get_mat_from_figure(fig, wanted_shape=(480*2,480*2)):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.resize(data, wanted_shape)
    return data

def video_create_movie_from_images_in_folder(images_path, frame_rate=25, output_folder=None, movie_name=None, flag_print_image_number=True):
    files_list = path_get_files_recursively(images_path, '')

    image_number_list = []
    for file in files_list:
        image_number_list.append(int(os.path.splitext(os.path.split(file)[1])[0]))
    sorted_indices_list = np.argsort(image_number_list)

    ### Get sample image: ###
    image = cv2.imread(files_list[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape)==2:
        height, width = image.shape
    elif len(image.shape)==3:
        height, width, channels = image.shape

    ### Create video: ###
    if output_folder is None:
        if movie_name is None:
            video_name = os.path.join(images_path, 'video_out.avi')
        else:
            video_name = os.path.join(images_path, movie_name)
    else:
        if movie_name is None:
            video_name = os.path.join(output_folder, 'video_out.avi')
        else:
            video_name = os.path.join(output_folder, movie_name)
    video_object = cv2.VideoWriter(video_name, 0, frame_rate, (width * 1, height * 1))
    for index in sorted_indices_list:
        # print(files_list[index])
        current_image = cv2.imread(files_list[index])
        video_object.write(current_image)
    print('done making video fucker')
    cv2.destroyAllWindows()
    video_object.release()


# from RapidBase.TrainingCore.misc.misc import read_image_cv2
def video_images_to_video(images_path, video_name='my_movie.avi'):
    # ### Images To Video: ###
    # images_path = r'/home/mafat/DataSets/Example Videos/LWIR_video_10'
    # video_name = 'LWIR_video_10.avi'

    image_filenames = read_image_filenames_from_folder(images_path)
    video_full_filename = os.path.join(images_path, video_name)
    single_image = read_image_cv2(image_filenames[0])
    H,W,C = single_image.shape
    number_of_images = len(image_filenames)
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_full_filename, 0, 25.0, (W, H))
    for frame_counter in np.arange(len(image_filenames)):
        current_frame = read_image_cv2(image_filenames[frame_counter])
        video_writer.write(current_frame)
    video_writer.release()

def video_images_to_video(images_path, video_name='my_movie.avi'):
    # ### Images To Video: ###
    # images_path = r'/home/mafat/DataSets/Example Videos/LWIR_video_10'
    # video_name = 'LWIR_video_10.avi'

    image_filenames = read_image_filenames_from_folder(images_path)
    video_full_filename = os.path.join(images_path, video_name)
    single_image = read_image_cv2(image_filenames[0])
    H,W,C = single_image.shape
    number_of_images = len(image_filenames)
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_full_filename, 0, 25.0, (W, H))
    for frame_counter in np.arange(len(image_filenames)):
        current_frame = read_image_cv2(image_filenames[frame_counter])
        current_frame = numpy_array_to_video_ready(current_frame)
        video_writer.write(current_frame)
    video_writer.release()

def video_numpy_array_to_video(input_tensor, video_name='my_movie.avi', FPS=25.0):
    T, H, W, C = input_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_name, 0, FPS, (W, H))
    for frame_counter in np.arange(T):
        current_frame = input_tensor[frame_counter]
        video_writer.write(current_frame)
    video_writer.release()

# import skvideo.io
def video_torch_array_to_video(input_tensor, video_name='my_movie.mp4', FPS=25.0, flag_stretch=False, output_shape=None):
    import skvideo
    import skvideo.io
    ### Initialize Writter: ###
    T, C, H, W = input_tensor.shape
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Be sure to use lower case
    # video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))

    ### Resize If Wanted: ###
    if output_shape is not None:
        output_tensor = torch.nn.functional.interpolate(input_tensor, size=output_shape)
    else:
        output_tensor = input_tensor

    ### Use Scikit-Video: ###
    output_numpy_array = numpy_array_to_video_ready(output_tensor.permute([0,2,3,1]).clamp(0, 255).cpu().numpy())
    if flag_stretch:
        output_numpy_array = scale_array_to_range(output_numpy_array, (0,255))
    skvideo.io.vwrite(video_name,
                      output_numpy_array,
                      outputdict={'-vcodec': 'libx264', '-b': '300000000'}) #chose the bitrate to be very high so no loss of information

    # for frame_counter in np.arange(T):
    #     current_frame = input_tensor[frame_counter]
    #     current_frame = current_frame.permute([1,2,0]).cpu().numpy()
    #     current_frame = BW2RGB(current_frame)
    #     current_frame = (current_frame * 255).clip(0,255).astype(np.uint8)
    #     video_writer.write(current_frame)
    # video_writer.release()

def video_torch_array_to_images(input_tensor, folder_path, flag_convert_bgr2rgb=True, flag_scale_by_255=False, flag_array_to_uint8=True):
    T,C,H,W = input_tensor.shape
    path_make_path_if_none_exists(folder_path)
    for i in np.arange(T):
        save_image_torch(folder_path,
                         filename=string_rjust(i, int(scientific_notation(T)[-2:])+1) + '.png',
                         torch_tensor=input_tensor[i],
                         flag_convert_bgr2rgb=flag_convert_bgr2rgb,
                         flag_scale_by_255=flag_scale_by_255,
                         flag_array_to_uint8=flag_array_to_uint8)


def change_filenames_rjust():
    filenames_list = path_get_all_filenames_from_folder('/home/mafat/Datasets/pngsh/pngs/2')
    for filename in filenames_list:
        numpy_filename = os.path.split(filename)[-1]
        dir_path = os.path.split(filename)[0]
        recording_number = str.split(numpy_filename, '.png')[0]
        recording_number = str.split(recording_number, '_')[-1]
        new_recording_number = string_rjust(int(recording_number), 5)
        new_numpy_filename = str.replace(numpy_filename, recording_number, new_recording_number)
        new_numpy_filename = os.path.join(dir_path, new_numpy_filename)
        os.rename(filename, new_numpy_filename)
# change_filenames_rjust()

def combine_two_movies_together(video_path1, video_path2):
    video_path1 = r'/home/mafat/DataSets/Example Videos/LWIR_video_10/LWIR_video_10.avi'
    video_path2 = r'/home/mafat/DataSets/Example Videos/LWIR_video_original/LWIR_video.mp4'
    final_movie_full_filename = r'/home/mafat/DataSets/Example Videos/LWIR_video_original/original_and_10_together.avi'

    video_stream1 = cv2.VideoCapture(video_path1)
    video_stream2 = cv2.VideoCapture(video_path2)
    counter = 0
    while video_stream1.isOpened():
        flag_frame_available1, current_frame1 = video_stream1.read()
        flag_frame_available2, current_frame2 = video_stream2.read()
        H1,W1,C1 = current_frame1.shape
        H2,W2,C2 = current_frame2.shape

        final_frame = np.concatenate([current_frame1,current_frame2], 1)  #concatenate width-wise
        if counter == 0:
            fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
            video_writer = cv2.VideoWriter(final_movie_full_filename, fourcc, 25.0, (W1+W2, H1))
        video_writer.write(final_frame)

        counter += 1

    video_writer.release()


def read_single_image_from_folder(path):
    #assuming folder is all images!!!
    for dirpath, _, fnames in sorted(os.walk(path)):
        for filename in fnames:
            return read_image_cv2(os.path.join(dirpath,filename))

def get_image_shape_from_path(path):
    if is_image_file(path):
        #path=image file
        image = read_image_cv2(path)
        shape = image.shape
    else:
        #path=folder:
        image = read_single_image_from_folder(path)
        shape = image.shape
    return shape

def path_keep_only_N_files(path, number_of_files_to_keep):
    files_list = os.listdir(path)
    number_of_files = len(files_list)
    number_of_files_to_delete = number_of_files - number_of_files_to_keep
    files_list.sort(key=lambda x: os.stat(os.path.join(path, x)).st_mtime)
    if number_of_files_to_delete > 0:
        for i in np.arange(number_of_files_to_delete):
            os.remove(os.path.join(path, files_list[i]))

def sort_files_by_date(full_filenames_list):
    full_filenames_list.sort(key=lambda x: os.path.getmtime(x))
    return full_filenames_list


def search_file(pattern='*', file_directory = os.getcwd()):
    return [os.path.basename(c) for c in glob.glob(file_directory + '/' + pattern)]

def path_get_residual_of_path_after_initial_path(full_path, initial_path):
    # return os.path.join(*str.split(full_path, initial_path)[1].split('\\')[1:])
    full_path = path_fix_path_for_linux(full_path)
    initial_path = path_fix_path_for_linux(initial_path)
    return os.path.join(str.split(full_path, initial_path)[1].split('/')[-1])

def join_string_list_with_delimiter_in_between(string_list,delimiter):
    joint_string = '';
    for i in range(0,len(string_list),1):
        if i>0:
            joint_string += delimiter;
        joint_string += string_list[i];
    return joint_string

### Colors: ###
def get_color_formula(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(np.pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(np.pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(np.pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(np.pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(2*np.pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(2*np.pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(2*np.pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(2*np.pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(4*np.pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*np.pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet

def gray2color(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]
    formula_triplet = get_color_formula(formula_id_triplet);
    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clamp(0,1)
    G = G.clamp(0,1)
    B = B.clamp(0,1)
    color_array = torch.cat([R,G,B], dim=1)
    return color_array

# ### Use Example: ###
# input_tensor = torch.Tensor(randn(1,1,100,100)).abs().clamp(0,1)
# color_tensor = gray2color(input_tensor,0)
# figure(1)
# imshow_torch(input_tensor[0].repeat((3,1,1)),0)
# figure(2)
# imshow_torch(color_tensor[0],0)



### Colors: ###
def get_color_formula_torch(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(np.pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(np.pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(np.pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(np.pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(2*np.pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(2*np.pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(2*np.pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(2*np.pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(4*np.pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*np.pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet



import numpy
def get_color_formula_numpy(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(np.pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.cos(np.pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(np.pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.cos(np.pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(2*np.pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.cos(2*np.pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.sin(2*np.pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.cos(2*np.pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.sin(4*np.pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*np.pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet


def gray2color_numpy(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
        # formula_id_triplet = [3,4,5]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]

    formula_triplet = get_color_formula_numpy(formula_id_triplet);

    input_min = input_array.min()
    input_max = input_array.max()
    input_array = to_range(input_array,0,1)
    # input_array = input_array/256

    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clip(0,1)
    G = G.clip(0,1)
    B = B.clip(0,1)
    if len(R.shape)==4:
        color_array = numpy.concatenate([R,G,B], 3)
    else:
        color_array = numpy.concatenate([R,G,B], 2)

    # input_array = input_array*256
    color_array = to_range(color_array, input_min, input_max)




    return color_array



def gray2color_torch(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]
    formula_triplet = get_color_formula_torch(formula_id_triplet);
    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clamp(0,1)
    G = G.clamp(0,1)
    B = B.clamp(0,1)

    if len(R.shape)==4:
        color_array = torch.cat([R,G,B], dim=1)
    else:
        color_array = torch.cat([R,G,B], dim=0)

    return color_array



def to_range(input_array, low, high):
    new_range_delta = high-low
    old_range_delta = input_array.max() - input_array.min()
    new_min = low
    old_min = input_array.min()
    input_array = ((input_array-old_min)*new_range_delta/old_range_delta) + new_min
    return input_array

def stretch_tensor_values(input_tensor, low, high):
    new_range_delta = high - low
    old_range_delta = input_tensor.max() - input_tensor.min()
    new_min = low
    old_min = input_tensor.min()
    input_array = ((input_tensor - old_min) * new_range_delta / old_range_delta) + new_min
    return input_array

def stretch_tensor_quantiles(input_tensor, min_quantile, max_quantile):
    min_value = input_tensor.quantile(min_quantile)
    max_value = input_tensor.quantile(max_quantile)
    return stretch_tensor_values(input_tensor.clamp(min_value,max_value), 0, 1)


