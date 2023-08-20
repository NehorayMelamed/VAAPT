# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from pygments.lexer import default
from torchvision import transforms
from torchvision_x.transforms import transforms_seg
import torch.utils.data as data
from PIL import Image
from random import shuffle


#ESRGAN:
# import ESRGAN_dataset
# import ESRGAN_Visualizers
# import ESRGAN_Optimizers
# import ESRGAN_Losses
# import ESRGAN_deep_utils
import ESRGAN_utils
# import ESRGAN_Models
# import ESRGAN_basic_Blocks_and_Layers
# import ESRGAN_OPT
from ESRGAN_utils import *
# from ESRGAN_deep_utils import *
# from ESRGAN_basic_Blocks_and_Layers import *
# from ESRGAN_Models import *
# from ESRGAN_dataset import *
# from ESRGAN_Losses import *
# from ESRGAN_Optimizers import *
# from ESRGAN_Visualizers import *
# from ESRGAN_dataset import *
# from ESRGAN_OPT import *

#Augmentations:
import PIL
import albumentations
from albumentations import *
import Albumentations_utils
import IMGAUG_utils
from IMGAUG_utils import *



IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    # '.png',
    # '.PNG',  #in ImageNet files there appears to be, at least in my computer, a problem with png and gif
    # '.ppm',
    # '.PPM',
    # '.bmp',
    # '.BMP',
]


IMG_EXTENSIONS_PNG = [
    # '.jpg',
    # '.JPG',
    # '.jpeg',
    # '.JPEG',
    '.png',
    '.PNG',  #in ImageNet files there appears to be, at least in my computer, a problem with png and gif
    # '.ppm',
    # '.PPM',
    # '.bmp',
    # '.BMP',
]


def is_image_file(filename, img_extentions=IMG_EXTENSIONS):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_image_file_png(filename, img_extentions=IMG_EXTENSIONS_PNG):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')



class DataSet_From_Numpy_List(data.Dataset):
    def __init__(self, numpy_images_list, transform=None):
        self.images_list = numpy_images_list
        self.transform = transform

    def __getitem__(self, index):
        #fetch the image:
        current_image = self.images_list[index]

        #If there is a transform apply it (whether it be from imgaug, albumentations or torchvision):
        if self.transform is not None:
            if type(self.transform) == iaa.Sequential:
                deterministic_Augmenter_Object = self.transform.to_deterministic()
                current_image = deterministic_Augmenter_Object.augment_image(current_image);
            elif type(self.transform) == albumentations.Compose:
                augmented_images_dictionary = self.transform(image=current_image);
                current_image = augmented_images_dictionary['image']
            elif type(self.transform) == torchvision.transforms.transforms.Compose:
                current_image = self.transform(current_image)

        #Return image:
        return current_image

    def __len__(self):
        return len(self.images_list)






class DataSet_From_Folder_To_RAM_max_number_of_images(data.Dataset):
    def __init__(self, path, transform=None, number_of_images=10, min_size=10, allowed_extentions=IMG_EXTENSIONS_PNG, flag_recursive=False):
        numpy_images_list = read_number_of_images_from_folder(path, number_of_images, min_size, allowed_extentions, flag_recursive);
        self.images_list = numpy_images_list
        self.transform = transform

    def __getitem__(self, index):
        #fetch the image:
        current_image = self.images_list[index]

        #If there is a transform apply it (whether it be from imgaug, albumentations or torchvision):
        if self.transform is not None:
            if type(self.transform) == iaa.Sequential:
                deterministic_Augmenter_Object = self.transform.to_deterministic()
                current_image = deterministic_Augmenter_Object.augment_image(current_image);
            elif type(self.transform) == albumentations.Compose:
                augmented_images_dictionary = self.transform(image=current_image);
                current_image = augmented_images_dictionary['image']
            elif type(self.transform) == torchvision.transforms.transforms.Compose:
                current_image = self.transform(current_image)

        #Return image:
        return current_image

    def __len__(self):
        return len(self.images_list)




class DataSet_From_Folder_To_RAM(data.Dataset):
    def __init__(self, path, transform=None, allowed_extentions=IMG_EXTENSIONS_PNG, flag_recursive=False):
        numpy_images_list = read_images_from_folder_to_list(path, allowed_extentions, flag_recursive);
        self.images_list = numpy_images_list
        self.transform = transform

    def __getitem__(self, index):
        #fetch the image:
        current_image = self.images_list[index]

        #If there is a transform apply it (whether it be from imgaug, albumentations or torchvision):
        if self.transform is not None:
            if type(self.transform) == iaa.Sequential:
                deterministic_Augmenter_Object = self.transform.to_deterministic()
                current_image = deterministic_Augmenter_Object.augment_image(current_image);
            elif type(self.transform) == albumentations.Compose:
                augmented_images_dictionary = self.transform(image=current_image);
                current_image = augmented_images_dictionary['image']
            elif type(self.transform) == torchvision.transforms.transforms.Compose:
                current_image = self.transform(current_image)

        #Return image:
        return current_image

    def __len__(self):
        return len(self.images_list)


# #Examples of experimentations with the framework which made possible creating the DataSet_From_Folder_To_RAM class:
# images_list = read_images_from_folder_to_list(OPT.test_path)
# bli = images_list[0]
# new_image1 = train_transform.augment_images(np.expand_dims(bli,0))
# new_image2 = train_transform.augment_image(bli)
# new_image2 = gaussblur(bli)
# new_image3 = iaa.Lambda(func_images=gaussblur,func_heatmaps=None,func_keypoints=None).augment_image(bli)
# new_image2-np.squeeze(new_image1)
# new_image4 = iaa.GaussianBlur(sigma=0.65).augment_image(bli)





class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader=default_loader):
        images = []
        for filename in os.listdir(root):
            if is_image_file(filename):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        try:
            img = self.loader(os.path.join(self.root, filename))
        except:
            return torch.zeros((3, 32, 32))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)




class ImageFolderRecursive(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader=default_loader):
        images = []
        filename_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn if os.path.getsize(os.path.join(dp, f)) > 34000]
        shuffle(filename_list)
        for filename in filename_list:
            if is_image_file(filename):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]

        try:
            img = self.loader(filename)
            s = img.size
            if min(s) <= 64:
                raise ValueError('image smaller than a block (32,32)')
            # img = img.resize((int(s[0]/2), int(s[1]/2)), Image.ANTIALIAS)
        except:
            print("problem loading " + filename)
            img = Image.new('RGB', [100, 100])

            # return torch.zeros((3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)







def put_image_filenames_in_txt_file(root, max_number_of_images=-1):
    max_number_of_images = 10;
    txt_file_path = 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi'
    txt_filename = 'imagenet_filenames.txt'
    images_folder_path = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet'

    log = open(os.path.join(txt_file_path, txt_filename), 'a')
    counter = 0;
    for dp,dn,fn in os.walk(txt_file_path):
        for f in fn:
            counter += 1;
            if counter==max_number_of_images and max_number_of_images!=-1:
                break;
            elif os.path.getsize(os.path.join(dp,f)) > 34000 and is_image_file(filename):
                log.write(os.path.join(dp,f)+'\n')




#TODO: need a way to get to wanted path's quickly- one way is a database object, but that's too much for now. another way maybe is a txt file including all image filenames where i can access lines quickly.
#      i don't have experience with things like that but it seems there are two ways: making sure all the lines are of the same lengthand saving as a variable and accessing indices in an array (or maybe using a list which means lines don't have to be the same size)
#      another way seems to be a module name linecache which uses "traceback" to retrieve source lines for inclusion in the formatted traceback
# class Images_From_TXT_File(data.Dataset):
#     def __init__(self, txt_path, transform, loader=default_loader):





class ImageFolderRecursive_MaxNumberOfImages(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader='PIL', max_number_of_images=-1, min_crop_size=100, allowed_extentions=IMG_EXTENSIONS_PNG, flag_recursive=True):
        #TODO: why isn't the inherited class' (data.Dataset) __init__ function being activated?
        images = []
        def is_image_file_png(filename, img_extentions=IMG_EXTENSIONS_PNG):
            return any(filename.endswith(extension) for extension in img_extentions)


        counter = 0;
        # root = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet'
        # max_number_of_images = 100
        for dp, dn, fn in os.walk(root):
            if counter > max_number_of_images and max_number_of_images != -1:
                break;
            for f in fn:
                filename = os.path.join(dp, f);
                if counter > max_number_of_images and max_number_of_images != -1:
                    break;
                elif is_image_file(filename, IMG_EXTENSIONS_PNG):
                    images.append('{}'.format(filename))
                    counter += 1;
            if flag_recursive == False:
                break;
        # shuffle(filename_list)



        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader
        self.min_crop_size = min_crop_size;
        self.extention = extention;

        if self.loader == 'PIL':
            self.loader_function = default_loader
        elif self.loader == 'CV':
            self.loader_function = read_image_cv2


    def __getitem__(self, index):
        filename = self.imgs[index]

        try:
            current_image = self.loader_function(filename)

            if self.loader == 'PIL':
                s = min(current_image.size)
            elif self.loader == 'CV':
                s = min(current_image.shape[0:2])

            if s < self.min_crop_size:
                print('image smaller than min initial crop size')
                raise ValueError('image smaller than a block (32,32)')
        except:
            print("problem loading " + filename)
            if self.loader == 'PIL':
                current_image = Image.new('RGB', [self.min_crop_size, self.min_crop_size])
            elif self.loader == 'CV':
                current_image = np.zeros((self.min_crop_size,self.min_crop_size,3))

        # If there is a transform apply it (whether it be from imgaug, albumentations or torchvision):
        if self.transform is not None:
            if type(self.transform) == iaa.Sequential:
                deterministic_Augmenter_Object = self.transform.to_deterministic()
                current_image = deterministic_Augmenter_Object.augment_image(current_image);
            elif type(self.transform) == albumentations.Compose:
                augmented_images_dictionary = self.transform(image=current_image);
                current_image = augmented_images_dictionary['image']
            elif type(self.transform) == torchvision.transforms.transforms.Compose:
                current_image = self.transform(current_image)

        return current_image

    def __len__(self):
        return len(self.imgs)




# root = 'F:/NON JPEG IMAGES\HR_images\General100'
# train_dataset = ImageFolderRecursive_MaxNumberOfImages(root,transform=None,loader='CV',max_number_of_images = -1, min_crop_size=100,extention='png')
# imshow(train_dataset[1])








class ImageFolderRecursive_MaxNumberOfImages_BinaryFiles(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, max_number_of_images=-1, min_crop_size=100, image_data_type_string='uint8'):
        #TODO: why isn't the inherited class' (data.Dataset) __init__ function being activated?
        images = []

        counter = 0;
        # root = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet'
        # max_number_of_images = 100
        for dp, dn, fn in os.walk(root):
            if counter > max_number_of_images and max_number_of_images != -1:
                break;
            for f in fn:
                filename = os.path.join(dp, f);
                counter += 1;
                if counter > max_number_of_images and max_number_of_images != -1:
                    break;
                else:
                    #TODO: add condition that file ends with .bin!!!!
                    images.append('{}'.format(filename))
        # shuffle(filename_list)

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader
        self.min_crop_size = min_crop_size;
        self.image_data_type_string = image_data_type_string


    def __getitem__(self, index):
        filename = self.imgs[index]

        try:
            fid_Read_X = open(filename, 'rb')
            frame_height = np.fromfile(fid_Read_X, 'float32', count=1);
            frame_width = np.fromfile(fid_Read_X, 'float32', count=1);
            number_of_channels = np.fromfile(fid_Read_X, 'float32', count=1);
            mat_shape = [frame_height, frame_width, number_of_channels]  # list or tuple?!?!?!?!!?
            total_images_number_of_elements = frame_height*frame_width*number_of_channels;
            img = np.fromfile(fid_Read_X, self.image_data_type_string, count=total_images_number_of_elements);
            img = img.reshape(mat_shape)
            s = img.size
            if min(s) < self.min_crop_size:
                raise ValueError('image smaller than a block (32,32)')
            # img = img.resize((int(s[0]/2), int(s[1]/2)), Image.ANTIALIAS)
        except:
            print("problem loading " + filename)
            img = Image.new('RGB', [self.min_crop_size, self.min_crop_size])

            # return torch.zeros((3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)




# images_folder,
# transform=None,
# loader='CV',
# number_of_images_per_video=25,
# max_number_of_videos=-1,
# crop_size=crop_size,
# flag_load_to_RAM=False,
# flag_output_YUV=False, flag_output_HSV=False, flag_output_channel_average=False,
# flag_recursive=True,
# flag_explicitely_make_tensor=True,
# flag_normalize_by_255=True

class Videos_In_Folders_DataSet(data.Dataset):
    #video frame sequences in folder with each sequence in a different folder (and no logic to enable time coherence to be able to guide hidden states...so this is a "stupid" dataset)
    def __init__(self,
                 root_folder,
                 transform=None,
                 loader='CV',
                 number_of_images_per_video=10, max_number_of_videos=-1,
                 crop_size=100,
                 flag_load_to_RAM=False,
                 flag_recursive=False,
                 flag_output_YUV=False, flag_output_HSV=False, flag_output_channel_average=False,
                 flag_explicitely_make_tensor=True,
                 flag_normalize_by_255=True):

        ### Note: ###
        #as of now i only use 'CV' as loader and disregard 'PIL'
        #as of now i read all images in a specific sub folder at once and using a transform should be thought about before rushed

        image_filenames_list_of_lists = []
        image_folders = []
        images = []

        # ### Trial Parameters: ###
        # root_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'
        # max_number_of_videos = 15
        # number_of_images_per_video = 10;
        # crop_size = 100;
        # flag_load_to_RAM = False
        # flag_recursive = True

        tic()
        folder_counter = 0;
        for directory_path, directory_name, file_names_in_sub_directory in os.walk(root_folder):
            #If number of videos/directories so far exceeds maximum then break:
            if folder_counter > max_number_of_videos and max_number_of_videos != -1:
                break;

            if len(file_names_in_sub_directory) >= number_of_images_per_video:
                #Add new element to list, representing a sub directory
                image_folders.append(directory_path)
                image_filenames_list_of_lists.append([])
                folder_counter += 1;

                #go over images in the folder and add them to the filenames list:
                images_counter = 0;
                for current_filename in file_names_in_sub_directory:
                    full_filename = os.path.join(directory_path, current_filename);
                    if images_counter == number_of_images_per_video:
                        break;
                    elif is_image_file(full_filename):
                        images_counter += 1
                        image_filenames_list_of_lists[folder_counter-1].append('{}'.format(full_filename))
                        if flag_load_to_RAM:
                            images.append(read_image_cv2(full_filename))

            if flag_recursive == False:
                break;
        del image_filenames_list_of_lists[0]
        toc('End Of DataSet')
        # shuffle(filename_list)

        self.root_folder = root_folder
        self.image_filenames_list_of_lists = image_filenames_list_of_lists
        self.image_folders = image_folders
        self.images_list = images;
        self.transform = transform
        # self.loader = default_loader
        self.number_of_images_per_video = number_of_images_per_video
        self.crop_size = crop_size
        self.flag_output_YUV = flag_output_YUV
        self.flag_output_HSV = flag_output_HSV
        self.flag_output_channel_average = flag_output_channel_average
        self.flag_explicitely_make_tensor = flag_explicitely_make_tensor
        self.flag_normalize_by_255 = flag_normalize_by_255;
        self.loader = loader;

    def __getitem__(self, index):
        #Read Images:
        specific_subfolder_filenames_list = self.image_filenames_list_of_lists[index]
        current_folder_images_numpy, current_folder_images_filenames = read_images_from_filenames_list_to_numpy_with_crop(specific_subfolder_filenames_list, crop_size=self.crop_size, number_of_channels=3, max_number_of_images=self.number_of_images_per_video)

        # #Transform:
        # #TODO: if i'm sticking to loading all images wanted at once (not necessarily the best thing) then i need to make sure that the transform does the same thing for every image
        # if self.transform is not None:
        #     img = self.transform(img)

        ### Now the format is: [time_steps, H, W, C]
        current_folder_images_numpy = np.ascontiguousarray(np.transpose(current_folder_images_numpy, [0,3,1,2]))  #--> [time_steps, C, H, W]

        if self.flag_normalize_by_255:
            current_folder_images_numpy /= 255

        if self.flag_explicitely_make_tensor:
            current_folder_images_numpy = torch.Tensor(current_folder_images_numpy)


        return current_folder_images_numpy

    def __len__(self):
        return len(self.image_filenames_list_of_lists)

















#####################################################################################################################################################
def turbulence_deformation_single_image(I, Cn2=5e-13, flag_clip=False):
    ### TODO: why the .copy() ??? ###
    # I = I.copy()

    # I = read_image_default()
    # I = crop_tensor(I,150,150)
    # imshow(I)

    ### Parameters: ###
    h = 100
    # Cn2 = 2e-13
    # Cn2 = 7e-17
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(I.shape[0] / PatchSize))
    PatchNumCol = int(np.round(I.shape[1] / PatchSize))
    shape = I.shape
    [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    if I.dtype == 'uint8':
        mv = 255
    else:
        mv = 1

    ### Get Random Motion Field: ###
    ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)

    ### Resize (small) Random Motion Field To Image Size: ###
    ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    ### Add Rescaled Flow Field To Meshgrid: ###
    X = X0 + ShiftMatX
    Y = Y0 + ShiftMatY

    ### Resample According To Motion Field: ###
    I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)

    ### Clip Result: ###
    if flag_clip:
        I = np.minimum(I, mv)
        I = np.maximum(I, 0)

    # imshow(I)

    return I




def get_turbulence_flow_field(H,W, batch_size, Cn2=2e-13):
    ### TODO: why the .copy() ??? ###
    # I = I.copy()

    ### Parameters: ###
    h = 100
    # Cn2 = 2e-13
    # Cn2 = 7e-17
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(H / PatchSize))
    PatchNumCol = int(np.round(W / PatchSize))
    shape = I.shape
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    if I.dtype == 'uint8':
        mv = 255
    else:
        mv = 1

    ### Get Random Motion Field: ###
    [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
    [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
    X_large = torch.Tensor(X_large).unsqueeze(-1)
    Y_large = torch.Tensor(Y_large).unsqueeze(-1)
    X_large = (X_large-W/2) / (W/2-1)
    Y_large = (Y_large-H/2) / (H/2-1)

    new_grid = torch.cat([X_large,Y_large],2)
    new_grid = torch.Tensor([new_grid.numpy()]*batch_size)

    ShiftMatX0 = torch.randn(batch_size, PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = torch.randn(batch_size, PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)
    ShiftMatX0 = ShiftMatX0 * W
    ShiftMatY0 = ShiftMatY0 * H

    ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0.unsqueeze(1), new_grid, mode='bilinear', padding_mode='reflection')
    ShiftMatY = torch.nn.functional.grid_sample(ShiftMatX0.unsqueeze(1), new_grid, mode='bilinear', padding_mode='reflection')

    ShiftMatX = ShiftMatX.squeeze()
    ShiftMatY = ShiftMatY.squeeze()

    # ### Resize (small) Random Motion Field To Image Size: ###
    # ShiftMatX0 = F.adaptive_avg_pool2d(ShiftMatX0, torch.Size([H,W]))
    # ShiftMatY0 = F.adaptive_avg_pool2d(ShiftMatY0, torch.Size([H,W]))

    return ShiftMatX, ShiftMatY








class get_turbulence_flow_field_object:
    def __init__(self,H,W, batch_size, Cn2=2e-13):
        ### Parameters: ###
        h = 100
        # Cn2 = 2e-13
        # Cn2 = 7e-17
        wvl = 5.3e-7
        IFOV = 4e-7
        R = 1000
        VarTiltX = 3.34e-6
        VarTiltY = 3.21e-6
        k = 2 * np.pi / wvl
        r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
        PixSize = IFOV * R
        PatchSize = 2 * r0 / PixSize

        ### Get Current Image Shape And Appropriate Meshgrid: ###
        PatchNumRow = int(np.round(H / PatchSize))
        PatchNumCol = int(np.round(W / PatchSize))
        shape = I.shape
        [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
        if I.dtype == 'uint8':
            mv = 255
        else:
            mv = 1

        ### Get Random Motion Field: ###
        [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
        [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
        X_large = torch.Tensor(X_large).unsqueeze(-1)
        Y_large = torch.Tensor(Y_large).unsqueeze(-1)
        X_large = (X_large-W/2) / (W/2-1)
        Y_large = (Y_large-H/2) / (H/2-1)

        new_grid = torch.cat([X_large,Y_large],2)
        new_grid = torch.Tensor([new_grid.numpy()]*batch_size)

        self.new_grid = new_grid;
        self.batch_size = batch_size
        self.PatchNumRow = PatchNumRow
        self.PatchNumCol = PatchNumCol
        self.VarTiltX = VarTiltX
        self.VarTiltY = VarTiltY
        self.R = R
        self.PixSize = PixSize
        self.H = H
        self.W = W

    def get_flow_field(self):
        ShiftMatX0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltX * self.R / self.PixSize)
        ShiftMatY0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltY * self.R / self.PixSize)
        ShiftMatX0 = ShiftMatX0 * self.W
        ShiftMatY0 = ShiftMatY0 * self.H

        ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')
        ShiftMatY = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')

        ShiftMatX = ShiftMatX.squeeze()
        ShiftMatY = ShiftMatY.squeeze()

        return ShiftMatX, ShiftMatY


# ##################
# # Lambda: #
#
# def func_images(images, random_state, parents, hooks):
#     images[:, ::2, :, :] = 0
#     return images
#
# def func_heatmaps(heatmaps, random_state, parents, hooks):
#     for heatmaps_i in heatmaps:
#         heatmaps.arr_0to1[::2, :, :] = 0
#     return heatmaps
#
# def func_keypoints(keypoints_on_images, random_state, parents, hooks):
#     return keypoints_on_images
#
#
# aug = iaa.Lambda(func_images = func_images,
#                  func_heatmaps = func_heatmaps,
#                  func_keypoints = func_keypoints)
# ##################




class ImageFolderRecursive_MaxNumberOfImages_Deformations(data.Dataset):
    ### An image dataloader more appropriate for loading numpy arrays and implementing geometric distortions + center cropping: ###

    def __init__(self, root, transform=None, max_number_of_images=-1, min_crop_size=120, crop_size=100, Cn2=5e-13,
                 loader='CV',
                 allowed_extensions=IMG_EXTENSIONS_PNG,
                 flag_base_transform=False,
                 flag_turbulence_transform=False,
                 flag_output_YUV=False, flag_output_HSV=False, flag_output_channel_average=False,
                 flag_recursive=True,
                 flag_explicitely_make_tensor=False,
                 flag_normalize_by_255=False):
        #TODO: why isn't the inherited class' (data.Dataset) __init__ function being activated?
        images = []

        IMG_EXTENSIONS_PNG = ['.png', '.PNG']
        def is_image_file_png(filename, img_extentions=IMG_EXTENSIONS_PNG):
            return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

        self.flag_output_YUV = flag_output_YUV;
        self.flag_output_HSV = flag_output_HSV;
        self.flag_output_channel_average = flag_output_channel_average

        counter = 0;
        # root = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet'
        # max_number_of_images = 100
        for dp, dn, fn in os.walk(root):
            if counter > max_number_of_images and max_number_of_images != -1:
                break;
            for f in fn:
                filename = os.path.join(dp, f);
                counter += 1;
                if counter > max_number_of_images and max_number_of_images != -1:
                    break;
                elif is_image_file(filename,allowed_extensions):
                    images.append('{}'.format(filename))
            if flag_recursive == False:
                break;
        # shuffle(filename_list)

        self.root = root
        self.image_filenames = images
        self.transform = transform
        self.flag_base_transform = flag_base_transform
        self.flag_turbulence_transform = flag_turbulence_transform
        self.loader = loader
        self.min_crop_size = min_crop_size;
        self.extention = allowed_extensions;
        self.flag_explicitely_make_tensor = flag_explicitely_make_tensor
        self.flag_normalize_by_255 = flag_normalize_by_255

        if self.loader == 'PIL':
            self.loader_function = default_loader
        elif self.loader == 'CV':
            self.loader_function = read_image_cv2
        self.Cn2 = Cn2

        self.cropping_transform = iaa.CropToFixedSize(width=crop_size, height=crop_size, position='center', name=None, deterministic=False)



    def __getitem__(self, index):
        filename = self.image_filenames[index]

        #####################################################################################
        ### Read Image: ###
        try:
            current_image = self.loader_function(filename)
            if self.loader == 'PIL':
                s = current_image.size[0:2]
            if self.loader == 'CV':
                s = current_image.shape[0:2]

            if min(s) < self.min_crop_size:
                print('image smaller then min crop size')
                raise ValueError('image smaller than a block (32,32)')
        except:
            print("problem loading " + filename)
            if self.loader == 'PIL':
                current_image = Image.new('RGB', [self.min_crop_size, self.min_crop_size])
            elif self.loader == 'CV':
                current_image = np.zeros((self.min_crop_size,self.min_crop_size,3))

        # ### Normalize To 1 and convert to float32: ###
        # current_image = current_image / 255;
        # current_image = float32(current_image)
        ######################################################################################



        ######################################################################################
        ### If there is a transform apply it to the SINGLE IMAGE (whether it be from imgaug, albumentations or torchvision): ###
        if self.flag_base_transform:
            if self.transform is not None:
                if type(self.transform) == iaa.Sequential:
                    deterministic_Augmenter_Object = self.transform.to_deterministic()
                    current_image = deterministic_Augmenter_Object.augment_image(current_image);
                elif type(self.transform) == albumentations.Compose:
                    augmented_images_dictionary = self.transform(image=current_image);
                    current_image = augmented_images_dictionary['image']
                elif type(self.transform) == torchvision.transforms.transforms.Compose:
                    current_image = self.transform(current_image)

        #(*). Now that i've finished the base transform part (which could require PIL.Image object) now i need this in numpy format: (*)#
        ### Normalize To 1 and convert to float32: ###
        if self.loader == 'PIL':
            current_image = np.array(current_image)
        if self.flag_normalize_by_255:
            current_image = current_image / 255;
        current_image = float32(current_image)

        ### Add Turbulence Deformation: ###
        if self.flag_turbulence_transform:
            current_image = turbulence_deformation_single_image(current_image, Cn2=self.Cn2)

        ### Crop Center: ###
        current_image = self.cropping_transform.augment_image(current_image)
        ######################################################################################


        ### Append Color Conversions If So Wanted: ###
        if self.flag_output_channel_average:
            current_image_intensity = np.sum(current_image, axis=2, keepdims=True)
            current_image = np.concatenate((current_image, current_image_intensity), axis=2)

        ### To pytorch convention: [H,W,C]->[C,H,W] ###
        current_image = np.ascontiguousarray(np.transpose(current_image,[2,0,1]))

        if self.flag_explicitely_make_tensor:
            current_image = torch.Tensor(current_image)

        return current_image

    def __len__(self):
        return len(self.image_filenames)














class DataSet_Numpy_Deformations(data.Dataset):
    ### An image dataloader more appropriate for loading numpy arrays and implementing geometric distortions + center cropping: ###

    def __init__(self, numpy_array, transform=None, crop_size=100, Cn2=5e-13, flag_base_transform=False, flag_turbulence_transform=False, flag_output_YUV=False, flag_output_HSV=False, flag_output_channel_average=False):
        #TODO: why isn't the inherited class' (data.Dataset) __init__ function being activated?

        counter = 0;

        self.images_list = numpy_array
        self.images_torch = torch.Tensor(np.transpose(numpy_array,[0,3,1,2]))

        self.transform = transform
        self.flag_base_transform = flag_base_transform
        self.flag_turbulence_transform = flag_turbulence_transform
        self.Cn2 = Cn2
        self.cropping_transform = iaa.CropToFixedSize(width=crop_size, height=crop_size, position='center', name=None, deterministic=False)
        self.images_torch_cropped = torch.Tensor(np.transpose(self.cropping_transform.augment_images(numpy_array), [0, 3, 1, 2]))


    def __getitem__(self, index):

        #####################################################################################
        current_image = self.images_list[index].copy();  ### Should i use .copy() here?!$
        current_image = float32(current_image)
        ######################################################################################



        ######################################################################################
        ### If there is a transform apply it to the SINGLE IMAGE (whether it be from imgaug, albumentations or torchvision): ###
        if self.flag_base_transform:
            if self.transform is not None:
                if type(self.transform) == iaa.Sequential:
                    deterministic_Augmenter_Object = self.transform.to_deterministic()
                    current_image = deterministic_Augmenter_Object.augment_image(current_image);
                elif type(self.transform) == albumentations.Compose:
                    augmented_images_dictionary = self.transform(image=current_image);
                    current_image = augmented_images_dictionary['image']
                elif type(self.transform) == torchvision.transforms.transforms.Compose:
                    current_image = self.transform(current_image)

        ### Add Turbulence Deformation: ###
        if self.flag_turbulence_transform:
            current_image = turbulence_deformation_single_image(current_image, Cn2=self.Cn2)

        ### Crop Center: ###
        current_image = self.cropping_transform.augment_image(current_image)
        ######################################################################################


        ### To pytorch convention: [H,W,C]->[C,H,W] ###
        current_image = np.ascontiguousarray(np.transpose(current_image,[2,0,1]))
        return current_image

    def __len__(self):
        return len(self.images_list)






#######################################################################################################################################################


# torch.nn.functional.affine_grid(torch.Tensor(cv2.getRotationMatrix2D((50,50), 0, 1)).unsqueeze(0), torch.Size([1,3,100,100]))[0,:,:,0]

# current_image = read_image_default()
# H,W,C = current_image.shape
# M = cv2.getRotationMatrix2D((W/2,H/2),45,0.75)
# new_image = cv2.warpAffine(current_image,M,(W,H))
# imshow(new_image)






















