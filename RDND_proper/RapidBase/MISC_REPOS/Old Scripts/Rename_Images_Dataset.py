


import os

IMG_EXTENSIONS_PNG = ['.png','.PNG']

def is_image_file(filename,allowed_endings=IMG_EXTENSIONS_NO_PNG):
    return any(list((filename.endswith(extension) for extension in allowed_endings)))

def get_image_filenames_from_folder(path,allowed_endings=IMG_EXTENSIONS_PNG, flag_recursive=False):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    image_filenames = []
    #Look recursively at all current and children path's in given path and get all image files
    for dirpath, _, fnames in sorted(os.walk(path)): #os.walk!!!!
        for fname in sorted(fnames):
            if is_image_file(fname,allowed_endings=allowed_endings):
                img_path = os.path.join(dirpath, fname)
                image_filenames.append(img_path)
        if flag_recursive == False:
            break;
    assert image_filenames, '{:s} has no valid image file'.format(path) #assert images is not an empty list
    return image_filenames


### Change Here: ###
images_dataset_super_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'


### Rename Files: ###
images_filenames = get_image_filenames_from_folder(images_dataset_super_folder,IMG_EXTENSIONS_PNG, True)
for image_filename in images_filenames:
    os.renames(image_filename, os.path.join(os.path.split(image_filename)[0] , os.path.split(image_filename)[1].split('.')[0].rjust(4,'0') + '.png'))











