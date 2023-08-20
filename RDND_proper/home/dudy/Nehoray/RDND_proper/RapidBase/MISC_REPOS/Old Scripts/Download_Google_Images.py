#Download Images From google images using python:


from google_images_download import google_images_download
import os
import glob

response = google_images_download.googleimagesdownload()   #class instantiation
arguments_dictionary = dict();

#Search Terms and number of images:
prefix_keywords_to_search_string = "macro"
main_search_string = "trees"; #can state many searchs with comma: "baloons, trees, humans"
suffix_keywords_to_search_string = "";
number_of_images = 20;

#Saved Image Names:
prefix_to_add_to_images = "";

#Image Type:
image_type = "photo" #Possible values: face, photo, clip-art, line-drawing, animated
image_format = "jpg" #Possible values: jpg, gif, png, bmp, svg, webp, ico

#Size Filter- Can Only Peak ONE Option for size stating:
flag_relative_or_exact_size = None #Possible Values: "exact"/"relative"/"none"
if flag_relative_or_exact_size == "relative":
    #(1). Images Larger Than Specified Size Below:
    relative_image_size = ">400*300"; #Possible values: large, medium, icon, >400*300, >640*480, >800*600, >1024*768, >2MP, >4MP, >6MP, >8MP, >10MP, >12MP, >15MP, >20MP, >40MP, >70MP
elif flag_relative_or_exact_size == "exact":
    #(2). Exact Size:
    exact_image_size = "1024,786";

#Directories:
output_directory = "C:/Users\dkarl\Desktop\GoogleImagesDownload"
chromedriver_full_path_to_exe = "C:/Users\dkarl\Desktop\GoogleImagesDownload\chromedriver_win32\chromedriver.exe"


#Build Dictionary for download function
arguments_dictionary['keywords'] = main_search_string;
arguments_dictionary['prefix_keywords'] = prefix_keywords_to_search_string
arguments_dictionary['suffix_keywords'] = suffix_keywords_to_search_string
arguments_dictionary['prefix'] = prefix_keywords_to_search_string
arguments_dictionary['limit'] = number_of_images;
arguments_dictionary['format'] = image_format
arguments_dictionary['type'] = image_type
arguments_dictionary['print_urls'] = False
arguments_dictionary['output_directory'] = output_directory
arguments_dictionary['chromedriver'] = chromedriver_full_path_to_exe
if flag_relative_or_exact_size == "relative":
    arguments_dictionary['size'] = relative_image_size;
elif flag_relative_or_exact_size == 'exact':
    arguments_dictionary['exact_size'] = exact_image_size

paths = response.download(arguments_dictionary)


#Change Downloaded Images Names to simple ordered:
pattern = '*';
for key in paths.keys():
    current_directory = os.path.join(output_directory, key);
    print('current directory: ' + current_directory)

    number_of_images_already_ordered = 0;
    for pathAndFilename in glob.iglob(os.path.join(current_directory, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, os.path.join(current_directory, str(number_of_images_already_ordered) + ext))
        number_of_images_already_ordered += 1







# arguments = {"keywords":search_string,
#              "limit":number_of_images,
#              "print_urls":True,
#              "output_directory":output_directory,
#              "chromedriver":chromedriver_full_path_to_exe}   #creating list of arguments
# paths = response.download(arguments)   #passing the arguments to the function
