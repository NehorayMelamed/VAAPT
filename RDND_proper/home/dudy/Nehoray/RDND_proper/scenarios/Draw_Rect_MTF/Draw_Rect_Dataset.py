from RapidBase.import_all import *
import math
from PIL import Image, ImageDraw

#finds the straight-line distance between two points
def distance(ax, ay, bx, by):
    return math.sqrt((by - ay)**2 + (bx - ax)**2)

#rotates point `A` about point `B` by `angle` radians clockwise.
def rotated_about(ax, ay, bx, by, angle):
    radius = distance(ax,ay,bx,by)
    angle += math.atan2(ay-by, ax-bx)
    return (
        round(bx + radius * math.cos(angle)),
        round(by + radius * math.sin(angle))
    )

image_size = (500,500)
square_center = (250,250)
square_length = 100
background_level = 80
rect_level = 150
rotation_angle = 5

background_levels_vec = (0,40,80,120,160,200,240)
rect_levels_vec = (10,50,90,130,170,210,250)

for rect_level in rect_levels_vec:
    for background_level in background_levels_vec:
        final_filepath = r'/home/mafat/DataSets/MTF/Rect_Background' + str(background_level) + '_Rect' + str(rect_level) + '_Angle' + str(rotation_angle)

        image = Image.new('L', image_size, background_level)
        draw = ImageDraw.Draw(image)

        square_vertices = (
            (square_center[0] + square_length / 2, square_center[1] + square_length / 2),
            (square_center[0] + square_length / 2, square_center[1] - square_length / 2),
            (square_center[0] - square_length / 2, square_center[1] - square_length / 2),
            (square_center[0] - square_length / 2, square_center[1] + square_length / 2)
        )

        square_vertices = [rotated_about(x,y, square_center[0], square_center[1], math.radians(rotation_angle)) for x,y in square_vertices]

        draw.polygon(square_vertices, fill=100)

        imshow(np.array(image))
        final_image = np.array(image)

        PSNR_vec = [np.inf,100,10,5,4,3,2,1]
        for PSNR in PSNR_vec:
            current_folder = os.path.join(final_filepath, str(PSNR))
            path_make_path_if_none_exists(current_folder)
            for noise_counter in np.arange(10):
                final_image_noisy = final_image + 1/np.sqrt(PSNR)*np.random.randn(*final_image.shape) * 255
                cv2.imwrite(os.path.join(current_folder,string_rjust(str(noise_counter), 4) + '.png'), final_image_noisy)








