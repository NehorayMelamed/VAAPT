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
final_filepath = r'/home/mafat/DataSets/MTF/Rect_Background' + str(background_level) + '_Rect' + str(rect_level) + '_Angle' + str(rotation_angle) + '.png'

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

cv2.imwrite(final_filepath, final_image)








