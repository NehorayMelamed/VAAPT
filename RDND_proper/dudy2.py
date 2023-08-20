from RapidBase.import_all import *

input_tensor = read_video_default_torch()
input_tensor = RGB2BW(input_tensor)
imshow_torch_video(BW2RGB(input_tensor), FPS=10)





