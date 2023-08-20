import torch
from os import path as osp

# Paths
base_path = '/home/mafat/Desktop/Omer'
vrt_path = osp.join(base_path, 'VRT')
data_path = osp.join(vrt_path, 'testsets')
dataset = 'coco30_shift=8_blur=8_steps=5_dim=[512,400]_frames=50'
results_dir = 'coco_30_results/007_VRT_videodeblurring_REDS'

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
    return model

def get_frames(num_img, num_frame): # Returns a tuple (orig_fra, blur_frame, vrt_frame)

    folder = str(num_img).zfill(3)
    pic = str(num_frame).zfill(8)
    format = '.png' # or file, Path, PIL, OpenCV, numpy, list
    orig_frame = osp.join(data_path,dataset,'Clean', folder, pic + format)
    blur_frame = osp.join(data_path,dataset,'Noisy', folder, pic + format)
    vrt_frame = osp.join(vrt_path, results_dir, folder, pic + format)

    return orig_frame, blur_frame, vrt_frame

def get_yolo_results(model, frames):

    results = []
    for frame in frames:
        results.append(model(frame)) # Inference
    return results

def show_results(results):

    for result in results:
        result.show()

def get_results_dfs(results):

    dfs = []
    for result in results:
        dfs.append(result.pandas().xyxy[0])
    return dfs

def get_metrics(results_dfs):


model = load_model()

num_img = 10
num_frame = 36
frames = get_frames(num_img, num_frame)
results = get_yolo_results(model, frames)
show_results(results[:3])

results[2].save()
results[1].save()
results[0].save()
dfs = get_results_dfs(results)

dfs[0]

results.save()


#metrics:

#) intersection with hr
#) num of detections compared to hr:
#) avg confidence compared to hr
# add metric with symmetric 1/2 + 1/2, look at both metrics and add final metric
# good - bad or good/bad or good - bad / bad or something
# add regular metric or edit it with confidences
# etcs