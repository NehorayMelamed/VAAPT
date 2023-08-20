from RapidBase.import_all import *
from RDND_proper.KAIR_light.models.vrt_models import VRT_SR_6_frames_checkpoint, VRT_SR_6_frames_checkpoint_img_sz_160

# Paths
# model_sr_path = '/raid/Pytorch_Checkpoints/2022-07-18/Model_New_Checkpoints/visDroen_SR_8_to_2_bw.py/visDroen_SR_8_to_2_bw.py_TEST1_Step37875.tar'
# model_sr_path = '/raid/Pytorch_Checkpoints/2022-06-26/Model_New_Checkpoints/drones_SR_BW[160,160].py/drones_SR_BW[160,160].py_TEST1_Step38925.tar'
model_sr_path = '/raid/Pytorch_Checkpoints/2022-06-26/Model_New_Checkpoints/drones_SR_BW_noised.py/drones_SR_BW_noised.py_TEST1_Step10350.tar'
inference_path ='/raid/datasets/third_fixed_images/004'
output_path = '/raid/inference/RVRT_SR_inference8'

# Setting Params
train_device = 3
test_device = 4
num_frames = 9 # Multiple of 3
w = 480 #2048
h =  640 #2048

# Model
model = VRT_SR_6_frames_checkpoint_img_sz_160(False, checkpoint_path=model_sr_path, train_device=train_device,
                                       test_device=test_device)
# print("Loading video")
video = read_video_default_torch(video_dir=inference_path, size=(w,h), flag_convert_to_rgb=0)
video = RGB2BW(video[:num_frames])

def SR(model, video):
    video = video/video.max()
    video = video.unsqueeze(0).to(train_device)
    B, T, C, H, W = video.shape
    video = video.reshape(B, T // 3, C * 3, H, W)
    print("Running model")
    output = model.test_video(video)
    B, T, C, H, W = output.shape
    output = output.reshape(B, T * 3, C // 3, H, W)
    print ('Inference finished')
    return output[0]
#
# imshow_torch(output[0][0]);plt.show()
# imshow_torch(video[0]);plt.show()