import os.path

BASE_PROJECT = os.path.dirname(os.path.abspath(__file__))
RDND_BASE_PATH = os.path.join(BASE_PROJECT, "RDND_proper")
GROUNDED_SEGMENT_ANYTHING_BASE_PATH = os.path.join(BASE_PROJECT, "Grounded_Segment_Anything")
OUTPUT = "output"
CAR_LICENSE_PLATE_RECOGNITION = "CAR_LICENSE_PLATE_RECOGNITION"


path_GroundingDINO_SwinT_OGC_PY = os.path.join(GROUNDED_SEGMENT_ANYTHING_BASE_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
path_groundingdino_swint_ogc_PTH = os.path.join(GROUNDED_SEGMENT_ANYTHING_BASE_PATH,"groundingdino_swint_ogc.pth")
path_sam_vit_h_4b8939 = os.path.join(GROUNDED_SEGMENT_ANYTHING_BASE_PATH, "sam_vit_h_4b8939.pth")
path_ema_ckpt = os.path.join(RDND_BASE_PATH, "EMA", "ckpt")





### denoise

path_restore_ckpt_denoise_flow_former = os.path.join(RDND_BASE_PATH, "models", "FlowFormer", "check_points", "sintel.pth")
path_checkpoint_latest_things = os.path.join(RDND_BASE_PATH, "models", "irr", "checkpoints", "checkpoint_latest_things.ckpt")


### deblur

RVRT_deblur_shoval_train_py_blur20_TEST1_Step60000 = os.path.join(BASE_PROJECT, "Omer", "to_neo", "RVRT_deblur_shoval_train.py_blur20_TEST1_Step60000.tar")


### Directory path to save result for the detedtion process

DETECTION_DIRECTORY_PATH = os.path.join(BASE_PROJECT, OUTPUT, CAR_LICENSE_PLATE_RECOGNITION)