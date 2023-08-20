import os

import torch

from PARAMETER import path_sam_vit_h_4b8939, path_GroundingDINO_SwinT_OGC_PY, path_groundingdino_swint_ogc_PTH
from grounded_sam_demo import grounded_segment_anything, load_model


def process_images_in_dir(input_dir, text_prompt, base_output_dir, grounded_segment_anything_args, use_biggest_mask=False, logger=None):

    print("process_images_in_dir")
    """
    Process each image in a directory.
    Args:
    - input_dir (str): The directory where the input images are located.
    - text_prompt (str): The text input to the model.
    - base_output_dir (str): The base directory where the output will be saved.
    - grounded_segment_anything_args (dict): Dictionary containing other arguments to pass to the grounded_segment_anything function.
    """

    # Get a list of all the images in the directory
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if
                   fname.endswith(('.jpg', '.jpeg', '.png'))]
    print(image_paths)
    c = 0
    # model = load_model(path_GroundingDINO_SwinT_OGC_PY, path_groundingdino_swint_ogc_PTH, device=grounded_segment_anything_args["device"]).eval()

    for image_path in image_paths:
        torch.cuda.empty_cache()
        print(f"Preforming segmentation ({str(c)}/{str(len(image_paths))}) for image - {image_path}\n with prompt "
              f"text - {text_prompt}")
        logger.log(f"Preforming segmentation ({str(c)}/{str(len(image_paths))}) for image - {image_path}\n with prompt "
              f"text - {text_prompt}")
        # create a unique output directory for each image based on its name
        frame_index_number = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(base_output_dir, frame_index_number)
        os.makedirs(output_dir, exist_ok=True)

        grounded_segment_anything(
            input_image=image_path,
            text_prompt=text_prompt,
            output_dir=output_dir,
            get_bigger_mask=use_biggest_mask,
            index_frame_name=frame_index_number,
            # model=model,
            **grounded_segment_anything_args,
        )
        c+=1
    return True


# # usage
# grounded_segment_anything_args = {
#     'desired_size': (512, 512),
#     'sam_checkpoint': path_sam_vit_h_4b8939,
#     'grounded_checkpoint': path_groundingdino_swint_ogc_PTH,
#     'config_file': path_GroundingDINO_SwinT_OGC_PY,
#     'sam_hq_checkpoint': None,
#     'use_sam_hq': False,
#     'box_threshold': 0.3,
#     'text_threshold': 0.25,
#     'device': "cpu"
# }
#

# process_images_in_dir("/home/nehoray/PycharmProjects/Shaback/Grounded_Segment_Anything/for_yoav/Frames", "car", "/home/nehoray/PycharmProjects/Shaback/Grounded_Segment_Anything/for_yoav/output", grounded_segment_anything_args)
# /home/nehoray/PycharmProjects/Shaback/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
# /home/nehoray/PycharmProjects/Shaback/Grounded_Segment_Anything/GroundingDINO/groundingdino/config