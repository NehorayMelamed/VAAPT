import argparse
import os
import copy
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PARAMETER import path_GroundingDINO_SwinT_OGC_PY, path_sam_vit_h_4b8939, path_groundingdino_swint_ogc_PTH

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    

# export CUDA_VISIBLE_DEVICES=0
# python grounded_sam_demo.py \
#   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#   --grounded_checkpoint groundingdino_swint_ogc.pth \
#   --sam_checkpoint sam_vit_h_4b8939.pth \
#   --input_image assets/demo1.jpg \
#   --output_dir "outputs" \
#   --box_threshold 0.3 \
#   --text_threshold 0.25 \
#   --text_prompt "bear" \
#   --device "cuda"


def grounded_segment_anything(input_image, text_prompt, output_dir, desired_size=(512, 512),
                              sam_checkpoint=path_sam_vit_h_4b8939, grounded_checkpoint=path_groundingdino_swint_ogc_PTH,
                              config_file=path_GroundingDINO_SwinT_OGC_PY,
                              sam_hq_checkpoint=None, use_sam_hq=False, box_threshold=0.3, text_threshold=0.25,
                              device="cpu", get_bigger_mask=False, index_frame_name=""): #, model=None):
    """
    Function to perform Grounded Segment Anything operation.
    index_frame_name: If needed it represents the frame index
    """

    # cfg
    grounded_checkpoint = grounded_checkpoint  # path of the model
    sam_checkpoint = sam_checkpoint
    sam_hq_checkpoint = sam_hq_checkpoint
    use_sam_hq = use_sam_hq
    image_path = input_image
    text_prompt = text_prompt
    output_dir = output_dir
    box_threshold = box_threshold
    text_threshold = text_threshold
    device = device
    print(device)
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    # if model is None:
    model = load_model(config_file, grounded_checkpoint, device=device).eval()

    # visualize raw image
    image_pil.save(os.path.join(output_dir, f"{index_frame_name}_raw_image.png"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # draw output image
    from collections import defaultdict
    label_count = defaultdict(int)

    max_mask = 0
    max_mask_info = None

    for idx, (mask, box, label) in enumerate(zip(masks, boxes_filt, pred_phrases)):
        label_name = label.split('(')[0]  # extract the label name before '('
        label_count[label_name] += 1

        mask_area = torch.sum(mask).item()  # this gives the number of pixels in the mask

        if get_bigger_mask:
            if mask_area > max_mask:
                max_mask = mask_area
                max_mask_info = (mask, box, label)
        else:
            # Create a new directory for this object
            object_dir = os.path.join(output_dir, f'{label_name}_{label_count[label_name]}')
            os.makedirs(object_dir, exist_ok=True)

            # Save the mask data
            torch.save(mask, os.path.join(object_dir, f"{index_frame_name}_segmentation.pt"))


            # Draw the segmentation on the image
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            show_mask(mask.cpu().numpy(), ax, random_color=True)
            show_box(box.numpy(), ax, label)
            ax.axis('off')

            fig.savefig(os.path.join(object_dir, f"{index_frame_name}_grounded_sam_output.png"))
            plt.close(fig)

    if get_bigger_mask and max_mask_info is not None:
        # This block will be executed if `get_bigger_mask` is True and there was at least one mask generated
        mask, box, label = max_mask_info

        # Create a new directory for this object
        object_dir = os.path.join(output_dir, f'{label.split("(")[0]}_biggest')
        os.makedirs(object_dir, exist_ok=True)

        # Save the mask data
        torch.save(mask, os.path.join(object_dir, f"{index_frame_name}_segmentation.pt"))

        # Draw the segmentation on the image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        show_mask(mask.cpu().numpy(), ax, random_color=True)
        show_box(box.numpy(), ax, label)
        ax.axis('off')

        # Save the output image
        fig.savefig(os.path.join(object_dir, "grounded_sam_output.jpg"))
        plt.close(fig)


### main
# grounded_segment_anything("/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/resized_image.png", "car", "/home/nehoray/PycharmProjects/Shaback/Grounded_Segment_Anything/output_car")
