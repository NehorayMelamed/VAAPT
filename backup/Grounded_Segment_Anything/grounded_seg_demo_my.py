"""Script for creating data from directory of images
"""
import glob
import re
import matplotlib.patches as patches
import torch.nn.functional as F
import random
import os
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt




def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])
    return image



def show_box(box, ax, label, color=None):
    if color is None:
        color = np.random.rand(3)
    x, y, w, h = box[:4]
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(
        x, y, label, fontsize=12, bbox=dict(facecolor=color, alpha=0.5), color="white"
    )


def show_mask(mask, ax, color=None, alpha=0.5):
    pass
    # if color is None:
    #     color = np.random.rand(3)
    # mask = (mask > 0.5).astype(np.uint8)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     cnt[:, :, 0] = cnt[:, :, 0].clip(0, mask.shape[1] - 1)
    #     cnt[:, :, 1] = cnt[:, :, 1].clip(0, mask.shape[0] - 1)
    #     cnt = cnt.astype(np.float32)
    #     cnt = cnt[:, 0, :]
    #     poly = Polygon(cnt, facecolor=color, edgecolor=color, alpha=alpha)
    #     ax.add_patch(poly)

def get_largest_image_size(image_paths):
    max_width, max_height = 0, 0
    for image_path in image_paths:
        img = Image.open(image_path)
        width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    return max_width, max_height

def load_image(image_path, max_width, max_height):
    image_pil = Image.open(image_path).convert("RGB")
    image_pil = image_pil.resize((max_width, max_height), Image.BICUBIC)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    image = F.interpolate(image.unsqueeze(0), (max_height, max_width), mode='bilinear', align_corners=False).squeeze(0)
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

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device, max_width, max_height):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    mask_img = torch.zeros((max_height, max_width))
    for idx, box in enumerate(boxes_filt):
        x1, y1, x2, y2 = [int(coord.item()) for coord in box]
        mask_img[y1:y2, x1:x2] = idx + 1

    mask_img = F.interpolate(mask_img.unsqueeze(0).unsqueeze(0), (max_height, max_width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    return boxes_filt, pred_phrases, mask_img


def convert_tif_to_png(tif_path, png_path):
    try:
        # Open the TIF image
        tif_image = Image.open(tif_path)

        # Convert the image to RGB mode if it's not already
        if tif_image.mode != "RGB":
            tif_image = tif_image.convert("RGB")

        # Save the image in PNG format
        tif_image.save(png_path, "PNG")

        print(f"Image converted successfully and saved as {png_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_mask_data(output_dir, image_name, masks, boxes, pred_phrases, max_width, max_height):
    mask_output_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_output_dir, exist_ok=True)

    def get_word_at_beginning(text):
        match = re.match(r'^([\w\s]+)', text)
        if match:
            word = match.group(1)
            return word.replace(' ', '_')
        return "_"

    best_masks = {}
    for idx, (mask, box, pred_phrase) in enumerate(zip(masks, boxes, pred_phrases)):
        name_class = f"{get_word_at_beginning(pred_phrase)}_{idx}"  # Add the index to the name_class
        score = float(re.search(r'\((.+?)\)', pred_phrase).group(1))
        if name_class not in best_masks or score > best_masks[name_class][2]:
            best_masks[name_class] = (mask, box, score)

    for name_class, (mask, box, score) in best_masks.items():
        mask_path = os.path.join(mask_output_dir, f"{image_name}_mask_{name_class}.png")
        mask = (mask.cpu().numpy() * 255).astype(np.uint8)
        mask = np.squeeze(mask)
        mask_resized = cv2.resize(mask, (max_width, max_height), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(mask_resized).save(mask_path)

        mask_pt_path = os.path.join(mask_output_dir, f"{image_name}_mask_{name_class}.pt")
        torch.save(mask, mask_pt_path)

        print(f"Saved mask for image {image_name} with phrase '{name_class}({score})'")


def process_image(image_path, text_prompt, output_dir, model, predictor, box_threshold, text_threshold, device,
                  max_width, max_height):
    # load image
    image_pil, image = load_image(image_path, max_width, max_height)

    # get image_name
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # visualize raw image
    image_pil.save(os.path.join(output_dir, f"{image_name}_raw_image.jpg"))
    # run grounding dino model
    boxes_filt, pred_phrases, mask_img = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device, max_width, max_height
    )
    # initialize SAM
    image = torch.nn.Upsample(scale_factor=0.25)(image.unsqueeze(0)).squeeze(0)
    image = np.array(image.permute(1, 2, 0))
    image -= image.min()
    image *= (255 / image.max())
    image = image.astype(np.uint8)
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
        boxes=transformed_boxes.to(device), # torch (1, 4)
        multimask_output=False,
    )
    # Apply different colored masks
    for i, (box, mask) in enumerate(zip(boxes_filt, masks)):
        color = random_color()
        image = apply_mask(image, mask.cpu().numpy(), color)

    # Save the image with different colored masks
    output_path = os.path.join(output_dir, f"{image_name}_colored_masks.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    save_mask_data(output_dir, image_name, masks, boxes_filt, pred_phrases, max_width, max_height)

    return masks, boxes_filt, pred_phrases



def main(input_dir, text_prompt, output_base_dir, config_file, grounded_checkpoint, sam_checkpoint, box_threshold,
         text_threshold, device):
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))

    max_width, max_height = get_largest_image_size(image_paths)
    model = load_model(config_file, grounded_checkpoint, device=device)
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(output_base_dir, image_name)
        os.makedirs(output_dir, exist_ok=True)

        process_image(image_path, text_prompt, output_dir, model, predictor, box_threshold, text_threshold, device,
                      max_width, max_height)
#
#
# if __name__ == "__main__":
#     input_dir = "/home/nehoray/PycharmProjects/Shaback/data/images/ID_1_Type_car"
#     output_base_dir = "output_car"
#
#     # text_prompt =       "rectangle." \
#     #                     "concrete barricade." #\
#     text_prompt = "car." \
#                   "license plate"
#     # "sticker" \
#     # "Caption"\
    # "writing"\
    # "inscription"\
    #                 "lights"\
    #                     "rear of the car. " \
    # "fron of the car"\
    #             " side of the car."\
    #             "rear window." \
    #             "side window." \
    #             "dor side" \
    #               "license plate." \
    #               "car left light." \
    #               " car right light." \
    #               " car wheels."

    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    box_threshold = 0.3
    text_threshold = 0.25
    device = "cpu"
    main(input_dir, text_prompt, output_base_dir, config_file, grounded_checkpoint, sam_checkpoint, box_threshold, text_threshold, device)



