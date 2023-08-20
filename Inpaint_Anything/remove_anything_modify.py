
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from segment_anything import sam_model_registry, SamPredictor
from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask, show_points


def inpaint_images(
        input_dir,
        points_list,
        point_labels,
        dilate_kernel_size,
        output_dir,
        sam_model_type,
        sam_ckpt,
        lama_config,
        lama_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu"

):
    images_paths = sorted(Path(input_dir).glob('*.png'))

    if len(images_paths) != len(points_list):
        # raise ValueError("The number of images must match the number of points")
        print(f"The number of images - {len(images_paths)} must match the number of points - { len(points_list)}")
        print("Try again")
        return -1

    print("Loading models...")
    # Instantiate the model and predictor once
    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
    sam_model.to(device=device)
    predictor = SamPredictor(sam_model)
    print("Finished!")
    c = 0
    for image_path, point in zip(images_paths, points_list):
        img = load_img_to_array(str(image_path))

        # process and save the image, then delete it from memory
        process_and_save_image(img, image_path, point, point_labels, dilate_kernel_size, output_dir, sam_model_type,
                               sam_ckpt, lama_config, lama_ckpt, device, predictor=predictor)

        del img
        torch.cuda.empty_cache()  # clear memory cache
        print(f"Frame number {c} / {len(points_list)} finished!")
        c += 1

def process_and_save_image(img, image_path, point, point_labels, dilate_kernel_size, output_dir, sam_model_type, sam_ckpt, lama_config, lama_ckpt, device,predictor):
    masks, _, _ = predict_masks_with_sam(
        img,
        [point],
        [point_labels],
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
        predictor=predictor
    )

    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(image_path).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), point, point_labels, size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, lama_config, lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)

        del mask, mask_p, img_inpainted_p, img_inpainted  # Explicit deletion of each intermediate variable

    del img, masks, img_stem, out_dir  # Explicit deletion of each intermediate variable
    torch.cuda.empty_cache()  # Clear GPU memory



