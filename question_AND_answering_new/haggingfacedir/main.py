"""
Implementation of the next fit repository ->
https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
"""

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from RapidBase.Utils.IO.tic_toc import start_cuda, finish_cuda, gtic, gtoc
import torch


def get_model_and_stuff():
    gtic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    gtoc("Model loading time:")
    return model


def question_on_image(path_to_image, model, question):
    gtic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_image = Image.open(path_to_image)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    inputs = processor(raw_image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    result = processor.decode(out[0], skip_special_tokens=True)
    gtoc("Question time:")
    return result

#
# if __name__ == '__main__':
#     # pass
#     model = get_model_and_stuff()
#     image_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/data/photos/shaback/vehicles/Downtest_ch0007_00010000284000000/jeep/Downtest_ch0007_000100002840000006.jpg'
#     image_path = "/home/dudy/Nehoray/segment_anything_base_dir/Grounded-Segment-Anything/data/vehicles_with_stickers/1.jpg"
#     image_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/question_AND_answering/data/images/vehicles_with_stickers/2.jpg"
#     image_path = "/segment_anything_base_dir/Grounded-Segment-Anything/scripts/output/minivan_with_writing/masked_image_Downtest_ch0007_000100002840000004_mask_rear_window_side_window_3.jpg"
#     image_path = "/home/dudy/Nehoray/segment_anything_base_dir/Grounded-Segment-Anything/outputs/vehicle_with_stickers/Downtest_ch0007_000100002840000004/Downtest_ch0007_000100002840000004_raw_image.jpg"
#     # image_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/question_AND_answering/data/signals_data/LWIR_0548.tif"
#     # image_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/question_AND_answering/data/signals_data/MWIR_0125.tif'
#     image_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/question_AND_answering/data/signals_data/NIR_13-42-39.000-183.tif'
#     # image_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/question_AND_answering/data/signals_data/SWIR_0168.tif'
#
#     question = "how many cars are there?"
#     question = "what is behind the cars?"
#     question = "is the car window white?"
#     question = "is the car window black?"
#     question = "is the car window transparent?"
#     print(f"The question: \'{question}\'")
#     print(question_on_image(image_path, model=model, question=question))
#     #
