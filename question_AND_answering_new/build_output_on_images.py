import os
import json
import shutil
from pathlib import Path
import haggingfacedir.main as hagging_face
import Blip_implementation.main as blip
from data.people.questions import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def process_questions(image_path, raw_image, categories, model_hf, model_blip, vis_processors, txt_processors):
    json_output = {}
    txt_output = ""

    for category, questions in categories.items():
        json_output[category] = []
        txt_output += f"\n!!!!! {category} !!!!!\n"

        for question in questions:
            hugging_face_answer = hagging_face.question_on_image(image_path, model=model_hf, question=question.value)
            blip_answer = blip.question_on_image(raw_image, model=model_blip, question=question.value, vis_processors=vis_processors, txt_processors=txt_processors)

            json_output[category].append({
                "question": question.value,
                "hugging_face_answer": hugging_face_answer,
                "blip_answer": blip_answer
            })

            txt_output += f"Question:    {question.value}\n"\
                          f"Hugging face answer:   {hugging_face_answer}\n"\
                          f"Blip Answer:           {blip_answer}\n\n"

    return json_output, txt_output


if __name__ == '__main__':
    images_directory = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/photos/shaback/people/Downtest_ch0007_00010000284000000"
    results_directory = "/home/dudy/Nehoray/SHABACK_POC_NEW/question_AND_answering/output"

    # Models
    hugging_face_model = hagging_face.get_model_and_stuff()
    blip_model, vis_processors, txt_processors = blip.get_model_and_stuff(blip.Action.QandA)

    question_categories = {
        "Outfit": PersonQuestion.Outfit,
        "Visibility": PersonQuestion.Visibility,
        "Actions": PersonQuestion.Actions
    }

    for image_name in os.listdir(images_directory):
        image_path = os.path.join(images_directory, image_name)
        image_id = Path(image_name).stem
        image_output_dir = os.path.join(results_directory, image_id)
        os.makedirs(image_output_dir, exist_ok=True)

        # Save the image in the new directory
        shutil.copy(image_path, os.path.join(image_output_dir, image_name))

        # Read the image
        raw_image = Image.open(image_path).convert("RGB")
        image = mpimg.imread(image_path)
        plt.imshow(image)
        plt.axis('off')  # Remove axes and ticks
        plt.show(block=False)  # Show image window without blocking code execution

        # Perform questions and save results
        json_output, txt_output = process_questions(image_path, raw_image, question_categories, hugging_face_model, blip_model, vis_processors, txt_processors)

        # Save JSON and text files
        with open(os.path.join(image_output_dir, "results.json"), "w") as json_file:
            json.dump(json_output, json_file, indent=4)

        with open(os.path.join(image_output_dir, "results.txt"), "w") as txt_file:
            txt_file.write(txt_output)

        # Keep the image window open until it is closed manually
        # plt.show()
