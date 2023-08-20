import haggingfacedir.main as hagging_face
import Blip_implementation.main as blip
from data.people.questions import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


if __name__ == '__main__':
    image_path = "/home/nehoray/PycharmProjects/Shaback/question_AND_answering_new/output/Downtest_ch0007_0001000028400000012/Downtest_ch0007_0001000028400000012.jpg"
    raw_image = Image.open(image_path).convert("RGB")

    # Read the image
    image = mpimg.imread(image_path)
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Remove axes and ticks
    plt.show(block=False)  # Show image window without blocking code execution

    # Models
    hugging_face_model = hagging_face.get_model_and_stuff()
    blip_model, vis_processors, txt_processors = blip.get_model_and_stuff(blip.Action.QandA)



    print("\n!!!!! Outfit !!!!!")
    # Iterate over the questions in the Outfit enum and print them
    for question in PersonQuestion.Outfit:
        hugging_face_answer = hagging_face.question_on_image(image_path, model=hugging_face_model, question=question.value)
        blip_answer = blip.question_on_image(raw_image, model=blip_model, question=question.value, vis_processors=vis_processors,txt_processors=txt_processors)

        print(f"Question:    {question.value}\n"
              f"Hugging face answer:   {hugging_face_answer}\n"
              f"Blip Answer:           {blip_answer}\n\n")

    print("\n!!!!! Visibility !!!!!")
    # Iterate over the questions in the Outfit enum and print them
    for question in PersonQuestion.Visibility:
        hugging_face_answer = hagging_face.question_on_image(image_path, model=hugging_face_model, question=question.value)
        blip_answer = blip.question_on_image(raw_image, model=blip_model, question=question.value, vis_processors=vis_processors,txt_processors=txt_processors)

        print(f"Question:    {question.value}\n"
              f"Hugging face answer:   {hugging_face_answer}\n"
              f"Blip Answer:           {blip_answer}\n\n")

    print("\n!!!!! Actions !!!!!")
    # Iterate over the questions in the Outfit enum and print them
    for question in PersonQuestion.Actions:
        hugging_face_answer = hagging_face.question_on_image(image_path, model=hugging_face_model,
                                                             question=question.value)
        blip_answer = blip.question_on_image(raw_image, model=blip_model, question=question.value,
                                             vis_processors=vis_processors, txt_processors=txt_processors)

        print(f"Question:    {question.value}\n"
              f"Hugging face answer:   {hugging_face_answer}\n"
              f"Blip Answer:           {blip_answer}\n\n")


    # Keep the image window open until it is closed manually
    plt.show()