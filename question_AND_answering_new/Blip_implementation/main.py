"""
Implementation of the next fit repository ->
https://github.com/salesforce/lavis
"""


from enum import Enum
import torch
from PIL import Image
import torch
#ToDo import it as usual
# from lavis.models import load_model_and_preprocess # install lavis - pip install salesforce-lavis
from RapidBase.Utils.IO.tic_toc import start_cuda, finish_cuda, gtic, gtoc
# setup device to use





class Action(Enum):
    QandA = "question_and_answering"
    ImageDescriber = "image_describer"


def get_model_and_stuff(action: Action):
    name = "blip_caption"
    # model_type = 'base_coco'
    model_type = "large_coco"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" #ToDo
    if action.value == Action.ImageDescriber.value:
        gtic()
        from lavis.models import load_model_and_preprocess  # install lavis - pip install salesforce-lavis
        model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True,
                                                             device=device)
        gtoc("Loading model...")
        return model, vis_processors, _

    elif action.value == Action.QandA.value:
        gtic()
        from lavis.models import load_model_and_preprocess  # install lavis - pip install salesforce-lavis
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2",
                                                                          is_eval=True,
                                                                          device=device)
        gtoc("Loading model...")

        return model, vis_processors, txt_processors
    else:
        print("Failed to select model")
        exit(1)


def image_describer(raw_image, vis_processors, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" #ToDo
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    gtic()
    array_result = model.generate({"image": image})
    gtoc(pre_string="Answer time")
    # ['a large fountain spewing water into the air']
    return array_result[0]


def question_on_image(raw_image, question, vis_processors, txt_processors, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device= "cpu"#ToDo
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    gtic()
    array_answer = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
    gtoc(pre_string="Answer time")
    return array_answer[0]
    # ['singapore']


# if __name__ == '__main__':
#     pass

    ### Get moels and names
    # from lavis.models import model_zoo
    # print(model_zoo)
    #
    # image_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/photos/14220.jpg"
    #
    # # load sample image
    # ### For image Question and answering
    # model, vis_processors, txt_processors = get_model_and_stuff(Action.QandA)
    # raw_image = Image.open(image_path).convert("RGB")
    # question = "is it male or female"
    # print(question_on_image(raw_image=raw_image, question=question, model=model, vis_processors=vis_processors,
    #                         txt_processors=txt_processors))

    # ### For image DESCRIBER
    # model, vis_processors, _ = get_model_and_stuff(Action.ImageDescriber)
    # raw_image = Image.open(image_path).convert("RGB")
    # question = "How many cars there are ? "
    # print(image_describer(raw_image=raw_image, model=model, vis_processors=vis_processors))
