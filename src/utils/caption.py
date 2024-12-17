import os
import random
import re

import transformers
from PIL import Image
import requests
import copy
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoTokenizer
import torch
from huggingface_hub import login

login("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")

model_id = "/workspace/Meta-Llama-3-8B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_example(model, processor, image_path, task_prompt="<DESCRIPTION>",
                text_input="Describe this image in great detail"):
    prompt = task_prompt + text_input
    image = Image.open(image_path)
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        repetition_penalty=1.10,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt,
                                                      image_size=(image.width, image.height))
    return parsed_answer["<DESCRIPTION>"]


def get_subject(tokenizer, model_llm, caption):
    messages = [

        {"role": "user",
         "content": caption + f"take one word from this sentence only which is the subject of sentence"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model_llm.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]

    # final_caption = pipeline_llama(caption + f" summerise it in 10 words with explaining about the logo, colour shape and the text written on it")

    return tokenizer.decode(response, skip_special_tokens=True)


def create_prompt(caption, subject, modifier_token):
    """ this function is used to create the caption for the image """

    replacement = modifier_token + " " + subject
    result = caption.replace(' ' + subject, replacement)

    print("#######################################################################################")
    return result


def choose_random_file(directory):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, file))]

    if not files:
        print("No files found in the directory.")
        return None

    # Select a random file
    random_file = random.choice(files)

    return random_file


def summerised_caption(directory,modifier_token,  wrds=50):
    """This function is used to summerise the caption"""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained("gokaygokay/Florence-2-Flux-Large", trust_remote_code=True).to(
        device).eval()
    processor = AutoProcessor.from_pretrained("gokaygokay/Florence-2-Flux-Large", trust_remote_code=True)

    img_path = choose_random_file(directory)
    caption = run_example(model=model, processor=processor, image_path=img_path)
    print(caption)
    subject = get_subject(tokenizer, model_llm, caption)
    print(subject)
    messages = [

        {"role": "user",
         "content": caption + f" summerise it in 50 words with keeping details about the logo, colour shape or design and the text on it"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model_llm.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]

    print("____ imhere__")
    # final_caption = pipeline_llama(caption + f" summerise it in 10 words with explaining about the logo, colour shape and the text written on it")
    print(tokenizer.decode(response, skip_special_tokens=True))
    create_prompt(tokenizer.decode(response, skip_special_tokens=True),subject, modefier_token)
    return caption


import time

c = time.time()
print(summerised_caption("/workspace/src/captioning/tiramisu_lateral-min.jpg"))
print(time.time() - c)
