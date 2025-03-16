import argparse
import torch
from PIL import Image
import os
import numpy as np

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder_cls import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import time
import requests
from PIL import Image
from io import BytesIO
import re
import json
import random
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate import InitProcessGroupKwargs
import datetime
import sys


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def batch_list(samples, batch_size):
    """Batch the given list into sublists of specified max size."""
    return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]


def process_prompt(caption, args, mm_use_im_start_end, model_name):
    qs = caption  # +"Only answer me 'left' or 'right'."
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    # print(conv_mode)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt().replace(
        "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ",
        "",
    )
    return prompt


def parse_log_file(file_path):
    items = {}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                original_image = parts[0]
                item = {
                    "original_image": parts[0],
                    "generate_image": parts[1],
                    "concat_generate_image": parts[2],
                    "description": parts[3],
                }

                if original_image not in items:
                    items[original_image] = []

                # 检查是否已存在相同的描述
                if not any(existing_item["description"] == item["description"] for existing_item in items[original_image]):
                    items[original_image].append(item)

    return items


def get_data(input_data_path, log_file_path):
    nocap_prompt_pool = [
        "<image>\nComparing the left part and right part of this image, which part is more suitable for advertising?",
        "<image>\nThe left and right part of this image is one advertising image for a product, respectively, which is preferred by the user?",
        "<image>\nWhich part will bring more click-through rate in this image?",
        "<image>\nWhich side of this image, left or right, is more effective for advertising the product?",
        "<image>\nWhich part of the image, left or right, is more appealing?",
        "<image>\nIn this image, which side, left or right, is preferred by users?",
    ]

    with open(input_data_path) as f:
        data = json.load(f)
    log_dict = parse_log_file(log_file_path)

    new_list = []
    for idx in range(len(data)):
        temp_item = {}
        img_path = data[idx]["image"]

        if isinstance(data[idx]["answer"], list):
            image, answers = zip(*[(item["generate_image"], item["description"]) for item in log_dict[img_path]])

        temp_item["image_path"] = img_path
        temp_item["images"] = image
        temp_item["answers"] = answers
        caption = data[idx]["new_caption"].strip().replace('"', "").replace("\n", "")

        temp_item["caption"] = caption

        question = random.choice(nocap_prompt_pool).replace("<image>\n", "")
        temp_item["reward_question"] = question
        temp_item["prompt_question"] = data[idx]["question"]

        new_list.append(temp_item)
    return new_list


def get_substring_between_words(s, start_word, end_word):
    start_index = s.find(start_word)
    end_index = s.find(end_word, start_index + len(start_word))
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None
    return s[start_index + len(start_word) : end_index]


def process_data_dpo(sample):
    new_sample = {}
    img_path = sample["image"]
    img_name = img_path.split("/")[-1]
    sku, ext = os.path.splitext(img_name)
    new_sample["id"] = sku
    new_sample["image"] = img_path
    new_sample["chosen_conversations"] = [
        {
            "from": "human",
            "value": get_substring_between_words(sample["question"], "USER: ", " ASSISTANT:"),
        },
        {"from": "gpt", "value": sample["chose_answer"].strip().replace('"', "")},
    ]
    new_sample["reject_conversations"] = [
        {
            "from": "human",
            "value": get_substring_between_words(sample["question"], "USER: ", " ASSISTANT:"),
        },
        {"from": "gpt", "value": sample["refuse_answer"].strip().replace('"', "")},
    ]
    return new_sample


def generate_prompt(args):
    disable_torch_init()
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=540000))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    if args.model_path == "None":
        args.model_path = args.model_base
        args.model_base = None
        print("change args.model_path to args.model_base")
    model_name = get_model_name_from_path(args.model_path)
    task_list = get_data(args.input_data_path, args.log_file_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=accelerator.process_index
    )
    mm_use_im_start_end = model.config.mm_use_im_start_end

    print("total num: ", len(task_list))

    accelerator.wait_for_everyone()
    start = time.time()
    with accelerator.split_between_processes(task_list) as prompts:
        results = []
        for num, item in enumerate(prompts):
            if len(item["images"]) < 2 or len(item["answers"]) < 2:
                continue
            prompt = process_prompt(item["reward_question"], args, mm_use_im_start_end, model_name)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            item["reward_question"] = prompt

            image_1 = Image.open(item["images"][0]).convert("RGB")
            image_2 = Image.open(item["images"][1]).convert("RGB")

            image_1 = image_1.resize((512, 512))
            image_2 = image_2.resize((512, 512))

            image_1_np = np.array(image_1)
            image_2_np = np.array(image_2)

            images = [Image.fromarray(np.concatenate([image_1_np, image_2_np], axis=1))]

            image_sizes = [x.size for x in images]
            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                model_output = model.forward(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    return_dict=True,
                )
            if model_output.logits[0, 0] >= model_output.logits[0, 1]:
                result = {
                    "question": item["prompt_question"],
                    "chose_answer": item["answers"][0],
                    "refuse_answer": item["answers"][1],
                    "image": item["image_path"],
                    "caption": item["caption"],
                }
                new_result = process_data_dpo(result)
                results.append(new_result)
            else:
                result = {
                    "question": item["prompt_question"],
                    "chose_answer": item["answers"][1],
                    "refuse_answer": item["answers"][0],
                    "image": item["image_path"],
                    "caption": item["caption"],
                }
                new_result = process_data_dpo(result)
                results.append(new_result)

    results_gathered = gather_object(results)

    filter_results = []
    for item in results_gathered:
        if len(item.keys()) > 0:
            filter_results.append(item)
    formatted_nested_data = json.dumps(filter_results, indent=0, ensure_ascii=False)
    if accelerator.is_main_process:
        timediff = time.time() - start
        print("total_time: ", timediff)
        with open(args.output_data_path, "w") as file:
            file.write(formatted_nested_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--query", type=str)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_samples", type=int, default=None)
    parser.add_argument("--generate_nums", type=int, default=1)
    parser.add_argument("--output_data_path", type=str)

    parser.add_argument("--input_data_path", type=str)
    parser.add_argument("--log_file_path", type=str)
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        args.temperature = 0
        args.model_path = (
            "/home/chenxingye6/chenxingye/pretrained_models/llava-v1.6-vicuna-7b-finetune-ctr_contrast-multiwords_cor-middle-classmodel_best/"
        )
        args.input_data_path = "/home/chenxingye6/chenxingye/LLava-Fine-Tune/image_prompt_dataset/dpo_pretrained_llava_concat_mab_data_batch3200_lr2e-5_0,0.20_only_conditional_epoch_training/json_samples_translate_wp_ep6.json"
        args.output_data_path = "/home/chenxingye6/chenxingye/epoch_0.json"
        args.log_file_path = "/home/chenxingye6/chenxingye/LLava-Fine-Tune/image_prompt_dataset/dpo_pretrained_llava_concat_mab_data_batch3200_lr2e-5_0,0.20_only_conditional_epoch_training/epoch_6/log.txt"
    generate_prompt(args)
