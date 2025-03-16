import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
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


instruction_pool = [
    "Comparing the left part and right part of this image, which part is more suitable for the product '{}' ?",
    "The left and right part of this image is one advertising image for the product '{}', respectively, which is preferred by the user?",
    "Which part will bring more click-through rate in this image for product '{}' ?",
    "Between the left and right sections of this image, which one is more appropriate for showcasing the product '{}'?",
    "Considering the left and right halves of this image, which side better represents the product '{}'?",
    "Which side of this image, left or right, is more effective for advertising the product '{}'?",
    "For the product '{}', which part of the image, left or right, is more appealing?",
    "When looking at the left and right portions of this image, which part is more suitable for promoting the product '{}'?",
    "In this image, which side, left or right, is preferred by users for the product '{}'?",
    "Which half of this image, left or right, is more likely to attract user preference for the product '{}'?",
    "Which section of this image, left or right, is expected to generate a higher click-through rate for the product '{}'?",
    "For the product '{}', which side of the image do users find more engaging, left or right?",
    "Between the left and right sides of this image, which one is anticipated to drive more clicks for the product '{}'?",
]


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
    qs = caption.replace("<image>\n", "")
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
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt().replace(
        "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ",
        "",
    )
    return prompt


def generate_prompt(args):
    disable_torch_init()
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=540000))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    if args.model_path == "None":
        args.model_path = args.model_base
        args.model_base = None
        print("change args.model_path to args.model_base")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=accelerator.process_index
    )
    mm_use_im_start_end = model.config.mm_use_im_start_end

    f = open(args.base_data_path, "r")
    content = f.read()
    a = json.loads(content)
    print("total num: ", len(a))
    if args.batch_samples:
        a = random.sample(a, args.batch_samples)
        print("random select samples: ", args.batch_samples)

    accelerator.wait_for_everyone()
    start = time.time()
    with accelerator.split_between_processes(a) as prompts:
        new_list = []
        for num, item in enumerate(prompts):
            new_sample = item

            prompt = process_prompt(item["question"], args, mm_use_im_start_end, model_name)
            print(prompt, end="")
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            new_sample["question"] = prompt
            images = [Image.open("/root/datasets/jd/"+item["image"]).convert("RGB")]

            image_sizes = [x.size for x in images]
            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.bfloat16)
            if args.generate_nums > 1:
                outputs = []
            for i in range(args.generate_nums):
                with torch.inference_mode():
                    model_output = model.forward(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        return_dict=True,
                    )
                if model_output.logits[0, 0] >= model_output.logits[0, 1]:
                    output = "left"
                else:
                    output = "right"
                if args.generate_nums > 1:
                    outputs.append(output)
                else:
                    outputs = output

            print("Label: ", new_sample["target"], "Predict: ", outputs)
            if outputs == new_sample["target"]:
                new_list.append(1)
            else:
                new_list.append(0)

            new_sample["answer"] = outputs

    results_gathered = gather_object(new_list)
    if accelerator.is_main_process:
        timediff = time.time() - start
        Acc = sum(results_gathered) / len(results_gathered)
        print(args.model_path, args.base_data_path)
        print("Acc: ", Acc)
        print(f"time elapsed: {timediff}")


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
    parser.add_argument("--base_data_path", type=str)
    parser.add_argument("--output_data_path", type=str)
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        args.temperature = 0
        args.model_path = "/root/datasets/jd/model/llava-v1.6-vicuna-7b-ali-reward-model-caig"
        args.base_data_path = "/root/datasets/jd/ali_reward_testset_1000_0.05_0.20.json"
    generate_prompt(args)
