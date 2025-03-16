import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import time
from PIL import Image
import re
import os
import json
import random
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate import InitProcessGroupKwargs
import datetime
from tqdm import tqdm

instruction_pool = [
    "Design a concise Stable Diffusion prompt that takes the product caption '{}' and product image as inspiration to generate an appealing advertising image background for this product. (Directly give me the prompt without prefix explanation)",
    "Produce a short diffusion prompt considering the critical information in product caption '{}' and product image to create an advertisement background. (Directly give me the prompt without prefix explanation)",
    "Construct a brief background generation prompt to produce an advertising image based on product caption '{}' and product image.  (Directly give me the prompt without prefix explanation)",
    "Generate a short Stable Diffusion prompt that leverage the product caption '{}' and the visual elements of the product image to output an advertising background that underscores the product's attractiveness. (Directly give me the prompt without prefix explanation)",
    "Provide a succinct text to background diffusion model prompt suitable for this product according to caption '{}' and product image. (Directly give me the prompt without prefix explanation)",
    "Develop a compact prompt for Stable Diffusion to craft a background for an ad, using the product caption '{}' and the accompanying product image as creative influences. (Directly give me the prompt without prefix explanation)",
    "Draft a distilled text to image prompt to fabricate an ad background, infusing the essence of '{}' from the product caption and the picture to accentuate the product's features. (Directly give me the prompt without prefix explanation)",
    "Based on this product image and product caption '{}', formulate a brief diffusion prompt to synthesize a background tailored for advertising purposes. (Directly give me the prompt without prefix explanation)",
    "Take advantage of this product with its title '{}', build a succinct prompt for diffusion model aimed at generating an advertisement background. (Directly give me the prompt without prefix explanation)",
    "Make use of product title '{}' along with its image, assemble a terse stable diffusion model prompt to render an ad background that complements and highlights the product. (Directly give me the prompt without prefix explanation)",
]



def batch_list(samples, batch_size):
    """Batch the given list into sublists of specified max size."""
    return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]


def preprocess_epoch_dataset(base_data_path, batch_samples, epoch):
    f = open(base_data_path, "r")
    content = f.read()
    a = json.loads(content)
    total_samples = len(a)
    print("total num: ", total_samples)

    if batch_samples:
        start_index = (epoch * batch_samples) % total_samples

        selected_samples = []
        remaining = batch_samples
        while remaining > 0:
            samples_to_take = min(remaining, total_samples - start_index)
            selected_samples.extend(a[start_index : start_index + samples_to_take])
            remaining -= samples_to_take
            start_index = (start_index + samples_to_take) % total_samples
        a = selected_samples

        print(f"Selected {batch_samples} samples starting from index {(epoch * batch_samples) % total_samples}")
        print(f"Number of selected samples: {len(a)}")
    return a


def process_prompt(caption, args, mm_use_im_start_end, model_name):
    instruct = random.choice(instruction_pool)  # +" I will give you one example: {}".format(random.choice(example_pool))
    qs = instruct.format(caption.strip().replace('"', ""))
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
    prompt = conv.get_prompt()
    return prompt


def generate_prompt(args):
    disable_torch_init()
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=540000))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    train_dataset = preprocess_epoch_dataset(args.base_data_path, args.batch_samples, args.epoch)

    if args.model_path == "None":
        args.model_path = args.model_base
        args.model_base = None
        print("change args.model_path to args.model_base")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=accelerator.process_index
    )
    mm_use_im_start_end = model.config.mm_use_im_start_end

    accelerator.wait_for_everyone()
    start = time.time()
    with accelerator.split_between_processes(train_dataset) as prompts:
        new_list = []
        total = len(prompts)
        for num, item in tqdm(enumerate(prompts), total=total, desc="Generating prompts", disable=not accelerator.is_local_main_process):
            new_sample = item

            prompt = process_prompt(item["new_caption"], args, mm_use_im_start_end, model_name)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            new_sample["question"] = prompt
            images = [Image.open(item["image"]).convert("RGB")]
            image_sizes = [x.size for x in images]

            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

            if args.generate_nums > 1:
                outputs = []
            for i in range(args.generate_nums):
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=(True if args.temperature > 0 else False),  # if args.temperature > 0 else False,
                        temperature=args.temperature,  # args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if args.generate_nums > 1:
                    outputs.append(output)
                else:
                    outputs = output

            new_sample["answer"] = outputs
            new_list.append(new_sample)
            print("index: ", num, "prompt: ", new_sample)

    results_gathered = gather_object(new_list)
    formatted_nested_data = json.dumps(results_gathered, indent=0, ensure_ascii=False)
    if accelerator.is_main_process:
        timediff = time.time() - start
        output_path = args.output_data_path
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(args.output_data_path, "w", encoding="utf-8") as file:
            file.write(formatted_nested_data)
        print(f"time elapsed: {timediff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--vision_tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_samples", type=int, default=None)
    parser.add_argument("--generate_nums", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--base_data_path", type=str)
    parser.add_argument("--output_data_path", type=str)
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        args.model_path = "/root/datasets/jd/model/llava-v1.6-vicuna-7b-finetune-dpo-epoch0-merged"
        args.vision_tower = "/root/datasets/jd/model/clip-vit-large-patch14-336"
        args.output_data_path = "./output/output_prompt.json"
        args.generate_nums = 2
        args.epoch = 0
        args.batch_samples = None  # 4500
        args.temperature = 1.0
        args.base_data_path = "/root/datasets/jd/jd_pcpo_testset.json"
    generate_prompt(args)
