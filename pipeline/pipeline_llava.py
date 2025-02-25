# from diffusers import (
#     StableDiffusionControlNetPipeline,
#     ControlNetModel,
#     UNet2DConditionModel,
# )
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import PIL
import torch
import os
import dataclasses
from tqdm.auto import tqdm
from utils.utils import choose_scheduler, resize_and_canny, CannyDetector, concat_image
from utils.clip_text_custom_embedder import text_embeddings

import random
from datetime import datetime


from accelerate import Accelerator
from accelerate.logging import get_logger

import time
import json
import textwrap
from tqdm import tqdm

logger = get_logger(__name__, log_level="INFO")


@dataclasses.dataclass(frozen=False)
class TrainPolicyFuncData:
    tot_p_loss: float = 0
    tot_ratio: float = 0
    tot_kl: float = 0
    tot_grad_norm: float = 0


class T2I_CN:
    def __init__(self, args):
        # self.pipe=self.prepare_model(args)
        self.prompt = args.prompt
        self.negative_prompt = args.negative_prompt
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.width = args.width
        self.height = args.height
        self.controlnet_conditioning_scale = args.controlnet_conditioning_scale
        self.clip_skip = args.clip_skip  # -1
        self.batch_size = args.batch_size
        self.eta = args.eta
        self.save_path = args.save_path
        self.image_scale = args.image_scale
        self.preprocessor = CannyDetector(args.low_threshold, args.high_threshold)
        self.keep_loc = args.keep_loc
        self.save_concat = args.save_concat
        self.args = args
        self.concat_path = os.path.realpath(self.save_path) + "-concat"
        if self.save_concat:
            os.makedirs(self.concat_path, exist_ok=True)

    def trans_pool(self):

        self.img_list = []
        self.prompt_list = []

        f = open(self.args.data_path, "r")
        content = f.read()
        a = json.loads(content)
        for idx, item in enumerate(a):

            if isinstance(item["answer"], list):
                current_prompt = item["answer"]
                current_prompt = [per_prompt.strip().replace('"', "").replace("\n", "").replace(".", "") for per_prompt in current_prompt]
                self.prompt_list.extend(current_prompt)
                self.per_sku = len(current_prompt)
                current_image = item["image"]
                repeated_image = [current_image for i in range(self.per_sku)]
                self.img_list.extend(repeated_image)
            else:
                current_prompt = item["answer"].strip().replace('"', "").replace("\n", "").replace(".", "")
                current_image = item["image"]
                self.img_list.append(current_image)
                self.prompt_list.append(current_prompt)
                self.per_sku = 1

        print(
            "image_list, prompt",
            len(a),
            self.per_sku,
            len(self.img_list),
            len(self.prompt_list),
        )

    def prepare_model(self, args, device_map=None):
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path, use_safetensors=True)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(args.base_model_path, controlnet=controlnet)
        if args.lora_model_path:
            pipe.load_lora_weights(args.lora_model_path)
            pipe.fuse_lora(lora_scale=args.lora_scale)
            print("load lora finished!!")
        pipe.scheduler = choose_scheduler(args.sampler_name).from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device_map)
        return pipe

    def concat_prompt(
        self,
        image,
        prompt,
        font_size=50,
        text_color=(0, 0, 0),
        bg_color=(255, 255, 255),
    ):
        image_width, image_height = image.size

        # Create a new image with space for the text (using an estimated height for the default font)
        text_height = 10 + font_size  # Estimate the height based on font size
        if image_height == 1024:
            extended_height = image_height + 360  # Add some padding below the text
            font_size = 50
        else:
            extended_height = image_height + 180
            font_size = 25
        new_image = Image.new("RGB", (image_width, extended_height), color=bg_color)
        new_image.paste(image, (0, 0))

        # Create a drawing context for the new image
        draw = ImageDraw.Draw(new_image)

        # Use the default font provided by PIL
        font = ImageFont.truetype(
            "pipeline/Times_New_Roman.ttf",
            font_size,
        )

        lines = textwrap.wrap(prompt, width=36)  # Initial wrap, adjust 'width' as needed for your font and image size
        wrapped_text = "\n".join(lines)
        left, top, right, bottom = draw.multiline_textbbox((0, 0), wrapped_text, font)
        text_width, text_height = right - left, bottom - top
        text_position = (
            (image_width - text_width) // 2,
            image_height + 10,
        )  # Center the text vertically with some padding

        # Add the text
        draw.multiline_text(text_position, wrapped_text, fill=text_color, font=font, align="center")
        return new_image

    def post_process(
        self,
        init_image,
        mask_image,
        repainted_image,
        img_save_path,
        img_path,
        current_prompt,
        cri_attn=None,
        new_image=None,
    ):

        concat_prompt_path = os.path.join(img_save_path, "concat_prompt")
        os.makedirs(concat_prompt_path, exist_ok=True)

        # print("cat_path:", self.concat_path)
        cat_images = []
        rewards = []
        for idx in range(len(repainted_image)):
            current_time = datetime.now()
            format_time = current_time.strftime("%Y%m%d%H%M%S.%f")[:-4]
            trans_image_path = img_path[idx]
            name, ext = os.path.splitext(os.path.basename(trans_image_path))
            mask_image_arr = np.array(mask_image[idx].convert("L"))
            mask_image_arr = mask_image_arr[:, :, None]
            mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
            unmasked_unchanged_image_arr = (1 - mask_image_arr) * init_image[idx] + mask_image_arr * repainted_image[idx]
            unmasked_unchanged_image = PIL.Image.fromarray(unmasked_unchanged_image_arr.round().astype("uint8"))
            os.makedirs(img_save_path, exist_ok=True)
            output_path = os.path.join(img_save_path, "{}_{}{}".format(name, format_time, ext))
            output_concat_prompt_path = os.path.join(concat_prompt_path, "{}_{}{}".format(name, format_time, ext))

            if self.args.concat_prompt:
                concat_prompt_image = self.concat_prompt(unmasked_unchanged_image, current_prompt[idx])
                concat_prompt_image.save(output_concat_prompt_path)

            unmasked_unchanged_image.save(output_path)
            if self.save_concat:
                cat_image = concat_image(
                    trans_image_path,
                    output_path,
                    self.concat_path,
                    cri_attn=cri_attn,
                    new_img=new_image,
                )
                cat_images.append(cat_image / 1.0)
            self.log_output(
                img_path[idx],
                output_path,
                os.path.join(self.concat_path, os.path.basename(output_path)),
                current_prompt[idx],
            )
        return rewards

    def batch_list(self, samples, batch_size):
        """Batch the given list into sublists of specified max size."""
        return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]

    def inference_new(self, args):
        accelerator = Accelerator()

        device_map = f"cuda:{accelerator.process_index}"  # This creates "cuda:0", "cuda:1", etc.

        self.pipe = self.prepare_model(args, device_map=device_map)
        print("start inference:")
        self.trans_pool()

        accelerator.wait_for_everyone()
        with torch.no_grad():
            with accelerator.split_between_processes(list(zip(self.img_list, self.prompt_list))) as img_prompt:
                img_prompt_batches = self.batch_list(img_prompt, self.batch_size)
                total_batches = len(img_prompt_batches)
                pbar = tqdm(total=total_batches, desc="Processing batches", disable=not accelerator.is_local_main_process)
                for batch in img_prompt_batches:
                    batch_image, batch_prompt = zip(*batch)
                    batch_image = list(batch_image)
                    batch_prompt = list(batch_prompt)
                    init_image, mask_image, edge_image, post_masks = resize_and_canny(batch_image, self.preprocessor, args.image_scale, self.width, self.height, self.keep_loc, matting=False)

                    generators = [torch.Generator().manual_seed(42) for _ in range(len(edge_image))]
                    repainted_image = self.pipe(
                        prompt=batch_prompt,
                        negative_prompt=[args.negative_prompt] * len(edge_image),  # "irregular shape, extended shape, floating, table legs, pedestal, indistinct background," "irregular shape, extended shape, floating, table legs, pedestal, improper position, improper size"
                        clip_skip=self.clip_skip,
                        generator=generators,
                        image=init_image,
                        mask_image=mask_image,
                        control_image=edge_image,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        width=self.width,
                        height=self.height,
                        eta=self.eta,
                        controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                    ).images

                    self.post_process(init_image, post_masks, repainted_image, self.save_path, batch_image, batch_prompt)

                    pbar.update(1)
                pbar.close()

    def log_output(self, ori_path, gene_path, cat_path, prompt):
        with open(os.path.join(self.save_path, "log.txt"), "a") as f:
            f.write(ori_path + "\t" + gene_path + "\t" + cat_path + "\t" + prompt)
            f.write("\n")
            f.close()
