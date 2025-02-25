from diffusers import DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler
from .scheduling_ddim_extended import DDIMSchedulerExtended
from safetensors.torch import load_file
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import time
import cv2
import os


def resize_and_canny(trans_path, preprocessor, scale, width, height, keep_loc, matting=False):
     init_images, mask_images, edge_images, post_masks=list(), list(), list(), list()
     for img in trans_path:
        file_name, ext = os.path.splitext(img)
        if ext not in ['.png', '.jpg']:
            continue
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if keep_loc:
            image = resize_image(image, width)
        else:
            image = golden_ratio_center_zoom(image, (width, height), scale, matting)
        mask = image[:, :, -1]
        matrix = 255 - np.asarray(mask).astype(np.uint8) 
        post_masks.append(Image.fromarray(matrix))
        kernel = np.ones((20, 20), np.uint8)
        matrix = cv2.dilate(matrix, kernel, 1)
        mask_images.append(Image.fromarray(matrix))  
        
        white = image[:, :, :3].copy()
        white[mask / 255.0 < 0.5] = 0
        edge_image = preprocessor(np.asarray(white).astype(np.uint8))
        edge_image = edge_image[:, :, None]
        edge_image = np.concatenate([edge_image, edge_image, edge_image], axis=2)
        edge_image = Image.fromarray(edge_image)
        edge_images.append(edge_image) 
        rgb_image = image[:, :, :3].copy()
        init_images.append(Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)))
     return init_images, mask_images, edge_images, post_masks


class CannyDetector(object):
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, input_image):
        return cv2.Canny(input_image, self.low_threshold, self.high_threshold)


def choose_scheduler(scheduler_name):
    scheduler_dict = {
        "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "Euler": EulerDiscreteScheduler,
        "DDIM": DDIMScheduler,
        "DDIMext": DDIMSchedulerExtended,
    }
    if scheduler_name in scheduler_dict:
        print("=" * 19)
        print(scheduler_dict[scheduler_name])
        return scheduler_dict[scheduler_name]
    return DPMSolverSinglestepScheduler


def get_timestramp():
    now = time.time()
    return time.strftime("%Y%m%d%H%M%S", time.localtime(now)) + str(int((now - int(now)) * 1000000))


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    if input_image.shape[-1] == 3:
        input_image = white_img_add_mask(input_image)
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def make_image_grid(images, rows: int, cols: int, resize: int = None):
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split(".", 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems["lora_up.weight"].to(dtype)
        weight_down = elems["lora_down.weight"].to(dtype)
        alpha = elems["alpha"]
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


def white_img_add_mask(img):
    mask = np.expand_dims(np.uint8(np.where(img == [255, 255, 255], 0, 1).all(axis=2)) * 255, 2)
    new_img = np.concatenate([img, mask], axis=2)
    return new_img


def golden_ratio_center_zoom(src_image, input_size, scale=0.7, matting=False):
    if src_image.shape[-1] == 3:
        src_image = white_img_add_mask(src_image)

    W, H = input_size

    mask = src_image[:, :, -1]
    binary_mask = np.uint8(mask / 255.0 >= 0.5)

    x_sum = np.squeeze(np.sum(binary_mask, axis=0))

    if len(np.squeeze(np.nonzero(x_sum))) == 0:
        x1, x2 = 0, len(x_sum)
    else:
        x1 = np.squeeze(np.nonzero(x_sum))[0]  # bug
        x2 = np.squeeze(np.nonzero(x_sum))[-1]

    y_sum = np.squeeze(np.sum(binary_mask, axis=1))

    if len(np.squeeze(np.nonzero(y_sum))) == 0:
        y1, y2 = 0, len(y_sum)
    else:
        y1 = np.squeeze(np.nonzero(y_sum))[0]
        y2 = np.squeeze(np.nonzero(y_sum))[-1]

    if x1 == x2 or y1 == y2:
        x1, x2 = 0, len(x_sum)
        y1, y2 = 0, len(y_sum)

    if matting == True:
        if x1 == 0 or x2 == src_image.shape[1] or y1 == 0 or src_image.shape[0] == y2:
            src_image = cv2.resize(src_image, (W, H), interpolation=cv2.INTER_AREA)
            return src_image

    img_crop = src_image[y1:y2, x1:x2, :]

    max_h = int(H * scale)
    max_w = int(W * scale)

    crop_h, crop_w = img_crop.shape[:2]
    if crop_h >= crop_w:
        new_h = np.minimum(max_h, int(max_w / crop_w * crop_h))
        new_w = int(new_h / crop_h * crop_w)
    else:
        new_w = np.minimum(max_w, int(max_h / crop_h * crop_w))
        new_h = int(new_w / crop_w * crop_h)
    new_crop = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # golden_center_y = max(H // 2, int(H * 0.618) - int(H / 512 * 30))
    golden_center_y = H // 2
    # beg_x, beg_y = np.random.randint(0, size - new_w), np.random.randint(0, size - new_h)
    beg_x = np.maximum(0, (W - new_w) // 2)
    beg_y = int(golden_center_y - new_h / 2)
    out_img = np.zeros((H, W, 4))
    out_img[beg_y : beg_y + new_h, beg_x : beg_x + new_w, :] = new_crop
    return np.uint8(out_img)


def export_video_file(frames, video_path, fps=8, video_size=(512, 512)):
    f = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(video_path, f, fps, video_size)
    for frame in frames:
        videoWriter.write(frame)
    videoWriter.release()


def concat_image(trans_path, image_path, save_folder, cri_attn=None, new_img=None):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # print(trans_path)
    trans_image = cv2.imread(trans_path, cv2.IMREAD_UNCHANGED)
    if trans_image is not None:
        if trans_image.shape[-1] == 4:
            rgb = trans_image[:, :, :-1]
        else:
            rgb = trans_image
        mask = trans_image[:, :, -1]
        white = rgb.copy()
        white[mask / 255.0 < 0.5] = 255

        white = cv2.resize(white, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        save_path = os.path.join(save_folder, os.path.basename(image_path))
        if cri_attn is not None:
            attn = cv2.resize(cri_attn, (512, 512), interpolation=cv2.INTER_AREA)
            attn = cv2.cvtColor(attn, cv2.COLOR_GRAY2BGR)
            concat_image = np.concatenate([white, image], axis=1)
            concat_image1 = np.concatenate([concat_image, attn], axis=1)
            cv2.imwrite(save_path, concat_image1)
        else:
            concat_image = np.concatenate([white, image], axis=1)
            if new_img is not None:
                new_image_arr = cv2.imread(new_img, cv2.IMREAD_UNCHANGED)
                new_image_arr = cv2.resize(new_image_arr, (512, 512), interpolation=cv2.INTER_AREA)
                concat_image = np.concatenate([concat_image, new_image_arr], axis=1)
            cv2.imwrite(save_path, concat_image)


    return concat_image
