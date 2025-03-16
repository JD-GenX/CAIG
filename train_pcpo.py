import os
import json
import copy
import random
import logging
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import transformers
from transformers import TrainerCallback
from llava.model import *
from llava.constants import IGNORE_INDEX
from llava import conversation as conversation_lib
from llava.train.train import preprocess_multimodal, preprocess, safe_save_model_for_hf_trainer, TrainingArguments
import copy

import logging
from pcpo.llava_pcpo_trainer import LlavaPCPOTrainer

local_rank = None
from pcpo.class_utils import *


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class PatchMasker(torch.nn.Module):
    def __init__(self, patch_size=32, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def random_masking(self, x):
        N, C, H, W = x.shape
        patches_h, patches_w = H // self.patch_size, W // self.patch_size
        num_patches = patches_h * patches_w
        num_mask = int(num_patches * self.mask_ratio)

        patch_mask = torch.ones(N, patches_h, patches_w, device=x.device)
        mask_indices = torch.randperm(num_patches)[:num_mask]
        patch_mask.view(N, -1)[:, mask_indices] = 0

        mask = patch_mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        mask = F.adaptive_avg_pool2d(mask.unsqueeze(1), (H, W))
        mask = mask.expand(-1, C, -1, -1)

        x_masked = x * mask

        return x_masked, mask

    def forward(self, img):
        return self.random_masking(img)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        dpo_data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        aux_data_path: str = None,
    ):
        super(LazySupervisedDataset, self).__init__()

        dpo_data_dict = json.load(open(dpo_data_path, "r"))
        dpo_data_dict = self.dpo_process(dpo_data_dict)

        self.list_data_dict = dpo_data_dict
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.crop_scale = tuple(map(float, self.data_args.crop_scale.split(",")))
        self.min = 0.010  # 0.012
        self.max = 0.055  # 0.053
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])  # 将PIL Image转换为tensor
        self.to_pil = ToPILImage()
        self.patch_masker = PatchMasker()

    def __len__(self):
        return len(self.list_data_dict)

    def dpo_process(self, pope_data):
        pope_data_dict = []
        for idx in range(len(pope_data)):
            id = pope_data[idx]["id"]
            image = pope_data[idx]["image"]
            chosen = pope_data[idx]["chosen_conversations"]
            reject = pope_data[idx]["reject_conversations"]

            pope_data_dict.append(
                {
                    "id": id,
                    "image": image,
                    "chosen_conversations": chosen,
                    "reject_conversations": reject,
                }
            )
        return pope_data_dict

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        crop_or_drop = random.choice([True, False])
        # crop_or_drop = True
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if image_folder is None:
                image = Image.open(image_file).convert("RGB")
            else:
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")

            if crop_or_drop:
                image_tensor = self.to_tensor_transform(image)
                image_crop = self.to_pil(self.patch_masker(image_tensor.unsqueeze(0))[0].squeeze(0))
            else:
                random_element = random.choice(self.list_data_dict)

            if self.data_args.image_aspect_ratio == "pad":
                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            if crop_or_drop:
                if self.data_args.image_aspect_ratio == "pad":
                    image_crop = expand2square(image_crop, tuple(int(x * 255) for x in processor.image_mean))
                image_crop = processor.preprocess(image_crop, return_tensors="pt")["pixel_values"][0]
            else:
                image_crop = image

            chosen_sources = preprocess_multimodal(
                copy.deepcopy([e["chosen_conversations"] for e in sources]),
                self.data_args,
            )
            reject_sources = preprocess_multimodal(
                copy.deepcopy([e["reject_conversations"] for e in sources]),
                self.data_args,
            )
            if crop_or_drop:
                chosen_sources_noise = chosen_sources
            else:
                chosen_sources_noise = preprocess_multimodal(
                    copy.deepcopy([e["chosen_conversations"] for e in [random_element]]),
                    self.data_args,
                )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        chosen_data_dict = preprocess(
            chosen_sources,
            self.tokenizer,
            has_image=("image" in self.list_data_dict[i]),
        )
        reject_data_dict = preprocess(
            reject_sources,
            self.tokenizer,
            has_image=("image" in self.list_data_dict[i]),
        )

        if crop_or_drop:
            chosen_data_dict_noise = chosen_data_dict
        else:
            chosen_data_dict_noise = preprocess(
                chosen_sources_noise,
                self.tokenizer,
                has_image=("image" in random_element),
            )

        if isinstance(i, int):
            data_dict = dict(
                chosen_input_ids=chosen_data_dict["input_ids"][0],
                chosen_labels=chosen_data_dict["labels"][0],
                reject_input_ids=reject_data_dict["input_ids"][0],
                reject_labels=reject_data_dict["labels"][0],
                chosen_input_ids_noise=chosen_data_dict_noise["input_ids"][0],
                chosen_labels_noise=chosen_data_dict_noise["labels"][0],
            )

        if "image" in self.list_data_dict[i]:
            data_dict["images"] = image
            data_dict["images_crops"] = image_crop
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["images"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            data_dict["images_crops"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels, reject_input_ids, reject_labels, chosen_input_ids_noise, chosen_labels_noise = tuple(
            [instance[key] for instance in instances]
            for key in (
                "chosen_input_ids",
                "chosen_labels",
                "reject_input_ids",
                "reject_labels",
                "chosen_input_ids_noise",
                "chosen_labels_noise",
            )
        )
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels, batch_first=True, padding_value=IGNORE_INDEX)

        reject_input_ids = torch.nn.utils.rnn.pad_sequence(
            reject_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        reject_labels = torch.nn.utils.rnn.pad_sequence(reject_labels, batch_first=True, padding_value=IGNORE_INDEX)

        chosen_input_ids_noise = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids_noise,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        chosen_labels_noise = torch.nn.utils.rnn.pad_sequence(chosen_labels_noise, batch_first=True, padding_value=IGNORE_INDEX)

        chosen_input_ids = chosen_input_ids[:, : self.tokenizer.model_max_length]
        chosen_labels = chosen_labels[:, : self.tokenizer.model_max_length]
        reject_input_ids = reject_input_ids[:, : self.tokenizer.model_max_length]
        reject_labels = reject_labels[:, : self.tokenizer.model_max_length]
        chosen_input_ids_noise = chosen_input_ids_noise[:, : self.tokenizer.model_max_length]
        chosen_labels_noise = chosen_labels_noise[:, : self.tokenizer.model_max_length]

        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            chosen_input_ids_noise=chosen_input_ids_noise,
            chosen_labels_noise=chosen_labels_noise,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.tokenizer.pad_token_id),
            chosen_attention_mask_noise=chosen_input_ids_noise.ne(self.tokenizer.pad_token_id),
        )

        if "images" in instances[0]:
            images = [instance["images"] for instance in instances]
            images_crops = [instance["images_crops"] for instance in instances]

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
                batch["images_crops"] = torch.stack(images_crops)
            else:
                batch["images"] = images
                batch["images_crops"] = torch.stack(images_crops)
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        dpo_data_path=data_args.data_path,
        aux_data_path=data_args.aux_data_path,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


class SaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, "checkpoint-{}".format(state.global_step))
        if args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(kwargs["model"].named_parameters(), args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(kwargs["model"].named_parameters())
            if args.local_rank in [-1, 0]:
                kwargs["model"].config.save_pretrained(checkpoint_dir)
                kwargs["model"].save_pretrained(checkpoint_dir, state_dict=state_dict)
                torch.save(
                    non_lora_state_dict,
                    os.path.join(checkpoint_dir, "non_lora_trainables.bin"),
                )


def setup_llava_model(model_args, data_args, script_args):
    # local rank
    if "LOCAL_RANK" not in os.environ:
        local_rank = None
    else:
        local_rank = int(os.environ["LOCAL_RANK"])

    # device
    if "LOCAL_RANK" not in os.environ:
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = f"cuda:{local_rank}"

    compute_dtype = torch.float16 if script_args.fp16 else (torch.bfloat16 if script_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if script_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": device},
                load_in_4bit=script_args.bits == 4,
                load_in_8bit=script_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=script_args.bits == 4,
                    load_in_8bit=script_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=script_args.double_quant,
                    bnb_4bit_quant_type=script_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    if model_args.vision_tower is not None:
        if "mpt1" in model_args.model_name_or_path:
            pass
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=script_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if script_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if script_args.fp16 else (torch.bfloat16 if script_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

    if script_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if script_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=script_args.lora_dropout,
            bias=script_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if script_args.bits == 16:
            if script_args.bf16:
                model.to(torch.bfloat16)
            if script_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            model_max_length=script_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        pass
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=script_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if script_args.bf16 else torch.float16, device=device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = script_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = script_args.freeze_mm_mlp_adapter
        if script_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if script_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = script_args.mm_projector_lr
        script_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if script_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if script_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if script_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    return model, tokenizer


def main():

    parser = transformers.HfArgumentParser((ScriptArguments, ModelArguments, DataArguments))
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # setup llava model
    llava_policy_model, tokenizer = setup_llava_model(
        model_args=model_args,
        data_args=data_args,
        script_args=script_args,
    )

    script_args.lora_enable = False

    model_args.model_name_or_path = model_args.ref_model_name_or_path
    llava_ref_model, _ = setup_llava_model(
        model_args=model_args,
        data_args=data_args,
        script_args=script_args,
    )

    script_args.lora_enable = True

    for n, p in llava_ref_model.named_parameters():
        p.requires_grad = False

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=script_args.bf16,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        max_grad_norm=script_args.max_grad_norm,
        deepspeed=script_args.deepspeed,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        save_total_limit=script_args.save_total_limit,
        warmup_ratio=script_args.warmup_ratio,
        tf32=script_args.tf32,
        dataloader_num_workers=script_args.dataloader_num_workers,
        fp16=script_args.fp16,
        seed=script_args.seed,
        cache_dir=script_args.cache_dir,
        freeze_mm_mlp_adapter=script_args.freeze_mm_mlp_adapter,
        mpt_attn_impl=script_args.mpt_attn_impl,
        model_max_length=script_args.model_max_length,
        double_quant=script_args.double_quant,
        quant_type=script_args.quant_type,
        bits=script_args.bits,
        lora_enable=script_args.lora_enable,
        lora_r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        lora_weight_path=script_args.lora_weight_path,
        lora_bias=script_args.lora_bias,
        mm_projector_lr=script_args.mm_projector_lr,
        group_by_modality_length=script_args.group_by_modality_length,
    )

    dpo_trainer = LlavaPCPOTrainer(
        model=llava_policy_model,
        ref_model=llava_ref_model,
        pcpo_weight=script_args.pcpo_weight,
        args=training_args,
        beta=script_args.beta,
        gamma=script_args.gamma,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        **data_module,
    )

    dpo_trainer.add_callback(SaveCallback())

    dpo_trainer.train()
    dpo_trainer.save_state()

    llava_policy_model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(llava_policy_model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(llava_policy_model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            llava_policy_model.config.save_pretrained(training_args.output_dir)
            llava_policy_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=dpo_trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
