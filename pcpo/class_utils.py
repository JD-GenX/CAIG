from dataclasses import dataclass, field
from typing import Optional
import transformers



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    ref_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="weights/clip-vit-large-patch14-336/")
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    aux_data_path: str = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    crop_scale: str = field(default="0.25,0.75", metadata={"help": "Scale range for RandomResizedCrop transformation. Format: 'min,max'"})
    drop_scale: float = field(default=0.5)



# Define and parse arguments.
@dataclass
class ScriptArguments(transformers.TrainingArguments):
    """
    The arguments for the DPO training script.
    """

    # llava parameters
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: Optional[bool] = field(default=False, metadata={"help": "whether using lora fine-tuning model."})
    lora_r: Optional[int] = field(default=64, metadata={"help": "lora rank."})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout."})
    lora_weight_path: Optional[str] = field(default=None, metadata={"help": "path to lora weight."})
    lora_bias: Optional[str] = field(default="none", metadata={"help": "lora bias."})
    mm_projector_lr: Optional[float] = field(default=None, metadata={"help": "mm_projector learning rate."})
    group_by_modality_length: Optional[bool] = field(default=False, metadata={"help": "group_by_modality_length."})

    # beta
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss"})
    gamma: Optional[float] = field(default=0.5, metadata={"help": "the gamma parameter for aux loss"})
    pcpo_weight: Optional[float] = field(default=1.0, metadata={"help": "pcpo_weight"})
    pcpo_beta: Optional[float] = field(default=0.5, metadata={"help": "pcpo_beta"})
    
    # training parameters
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "maximum value of gradient norm"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "whether to use gradient checkpointing"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=-1, metadata={"help": "the saving frequency"})
    evaluation_strategy: Optional[str] = field(default="no", metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[int] = field(default=-1, metadata={"help": "the evaluation frequency"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "path to deepspeed config"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "whether to use bf16 weight"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "whether to use fp16 weight"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "number of training epochs"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "strategy used to save model"})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "limit number of saved model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "number of training epochs"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warmup ratio"})
    tf32: Optional[bool] = field(default=True, metadata={"help": "whether to use tf32"})
    dataloader_num_workers: Optional[int] = field(default=4, metadata={"help": "number of dataloader workers"})
    fsdp: Optional[str] = field(default="", metadata={"help": "whether to use fsdp"})
    local_rank: int = field(default=-1, metadata={"help": "local rank"})
    seed: Optional[int] = field(default=42, metadata={"help": "seed"})
    # instrumentation
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={"help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,' '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. ' 'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'},
    )
    run_name: Optional[str] = field(default="ppo_llava-1.5", metadata={"help": "name of the run"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={"help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See" "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"},
    )