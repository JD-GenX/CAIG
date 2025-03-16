NUMBER_OF_EPOCHS=5
PROMPT_VERSION=v1.6
MODEL_VERSION="v1.6-vicuna-7b"
batch_samples=8
learning_rate=2e-5
per_device_train_batch_size=1
gradient_accumulation_steps=1
pcpo_weight=1.0
pcpo_beta=0.5
OUTPUT_NAME=pcpo_llava_batch${batch_samples}_lr${learning_rate}_MaskImage0.75_RandomCaption${pcpo_weight}_beta${pcpo_beta}
pcpo_log_path="logs" # Customize your log path

base_data_path=tiny_dataset/tiny.json 
controlnet_model_path="lllyasviel/control_v11p_sd15_canny" # do not modify this
base_model_path="digiplay/majicMIX_realistic_v7" # do not modify this

vision_tower_path="/root/datasets/jd/model/clip-vit-large-patch14-336" # Set to your path !!
origin_prompt_llava_path="/root/datasets/jd/model/llava-v1.6-vicuna-7b-pretrain-caig" # Set to your path !!
reward_llava_path="/root/datasets/jd/model/llava-v1.6-vicuna-7b-ali-reward-model-caig" # Set to your path !!

prompt_llava_path=$pcpo_log_path/$OUTPUT_NAME/weights # do not modify this
ref_promt_llava_path=$prompt_llava_path/llava-v1.6-vicuna-7b-finetune-dpo-epoch0-merged # do not modify this
mkdir -p $prompt_llava_path
cp -r $origin_prompt_llava_path $ref_promt_llava_path


for (( i=0; i<NUMBER_OF_EPOCHS; i++ ))
do
    j=$((i + 1))
    echo "epoch:" $i
    echo "epoch $i, generate prompts----------" 

    LORA_PATH="$prompt_llava_path/llava-v1.6-vicuna-7b-finetune-dpo-epoch$i-merged" 
    WP_PATH="$pcpo_log_path/$OUTPUT_NAME/json_samples_translate_wp_ep$i.json"
    echo "load Lora_path: " $LORA_PATH
    echo "load Wp_path: " $WP_PATH


    start_time=$(date +%s)
    start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Prompt Generation Start Time: $start_time_formatted"

    accelerate launch inference_llava.py  --model-path $LORA_PATH --output_data_path $WP_PATH --generate_nums 2 --batch_samples $batch_samples --temperature 1.0 --base_data_path $base_data_path --epoch $i
    
    end_time=$(date +%s)
    end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Prompt Generation End Time: $end_time_formatted"

    dir="$pcpo_log_path/$OUTPUT_NAME/epoch_$i"
    if [ ! -d "$dir" ];then
        mkdir $dir
        echo "Create Dir" "$pcpo_log_path/$OUTPUT_NAME/epoch_$i"
    else
        echo "Dir Exists" "$pcpo_log_path/$OUTPUT_NAME/epoch_$i"
    fi
    
    start_time=$(date +%s)
    start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Image Generation Start Time: $start_time_formatted"

    rm -rf $pcpo_log_path/$OUTPUT_NAME/epoch_$i/log.txt
    accelerate launch sample_llava.py --batch_size 4 --base_model_path $base_model_path  --save_path $dir --controlnet_model_path $controlnet_model_path  --num_inference_steps 30 --sampler_name 'Euler a' --data_path $WP_PATH

    end_time=$(date +%s)
    end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Image Generation End Time: $end_time_formatted"


    start_time=$(date +%s)
    start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Reward scoring positive and negative sample generation start time: $start_time_formatted"

    accelerate launch pcpo_reward_gen.py --model_path  $reward_llava_path --input_data_path  $WP_PATH  --output_data_path $pcpo_log_path/$OUTPUT_NAME/epoch_$i.json  --log_file_path $pcpo_log_path/$OUTPUT_NAME/epoch_$i/log.txt
    
    end_time=$(date +%s)
    end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Reward scoring positive and negative sample generation end time: $end_time_formatted"


    echo "load LLava_path: " $prompt_llava_path/llava-v1.6-vicuna-7b-finetune-dpo-epoch$i-merged
    echo "load dpo_json_path: " "$JSON_DATA_PATH/epoch_$i.json"
    echo "output path: " $WP_PATH

    start_time=$(date +%s)
    start_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    echo "PCPO Start Time: $start_time_formatted"

    deepspeed train_pcpo.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LORA_PATH \
    --pcpo_weight $pcpo_weight \
    --ref_model_name_or_path $ref_promt_llava_path \
    --version $PROMPT_VERSION \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --data_path $pcpo_log_path/$OUTPUT_NAME/epoch_$i.json \
    --vision_tower $vision_tower_path\
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature "patch"\
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $prompt_llava_path/llava-$MODEL_VERSION-finetune-dpo-epoch$j-lora \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --gamma 0.1    \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --pcpo_beta $pcpo_beta

    end_time=$(date +%s)
    end_time_formatted=$(date +"%Y-%m-%d %H:%M:%S")
    execution_time=$((end_time - start_time))
    echo "PCPO End Time: $end_time_formatted"


    echo "epoch $i, merging lora----------" 
    python3 merge_lora_weights.py --model-path $prompt_llava_path/llava-$MODEL_VERSION-finetune-dpo-epoch$j-lora --model-base $LORA_PATH --save-model-path $prompt_llava_path/llava-v1.6-vicuna-7b-finetune-dpo-epoch$j-merged
done
