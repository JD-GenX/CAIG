prompt_nums=1
base_data_path="tiny_dataset/tiny.json"
OUTPUT_PATH="./output_demo" 

controlnet_model_path="lllyasviel/control_v11p_sd15_canny"
base_model_path="digiplay/majicMIX_realistic_v7"
Prompt_Model_Path="/root/datasets/jd/model/llava-v1.6-vicuna-7b-pretrain"
Prompt_Model_Output_Path="$OUTPUT_PATH/output_prompt.json"

accelerate launch inference_llava.py --model-path $Prompt_Model_Path --output_data_path $Prompt_Model_Output_Path --generate_nums $prompt_nums --base_data_path $base_data_path --temperature 1.0 

dir="$OUTPUT_PATH/epoch-new"
accelerate launch sample_llava.py --batch_size 4 --base_model_path $base_model_path  --save_path $dir --controlnet_model_path $controlnet_model_path  --num_inference_steps 30 --sampler_name 'Euler a' --data_path $Prompt_Model_Output_Path
