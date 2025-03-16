MODEL_PATH=/root/datasets/jd/model/llava-v1.6-vicuna-7b-ali-reward-model-caig
BASE_DATA_PATH=/root/datasets/jd/ali_reward_testset_1000_0.05_0.20.json
accelerate launch  eval_reward_model.py --temperature 0 --model_path $MODEL_PATH --base_data_path $BASE_DATA_PATH

