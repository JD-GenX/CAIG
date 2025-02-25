import argparse
from pipeline.pipeline_llava import T2I_CN


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, choices=["T2I", "T2I_CN"], default="T2I_CN")
    parser.add_argument("--use_webui_prompt", action="store_true")
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--controlnet_model_path", type=str)
    parser.add_argument("--lora_model_path", type=str, default=None)
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--clip_skip", default=None)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--image_scale", type=float, default=0.78)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--low_threshold", type=int, default=100)
    parser.add_argument("--high_threshold", type=int, default=200)
    parser.add_argument("--keep_loc", action="store_true")
    parser.add_argument("--save_concat", default=False, action="store_true")
    parser.add_argument("--concat_prompt", default=False, action="store_true")
    parser.add_argument("--sampler_name", type=str, choices=["Euler a", "DPM++ SDE Karras", "DDIM", "DDIMext"], default="DDIMext")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    if args.debug:
        args.base_model_path = "digiplay/majicMIX_realistic_v7"
        args.controlnet_model_path = "lllyasviel/control_v11p_sd15_canny"
        args.save_path = "/root/CAIG/output_demo/case"
        args.batch_size = 2  # 50
        args.num_inference_steps = 30
        args.sampler_name = "Euler a"
        args.data_path = "/root/CAIG/output_demo/output_prompt.json"

    print(args)
    pipe = T2I_CN(args)
    pipe.inference_new(args)
