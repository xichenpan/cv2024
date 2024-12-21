import os
from argparse import ArgumentParser

import transformers
from datasets import load_dataset
from omegaconf import OmegaConf

def main():
    args = OmegaConf.create()
    args.checkpoint_path = "/fsx/xichenpan/output/siglip_nextdit_int21oi_384_256_8/checkpoint-190000"
    args.data_dir = '/fsx/xichenpan/.cache'
    args.batch_size = 125
    args.num_workers = int(os.getenv("OMP_NUM_THREADS", 12))
    args.num_inference_steps = 30
    args.guidance_scale = 3.0
    args.text_guidance_scale = 3.0
    args.num_images_per_prompt = 1
    args.seed = 0
    args.override = True
    args.output_dir = (
        "/fsx/xichenpan/eval/coco/"
        + args.checkpoint_path.split("/")[-2]
        + "_"
        + args.checkpoint_path.split("/")[-1].split("-")[-1]
        + "_"
        + str(args.num_inference_steps)
        + "_"
        + str(args.guidance_scale)
        + "_"
        + str(args.text_guidance_scale)
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_name", type=str, default="/fsx/xichenpan/coco/val_shard_0.json")
    parser.add_argument("--output_dir", type=str, default="/fsx/xichenpan/eval/coco")
    args = parser.parse_args()
    main(args)
