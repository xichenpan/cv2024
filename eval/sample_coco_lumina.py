import os
from argparse import ArgumentParser

from omegaconf import OmegaConf
from datasets import load_from_disk
from diffusers import LuminaText2ImgPipeline
import torch
from tqdm import tqdm
import shutil


def main(args):
    # Convert args to OmegaConf and add additional parameters
    args = OmegaConf.create(vars(args))
    args.data_dir = "/fsx/xichenpan/.cache"
    args.batch_size = 1
    args.num_workers = int(os.getenv("OMP_NUM_THREADS", 12))
    args.num_inference_steps = 30
    args.guidance_scale = 3.0
    args.num_images_per_prompt = 1
    args.seed = 0
    args.output_dir = (
        args.output_dir + "/lumina" + "_" + str(args.num_inference_steps) + "_" + str(args.guidance_scale) + f"/{args.file_name.split('/')[-1]}"
    )
    print(args.output_dir)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_from_disk(args.file_name)

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i : i + args.batch_size]
        images = pipeline(
            batch["caption"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            clean_caption=False,
        ).images

        for j, image in enumerate(images):
            image = image.resize((256, 256))
            image.save(f"{args.output_dir}/{i+j}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_name", type=str, default="/fsx/xichenpan/coco/val_shard_0")
    parser.add_argument("--output_dir", type=str, default="/fsx/xichenpan/eval/coco")
    args = parser.parse_args()
    pipeline = LuminaText2ImgPipeline.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", torch_dtype=torch.bfloat16).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    main(args)
