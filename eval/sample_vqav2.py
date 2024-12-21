import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser

from omegaconf import OmegaConf
from datasets import load_from_disk
import torch
from tqdm import tqdm
import shutil
from pipelines.pipeline_soda import SODAPipeline
from trainer_utils import find_newest_checkpoint
from PIL import Image
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ]
)


def calculate_ssim(img1: Image.Image, img2: Image.Image):
    """Calculate SSIM between two PIL images."""
    # Apply transforms to resize and center crop both images
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    # Calculate SSIM
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(img1, img2).item()


def main(args):
    args = OmegaConf.create(vars(args))
    args.resume_from_checkpoint = "/fsx/xichenpan/output/flow_siglip_luminanexticl_512_trainall_1e4_64_vqav2_sft_8"
    args.resume_from_checkpoint = find_newest_checkpoint(args.resume_from_checkpoint)
    args.data_dir = "/fsx/xichenpan/.cache"
    args.batch_size = 10
    args.num_workers = int(os.getenv("OMP_NUM_THREADS", 12))
    args.num_inference_steps = 30
    args.guidance_scale = 1.0
    args.text_guidance_scale = 50.0
    args.num_images_per_prompt = 1
    args.seed = 0
    args.output_dir = (
        "/fsx/xichenpan/eval/vqav2/"
        + args.resume_from_checkpoint.split("/")[-2]
        + "_"
        + args.resume_from_checkpoint.split("/")[-1]
        + "_"
        + str(args.num_inference_steps)
        + "_"
        + str(args.guidance_scale)
        + "_"
        + str(args.text_guidance_scale)
    )
    print(args.output_dir)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = SODAPipeline.from_pretrained(args.resume_from_checkpoint, ignore_mismatched_sizes=True)
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)

    # Load the dataset using load_from_disk
    dataset = load_from_disk(args.file_name)
    total_images = 0
    true_images = 0

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i : i + args.batch_size]
        # Generate image from the question and answer pair
        source_images = sum(batch["source_images"], [])
        x_source = pipeline.encoder.process(source_images)
        images = pipeline(
            x_source,
            caption=batch["prompt"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            text_guidance_scale=args.text_guidance_scale,
            num_images_per_prompt=1,
            need_preprocess=False,
        ).images

        # calculate accurancy
        source_images_0 = [example[0] for example in batch["source_images"]]
        source_images_1 = [example[1] for example in batch["source_images"]]
        target_images = batch["target_image"]
        gt_idx = [0 if src_0 == target else 1 for src_0, src_1, target in zip(source_images_0, source_images_1, target_images)]
        pred_idx = [
            0 if calculate_ssim(pred, src_0) > calculate_ssim(pred, src_1) else 1
            for pred, src_0, src_1 in zip(images, source_images_0, source_images_1)
        ]
        true_images += sum(1 for gt, pred in zip(gt_idx, pred_idx) if gt == pred)
        total_images += len(gt_idx)

        print(f"Accuracy: {true_images / total_images}")

        for j, (image, target_image) in enumerate(zip(images, batch["target_image"])):
            # Resize both images to 512x512
            image = image.resize((512, 512))
            target_image = target_image.resize((512, 512))

            # Create a new image that's wide enough for both images
            combined = Image.new("RGB", (1024, 512))

            # Paste the generated image first
            combined.paste(image, (0, 0))

            # Paste the target image
            combined.paste(target_image, (512, 0))

            combined.save(f"{args.output_dir}/{i+j}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_name", type=str, default="/fsx/xichenpan/vqav2_converted/val/vqav2_shard_0")
    parser.add_argument("--output_dir", type=str, default="/fsx/xichenpan/eval/vqav2")
    args = parser.parse_args()
    main(args)
