import os
import shutil

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import v2
from tqdm import tqdm


def main():
    args = OmegaConf.create()
    args.image_folder = "/fsx/xichenpan/eval/coco/lumina_30_3.0"
    args.batch_size = 125
    args.num_workers = int(os.getenv("OMP_NUM_THREADS", 12))

    accelerator = Accelerator(mixed_precision="bf16")
    fid = FrechetInceptionDistance(normalize=True)
    fid = accelerator.prepare_model(fid, evaluation_mode=True)

    gt_images = load_dataset(
        "sayakpaul/coco-30-val-2014",
        split="train",
        cache_dir="/fsx/xichenpan/.cache",
        trust_remote_code=True,
        num_proc=16,
    )
    gt_images = gt_images.remove_columns("caption")

    preprocess_image = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(256),
            v2.ToTensor(),
        ]
    )

    def preprocess_fn(batch):
        batch["image"] = [preprocess_image(image.convert("RGB")) for image in batch["image"]]
        return batch

    def collate_fn(batch):
        return torch.stack([item["image"] for item in batch], dim=0)

    gt_images.set_transform(preprocess_fn)
    accelerator.print("Number of real images: ", len(gt_images))
    gt_images_loader = torch.utils.data.DataLoader(
        gt_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    gt_images_loader = accelerator.prepare(gt_images_loader)

    for batch in tqdm(gt_images_loader):
        fid.update(batch, real=True)

    generated_images = load_dataset("imagefolder", data_dir=args.image_folder, split="validation", num_proc=16)
    generated_images.set_transform(preprocess_fn)
    accelerator.print("Number of fake images: ", len(generated_images))
    generated_images = torch.utils.data.DataLoader(
        generated_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    generated_images = accelerator.prepare(generated_images)

    for batch in tqdm(generated_images):
        fid.update(batch, real=False)
    accelerator.print("FID: ", fid.compute())


if __name__ == "__main__":
    main()
