import os

import torch
from PIL import Image
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.multimodal.clip_score import CLIPScore as CLIP_TScore
from tqdm import tqdm
from trainer_utils import find_newest_checkpoint

from pipelines.pipeline_soda import SODAPipeline
from app import randomize_seed_fn
from metrics.clip_score import CLIPIScore as CLIP_IScore
from metrics.clip_score import CLIPTScore as CLIP_TScore
from metrics.dino_score import DINOScore as DINO_Score
from metrics.dreambench_prompts import *


class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, files):
        self.args = args
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        object_name, object_id, image_id, prompt = image_path.split("/")[-1].split(".")[0].split("+")
        image = Image.open(image_path).convert("RGB")
        real_image = Image.open(os.path.join(self.args.data_dir, object_name, object_id + ".jpg")).convert("RGB")

        return image, real_image, prompt


def image_collate_fn(batch):
    image = [x[0] for x in batch]
    real_image = [x[1] for x in batch]
    prompt = [x[2] for x in batch]
    return image, real_image, prompt


class DreamBench_Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        # Traverse all images in the dataset
        self.image_paths = []
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths) * 25

    def __getitem__(self, index):
        image_path = self.image_paths[index // 25]
        real_image = Image.open(image_path).convert("RGB")
        object_id = image_path.split("/")[-1].split(".")[0]
        object_name = image_path.split("/")[-2]
        if object_name in OBJECT:
            object_class = OBJECT[object_name]
            prompt = SODA_OBJECT_PROMPTS[index % 25]
        else:
            object_class = LIVE_OBJECT[object_name]
            prompt = SODA_LIVE_OBJECT_PROMPTS[index % 25]

        prompt = prompt.format(object_class)
        return object_name, object_id, real_image, prompt


def dreambench_collate_fn(batch):
    object_name = [x[0] for x in batch]
    object_id = [x[1] for x in batch]
    real_image = [x[2] for x in batch]
    prompt = [x[3] for x in batch]
    return object_name, object_id, real_image, prompt


def main():
    args = OmegaConf.create()
    args.resume_from_checkpoint = "/fsx/xichenpan/output/flow_siglip_luminanext_512_trainall_1e4_64_6p5m_inst_32"
    args.resume_from_checkpoint = find_newest_checkpoint(args.resume_from_checkpoint)
    args.data_dir = "/fsx/xichenpan/dreambench/dreambooth/dataset"
    args.batch_size = 4
    args.num_workers = 4
    args.num_inference_steps = 30
    args.guidance_scale = 4.5
    args.text_guidance_scale = 7.5
    args.num_images_per_prompt = 4
    args.seed = 0

    args.output_dir = (
        "/fsx/xichenpan/eval/dreambench/"
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

    accelerator = Accelerator()
    if accelerator.is_main_process and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dino_score = DINO_Score(model_name_or_path="dino_vits16")
    clip_i_score = CLIP_IScore(model_name_or_path="openai/clip-vit-base-patch32")
    clip_t_score = CLIP_TScore(model_name_or_path="openai/clip-vit-base-patch32")

    dino_score = accelerator.prepare_model(dino_score, evaluation_mode=True)
    clip_i_score = accelerator.prepare_model(clip_i_score, evaluation_mode=True)
    clip_t_score = accelerator.prepare_model(clip_t_score, evaluation_mode=True)

    # stat existing images in output_dir
    image_paths = list()
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    if len(image_paths) >= 3000:
        accelerator.print("Already generated enough images")
        dataset = Image_Dataset(args, image_paths)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=image_collate_fn,
        )
        dataloader = accelerator.prepare(dataloader)
        accelerator.print("Number of Images: ", len(dataset))

        for batch in tqdm(dataloader):
            images, real_images, prompts = batch
            dino_score.update(images, real_images)
            clip_i_score.update(images, real_images)
            clip_t_score.update(images, prompts)
        accelerator.print("Computing Scores...")
        accelerator.print("DINO Score: ", dino_score.compute())
        accelerator.print("CLIP Image Score: ", clip_i_score.compute())
        accelerator.print("CLIP Text Score: ", clip_t_score.compute())
        return
    else:
        # clear all existing images
        if accelerator.is_main_process:
            for root, dirs, files in os.walk(args.output_dir):
                for file in files:
                    if file.endswith(".png"):
                        os.remove(os.path.join(root, file))

    model = SODAPipeline.from_pretrained(args.resume_from_checkpoint, ignore_mismatched_sizes=True)
    model = model.to(device=accelerator.device, dtype=torch.bfloat16)

    dataset = DreamBench_Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=dreambench_collate_fn,
    )
    accelerator.print("Number of Images: ", len(dataset))

    model, dataloader = accelerator.prepare(model, dataloader)

    kwargs = {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "text_guidance_scale": args.text_guidance_scale,
        "num_images_per_prompt": args.num_images_per_prompt,
    }

    for batch_id, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        object_name, object_id, real_image, prompt = batch

        # generate images
        randomize_seed_fn(args.seed, False)
        images = model(real_image, prompt, **kwargs).images

        # save image
        for image_id, image in enumerate(images):
            batch_idx = image_id // args.num_images_per_prompt
            pos = (
                batch_id * accelerator.num_processes * args.batch_size * args.num_images_per_prompt
                + image_id * accelerator.num_processes
                + accelerator.process_index
            )
            name = "+".join([object_name[batch_idx], object_id[batch_idx], str(pos), prompt[batch_idx]])
            images[image_id].save(os.path.join(args.output_dir, "{}.png".format(name)))

        real_image = [img for img in real_image for _ in range(args.num_images_per_prompt)]
        dino_score.update(images, real_image)
        clip_i_score.update(images, real_image)
        clip_t_score.update(images, [p for p in prompt for _ in range(args.num_images_per_prompt)])

        accelerator.print("Number of Samples: ", (dino_score.n_samples * accelerator.num_processes).item())
        accelerator.print("DINO Score: ", (dino_score.compute()).item())
        accelerator.print("CLIP Image Score: ", (clip_i_score.compute()).item())
        accelerator.print("CLIP Text Score: ", (clip_t_score.compute()).item())

    accelerator.print("Number of Samples: ", (dino_score.n_samples * accelerator.num_processes).item())
    accelerator.print("DINO Score: ", (dino_score.compute()).item())
    accelerator.print("CLIP Image Score: ", (clip_i_score.compute()).item())
    accelerator.print("CLIP Text Score: ", (clip_t_score.compute()).item())


if __name__ == "__main__":
    main()
