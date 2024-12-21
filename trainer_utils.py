import os
import random

import torch
import torchvision.transforms as v2
import torchvision.transforms.functional as F
import yaml
from requests.packages import target
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import Dataset


class AddGaussianNoise:
    def __init__(self, sigma=0.10):
        self.sigma = sigma

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        dtype = tensor.dtype

        tensor = tensor.float()
        out = tensor + self.sigma * torch.randn_like(tensor)

        if out.dtype != dtype:
            out = out.to(dtype)
        return out


class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, tensor):
        return self.processor(tensor, return_tensors="pt")["pixel_values"].squeeze(0)


def possible_override_args(override_args, model_args, data_args, training_args):
    if hasattr(override_args, "override") and not override_args.override:
        return model_args, data_args, training_args
    if hasattr(override_args, "config_file") and override_args.config_file is not None:
        yaml_file = os.path.join("configs", override_args.config_file)
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)

        for key, value in config.items():
            if hasattr(model_args, key):
                setattr(model_args, key, value)
            if hasattr(data_args, key):
                setattr(data_args, key, value)
            if hasattr(training_args, key):
                setattr(training_args, key, value)

    return model_args, data_args, training_args


def get_full_dirs(training_args):
    if not os.path.isabs(training_args.output_dir):
        training_args.output_dir = os.path.join(training_args.base_dir, training_args.output_dir)
    if not os.path.isabs(training_args.data_dir):
        training_args.data_dir = os.path.join(training_args.base_dir, training_args.data_dir)
    if not os.path.isabs(training_args.logging_dir):
        training_args.logging_dir = os.path.join(training_args.base_dir, training_args.logging_dir)
    return training_args


def find_newest_checkpoint(checkpoint_path):
    # see if checkpoint_path's child contains pt or safetensors or pth
    if os.path.isdir(checkpoint_path) and any(x.endswith(("pt", "safetensors", "pth")) for x in os.listdir(checkpoint_path)):
        return checkpoint_path

    else:
        return get_last_checkpoint(checkpoint_path)


def randomly_mask_patches(img, patch_size=16, mask_ratio=0.75):
    # Convert the input image to tensor format
    img_tensor = F.to_tensor(img)

    # Get the width and height of the image
    _, height, width = img_tensor.shape

    # Calculate how many patches can be divided horizontally and vertically
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    # Calculate the total number of patches
    total_patches = num_patches_x * num_patches_y

    # Calculate the number of patches to be masked
    num_masked_patches = int(total_patches * mask_ratio)

    # Generate the indices for all patches
    patch_indices = [(i, j) for i in range(num_patches_y) for j in range(num_patches_x)]

    # Randomly select the indices of patches to be masked
    masked_patch_indices = random.sample(patch_indices, num_masked_patches)

    # Iterate through all patches that need to be masked and perform the masking operation
    for i, j in masked_patch_indices:
        img_tensor[
            :,
            i * patch_size : (i + 1) * patch_size,
            j * patch_size : (j + 1) * patch_size,
        ] = 0  # Use 0 to represent masking

    # Convert the tensor back to a PIL image
    masked_img = F.to_pil_image(img_tensor)

    return masked_img


def get_random_transform(source_transform, target_transform):
    low_level_augmentations = [
        v2.Lambda(lambda img: F.equalize(img)),
        v2.Lambda(lambda img: F.invert(img)),
        v2.GaussianBlur(kernel_size=random.choice([49, 81, 121]), sigma=random.uniform(9.0, 21.0)),
        v2.ColorJitter(
            brightness=random.uniform(0.5, 2.0),
            contrast=random.uniform(0.5, 2.0),
            saturation=random.uniform(0.5, 2.0),
            hue=random.uniform(0, 0.5),
        ),
        v2.Lambda(lambda img: F.rotate(img, angle=random.uniform(-45, 45))),
        v2.Lambda(
            lambda img: F.affine(
                img,
                angle=0,
                translate=(random.randint(20, 50), random.randint(20, 50)),
                scale=1.0,
                shear=random.uniform(10, 30),
            )
        ),
        v2.Grayscale(num_output_channels=3),
        v2.Lambda(lambda img: F.hflip(img)),
        v2.Lambda(lambda img: F.vflip(img)),
        v2.Lambda(
            lambda img: F.perspective(
                img,
                startpoints=[
                    (0, 0),
                    (0, img.height),
                    (img.width, img.height),
                    (img.width, 0),
                ],
                endpoints=[
                    (random.randint(0, 20), random.randint(0, 20)),
                    (0, img.height),
                    (img.width - random.randint(20, 50), img.height),
                    (img.width, img.height - random.randint(20, 50)),
                ],
            )
        ),
        v2.Lambda(lambda img: randomly_mask_patches(img, patch_size=img.width // 16, mask_ratio=0.75)),
        v2.Lambda(lambda img: F.to_pil_image(F.to_tensor(img) + torch.randn((len(img.getbands()), img.height, img.width)) * random.uniform(0.2, 0.5))),
        # random.choice([
        #     v2.Lambda(lambda img: F.adjust_sharpness(img, sharpness_factor=random.uniform(0.1, 0.3))),
        #     v2.Lambda(lambda img: F.adjust_sharpness(img, sharpness_factor=random.uniform(5, 10)))
        # ]),
        random.choice(
            [
                v2.Lambda(lambda img: F.adjust_brightness(img, brightness_factor=random.uniform(0.1, 0.3))),
                v2.Lambda(lambda img: F.adjust_brightness(img, brightness_factor=random.uniform(3, 5))),
            ]
        ),
        random.choice(
            [
                v2.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=random.uniform(0.1, 0.3))),
                v2.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=random.uniform(3, 5))),
            ]
        ),
        random.choice(
            [
                v2.Lambda(lambda img: F.adjust_saturation(img, saturation_factor=random.uniform(0.1, 0.3))),
                v2.Lambda(lambda img: F.adjust_saturation(img, saturation_factor=random.uniform(3, 5))),
            ]
        ),
    ]

    chosen_augmentation = random.choice(low_level_augmentations)

    source_transform_list = source_transform.transforms
    target_transform_list = target_transform.transforms

    return (
        v2.Compose(source_transform_list[:2] + [chosen_augmentation] + source_transform_list[2:]),
        v2.Compose(target_transform_list[:2] + [chosen_augmentation] + target_transform_list[2:]),
    )


class ConcatenatedDataset(Dataset):
    def __init__(self, datasets, probs=None):
        self.datasets = list(datasets.values())
        self.dataset_names = list(datasets.keys())
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.length = sum(self.dataset_lengths)

        if probs is None:
            # Default to uniform probabilities
            self.probs = [1.0 / len(self.datasets)] * len(self.datasets)
        else:
            # Get probabilities in same order as datasets
            self.probs = [probs[name] for name in self.dataset_names]
            # Normalize probabilities to sum to 1
            total = sum(self.probs)
            self.probs = [p / total for p in self.probs]

        # Calculate cumulative probabilities for sampling
        self.cumprobs = []
        cumsum = 0
        for p in self.probs:
            cumsum += p
            self.cumprobs.append(cumsum)

    def __getitem__(self, idx):
        # Sample dataset according to probabilities
        rand = random.random()
        for dataset_idx, threshold in enumerate(self.cumprobs):
            if rand <= threshold:
                # Generate random index for selected dataset
                rand_idx = random.randint(0, self.dataset_lengths[dataset_idx] - 1)
                return self.datasets[dataset_idx][rand_idx]

    def __len__(self):
        return self.length


i2i_instructions = [
    f"{instr} {abs} {obj}"
    for instr in ["Generate", "Give", "Make", "Create", "Show", "Display", "Produce", "Render", "Output", "Synthesize"]
    for abs in [
        "an identical",
        "a similar",
        "the same",
        "a matching",
        "an equivalent",
    ]
    for obj in [
        "image",
        "object",
        "one",
        "version",
        "picture",
        "rendering",
    ]
]
