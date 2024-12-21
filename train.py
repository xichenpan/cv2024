import io
import os
import re
import shutil
from dataclasses import dataclass, field
import random

import PIL.Image
import torch
import transformers
import wandb
from datasets import load_dataset, concatenate_datasets, Image, load_from_disk
from huggingface_hub import login
from tabulate import tabulate
from torchvision.transforms import v2
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data.dataset import ConcatDataset
from glob import glob

from models.llavaov import LLaVAOV
from models.soda import SODAConfig, SODA
from trainer import SODATrainer, SODACallback, VLMEvalCallback
from trainer_utils import (
    possible_override_args,
    find_newest_checkpoint,
    get_full_dirs,
    get_random_transform,
    ConcatenatedDataset,
    i2i_instructions,
)

try:
    login(token="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")
except:
    pass

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "SODA"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from PIL import PngImagePlugin

PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)


@dataclass
class OverrideArguments:
    override: bool = True
    config_file: str = "siglip_384_luminanext_512_flow_proj.yaml"


@dataclass
class ModelArguments:
    _gradient_checkpointing: bool = False
    encoder_id: str = "google/siglip-so400m-patch14-384"
    # encoder_id: str = "google/siglip-large-patch16-256"
    # encoder_id: str = "openai/clip-vit-large-patch14"
    text_encoder: bool = False
    from_scratch: bool = False
    diffusion_model: str = "nextdit"  # hypersd, lcm, nextdit, nextdit-crossattn
    pooler_output: bool = True
    vae_id: str = "stabilityai/sdxl-vae"
    in_channels: int = 4
    vae_downsample_f: int = 8
    noise_scheduler_id: str = "facebook/DiT-XL-2-256"
    scheduler_id: str = "facebook/DiT-XL-2-256"
    mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    unet_id: str = "benjamin-paine/stable-diffusion-v1-5"
    lora_repo: str = "ByteDance/Hyper-SD"
    lora_ckpt: str = "Hyper-SD15-1step-lora.safetensors"
    load_lora = True
    loss_type: str = "diff"
    learn_sigma: bool = False
    num_sample_steps: int = 50
    start_timestep_idx: int = 0
    num_grad_steps: int = 1
    cut_off_sigma: float = 1.0
    first_step_ratio: float = -1
    as_latents: bool = False
    num_pooled_tokens: int = -1
    drop_prob: float = 0.1
    use_ema: bool = False
    ema_decay: float = 0.999
    modules_to_freeze: tuple[str] = ("vae", "transformer")
    modules_to_unfreeze: tuple[str] = ()


@dataclass
class DataArguments:
    train_datasets: dict[str, float] = field(
        default_factory=lambda: {
            # "imagenet": 1.2,
            # "imagenet21k": 14.0,
            # "openimages": 9.0,
            "cc12m": -1,
            "journeydb": -1,
            "mmc4": -1,
            "vqav2": -1,
        }
    )
    gen_eval_dataset: str = "mmc4"
    source_image_size: int = 384
    target_image_size: int = 256


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # deepspeed = "deepspeed_config.json"
    base_dir: str = "/fsx/xichenpan"
    output_dir: str = "output"
    data_dir: str = ".cache"
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    eval_delay: int = 0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch_fused"
    max_steps: int = int(1e10)
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 10000
    logging_dir: str = "log"
    logging_steps: int = 32
    save_steps: int = 5000
    save_total_limit: int = 1
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    # data_seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = 4
    datasets_num_proc: int = os.getenv("OMP_NUM_THREADS", 12)
    dataloader_persistent_workers: bool = False
    dataloader_drop_last: bool = False
    dataloader_prefetch_factor: int = 2
    remove_unused_columns: bool = False
    run_name: str = "test"
    report_to: str = "wandb"
    ddp_find_unused_parameters: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: str = None
    keys_to_ignore: tuple[str] = None
    eval_dataset: tuple[str] = ("MMVP", "Winoground")
    disable_cfg: bool = False


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((OverrideArguments, ModelArguments, DataArguments, TrainingArguments))
    override_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args, data_args, training_args = possible_override_args(override_args, model_args, data_args, training_args)
    training_args = get_full_dirs(training_args)

    assert data_args.target_image_size % model_args.vae_downsample_f == 0, f"Image size must be divisible by {model_args.vae_downsample_f}"
    input_size = data_args.target_image_size // model_args.vae_downsample_f

    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = find_newest_checkpoint(training_args.resume_from_checkpoint)
        model = SODA.from_pretrained(
            training_args.resume_from_checkpoint,
            input_size=input_size,
            ignore_mismatched_sizes=True,
            **model_args.__dict__,
        )
        if training_args.keys_to_ignore:
            state_dict = model.state_dict()

            for pat in training_args.keys_to_ignore:
                state_dict = {k: v for k, v in state_dict.items() if re.search(pat, k) is None}

            config = SODAConfig.from_pretrained(
                training_args.resume_from_checkpoint,
                input_size=input_size,
                **model_args.__dict__,
            )
            model = SODA(config)
            model.load_state_dict(state_dict, strict=False)
    else:
        model = SODA(
            config=SODAConfig(
                input_size=input_size,
                **model_args.__dict__,
            ),
        )
    model.init_copy()
    data_args.source_image_size = model.source_image_size

    train_datasets = {}
    if "imagenet" in data_args.train_datasets:
        train_dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            trust_remote_code=True,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["imagenet"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["imagenet"] * 1000000)))
        train_dataset = train_dataset.remove_columns("label")
        train_datasets["imagenet"] = train_dataset
    if "imagenet21k" in data_args.train_datasets:
        data_files = glob(f"{training_args.base_dir}/imagenet-w21-wds/*.tar")
        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files[0] if training_args.run_name == "test" else data_files,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["imagenet21k"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["imagenet21k"] * 1000000)))
        if "jpg" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("jpg", "image")
        else:
            train_dataset = train_dataset.rename_column("jpeg", "image")
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != "image"])
        train_datasets["imagenet21k"] = train_dataset
    if "openimages" in data_args.train_datasets:
        data_files = glob(f"{training_args.base_dir}/open-images/data/train/*.tar")
        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files[0] if training_args.run_name == "test" else data_files,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["openimages"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["openimages"] * 1000000)))
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != "image"])
        train_datasets["openimages"] = train_dataset
    if "cc12m_i2i" in data_args.train_datasets:
        data_files = glob(f"{training_args.base_dir}/cc12m-wds/*.tar")
        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files[0] if training_args.run_name == "test" else data_files,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["cc12m_i2i"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["cc12m_i2i"] * 1000000)))
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (["image", "caption"])])
        train_datasets["cc12m_i2i"] = train_dataset
    if "cc12m_t2i" in data_args.train_datasets:
        data_files = glob(f"{training_args.base_dir}/cc12m-wds/*.tar")
        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files[0] if training_args.run_name == "test" else data_files,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["cc12m_t2i"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["cc12m_t2i"] * 1000000)))
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (["image", "caption"])])
        train_datasets["cc12m_t2i"] = train_dataset

    if "shutterstocki2i" in data_args.train_datasets:
        data_files = (
            glob("/fsx-shutterstock-image/dataset/first_cleaned/*/webdataset_512/*/*.tar")[:7000]
            + glob("/fsx-shutterstock-image/dataset/second_batch/*/webdataset_512/*/*.tar")[:3000]
        )
        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files[0] if training_args.run_name == "test" else data_files,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (["image", "caption"])])
        train_datasets["shutterstocki2i"] = train_dataset

    if "shutterstockt2i" in data_args.train_datasets:
        data_files = (
            glob("/fsx-shutterstock-image/dataset/first_cleaned/*/webdataset_512/*/*.tar")[:7000]
            + glob("/fsx-shutterstock-image/dataset/second_batch/*/webdataset_512/*/*.tar")[:3000]
        )
        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files[0] if training_args.run_name == "test" else data_files,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (["image", "caption"])])
        train_datasets["shutterstockt2i"] = train_dataset

    if "journeydb" in data_args.train_datasets:
        train_dataset = load_dataset(
            "json",
            data_files=f"{training_args.base_dir}/JourneyDB/data/train/train_anno_realease_repath_cleaned.json",
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["journeydb"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["journeydb"] * 1000000)))
        train_dataset = train_dataset.map(
            lambda x: {"image": f"{training_args.base_dir}/JourneyDB/data/train/imgs" + x["image"][1:]},
            num_proc=training_args.datasets_num_proc,
        )
        train_datasets["journeydb"] = train_dataset
    if "mmc4" in data_args.train_datasets:
        data_files = glob(f"{training_args.base_dir}/mmc4_grouped/*/*.arrow")
        train_dataset = load_dataset(
            "arrow",
            data_files=data_files[0] if training_args.run_name == "test" else data_files,
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["mmc4"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["mmc4"] * 1000000)))
        train_datasets["mmc4"] = train_dataset
    if "vqav2" in data_args.train_datasets:
        # train_dataset = load_dataset(
        #     "arrow",
        #     data_files=f"{training_args.base_dir}/vqav2_converted/train/*/*.arrow",
        #     cache_dir=training_args.data_dir,
        #     split="train",
        #     num_proc=training_args.datasets_num_proc,
        # )
        train_dataset = load_dataset(
            "xcpan/vqav2",
            cache_dir="/data/home/xichenpan/vqav2",
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["vqav2"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["vqav2"] * 1000000)))
        train_datasets["vqav2"] = train_dataset
    if "coco" in data_args.train_datasets:
        train_dataset = load_dataset(
            "xcpan/coco2017",
            cache_dir="/data/home/xichenpan/coco2017",
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if data_args.train_datasets["coco"] > 0:
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(range(int(data_args.train_datasets["coco"] * 1000000)))
        train_datasets["coco"] = train_dataset

    source_transform = model.source_transform

    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
            v2.ToTensor(),
            v2.Normalize([0.5], [0.5]),
        ]
    )

    ground_truth_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
        ]
    )

    def i2i_process_fn(batch):
        images = batch["image"]
        captions = batch["caption"] if "caption" in batch else [random.choice(i2i_instructions) for _ in range(len(images))]
        captions = [cap.replace("<i>", "") if cap is not None else random.choice(i2i_instructions) for cap in captions]
        for i in range(len(images)):
            try:
                images[i] = PIL.Image.open(io.BytesIO(images[i]["bytes"]) if images[i]["bytes"] is not None else images[i]["path"]).convert("RGB")
            except:
                images[i] = None
                captions[i] = ""

        batch["x_target"] = [target_transform(image) if image is not None else None for image in images]
        rand_probs = torch.rand((len(images), 1))
        null_caption_mask = rand_probs < 0.2 if not training_args.disable_cfg else torch.zeros_like(rand_probs).bool()
        null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3) if not training_args.disable_cfg else torch.zeros_like(rand_probs).bool()
        captions = [caption if not null_caption_mask[i] else "" for i, caption in enumerate(captions)]
        images = [image if not null_image_mask[i] else PIL.Image.new("RGB", (image.width, image.height)) for i, image in enumerate(images)]
        if not model_args.diffusion_model == "llavaov":
            batch["x_source"] = [source_transform(image).unsqueeze(0) if image is not None else None for image in images]
            batch["caption"], batch["attn_mask"] = model.tokenize(captions)
        else:
            batch["caption"], batch["x_source"] = captions, images

        while all(x is None for x in batch["x_target"]):
            randidx = torch.randint(0, len(train_dataset), (1,)).item()
            batch = train_dataset[randidx]
            # Expand single items into batches
            batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value] for key, value in batch.items()}
        return batch

    def i2i_eval_process_fn(batch):
        images = batch["image"]
        captions = batch["caption"] if "caption" in batch else [random.choice(i2i_instructions) for _ in range(len(images))]
        captions = [cap.replace("<i>", "") if cap is not None else random.choice(i2i_instructions) for cap in captions]
        if not model_args.diffusion_model == "llavaov":
            batch["x_source"] = [source_transform(image.convert("RGB")) for image in images]
            batch["caption"], batch["attn_mask"] = model.tokenize(captions)
        else:
            batch["caption"], batch["attn_mask"], batch["x_source"], batch["image_sizes"] = model.tokenize(captions, images)

        keys_to_delete = [key for key in list(batch.keys()) if key not in ["x_source", "caption", "attn_mask", "image_sizes"]]
        for key in keys_to_delete:
            del batch[key]
        return batch

    def t2i_process_fn(batch):
        images = batch["image"]
        captions = batch["caption"] if "caption" in batch else [random.choice(i2i_instructions) for _ in range(len(images))]
        captions = [cap.replace("<i>", "") if cap is not None else random.choice(i2i_instructions) for cap in captions]
        for i in range(len(images)):
            try:
                images[i] = PIL.Image.open(io.BytesIO(images[i]["bytes"]) if images[i]["bytes"] is not None else images[i]["path"]).convert("RGB")
            except:
                images[i] = None
                captions[i] = ""

        batch["x_target"] = [target_transform(image) if image is not None else None for image in images]
        rand_probs = torch.rand((len(images), 1))
        null_caption_mask = rand_probs < 0.2 if not training_args.disable_cfg else torch.zeros_like(rand_probs).bool()
        captions = [caption if not null_caption_mask[i] else "" for i, caption in enumerate(captions)]
        if not model_args.diffusion_model == "llavaov":
            batch["caption"], batch["attn_mask"] = model.tokenize(captions)
        else:
            batch["caption"] = captions

        while all(x is None for x in batch["x_target"]):
            randidx = torch.randint(0, len(train_dataset), (1,)).item()
            batch = train_dataset[randidx]
            # Expand single items into batches
            batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value] for key, value in batch.items()}
        return batch

    def t2i_eval_process_fn(batch):
        captions = batch["caption"] if "caption" in batch else [random.choice(i2i_instructions) for _ in range(len(batch["image"]))]
        captions = [cap.replace("<i>", "") if cap is not None else random.choice(i2i_instructions) for cap in captions]
        batch["caption"], batch["attn_mask"] = model.tokenize(captions)
        keys_to_delete = [key for key in list(batch.keys()) if key not in ["caption", "attn_mask"]]
        for key in keys_to_delete:
            del batch[key]
        return batch

    def inst_process_fn(batch):
        source_images = batch["source_images"] if "source_images" in batch else None
        caption = [prompt.replace("<i>", "") for prompt in batch["prompt"]]
        rand_probs = torch.rand((len(batch["target_image"]), 1))
        null_caption_mask = rand_probs < 0.2 if not training_args.disable_cfg else torch.zeros_like(rand_probs).bool()
        null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3) if not training_args.disable_cfg else torch.zeros_like(rand_probs).bool()
        caption = [caption if not null_caption_mask[i] else "" for i, caption in enumerate(caption)]
        source_images = (
            [(image if not null_image_mask[i] else [PIL.Image.new("RGB", (img.width, img.height)) for img in image]) for i, image in enumerate(source_images)]
            if source_images is not None
            else None
        )

        if not model_args.diffusion_model == "llavaov":
            if source_images is not None:
                batch["x_source"] = [torch.stack([source_transform(img.convert("RGB")) for img in images], dim=0) for images in source_images]
            batch["caption"], batch["attn_mask"] = model.tokenize(caption)
        else:
            if source_images is not None:
                batch["caption"], batch["x_source"] = caption, source_images
            else:
                batch["caption"] = caption

        batch["x_target"] = [target_transform(img.convert("RGB")) for img in batch["target_image"]]
        batch["gt_idx"] = torch.tensor([0 if src[0] == tgt else 1 for src, tgt in zip(batch["source_images"], batch["target_image"])])
        return batch

    def inst_eval_process_fn(batch):
        source_images = batch["source_images"] if "source_images" in batch else None
        caption = [prompt.replace("<i>", "") for prompt in batch["prompt"]]

        if not model_args.diffusion_model == "llavaov":
            if source_images is not None:
                batch["x_source"] = [torch.stack([source_transform(img.convert("RGB")) for img in images], dim=0) for images in source_images]
            batch["caption"], batch["attn_mask"] = model.tokenize(caption)
        else:
            if source_images is not None:
                batch["caption"], batch["attn_mask"], batch["x_source"], batch["image_sizes"] = model.tokenize(caption, source_images)
            else:
                batch["caption"], batch["attn_mask"] = model.tokenize(caption)

        batch["gt_idx"] = torch.tensor([0 if src[0] == tgt else 1 for src, tgt in zip(batch["source_images"], batch["target_image"])])
        keys_to_delete = [key for key in list(batch.keys()) if key not in ["x_source", "caption", "attn_mask", "gt_idx"]]
        for key in keys_to_delete:
            del batch[key]
        return batch

    def collate_fn(batch):
        none_idx = [i for i, example in enumerate(batch) if example["x_target"] is None]
        if len(none_idx) > 0:
            batch = [example for i, example in enumerate(batch) if i not in none_idx]
        return_dict = {
            "x_target": torch.stack([example["x_target"] for example in batch]),
        }
        if model_args.diffusion_model == "llavaov":
            return_dict["caption"], return_dict["attn_mask"], return_dict["x_source"], return_dict["image_sizes"] = model.tokenize(
                [example["caption"] for example in batch],
                [example["image"] for example in batch if "image" in example],
            )
        else:
            if any("x_source" in example for example in batch):
                # Get sequences and their lengths
                sequences = [example.get("x_source", torch.zeros(1, 3, data_args.source_image_size, data_args.source_image_size)) for example in batch]
                lengths = torch.LongTensor(
                    [
                        (
                            seq.size(0) * (model_args.num_pooled_tokens + 2 if model_args.num_pooled_tokens > 0 else 731)
                            # seq.size(0) * (model_args.num_pooled_tokens if model_args.num_pooled_tokens > 0 else 729)
                            if example.get("x_source", None) is not None
                            else 0
                        )
                        for example, seq in zip(batch, sequences)
                    ]
                )
                # Use pad_sequence to pad
                padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
                # Create attention mask from lengths
                max_len = lengths.max()
                source_attn_mask = torch.arange(max_len)[None, :] < lengths[:, None]
                return_dict["x_source"] = padded
            if any("caption" in example for example in batch):
                return_dict["caption"] = torch.nn.utils.rnn.pad_sequence([example.get("caption", torch.zeros(1)) for example in batch], batch_first=True)
                return_dict["attn_mask"] = torch.nn.utils.rnn.pad_sequence([example.get("attn_mask", torch.zeros(1)) for example in batch], batch_first=True)

                if "x_source" in return_dict and not model_args.diffusion_model == "llavaov" and return_dict["attn_mask"] is not None:
                    return_dict["attn_mask"] = torch.cat([source_attn_mask, return_dict["attn_mask"]], dim=1)
            if any("gt_idx" in example for example in batch):
                return_dict["gt_idx"] = torch.stack([example["gt_idx"] for example in batch])
        return return_dict

    eval_dataset = train_datasets[data_args.gen_eval_dataset].select(range(training_args.world_size))
    gt_images = eval_dataset["target_image"] if "target_image" in eval_dataset.column_names else eval_dataset["image"]
    gt_images = [ground_truth_transform(image.convert("RGB")) for image in gt_images]

    if data_args.gen_eval_dataset in ["imagenet", "imagenet21k", "openimages", "cc12m_i2i", "shutterstocki2i"]:
        eval_dataset.set_transform(i2i_eval_process_fn)
    elif data_args.gen_eval_dataset in ["cc12m_t2i", "journeydb", "shutterstockt2i", "coco"]:
        eval_dataset.set_transform(t2i_eval_process_fn)
    elif data_args.gen_eval_dataset in ["mmc4", "vqav2"]:
        eval_dataset.set_transform(inst_eval_process_fn)
    else:
        raise ValueError(f"Unknown gen_eval_dataset: {data_args.gen_eval_dataset}")

    for dataset_name, train_dataset in train_datasets.items():
        if dataset_name in ["imagenet", "imagenet21k", "openimages", "cc12m_i2i", "shutterstocki2i"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column("image", Image(decode=False))
            train_datasets[dataset_name].set_transform(i2i_process_fn)
        elif dataset_name in ["cc12m_t2i", "journeydb", "shutterstockt2i", "coco"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column("image", Image(decode=False))
            train_datasets[dataset_name].set_transform(t2i_process_fn)
        elif dataset_name in ["mmc4", "vqav2"]:
            train_datasets[dataset_name].set_transform(inst_process_fn)
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        train_datasets[dataset_name] = train_datasets[dataset_name].shuffle(seed=training_args.data_seed)

    # if more than one dataset in the dict, concatenate them
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(list(train_datasets.values()))
    else:
        train_dataset = train_datasets[list(train_datasets.keys())[0]]

    trainer = SODATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[
            SODACallback(),
            VLMEvalCallback(eval_dataset=training_args.eval_dataset),
        ],
    )

    training_args.output_dir = str(os.path.join(training_args.output_dir, training_args.run_name))
    if trainer.is_world_process_zero():
        if training_args.overwrite_output_dir and os.path.exists(training_args.output_dir):
            shutil.rmtree(training_args.output_dir)
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
        print(f"Training dataset size: {len(train_dataset)}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.log_images({"gt_images": [wandb.Image(image) for image in gt_images]})
    # if training_args.run_name == "test":
    trainer.evaluate()
    trainer.train(resume_from_checkpoint=last_checkpoint)
