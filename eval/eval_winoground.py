import os
from dataclasses import dataclass

import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import SiglipProcessor, SiglipModel

from models.soda import SODA
from trainer_utils import find_newest_checkpoint

login(token="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")
os.environ["WANDB_PROJECT"] = "SODA"
USER_NAME = os.popen("whoami").read().strip()


def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]


def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]


def group_correct(result):
    return image_correct(result) and text_correct(result)


def dict_to_device(d, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if d[k].dtype == torch.float32 else d[k].dtype) for k, v in
            d.items()}


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_id: str = "google/siglip-so400m-patch14-384"
    output_dir: str = f'/fsx-project/{USER_NAME}/output'
    data_dir: str = f'/fsx-project/{USER_NAME}/.cache'
    per_device_eval_batch_size: int = 1024
    logging_dir: str = f'/fsx-project/{USER_NAME}/log'
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = os.getenv("OMP_NUM_THREADS", 24)
    dataloader_persistent_workers: bool = False
    dataloader_drop_last: bool = False
    dataloader_prefetch_factor: int = 2
    remove_unused_columns: bool = False
    # resume_from_checkpoint: str = f"/fsx-project/{USER_NAME}/stage1_256_420k"
    resume_from_checkpoint: str = f"/fsx-project/{USER_NAME}/output/sample_mse_1e7_50_start0_gradall_cfg7p5"


@torch.inference_mode()
def main():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args, = parser.parse_args_into_dataclasses()
    args.resume_from_checkpoint = find_newest_checkpoint(args.resume_from_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiglipModel.from_pretrained(args.model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
                                        device_map=device)
    processor = SiglipProcessor.from_pretrained(args.model_id)
    # model.vision_model = SODA.from_pretrained(args.resume_from_checkpoint, ignore_mismatched_sizes=True).encoder.model.vision_model
    model = model.to(device)
    dataset = load_dataset("facebook/winoground", trust_remote_code=True, cache_dir=args.data_dir, split="test",
                           num_proc=args.dataloader_num_workers)

    winoground_clip_scores = []
    for example in tqdm(dataset):
        input_c0_i0 = dict_to_device(processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")],
                                               padding="max_length", return_tensors="pt"), device)
        input_c1_i0 = dict_to_device(processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")],
                                               padding="max_length", return_tensors="pt"), device)
        input_c0_i1 = dict_to_device(processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")],
                                               padding="max_length", return_tensors="pt"), device)
        input_c1_i1 = dict_to_device(processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")],
                                               padding="max_length", return_tensors="pt"), device)
        output_c0_i0 = model(**input_c0_i0)
        output_c1_i0 = model(**input_c1_i0)
        output_c0_i1 = model(**input_c0_i1)
        output_c1_i1 = model(**input_c1_i1)
        clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
        clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
        clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
        clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
        winoground_clip_scores.append(
            {"id": example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0,
             "c1_i1": clip_score_c1_i1})

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(winoground_clip_scores)
    print("text score:", text_correct_count / denominator)
    print("image score:", image_correct_count / denominator)
    print("group score:", group_correct_count / denominator)


if __name__ == '__main__':
    main()
