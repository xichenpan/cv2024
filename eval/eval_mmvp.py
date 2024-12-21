import csv
import os
from dataclasses import dataclass

import torch
import transformers
from PIL import Image
from huggingface_hub import login
from tqdm import tqdm
from transformers import SiglipProcessor, SiglipModel

from models.soda import SODA
from trainer_utils import find_newest_checkpoint

login(token="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")
os.environ["WANDB_PROJECT"] = "SODA"
USER_NAME = os.popen("whoami").read().strip()


def dict_to_device(d, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if d[k].dtype == torch.float32 else d[k].dtype) for k, v in
            d.items()}


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_id: str = "google/siglip-so400m-patch14-384"
    output_dir: str = f'/fsx-project/{USER_NAME}/output'
    data_dir: str = f'/fsx-project/{USER_NAME}/MMVP_VLM'
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
    model.vision_model = SODA.from_pretrained(args.resume_from_checkpoint, ignore_mismatched_sizes=True).encoder.model.vision_model
    model = model.to(device)

    image_dir = os.path.join(args.data_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(args.data_dir, 'Questions.csv')

    csv_outfile = open('output.csv', 'w', newline='')
    categories = [
        'Orientation and Direction', 'Presence of Specific Features',
        'State and Condition', 'Quantity and Count',
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader)):
            qid0, qtype0, statement0 = row

            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid1, qtype1, statement1 = row

            qid0, qid1 = int(qid0), int(qid1)

            img0 = Image.open(os.path.join(image_dir, qtype0, f'{qid0}.jpg')).convert("RGB")
            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg')).convert("RGB")

            caption0 = 'a photo of ' + statement0
            caption1 = 'a photo of ' + statement1

            input_c0_i0 = dict_to_device(
                processor(text=[caption0], images=[img0], padding="max_length", return_tensors="pt"), device)
            input_c1_i0 = dict_to_device(
                processor(text=[caption1], images=[img0], padding="max_length", return_tensors="pt"), device)
            input_c0_i1 = dict_to_device(
                processor(text=[caption0], images=[img1], padding="max_length", return_tensors="pt"), device)
            input_c1_i1 = dict_to_device(
                processor(text=[caption1], images=[img1], padding="max_length", return_tensors="pt"), device)

            output_c0_i0 = model(**input_c0_i0)
            output_c1_i0 = model(**input_c1_i0)
            output_c0_i1 = model(**input_c0_i1)
            output_c1_i1 = model(**input_c1_i1)
            clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
            clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
            clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
            clip_score_c1_i1 = output_c1_i1.logits_per_image.item()

            pred0 = "img0" if clip_score_c0_i0 > clip_score_c0_i1 else "img1"
            pred1 = "img0" if clip_score_c1_i0 > clip_score_c1_i1 else "img1"

            gt0 = "img0" if qid0 % 2 == 1 else "img1"
            gt1 = "img0" if qid1 % 2 == 1 else "img1"

            current_category = categories[num_pairs // 15]
            if pred0 == gt0 and pred1 == gt1:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        csv_outfile.close()

    # Calculate percentage accuracies
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    pair_accuracies["Overall"] = sum(pair_accuracies.values()) / len(pair_accuracies)
    from tabulate import tabulate
    print(tabulate(pair_accuracies.items(), headers=['Category', 'Accuracy']))


if __name__ == '__main__':
    main()
