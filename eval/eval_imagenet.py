import os
from dataclasses import dataclass

import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F

from trainer_utils import find_newest_checkpoint

login(token="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")
os.environ["WANDB_PROJECT"] = "SODA"
USER_NAME = os.popen("whoami").read().strip()

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def dict_to_device(d, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if d[k].dtype == torch.float32 else d[k].dtype) for k, v in
            d.items()}


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 0, True, True)[1].t()
    correct = pred.eq(target)
    return [correct[:, :k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item() for k in topk]


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # model_id: str = "google/siglip-large-patch16-256"
    model_id: str = "google/siglip-so400m-patch14-384"
    output_dir: str = f'/fsx-project/{USER_NAME}/output'
    data_dir: str = f'/fsx-project/{USER_NAME}/.cache'
    per_device_eval_batch_size: int = 500
    logging_dir: str = f'/fsx-project/{USER_NAME}/log'
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = os.getenv("OMP_NUM_THREADS", 24)
    dataloader_persistent_workers: bool = False
    dataloader_drop_last: bool = False
    dataloader_prefetch_factor: int = 2
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
    dataset = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True, cache_dir=args.data_dir,
                           split="validation", num_proc=args.dataloader_num_workers)

    def collate_fn(batch):
        return {
            "image": [item['image'].convert("RGB") for item in batch],
            "label": [item['label'] for item in batch],
        }

    dataloaders = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_persistent_workers,
        drop_last=args.dataloader_drop_last,
        prefetch_factor=args.dataloader_prefetch_factor,
        collate_fn=collate_fn,
    )

    zeroshot_weights = []
    for classname in tqdm(dataset.features["label"].names):
        texts = [template.format(classname) for template in imagenet_templates]
        texts = dict_to_device(processor(text=texts, padding="max_length", return_tensors="pt"), device)
        class_embeddings = model.get_text_features(**texts)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights)

    top1, top5, n = 0., 0., 0.
    for example in tqdm(dataloaders):
        image = dict_to_device(processor(images=example['image'], return_tensors="pt"), device)
        image_features = model.get_image_features(**image)
        image_features = F.normalize(image_features, p=2, dim=-1)
        logits = torch.matmul(zeroshot_weights, image_features.t()) * model.logit_scale.exp() + model.logit_bias
        label = torch.LongTensor(example['label']).to(device).unsqueeze(-1)
        # measure accuracy
        acc1, acc5 = accuracy(logits, label, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += len(example['image'])
        print("Top-1 accuracy: {:.2f}, Top-5 accuracy: {:.2f}".format((top1 / n) * 100, (top5 / n) * 100))

    print(f"Final results:")
    print(f"Top-1 accuracy: {(top1 / n) * 100:.2f}")
    print(f"Top-5 accuracy: {(top5 / n) * 100:.2f}")


if __name__ == '__main__':
    main()
