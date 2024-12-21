import os
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from datasets.features import Image
import PIL.Image
import io


def create_full_path(example):
    example["image"] = os.path.join("/fsx/xichenpan/coco2017", example["file_name"])
    return example


def filter_bad_images(example):
    try:
        PIL.Image.open(example["image"])
    except:
        return False
    return True


def main(args):
    dataset = load_dataset("phiyodr/coco2017", split="train")
    dataset = dataset.map(create_full_path, num_proc=64)
    dataset = dataset.filter(filter_bad_images, num_proc=64)
    dataset = dataset.cast_column("image", Image(decode=True))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    # Process each split in the dataset
    num_shards = (len(dataset) + args.shard_size - 1) // args.shard_size
    for shard_idx in tqdm(range(num_shards)):
        start_idx = shard_idx * args.shard_size
        end_idx = min((shard_idx + 1) * args.shard_size, len(dataset))
        shard = dataset.select(range(start_idx, end_idx))
        shard.save_to_disk(os.path.join(args.output_dir, "train", f"train-{shard_idx}-of-{num_shards}"))

    dataset = load_dataset("phiyodr/coco2017", split="validation")
    dataset = dataset.map(create_full_path, num_proc=64)
    dataset = dataset.filter(filter_bad_images, num_proc=64)
    dataset = dataset.cast_column("image", Image(decode=True))

    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)
    num_shards = (len(dataset) + args.shard_size - 1) // args.shard_size
    for shard_idx in tqdm(range(num_shards)):
        start_idx = shard_idx * args.shard_size
        end_idx = min((shard_idx + 1) * args.shard_size, len(dataset))
        shard = dataset.select(range(start_idx, end_idx))
        shard.save_to_disk(os.path.join(args.output_dir, "val", f"val-{shard_idx}-of-{num_shards}"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/fsx/xichenpan/coco2017")
    parser.add_argument("--output_dir", type=str, default="/fsx/xichenpan/coco2017_sharded")
    parser.add_argument("--shard_size", type=int, default=500, help="number of pairs per shard")
    args = parser.parse_args()
    main(args)
