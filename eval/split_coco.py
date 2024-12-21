from datasets import load_dataset
import math
import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shard_size", type=int, default=500)
args = parser.parse_args()

# Load the original dataset
coco = load_dataset(
    "sayakpaul/coco-30-val-2014",
    split="train",
    cache_dir="/fsx/xichenpan/.cache",
    trust_remote_code=True,
    num_proc=16,
)

shutil.rmtree("/fsx/xichenpan/coco", ignore_errors=True)
os.makedirs("/fsx/xichenpan/coco", exist_ok=True)

# Calculate number of shards
num_shards = math.ceil(len(coco) / args.shard_size)

# Split and save shards
for i in range(num_shards):
    start_idx = i * args.shard_size
    end_idx = min((i + 1) * args.shard_size, len(coco))

    # Create shard using select() directly on coco dataset
    shard = coco.select(range(start_idx, end_idx))

    # Save shard
    shard.save_to_disk(f"/fsx/xichenpan/coco/val_shard_{i}")
