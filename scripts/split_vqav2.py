"""
this file is to shard the vqav2 dataset into multiple json files, each containing 100 pairs
"""

import datasets
from argparse import ArgumentParser
import os
import json
import PIL
import numpy as np
from tqdm import tqdm
import shutil

def aggregate_pairs(annotations_path, questions_path, complementarity_path, images_dir, split):
    annotations = json.load(open(annotations_path, "r"))["annotations"]
    annotations = {ann["question_id"]: ann for ann in annotations}
    questions = json.load(open(questions_path, "r"))["questions"]
    questions = {q["question_id"]: q for q in questions}
    complementarity = json.load(open(complementarity_path, "r"))

    pairs = []

    for q1, q2 in tqdm(complementarity, total=len(complementarity)):
        q1_question = questions[q1]
        q2_question = questions[q2]
        q1_ann = annotations[q1]
        q2_ann = annotations[q2]

        q1_image_path = os.path.join(images_dir, f"COCO_{split}2014_{q1_question['image_id']:012d}.jpg")
        q2_image_path = os.path.join(images_dir, f"COCO_{split}2014_{q2_question['image_id']:012d}.jpg")

        question1 = q1_question["question"]
        question2 = q2_question["question"]
        answer1 = q1_ann["multiple_choice_answer"]
        answer2 = q2_ann["multiple_choice_answer"]

        if answer1 == answer2 or question1 != question2:
            continue

        pairs.append((q1_image_path, question1, answer1, q2_image_path, question2, answer2))

    return pairs


def main(args):
    train_annotations_path = os.path.join(args.data_dir, "v2_mscoco_train2014_annotations.json")
    train_questions_path = os.path.join(args.data_dir, "v2_OpenEnded_mscoco_train2014_questions.json")
    train_images_dir = os.path.join(args.data_dir, "train2014")
    train_complementarity_path = os.path.join(args.data_dir, "v2_mscoco_train2014_complementary_pairs.json")
    val_annotations_path = os.path.join(args.data_dir, "v2_mscoco_val2014_annotations.json")
    val_questions_path = os.path.join(args.data_dir, "v2_OpenEnded_mscoco_val2014_questions.json")
    val_images_dir = os.path.join(args.data_dir, "val2014")
    val_complementarity_path = os.path.join(args.data_dir, "v2_mscoco_val2014_complementary_pairs.json")

    train_pairs = aggregate_pairs(train_annotations_path, train_questions_path, train_complementarity_path, train_images_dir, "train")
    val_pairs = aggregate_pairs(val_annotations_path, val_questions_path, val_complementarity_path, val_images_dir, "val")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)

    n_shards = len(train_pairs) // args.shard_size + (1 if len(train_pairs) % args.shard_size != 0 else 0)
    
    for i in range(n_shards):
        start_idx = i * args.shard_size
        end_idx = min((i + 1) * args.shard_size, len(train_pairs))
        shard_pairs = train_pairs[start_idx:end_idx]
        
        with open(os.path.join(args.output_dir, "train", f"vqav2_shard_{i}.json"), "w") as f:
            json.dump(shard_pairs, f)
        print(f"Saved train shard {i} with {len(shard_pairs)} pairs")

    n_shards = len(val_pairs) // args.shard_size + (1 if len(val_pairs) % args.shard_size != 0 else 0)
    
    for i in range(n_shards):
        start_idx = i * args.shard_size
        end_idx = min((i + 1) * args.shard_size, len(val_pairs))
        shard_pairs = val_pairs[start_idx:end_idx]

        with open(os.path.join(args.output_dir, "val", f"vqav2_shard_{i}.json"), "w") as f:
            json.dump(shard_pairs, f)
        print(f"Saved val shard {i} with {len(shard_pairs)} pairs")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/fsx/xichenpan/vqav2")
    parser.add_argument("--output_dir", type=str, default="/fsx/xichenpan/vqav2_sharded")
    parser.add_argument("--shard_size", type=int, default=500, help="number of pairs per shard")
    args = parser.parse_args()
    main(args)
