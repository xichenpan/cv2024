"""
this file is to save the vqav2 dataset to the huggingface dataset format

First using the complementarity pairs to find the 
"""

import datasets
from argparse import ArgumentParser
import json
import PIL
from datasets.features import Sequence, Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from tqdm import tqdm

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {
        "role": "user",
        "content": "Based on the provided question and answer, create a problem statement for the multiple choice question. Remember the problem statement should be concise and short, and the instructions (especially the verb) should be diverse and clear. Try to use the expressions from the original question and answer as much as possible.",
    },
    {"role": "user", "content": "The question is: How many people are on the hill?; The answer is: 8"},
    {"role": "assistant", "content": "Choose the image with 8 people on the hill."},
    {"role": "user", "content": "The question is: Is any of the fruit sliced?; The answer is: no"},
    {"role": "assistant", "content": "Give me the image with no sliced fruit."},
    {"role": "user", "content": "The question is: What type of fruit is in the bottom right corner?; The answer is: orange"},
    {"role": "assistant", "content": "Generate a image with orange in the bottom right corner."},
    {"role": "user", "content": "The question is: Is their hair long or short?; The answer is: long"},
    {"role": "assistant", "content": "Pick the image with long hair."},
    {"role": "user", "content": "The question is: Does this chair look new?; The answer is: no"},
    {"role": "assistant", "content": "Select the image showing an old chair."},
    {"role": "user", "content": "The question is: What is the name of this Inn?; The answer is: none"},
    {"role": "assistant", "content": "Find the image with no name of the Inn."},
]


def get_prompt(question, answer):
    text = tokenizer.apply_chat_template(
        messages
        + [
            {
                "role": "user",
                "content": "Now following the demonstration, create a problem statement for the following question and answer:\nThe question is: "
                + question
                + "; The answer is: "
                + answer,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main(args):
    data = json.load(open(args.file_name, "r"))
    dataset = {
        "source_images": [],
        "source_captions": [],
        "prompt": [],
        "target_image": [],
        "target_caption": [],
    }
    for q1_image_path, question1, answer1, q2_image_path, question2, answer2 in tqdm(data):
        try:
            image1 = PIL.Image.open(q1_image_path)
            image2 = PIL.Image.open(q2_image_path)
        except:
            continue

        try:
            prompt = get_prompt(question1, answer1)
            dataset["source_images"].append([image1, image2] if random.random() < 0.5 else [image2, image1])
            dataset["source_captions"].append([question2 + "<answer>" + answer2, question1 + "<answer>" + answer1])
            dataset["prompt"].append(prompt)
            dataset["target_image"].append(image1)
            dataset["target_caption"].append("")
        except:
            pass

        try:
            prompt = get_prompt(question2, answer2)
            dataset["source_images"].append([image1, image2] if random.random() < 0.5 else [image2, image1])
            dataset["source_captions"].append([question1 + "<answer>" + answer1, question2 + "<answer>" + answer2])
            dataset["prompt"].append(prompt)
            dataset["target_image"].append(image2)
            dataset["target_caption"].append("")
        except:
            pass

    dataset = datasets.Dataset.from_dict(dataset)
    dataset = dataset.cast_column("source_images", Sequence(Image(decode=True)))
    dataset = dataset.cast_column("target_image", Image(decode=True))
    dataset.save_to_disk(f"{args.output_dir}/{args.file_name.split('/')[-1].split('.')[0]}")

    # # Push to hub
    # dataset.push_to_hub(
    #     f"xcpan/{args.output_dir.split('/')[-1]}",
    #     split=args.file_name.split(".")[0].split("_")[-1],
    #     private=True,
    # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_name", type=str, default="/fsx/xichenpan/vqav2_sharded/train/vqav2_shard_360.json")
    parser.add_argument("--output_dir", type=str, default="/fsx/xichenpan/vqav2_converted/train")
    args = parser.parse_args()
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    main(args)
