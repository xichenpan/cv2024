import datasets
from argparse import ArgumentParser
from datasets.features import Sequence, Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {
        "role": "user",
        "content": """Given a caption, follow these steps:
        1) Identify and extract the main object from the caption.
        2) Think of a similar but different object that could be confused with the main object - this will be the hard negative.
        3) Create a new caption by inserting the hard negative object Immediately after the main object, using connectors like 'but not', 'instead of', 'rather than', or 'not a'.
        Important rules:
        1) Keep the original sentence structure intact
        2) Place the hard negative object immediately after the main object, instead of adding the hard negative at the end of the caption
        """,
    },
    {"role": "user", "content": "A man with a red helmet on a small moped on a dirt road."},
    {
        "role": "assistant",
        "content": "Find main object: [a small moped]; Find hard negative: [a bicycle]; Now we put the hard negative Immediately after the main object to get the new caption: [A man with a red helmet on a small moped, but not a bicycle, on a dirt road].",
    },
    {"role": "user", "content": "A woman wearing a net on her head cutting a cake."},
    {
        "role": "assistant",
        "content": "Find main object: [a cake]; Find hard negative: [a bread]; Now we put the hard negative Immediately after the main object to get the new caption: [A woman wearing a net on her head cutting a cake instead of a bread].",
    },
    {"role": "user", "content": "A child holding a flowered umbrella and petting a yak."},
    {
        "role": "assistant",
        "content": "Find main object: [a yak]; Find hard negative: [a horse]; Now we put the hard negative Immediately after the main object to get the new caption: [A child holding a flowered umbrella petting a yak rather than a horse].",
    },
    {"role": "user", "content": "A young boy standing in front of a computer keyboard."},
    {
        "role": "assistant",
        "content": "Find main object: [a young boy]; Find hard negative: [a young girl]; Now we put the hard negative Immediately after the main object to get the new caption: [A young boy, not a young girl, standing in front of a computer keyboard].",
    },
    {"role": "user", "content": "A woman in a room with a cat."},
    {
        "role": "assistant",
        "content": "Find main object: [a cat]; Find hard negative: [a dog]; Now we put the hard negative Immediately after the main object to get the new caption: [A woman in a room with a cat but not a dog].",
    },
    {"role": "user", "content": "A commercial stainless kitchen with a pot of food cooking."},
    {
        "role": "assistant",
        "content": "Find main object: [commercial kitchen]; Find hard negative: [home kitchen]; Now we put the hard negative Immediately after the main object to get the new caption: [A commercial stainless kitchen, rather than a home kitchen, with a pot of food cooking].",
    },
    {"role": "user", "content": "A wooden ball on top of a wooden stick."},
    {
        "role": "assistant",
        "content": "Find main object: [a wooden ball]; Find hard negative: [a basketball]; Now we put the hard negative Immediately after the main object to get the new caption: [A wooden ball, instead of a basketball, on top of a wooden stick].",
    },
    {"role": "user", "content": "The view of a large bathroom with a walk in closet."},
    {
        "role": "assistant",
        "content": "Find main object: [bathroom]; Find hard negative: [bedroom]; Now we put the hard negative Immediately after the main object to get the new caption: [The view of a large bathroom, not a bedroom, with a walk in closet].",
    },
    {"role": "user", "content": "There are two sinks next to two mirrors."},
    {
        "role": "assistant",
        "content": "Find main object: [sinks]; Find hard negative: [toilet]; Now we put the hard negative Immediately after the main object to get the new caption: [There are two sinks, instead of a toilet, next to two mirrors].",
    },
    {"role": "user", "content": "A white urinal mounted to a bathroom wall."},
    {
        "role": "assistant",
        "content": "Find main object: [urinal]; Find hard negative: [toilet]; Now we put the hard negative Immediately after the main object to get the new caption: [A white urinal, rather than a toilet, mounted to a bathroom wall].",
    },
    {"role": "user", "content": "A giraffe grazing from a tree with a rock wall in the background"},
    {
        "role": "assistant",
        "content": "Find main object: [giraffe]; Find hard negative: [elephant]; Now we put the hard negative Immediately after the main object to get the new caption: [A giraffe, not an elephant, grazing from a tree with a rock wall in the background].",
    },
    {"role": "user", "content": "A man using a cross walk to cross a street."},
    {
        "role": "assistant",
        "content": "Find main object: [man]; Find hard negative: [cyclist]; Now we put the hard negative Immediately after the main object to get the new caption: [A man, instead of a cyclist, using a cross walk to cross a street].",
    },
]


def get_caption(caption):
    text = tokenizer.apply_chat_template(
        messages + [{"role": "user", "content": "Now following the demonstration, generate a new caption for the following caption:\n" + caption}],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    try:
        start_idx = response.rfind("[")
        if start_idx == -1:
            return None
        end_idx = response.find("]", start_idx)
        if end_idx == -1:
            return None
        return response[start_idx + 1 : end_idx]
    except:
        return None


def main(args):
    dataset = datasets.load_from_disk(args.file_name)
    # dataset = dataset.select(range(50))
    new_dataset = {
        "caption": [],
        "image": [],
    }
    for example in tqdm(dataset):
        image = example["image"]
        caption = example["captions"][0]

        try:
            caption = get_caption(caption)
            if caption is not None:
                new_dataset["caption"].append(caption)
                new_dataset["image"].append(image)
        except:
            pass

    dataset = datasets.Dataset.from_dict(new_dataset)
    dataset = dataset.cast_column("image", Image(decode=True))
    dataset.save_to_disk(f"{args.output_dir}/{args.file_name.split('/')[-1].split('.')[0]}")

    # # Push to hub
    # dataset.push_to_hub(
    #     "xcpan/coco2017_converted",
    #     split=args.file_name.split("-")[2],
    #     private=True,
    # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_name", type=str, default="/fsx/xichenpan/coco2017_sharded/train/train-0-of-237")
    parser.add_argument("--output_dir", type=str, default="/fsx/xichenpan/coco2017_converted/train")
    args = parser.parse_args()
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    main(args)
