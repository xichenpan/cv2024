import transformers
from datasets import load_dataset

from train import OverrideArguments, ModelArguments, DataArguments, TrainingArguments
from trainer_utils import possible_override_args, get_full_dirs

# Parsing the training arguments
parser = transformers.HfArgumentParser(
    (OverrideArguments, ModelArguments, DataArguments, TrainingArguments)
)
override_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Setting up the arguments
model_args, data_args, training_args = possible_override_args(override_args, model_args, data_args, training_args)
training_args = get_full_dirs(training_args)

# Load the dataset and sort by label
train_dataset = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True, cache_dir=training_args.data_dir,
                             split="train", num_proc=training_args.datasets_num_proc)
train_dataset = train_dataset.sort("label")


# Copy the image column to image1 and image2, then shift image2 by 1
def copy_and_shift_images(example, idx):
    return {
        "image1": example["image"],
        "image2": train_dataset[idx - 1]["image"] if idx > 0 else example["image"]
        # Use the first image for the first row
    }


# Apply the mapping
train_dataset = train_dataset.map(copy_and_shift_images, with_indices=True, num_proc=training_args.datasets_num_proc)

train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
    ["image1", "image2"])])

train_dataset.save_to_disk(f"{training_args.base_dir}/int1k_paired")
print("Finished processing and saving the paired dataset.")
