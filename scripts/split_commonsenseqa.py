from datasets import load_dataset
from huggingface_hub import login

login(token="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")
dataset = load_dataset("tau/commonsense_qa", split="train", use_auth_token=True)
print(dataset)
