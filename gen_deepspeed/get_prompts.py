# generate prompts

import torch
# from torch.utils.data import Dataset, DataLoader

import os, sys
from functools import partial
import transformers


sys.path.append(os.path.join(os.environ["HOME"], 'school'))  # School path
GSM8K_PATH = os.path.join(os.environ["HOME"], 'gsm8k')
sys.path.append(GSM8K_PATH)  # GSM8K_PATH
from ..utils import CoTDataset, collate_cot_batch, get_tokenizers
from prompts import *
from grade_school_math.dataset import get_examples, GSMDataset

## Args / params

MODEL_NAME = 'meta-llama/Llama-2-7b-hf'
prefix_tokenizer, suffix_tokenizer = get_tokenizers(MODEL_NAME)

def get_dataset():

    ## Prompt

    prompt_cot = get_cot_hub_prompt(raw=False)
    collate = partial(collate_cot_batch, prefix_tokenizer=prefix_tokenizer, suffix_tokenizer=suffix_tokenizer)

    train_examples = get_examples("train", data_path=os.path.join(GSM8K_PATH, 'grade_school_math/data'))
    dataset_train = CoTDataset(train_examples, prompt_cot)
    print(f"{len(dataset_train)=}")

    batch_size=3
    for i in range(0, len(dataset_train), batch_size):
        batch = [dataset_train[i+j]['cot8_prefix'] for j in range(batch_size)]
        break

    batch_pt = prefix_tokenizer(batch, return_tensors='pt', padding='longest')

    # batch_pt['input_ids'].shape

    print(batch)
    print(batch_pt['input_ids'].shape)