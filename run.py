# based on https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/inference-test.py

import transformers

import deepspeed
import os
import sys
import torch
import time
import pickle

from gen_deepspeed.ds_utils import DSPipeline, Performance

# ds_utils.py is unchanged utils.py from https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/utils.py
# subclassing to use custom generation
from gen_deepspeed.arguments import parser
from prompts import *

# from utils import CoTDataset, collate_cot_batch, get_tokenizers

GSM8K_PATH = os.path.join(os.environ["HOME"], "gsm8k")
sys.path.append(GSM8K_PATH)  # GSM8K_PATH
from grade_school_math.dataset import get_examples  # , GSMDataset


class CustomDSPipeline(DSPipeline):
    def __init__(
        self,
        model_name,
        dtype=torch.float16,
        device=-1,
    ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

        if self.dtype == torch.float16:
            self.model.half()

    def generate_outputs(self, input_ids, **kwargs):
        input_ids = input_ids.to(self.device)
        self.model.cuda().to(self.device)

        outputs = self.model.generate(
            inputs=input_ids, generation_config=None, **kwargs
        )
        return outputs


def get_dataset(pickle_path=None):
    if pickle_path is not None:
        with open(pickle_path, "rb") as f:
            input_sentences = pickle.load(f)
    else:
        train_examples = get_examples(
            "train", data_path=os.path.join(GSM8K_PATH, "grade_school_math/data")
        )
        input_sentences = [te["question"] for te in train_examples]

    print(f"{len(input_sentences)=}")
    return input_sentences


def main(args):
    if args.hf_baseline and args.world_size > 1:
        raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

    data_type = getattr(torch, args.dtype)

    if args.local_rank == 0:
        # see_memory_usage("before init", True)
        t0 = time.time()

    pipe = CustomDSPipeline(
        model_name=args.model,
        dtype=data_type,
        #   is_meta=False,  # args.use_meta_tensor,
        device=args.local_rank,
        #   checkpoint_path=args.checkpoint_path
    )

    if args.local_rank == 0:
        print(f"initialization time: {(time.time()-t0):.3f} s")
        # see_memory_usage("after init", True)

    pipe.model = deepspeed.init_inference(
        pipe.model,
        dtype=data_type,
        mp_size=args.world_size,
        replace_with_kernel_inject=args.use_kernel,
        max_tokens=args.max_tokens,
        save_mp_checkpoint_path=args.save_mp_checkpoint_path,
    )

    dataset = get_dataset()
    prompt_cot = get_cot_hub_prompt(raw=False)
    prefix_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    prefix_tokenizer.padding_side = "left"
    prefix_tokenizer.pad_token = prefix_tokenizer.eos_token

    results = []
    times = []

    batch_size = args.batch_size
    for i in range(args.start_id, min(args.end_id, len(dataset)), batch_size):
        start = time.time()
        data_ids = [i + j for j in range(batch_size)]
        batch0 = [dataset[j] for j in data_ids]
        batch1 = [
            f"{prompt_cot}\nQuestion: {q}Let's think step by step:" for q in batch0
        ]
        batch_pt = prefix_tokenizer(batch1, return_tensors="pt", padding="longest")

        if args.local_rank == 0:
            print(f"batch # {i:>3}  ", end="")
        
        q_length = batch_pt["input_ids"].shape[-1]
        torch.cuda.synchronize()

        output_tokens = pipe.generate_outputs(
            batch_pt["input_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=(not args.greedy),
            num_return_sequences=8,
            top_k=50,
            top_p=0.95,
            use_cache=True,
            pad_token_id=2,  # prefix_tokenizer.eos_token_ids
        )
        output_tokens = output_tokens[:, q_length:]  # stripping the prompt and question
        outputs = prefix_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        if args.local_rank == 0:
            print(f"time: {end - start:.3f} s.")
            res = dict(
                i=i,
                data_ids=data_ids,
                rank=args.local_rank,
                time=-1 if args.local_rank != 0 else end - start,
                inputs=batch0,
                outputs=outputs,
                # output_tokens=output_tokens,
                # batch_pt=batch_pt
            )
            results.append(res)

        # break

    print("PERF STATS:")
    Performance.print_perf_stats(
        map(lambda t: t / args.max_new_tokens, times),
        pipe.model.config,
        args.dtype,
        args.batch_size,
    )

    import json
    results_j = json.dumps(results)
    with open("results.json", "w") as outfile:
        outfile.write(results_j)
    # torch.save(results, 'results.pt')

    print("prefix_tokenizer.eos_token_id", prefix_tokenizer.eos_token_id)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
