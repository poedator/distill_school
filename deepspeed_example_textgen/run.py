import math
import os
import time

import deepspeed
import torch
from arguments import parser
from deepspeed.runtime.utils import see_memory_usage
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils import DSPipeline, Performance

# if __name__ == '__main__':
args = parser.parse_args()

if args.hf_baseline and args.world_size > 1:
    raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

data_type = getattr(torch, args.dtype)

if args.local_rank == 0:
    see_memory_usage("before init", True)
    t0 = time.time()

pipe = DSPipeline(
    model_name=args.model,
    dtype=data_type,
    device=args.local_rank,
    # checkpoint_path=args.checkpoint_path,
)

if args.local_rank == 0:
    print(f"initialization time: {(time.time()-t0):.3f} s")
    see_memory_usage("after init", True)

ds_kwargs = dict()

pipe.model = deepspeed.init_inference(
    pipe.model,
    dtype=data_type,
    mp_size=args.world_size,
    replace_with_kernel_inject=args.use_kernel,
    max_tokens=args.max_tokens,
    # save_mp_checkpoint_path=args.save_mp_checkpoint_path,
    **ds_kwargs,
)

if args.local_rank == 0:
    see_memory_usage("after init_inference", True)

############################################################
import pickle

with open("texts.pkl", "rb") as f:
    input_sentences = pickle.load(f)


iters = 3
results = []
times = []

for i in range(0, iters * args.batch_size, args.batch_size):
    inputs = input_sentences[i : i + args.batch_size]

    start = time.time()

    if args.local_rank == 0:
        print(f"batch # {i:>3}  ", end="")
    torch.cuda.synchronize()
    outputs = pipe(
        inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=(not args.greedy),
        num_return_sequences=8,
        top_k=50,
        top_p=0.95,
        use_cache=True,
        pad_token_id=2,  # SUKABLYAT prefix_tokenizer.eos_token_ids
    )
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
    if args.local_rank == 0:
        print(f"time: {end - start:.3f} s.")
        res = dict(
            i=i,
            rank=args.local_rank,
            time=-1 if args.local_rank != 0 else end - start,
            inputs=inputs,
            outputs=outputs,
        )
        results.append(res)


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

# print(prefix_tokenizer.eos_token_id)



import json

# Load the JSON data from a file
with open('results.json', 'r') as file:
    data = json.load(file)

# Print the data in a nicely formatted manner
print(json.dumps(data, indent=4, ensure_ascii=False))
