from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
import time
from utils import DSPipeline, Performance
from deepspeed.runtime.utils import see_memory_usage
from arguments import parser

if __name__ == '__main__':
    args = parser.parse_args()

    if args.hf_baseline and args.world_size > 1:
        raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

    data_type = getattr(torch, args.dtype)

    if args.local_rank == 0:
        see_memory_usage("before init", True)

    t0 = time.time()

    pipe = DSPipeline(model_name=args.model,
                      dtype=data_type,
                      is_meta=args.use_meta_tensor,
                      device=args.local_rank,
                      checkpoint_path=args.checkpoint_path)

    if args.local_rank == 0:
        print(f"initialization time: {(time.time()-t0) * 1000}ms")
        see_memory_usage("after init", True)

    if args.use_meta_tensor:
        ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
    else:
        ds_kwargs = dict()

    # Use DeepSpeed Hybrid Engine for inference
    if args.test_hybrid_engine:
        ds_config = {"train_batch_size": args.batch_size, "fp16": {"enabled": True if data_type==torch.half else False}, 
                     "hybrid_engine": {"enabled": False}}
        pipe.model, *_ = deepspeed.initialize(model=pipe.model, config=ds_config)
        pipe.model.eval()
    # If not trying with the HuggingFace baseline, use DeepSpeed Inference Engine
    else:
        if not args.hf_baseline:
            pipe.model = deepspeed.init_inference(pipe.model,
                                        dtype=data_type,
                                        mp_size=args.world_size,
                                        replace_with_kernel_inject=args.use_kernel,
                                        max_tokens=args.max_tokens,
                                        save_mp_checkpoint_path=args.save_mp_checkpoint_path,
                                        **ds_kwargs
                                        )

    if args.local_rank == 0:
        see_memory_usage("after init_inference", True)

    ############################################################
    import pickle
    with open("texts.pkl", 'rb') as f:
        input_sentences = pickle.load(f)
    
    assert args.batch_size <= len(input_sentences), f"this shitty benchmark only has {len(input_sentences)} sequences"
    
    inputs = input_sentences[:args.batch_size]

    iters = 30 if args.test_performance else 2 #warmup
    times = []
    for i in range(iters):
        print("Attempt #", i)
        torch.cuda.synchronize()
        start = time.time()
        outputs = pipe(inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(not args.greedy),
                       num_return_sequences=8,
        top_k = 50,
        top_p = 0.95,
        use_cache=True,
        pad_token_id=2,#SUKABLYAT prefix_tokenizer.eos_token_ids
                      )
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        print("Attempt took", times[-1], 's.')
    print(f"generation time is {times[1]} sec")

    if args.local_rank == 0:
        for i, o in zip(inputs, outputs):
            print(f"\nin={i}\nout={o}\n{'-'*60}")
        if args.test_performance:
            Performance.print_perf_stats(map(lambda t: t / args.max_new_tokens, times), pipe.model.config, args.dtype, args.batch_size)

