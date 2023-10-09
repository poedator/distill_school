import transformers
import time
from contextlib import contextmanager
import torch

@contextmanager
def suspend_nn_inits():
    skip = lambda *args, **kwargs: None
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring
        
start = time.time()

with suspend_nn_inits():
    model_name = "meta-llama/Llama-2-7b-hf"

model =  transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype='auto')


elapsed = time.time() - start

print (f"{elapsed=:.3f}")
print('-'*80)
# print(model)
