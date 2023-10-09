from contextlib import contextmanager

import torch
import transformers
import peft
import numpy as np
from tqdm.auto import trange, tqdm
from matplotlib import pyplot as plt



import time

# ======= LOADING

@contextmanager
def suspend_nn_inits():
    skip = lambda *args, **kwargs: None  
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  #saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  #replacing     
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring
        
        
def load_model(model_name, load_in_8bit = False, torch_dtype=torch.float16, device_map=None):
    with suspend_nn_inits():
        if load_in_8bit:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype,  
                                                                      device_map=device_map)
    return model


def to_teacher(model):
    for m in model.modules():
        if isinstance(m, peft.tuners.lora.LoraLayer):
            m.active_adapter = None


def to_student(model, adapter_name='default'):
    for m in model.modules():
        if isinstance(m, peft.tuners.lora.LoraLayer):
            m.active_adapter = adapter_name
            
# ==== INFERENCE =======================================

def model_inference(prompt, model, tokenizer, stop_token_id=None, max_new_tokens=1024, 
                    num_return_sequences=1, use_cache=True):

    # generate_kwargs = dict()  # fill depending on mode
    # commented options are for beam search
    # beams_per_seq = 2
    # num_beams = num_return_sequences * beams_per_seq
    # num_beam_groups = None if num_return_sequences == 1 else num_return_sequences
    # diversity_penalty = None if num_return_sequences == 1 else 0.6
    
    prompt_t = tokenizer(prompt, return_tensors='pt').to('cuda')
    max_length = prompt_t['input_ids'].shape[-1] + max_new_tokens
    eos_token_id = stop_token_id or tokenizer.eos_token_id

    # with torch.inference_mode():
    with torch.no_grad():
        rr = model.generate(prompt_t['input_ids'], 
                                max_length=max_length, 
                                attention_mask=prompt_t['attention_mask'],
                                # num_beams=num_beams,
                                eos_token_id = eos_token_id,
                                temperature = 1.0,
                                num_return_sequences=num_return_sequences,
                                # diversity_penalty=diversity_penalty,
                                # num_beam_groups=num_return_sequences,
                                do_sample = True,
                                top_k = 50,
                                top_p = 0.95,
                                use_cache=use_cache
                           )
    results = [tokenizer.decode(r[prompt_t['input_ids'].shape[-1]:], 
                                skip_special_tokens=True
                               ).strip() 
               for r in rr]
    
    return results

    
    
import requests

def model_inference_remote(prompt, stop_token_id=None, num_return_sequences=1, port=5000, *args, **kwargs):
    url = f"http://localhost:{port}/generate_line"
    data = {"prompt": prompt, 
            'stop_token_id': stop_token_id,
            'num_return_sequences': num_return_sequences,
           }
    
    response = requests.post(url, json=data)
    assert response.status_code==200, f"Request error: response status code = {response.status_code}"
    try:
        result = response.json().get('output')
        return result
    except:
        print("DECODE ERROR!")
        return response
    
import re


# ============= D A T A ====================

def get_tokenizers(model_name):
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # tokenizer = transformers.LlamaTokenizer.from_pretrained(MODEL_NAME)  # for Deacpoda
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    suffix_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    suffix_tokenizer.padding_side = 'right'
    if suffix_tokenizer.pad_token is None:
        suffix_tokenizer.pad_token = suffix_tokenizer.eos_token

    prefix_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    prefix_tokenizer.padding_side = 'left'
    if prefix_tokenizer.pad_token is None:
        prefix_tokenizer.pad_token = prefix_tokenizer.eos_token
                
    return prefix_tokenizer, suffix_tokenizer


from torch.utils.data import Dataset, DataLoader

def join_inputs(left_inputs, right_inputs, device=None):
    """joins inputs in post-tokenizer dict form"""
    assert left_inputs.keys() == right_inputs.keys()
    
    out = dict()
    for key in left_inputs:
        assert left_inputs[key].ndim == right_inputs[key].ndim == 2
        out[key] = torch.cat([left_inputs[key], right_inputs[key]], dim=1)
        if device is not None: 
            out[key] = out[key].to(device)
        
    return out

class CoTDataset(Dataset):
    """for CoT prompting from GSM8k dataset"""
    def __init__(self, source_set, prompt0):
        self.source_set = source_set
        self.prompt0 = prompt0.strip()
        # self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.source_set)
    
    def __getitem__(self, idx):
        ex = ex_split(self.source_set[idx])

        steps = ' '.join([s for s in ex['steps'] if s])
        answer = int(ex['answer']) if int(ex['answer']) == ex['answer'] else ex['answer']

        # teacher_prefix = f"{self.prompt0}\nQuestion: {ex['question']}\n"
        teacher_prefix = f"{self.prompt0}\nQuestion: {ex['question']}\nLet's think step by step: {steps}\n"
        cot8_prefix = f"{self.prompt0}\nQuestion: {ex['question']}\nLet's think step by step:"
        student_prefix = f"Question: {ex['question']}\n"
        # main_text = f"Let's think step by step: {steps} The answer is {answer}."
        main_text = f"The answer is {answer}."
                
        out = dict(teacher_prefix = teacher_prefix,
                    student_prefix = student_prefix,
                    cot8_prefix = cot8_prefix,
                    main_text = main_text
                   )
        return out
    

def collate_cot_batch(batch, prefix_tokenizer, suffix_tokenizer):
    teacher_prefix_inputs = prefix_tokenizer([item["teacher_prefix"] for item in batch], return_tensors='pt', padding=True)
    student_prefix_inputs = prefix_tokenizer([item["student_prefix"] for item in batch], return_tensors='pt', padding=True)
    main_inputs = suffix_tokenizer([item["main_text"] for item in batch], return_tensors='pt', padding=True, add_special_tokens=False)
    main_inputs_length = main_inputs['input_ids'].shape[1]

    out = dict (
        teacher_prefix_inputs = teacher_prefix_inputs,
        student_prefix_inputs = student_prefix_inputs,
        main_inputs = main_inputs,
        teacher_batch = join_inputs(teacher_prefix_inputs, main_inputs),
        student_batch = join_inputs(student_prefix_inputs, main_inputs),
        main_inputs_length = main_inputs['input_ids'].shape[1],
        main_inputs_attn_mask = main_inputs['attention_mask'],
           )     
    return out


def ex_split(ex):
    # split GSM8k example into (question, steps, answer)    
    steps, answer = re.split(r'\#\#+', ex['answer'])
    return dict(
        question = ex['question'].strip(),
        # steps_raw = [s.strip() for s in steps.split('\n')],
        steps = [re.sub(r'<<.*>>', '', s.strip()) for s in steps.split('\n')],
        answer = float(answer.replace('<|endoftext|>','').replace(',', '').strip()),
        )


def ex_2_cot(ex):
    # repack GSM8k example into string for CoT format
    ex_ = ex_split(ex)
    steps_sep = '\n'

    out =(
        f"Question:\n{ex_['question']}"
        f"Solution:\n{steps_sep.join(ex_['steps'])}"
        f"Answer:\n<{ex_['answer']}>"
        )
    return out
    
    
def get_cot_prompt(dataset_, ex_ids, q_id):
    # creates CoT prompt using GSM8k dataset for Qs and A
    out = "For each Question describe solution using simple interim questions and then provide answer between < >:\n" + \
        '\n'.join([ex_2_cot(dataset_[i]) for i in ex_ids]) + '\n' + \
        dataset_[q_id]['question'] + \
        'Solution:\n' 
    return out

# ++++++++++++++++++++++ T R A I N ++++++++++++++++++++++++

def validation_epoch(model, val_loader, device):
    print ('VALIDATION')
    num, den = 0, 0
    val_log = []
    for i, batch in enumerate(tqdm(val_loader)):

        teacher_batch = {k:v.to(device) for k, v in batch['teacher_batch'].items()}
        student_batch = {k:v.to(device) for k, v in batch['student_batch'].items()}

        with torch.no_grad():

            to_teacher(model)
            ref_probs = torch.softmax(model(**teacher_batch, use_cache=False).logits, dim=-1)
            ref_probs = ref_probs[:, -batch['main_inputs_length']:, :]

            main_inputs_attn_mask = student_batch['attention_mask'][:, -batch['main_inputs_length']:]

            to_student(model)
            student_logprobs = torch.log_softmax(model(**student_batch, use_cache=False).logits, dim=-1) \
                [:, -batch['main_inputs_length']:, :]

            ref_probs = ref_probs.to(student_logprobs.device)
            loss = - ((ref_probs * student_logprobs).sum(-1) * main_inputs_attn_mask).sum() / main_inputs_attn_mask.sum()

            teacher_entropy = -(
                (ref_probs * torch.log(ref_probs+1e-9)).sum(-1) * main_inputs_attn_mask
            ).sum() / main_inputs_attn_mask.sum()
            delta_entropy1 = (loss - teacher_entropy).item()
            num1 = ((ref_probs.argmax(-1) == student_logprobs.argmax(-1)).float() * main_inputs_attn_mask).sum().item()
            den1 = main_inputs_attn_mask.sum().item()
            acc1 = num1 / den1
            num += num1
            den += den1
            val_log.append((loss.item(), delta_entropy1, acc1))
        
            print(f"{str(i)+':':>4} | batch results: {delta_entropy1=:.4f}  {acc1=:.4f}")
            torch.cuda.empty_cache()
            
    acc = num / den
    delta_entropy = np.mean([l[1] for l in val_log])
    print(f"val results: {acc=:.4f}  {delta_entropy=:.4f}")
    return val_log

def train_epoch(model, optimizer, train_loader, args, device):
    train_log = [] 
    NUM_ACCUMULATION_STEPS = args.batch_size / args.per_device_train_batch_size
    
    for i, batch in enumerate(tqdm(train_loader)):

        teacher_batch = {k:v.to(device) for k, v in batch['teacher_batch'].items()}
        student_batch = {k:v.to(device) for k, v in batch['student_batch'].items()}

        to_teacher(model)
        with torch.no_grad():
            ref_probs = torch.softmax(model(**teacher_batch, use_cache=False).logits, dim=-1)
            ref_probs = ref_probs[:, -batch['main_inputs_length']:, :]


        main_inputs_attn_mask = student_batch['attention_mask'][:, -batch['main_inputs_length']:]

        to_student(model)
        student_logprobs = torch.log_softmax(model(**student_batch, use_cache=False).logits, dim=-1) \
            [:, -batch['main_inputs_length']:, :]

        # distillation_xent loss
        ref_probs = ref_probs.to(device)
        loss = - ((ref_probs * student_logprobs).sum(-1) * main_inputs_attn_mask).sum() / main_inputs_attn_mask.sum()

        loss = loss / NUM_ACCUMULATION_STEPS
        loss.backward()
        
        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # teacher_entropy_for_show_only_do_not_use
        with torch.no_grad():
            teacher_entropy = -((ref_probs * torch.log(ref_probs+1e-9)).sum(-1) * main_inputs_attn_mask).sum() \
                / main_inputs_attn_mask.sum()
            delta_entropy = (loss * NUM_ACCUMULATION_STEPS - teacher_entropy).item()
            acc = (((ref_probs.argmax(-1) == student_logprobs.argmax(-1)).float() * main_inputs_attn_mask).sum() \
                / main_inputs_attn_mask.sum()).item()

        if i % 16 == 0:
            print(f"{str(i)+':':>4} loss={loss * NUM_ACCUMULATION_STEPS:.4f}  {acc=:.4f}  {delta_entropy=:.4f}")
            torch.cuda.empty_cache()

        train_log.append(dict(i=i, loss=loss.item(), acc=acc, 
                        delta_entropy=delta_entropy, 
                        main_inputs_length=batch['main_inputs_length'],
                        teacher_entropy=teacher_entropy.item(),
                           )
                      )
    return train_log

def plot_history(history, window=1):
    # args: history dict, moving average window int

    fig, axs = plt.subplots(1, 2, figsize = (10,4))
    data = [np.convolve([h['delta_entropy'] for h in history], np.ones(window), 'valid') / window,
            np.convolve([h['acc'] for h in history], np.ones(window), 'valid') / window]
    titles = ('student - teacher x-entropy diff', 'accuracy')
    for i, ax in enumerate(axs):
        ax.plot(data[i])
        ax.set_title(titles[i])
        ax.set_xlabel('steps')
        ax.grid()
    fig.suptitle(f'train overfit observed, moving averages with {window=}', fontsize=16);
    plt.tight_layout()

# ============ T E S T I N G =====================

def model_inference(prompt, model, tokenizer, stop_token_id=None, max_new_tokens=1024, 
                    num_return_sequences=1, use_cache=True, device='cuda'):    
    prompt_t = tokenizer(prompt, return_tensors='pt').to(device)
    max_length = prompt_t['input_ids'].shape[-1] + max_new_tokens
    eos_token_id = stop_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        rr = model.generate(input_ids=prompt_t['input_ids'], 
                                max_length=max_length, 
                                attention_mask=prompt_t['attention_mask'],
                                eos_token_id = eos_token_id,
                                num_return_sequences=num_return_sequences,
                                do_sample = True,
                                top_k = 50,
                                top_p = 0.95,
                                use_cache=use_cache,
                           )
    results = [tokenizer.decode(r[prompt_t['input_ids'].shape[-1]:], 
                                skip_special_tokens=True
                               ).strip() 
               for r in rr]
    
    return results


def test_gsm(model, tokenizer, template, test_dataset, end_of_answer_token_id = 13, verbose=True, 
             max_new_tokens=32, num_return_sequences = 8, test_range = (0, 200)):
    log = []

    pbar = trange(*test_range)
    for q_id in pbar:
        if verbose:
            print(q_id, end=': ')
        start = time.time()
        ex = ex_split(test_dataset[q_id])

        prompt = template.format(ex['question'])
        res = model_inference(prompt, model, tokenizer, 
                              stop_token_id=end_of_answer_token_id, 
                              num_return_sequences=num_return_sequences, 
                              max_new_tokens=max_new_tokens)
        aaa = [answer_cleansing(r) for r in res]

        mode, confidence = get_mode(aaa, return_confidence=True)
        verdict = mode == float(ex['answer'])
        elapsed = round(time.time() - start, 1)
        if verbose:
            print(f"{verdict} {mode}{'=' if verdict else '≠'}{ex['answer']} c={confidence:.1f}, t={elapsed:.1f}s, ans:{aaa}")
        log.append(dict(
            q_id=q_id,
            verdict=verdict,
            mode=mode,
            confidence=confidence, 
            gt = ex['answer'], 
            elapsed = elapsed, 
            answers = aaa, 
            responses = res,
        ))        
        running_acc = np.mean([int(l['verdict']) for l in log])
        pbar.set_description(f"running acc: {running_acc:.1%}")

    print(score_log(log))
    return log


def score_log(log):
    # returns majority accuracy and average accuracy
    
    maj_acc = sum([int(l['verdict']) for l in log]) / len(log)  # Maj@1:

    num, den = 0, 0
    for l in log:
        all_answers = l['answers']
        gt = l['gt']
        for a in all_answers:
            num += (a == gt)
            den += 1
    avg_acc = num / den, 4
    
    maj_acc = round(maj_acc, 4)
    avg_acc = round(avg_acc, 4)

    return maj_acc, avg_acc


import re

pattern_float = re.compile(r'(-?[\d\,]*\d\.?\d*)', re.IGNORECASE)

def parse_answer(res_):
    # extract last float number from input string
    match = pattern_float.findall(res_)
    if match:
        return float(match[-1].replace(',', ''))

def get_mode(data, return_confidence=False):
    """Calculates the mode of the given data. (MAJORITY VOTE)"""
    dd = [d for d in data if d is not None]
    if dd:
        unique_elements, counts_elements = np.unique(dd, return_counts=True)
        mode = unique_elements[np.argmax(counts_elements)]
        if return_confidence:
            return mode, max(counts_elements) / len(data)
        else:
            return mode
    else:
        if return_confidence:
            return None, 0
        else:
            return 0


def get_answer_line(xxx):
    # returns first line with word 'answer' or last line
    for xx in xxx.split('\n'):
        if 'answer' in xx.lower():
            break
    return xx


def gen_test(model, tokenizer, prompt="Here is a story:", max_new_tokens=128, device=torch.device('cuda'), warmup=True):
    # test generation speed
    prompt_t = tokenizer(prompt, return_tensors='pt').to('cuda')
    max_length = prompt_t['input_ids'].shape[-1] + max_new_tokens
    if warmup:
        model.generate(prompt_t['input_ids'], max_length=prompt_t['input_ids'].shape[-1] + 4)
    
    start = time.perf_counter()
    with torch.no_grad():
        rr = model.generate(prompt_t['input_ids'], 
                            max_length=max_length, 
                            attention_mask=prompt_t['attention_mask'],
                           )
    elapsed = time.perf_counter() - start
    gen_tokens_count = rr[0].shape[-1] - prompt_t['input_ids'].shape[-1]
    assert gen_tokens_count == max_new_tokens
    generate_speed = gen_tokens_count / elapsed
    print (f"{elapsed=:.2f}, {generate_speed=:.2f} tokens/sec")
    return round(generate_speed, 2)


def answer_cleansing(pred, fs_trigger=None, method="zero_shot", dataset="gsm8k", verbose = False):
    # extracts float answer value from answer string
    # based on https://github.com/kojima-takeshi188/zero_shot_cot/
    
    fs_trigger = fs_trigger or 'The answer is'  # direct_answer_trigger_for_fewshot
    if verbose:
        print("pred_before : " + pred)
    
    if method in ("few_shot", "few_shot_cot"):
        preds = pred.split(fs_trigger)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    if dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    if verbose:
        print("pred_after : " + pred)
    
    try:
        pred = float(pred)
    except:
        pass
    
    return pred