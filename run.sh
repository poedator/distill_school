# cd $HOME/school/gen_deepspeed

# CUDA_VISIBLE_DEVICES=5,7 \
HF_HOME=/mnt/LLM \
deepspeed  --master_port 29771 --include=localhost:0,1 \
# run.py --model PY007/TinyLlama-1.1B-intermediate-step-480k-1T  --batch_size 1  --dtype float16 --max_tokens 2048 --max_new_tokens 256 --end_id 40
run.py --model meta-llama/Llama-2-7b-hf  --batch_size 1  --dtype float16 --max_tokens 2048 --max_new_tokens 256 --end_id 40

#  --num_gpus 2 
# --include=localhost:0,1  == alternative to cuda_visible_devices