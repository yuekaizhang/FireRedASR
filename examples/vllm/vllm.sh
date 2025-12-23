
# uv pip install vllm==0.12.0 --torch-backend=auto

export CUDA_VISIBLE_DEVICES=1
model_dir=/workspace_yuekai/asr/FireRedASR/pretrained_models/FireRedASR-LLM-L/new_llm

vllm serve $model_dir --runner generate \
  --max-model-len 4096 --enable-prompt-embeds



# vllm serve Qwen/Qwen3-0.6B --runner generate \
#   --max-model-len 4096 --enable-prompt-embeds