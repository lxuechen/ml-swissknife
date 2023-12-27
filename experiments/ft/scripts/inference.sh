#python inference.py \
#  --model_name_or_path "/self/scr-ssd/lxuechen/working_dir/phi2" \
#  --cache_dir "/self/scr-ssd/lxuechen/cache"

CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_name_or_path "/self/scr-ssd/lxuechen/working_dir/phi-2-dpo" \
  --cache_dir "/self/scr-ssd/lxuechen/cache"

CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_name_or_path "/self/scr-ssd/lxuechen/working_dir/phi-2-dpo-v2" \
  --cache_dir "/self/scr-ssd/lxuechen/cache"

CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_name_or_path "/self/scr-ssd/lxuechen/working_dir/phi-2-dpo-v3" \
  --cache_dir "/self/scr-ssd/lxuechen/cache"
