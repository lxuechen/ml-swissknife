# v2
#torchrun --nproc_per_node=2 --master_port=1236 dpo.py \
#    --model_name_or_path "/self/scr-ssd/lxuechen/working_dir/phi-2-sft" \
#    --bf16 True \
#    --output_dir "/self/scr-ssd/lxuechen/working_dir/phi-2-dpo-v2" \
#    --num_train_epochs 2 \
#    --max_size 10000 \
#    --per_device_train_batch_size 1 \
#    --per_device_eval_batch_size 1 \
#    --gradient_accumulation_steps 32 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 2000 \
#    --save_total_limit 1 \
#    --learning_rate 3e-5 \
#    --beta 0.1 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --cache_dir "/self/scr-ssd/lxuechen/cache" \
#    --save_raw_state_dict True \
#    --seed 42 \
#    --fsdp "full_shard auto_wrap" \
#    --fsdp_transformer_layer_cls_to_wrap "ParallelBlock" \
#    --report_to "none"

# v3
torchrun --nproc_per_node=4 --master_port=1236 dpo.py \
    --model_name_or_path "/self/scr-ssd/lxuechen/working_dir/phi-2-sft" \
    --bf16 True \
    --output_dir "/self/scr-ssd/lxuechen/working_dir/phi-2-dpo-v3" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --beta 0.1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --cache_dir "/self/scr-ssd/lxuechen/cache" \
    --save_raw_state_dict True \
    --seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "ParallelBlock" \
    --report_to "none"
