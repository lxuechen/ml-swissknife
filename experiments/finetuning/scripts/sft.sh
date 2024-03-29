python sft.py \
    --model_name_or_path "microsoft/phi-2" \
    --bf16 True \
    --output_dir "/self/scr-ssd/lxuechen/working_dir/phi2" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --cache_dir "/self/scr-ssd/lxuechen/cache" \
    --save_raw_state_dict True
