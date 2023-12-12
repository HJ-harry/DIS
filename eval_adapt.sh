#!/bin/bash

lr_list=(1e-5 1e-4 1e-3)
iter_list=(3 5 10)
adapt_method_list=("lora" "full")
lora_rank=4

for lr in "${lr_list[@]}"; do
    echo $lr
    for iter in "${iter_list[@]}"; do
        echo $iter
        for adapt_method in "${adapt_method_list[@]}"; do    
            echo $adpat_method
            python main.py \
            --ni \
            --config model_imagenet_data_celeba.yml \
            --path_y celeba_hq \
            --eta 0.85 \
            --deg "sr_averagepooling" \
            --deg_scale 8 \
            --sigma_y 0.0 \
            --adapt_lr $lr \
            --adapt_iter $iter \
            --adapt_method $adapt_method \
            --lora_rank $lora_rank \
            --use_adapt \
            --use_additional_dc \
            --use_dc_before_adapt \
            --tv_penalty 0 \
            --dc_before_adapt_method "dds" \
            -i ./results_adapt
        done
    done
done