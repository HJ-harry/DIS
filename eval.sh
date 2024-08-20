#!/bin/bash


# DDNM
python main.py \
--ni \
--config imagenet_256.yml \
--path_y imagenet \
--eta 1.0 \
--deg "inpainting_random" \
--deg_scale 4 \
--sigma_y 0.0 \
--method "dps" \
-i ./results

