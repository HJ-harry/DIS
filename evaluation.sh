#!/bin/bash
python main.py \
--ni \
--config imagenet_256.yml \
--path_y imagenet \
--eta 0.85 \
--deg "deblur_gauss" \
--deg_scale 16 \
--sigma_y 0.01 \
-i imagenet_deblur_gauss \
--add_noise
