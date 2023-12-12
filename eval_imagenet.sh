#!/bin/bash

python main.py \
--ni \
--config imagenet_256.yml \
--path_y imagenet \
--eta 0.85 \
--deg "deblur_gauss" \
--deg_scale 8.0 \
--sigma_y 0.01 \
--T_sampling 1000 \
--method ddnm \
--add_noise \
-i ./results_230404