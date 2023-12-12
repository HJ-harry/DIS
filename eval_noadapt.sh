#!/bin/bash
python main.py \
--ni \
--config imagenet_512.yml \
--path_y imagenet \
--eta 0.85 \
--deg "sr_averagepooling" \
--deg_scale 8 \
--sigma_y 0.0 \
--method "dds" \
-i ./tmp