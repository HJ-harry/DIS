from typing import Optional
import torch 
import torch.nn as nn
from torch import Tensor

from adapt.lora import inject_trainable_lora_extended

import itertools

def tv_loss(x):

    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.mean(dh[..., :-1, :] + dw[..., :, :-1])

def _score_model_adpt(
    score: nn.Module, 
    im_size: Optional[int] = None, 
    method: str = 'decoder',
    r: int = 4,
    ) -> None:
    
    for name, param in score.named_parameters():
        param.requires_grad = False

    if method == 'full':
        for name, param in score.named_parameters():
            param.requires_grad = True
    elif method == 'decoder': 
        for name, param in score.out.named_parameters():
            if not "emb_layers" in name:
                param.requires_grad = True
        for name, param in score.output_blocks.named_parameters():
            if not "emb_layers" in name:
                param.requires_grad = True
    elif method == 'lora':
        """ 
        Implement LoRA: https://arxiv.org/pdf/2106.09685.pdf 

        Adding LoRA modules to nn.Conv1d, nn.Conv2d (should we also add to nn.Linear?)
         + retraining all biases (only a negligible number of parameters)
        """
        score.requires_grad_(False)

        for name, param in score.named_parameters():
            if "bias" in name and not "emb_layers" in name:
                param.requires_grad = True

        lora_params, train_names = inject_trainable_lora_extended(score, r=r) 

        new_num_params = sum([p.numel() for p in score.parameters()])
        trainable_params = sum([p.numel() for p in score.parameters() if p.requires_grad])
        print(f'% of trainable params: {trainable_params/new_num_params*100}')
    else: 
        raise NotImplementedError
    
    return score