import torch
from tqdm import tqdm
from torch.autograd import grad
import torchvision.utils as tvu
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

from functools import partial

from adapt.lora import tune_lora_scale
from adapt.adaptation import tv_loss


class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def precompute_posterior_variance(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.zeros(1).to(betas.device), alphas_cumprod[:-1]], dim=0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return posterior_variance

def compute_posterior_variance(posterior_variance, t):
    return posterior_variance.index_select(0, t).view(-1, 1, 1, 1)


def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def ddnm_diffusion(x, model, b, eta, A, Ap, y, cls_fn=None, classes=None, 
                   args=None, config=None, save_progress=False):
    # setup iteration variables
    skip = config.diffusion.num_diffusion_timesteps // args.T_sampling
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = range(0, 1000, skip)
    times_next = [-1] + list(times[:-1])
    times_pair = zip(reversed(times), reversed(times_next))
    
    # reverse diffusion sampling
    for i, j in tqdm(times_pair):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        # 0. adaptation
        # =================================================
        torch.autograd.set_detect_anomaly(True)
        model.eval()
        optim = torch.optim.Adam(model.parameters(), lr=args.adapt_lr)
        for i in range(args.adapt_iter):
            optim.zero_grad()
            
            xt.requires_grad_()
            et = model(xt.float(), t.float())
            et = et[:, :1]
            
            xhat0 = (xt - et * (1 - at).sqrt()) / at.sqrt()
            if args.use_dc_before_adapt:
                if args.dc_before_adapt_method == "dds":
                    residual = torch.linalg.norm(A(xhat0) - y) ** 2
                    # gradient w.r.t. xhat0
                    gradient = grad(outputs=residual, inputs=xhat0)[0]
                elif args.dc_before_adapt_method == "dps":
                    residual = torch.linalg.norm(A(xhat0) - y)
                    # gradient w.r.t. xt (product with U-Net Jacobian)
                    gradient = grad(outputs=residual, inputs=xt, retain_graph=True)[0]
                # gradient step
                xhat0 = xhat0 - gradient * 1.0
            loss = torch.mean((A(xhat0) - y)**2) + float(args.tv_penalty) * tv_loss(xhat0)
            loss.backward()
            optim.step()
            xt.detach()
        model.eval()
        # end adaptation
        # =================================================
        
        # ddnm sampling
        with torch.no_grad():
            # 1. NFE
            et = model(xt, t)
            et = et[:, 0]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # x_sv = clear(x0_t)
            # plt.imsave(str(args.image_folder / f"progress" / f"reco_{str(j).zfill(3)}.png"), x_sv, cmap='gray')

            if args.use_additional_dc:
                # 2. (optional) projection data consistency
                x0_t = x0_t - Ap(A(x0_t) - y)

            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            
            if args.ddim_eps_choice == "old":
                tune_lora_scale(model, alpha=0.0)
                et_noise = model(xt.float(), t.float())
                et_noise = et_noise[:, :1]
                # turn back on
                tune_lora_scale(model, alpha=1.0)
            else:
                et_noise = et

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

