import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
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
            if cls_fn == None:
                et = model(xt, t)
            else:
                classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            x0_t_hat = x0_t - A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())

            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def dps_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None,
                  alpha_schedule=None, coeff_schedule="ddim"):
    # setup iteration variables
    skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
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

        # turn on gradient
        xt.requires_grad_()
        if cls_fn == None:
            et = model(xt, t)
        else:
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        # 0. x0hat prediction
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_t = torch.clip(x0_t, -1.0, 1.0)

        # 1. DPS gradient step to x0hat
        residual = A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
        residual_norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xt)[0]

        # 4. DDIM sampling (add noise back, point towards next step)
        c1 = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
        c2 = (1 - at_next - c1**2).sqrt()
        
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et - norm_grad

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

        # turn off gradient
        xt.detach_()

    return [xs[-1]], [x0_preds[-1]]