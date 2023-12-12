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
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
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
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

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
    times = get_schedule_jump(config.time_travel.T_sampling, 
                              config.time_travel.travel_length, 
                              config.time_travel.travel_repeat,
                              )
    time_pairs = list(zip(times[:-1], times[1:]))
    # reverse diffusion sampling
    for i, j in tqdm(time_pairs):
        i, j = i*skip, j*skip
        if j<0: j=-1

        if j < i: # normal sampling 
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

            # 1. MCG gradient step to x0hat
            residual = A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            residual_norm = torch.linalg.norm(residual) ** 2
            # residual_norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xt)[0]

            # 4. DDIM sampling (add noise back, point towards next step)
            c1 = eta * ((1 - at / at_next) * (1 - at_next)/(1 - at)).sqrt()
            c2 = (1 - at_next - c1**2).sqrt()
            # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et - alpha * norm_grad
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et - at.sqrt() * norm_grad

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

        # turn off gradient
        xt.detach_()

    return [xs[-1]], [x0_preds[-1]]



def dds_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None,
                  alpha_schedule=None, coeff_schedule="ddim"):
    # setup iteration variables
    skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = get_schedule_jump(config.time_travel.T_sampling, 
                              config.time_travel.travel_length, 
                              config.time_travel.travel_repeat,
                              )
    time_pairs = list(zip(times[:-1], times[1:]))
    # reverse diffusion sampling
    for i, j in tqdm(time_pairs):
        i, j = i*skip, j*skip
        if j<0: j=-1

        if j < i: # normal sampling 
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            with torch.no_grad():
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
            x0_t.requires_grad_()

            # 1. MCG gradient step to x0hat
            residual = A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            residual_norm = torch.linalg.norm(residual) ** 2
            # residual_norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=x0_t)[0]

            # 4. DDIM sampling (add noise back, point towards next step)
            c1 = eta * ((1 - at / at_next) * (1 - at_next)/(1 - at)).sqrt()
            c2 = (1 - at_next - c1**2).sqrt()
            # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et - alpha * norm_grad
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et - at.sqrt() * norm_grad

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

        # turn off gradient
        xt.detach_()

    return [xs[-1]], [x0_preds[-1]]



def pigdm_diffusion(x, model, b, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None,
                    coeff_schedule="ddrm"):
    """
    My reproducing of Kawar et al. \PiGDM
    """
    # setup iteration variables
    skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = get_schedule_jump(config.time_travel.T_sampling, 
                              config.time_travel.travel_length, 
                              config.time_travel.travel_repeat,
                              )
    time_pairs = list(zip(times[:-1], times[1:]))
    # reverse diffusion sampling
    for i, j in tqdm(time_pairs):
        i, j = i*skip, j*skip
        if j<0: j=-1 

        if j < i: # normal sampling 
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

            # 0. x0hat prediction
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # turn on gradient
            # x0_t.requires_grad_()

            # 1. PIGDM VJP (noiseless)
            if sigma_y == 0.:
                mat = A_funcs.A_pinv(y.reshape(y.size(0), -1)) - A_funcs.A_pinv(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)))
                mat_x = (mat.detach() * x0_t.reshape(x0_t.size(0), -1)).sum()
                guidance = torch.autograd.grad(outputs=mat_x, inputs=x0_t)[0]
            # 1. PIGDM VJP (noisy)
            else:
                mat = A_funcs.A_pinv(y.reshape(y.size(0), -1)) - A_funcs.A_pinv(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)))
                mat_x = (mat.detach() * x0_t.reshape(x0_t.size(0), -1)).sum()
                guidance = torch.autograd.grad(outputs=mat_x, inputs=x0_t)[0]
            x0_t += guidance
            # turn off gradient
            x0_t.detach_()
            xt.detach_()
            et = et.detach()
            # x0_t += at.sqrt() * guidance

            # 4. DDIM sampling (add noise back, point towards next step)
            if coeff_schedule == "ddrm":
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            elif coeff_schedule == "ddim":
                c1 = eta * ((1 - at / at_next) * (1 - at_next)/(1 - at)).sqrt()
                c2 = (1 - at_next - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
