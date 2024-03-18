from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from inject_vida import inject_trainable_vida
from time import time
import logging


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (384, 384, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher, alpha_vida):#, iteration):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    # return ema_model
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        if "vida_" in name:
            ema_param.data[:] = alpha_vida * ema_param[:].data[:] + (1 - alpha_vida) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class ViDA(nn.Module):
    """ViDA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, ema=0.99, ema_vida = 0.99, rst_m=0.1, anchor_thr=0.9, unc_thr = 0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "ViDA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.alpha_teacher = ema
        self.alpha_vida = ema_vida
        self.rst = rst_m
        self.thr = unc_thr

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def set_scale(self, update_model, high, low):
        for name, module in update_model.named_modules():
            if hasattr(module, 'scale1'):
                module.scale1 = low.item()
            elif hasattr(module, 'scale2'):
                module.scale2 = high.item()
        # print('2')
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        self.model_ema.eval()
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        # Augmentation-averaged Prediction
        N = 10 
        outputs_uncs = []
        for i in range(N):
            outputs_  = self.model_ema(self.transform(x)).detach()
            outputs_uncs.append(outputs_)
        outputs_unc = torch.stack(outputs_uncs)
        variance = torch.var(outputs_unc, dim=0)
        uncertainty = torch.mean(variance) * 0.1
        if uncertainty>= self.thr:
            lambda_high = 1+uncertainty
            lambda_low = 1-uncertainty
        else:
            lambda_low = 1+uncertainty
            lambda_high = 1-uncertainty
        self.set_scale(update_model = model, high = lambda_high, low = lambda_low)
        self.set_scale(update_model = self.model_ema, high = lambda_high, low = lambda_low)
        standard_ema = self.model_ema(x)
        outputs = self.model(x)
        # Student update
        loss = (softmax_entropy(outputs, standard_ema.detach())).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher= self.alpha_teacher, alpha_vida = self.alpha_vida)
        # Stochastic restore
        if True:
            for npp, p in model.named_parameters():
                if p.requires_grad:
                    mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                    with torch.no_grad():
                        p.data = self.model_state[npp] * mask + p * (1.-mask)

            # for nm, m  in self.model.named_modules():
            #     for npp, p in m.named_parameters():
            #         if npp in ['weight', 'bias'] and p.requires_grad:
            #             mask = (torch.rand(p.shape)<self.rst).float().cuda() 
            #             with torch.no_grad():
            #                 p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return standard_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    vida_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'vida_' in name:
            vida_params_list.append(param)
        else:
            model_params_lst.append(param)     
    return model_params_lst, vida_params_list


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, cfg):
    """Configure model for use with tent."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cpu()
    vida_params, vida_names = inject_trainable_vida(model = model, target_replace_module = ["CrossAttention", "Attention"], \
            r = cfg.TEST.vida_rank1, r2 = cfg.TEST.vida_rank2)
    if cfg.TEST.ckpt!=None:
        checkpoint = torch.load(cfg.TEST.ckpt)
        model.load_state_dict(checkpoint['model'], strict=True)

    # if cfg.TEST.ckpt!=None:
    #     checkpoint = torch.load(cfg.TEST.ckpt)
    #     model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.train()
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
