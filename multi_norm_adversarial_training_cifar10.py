#!/usr/bin/env python
# coding: utf-8

# # CS 521 HW3, Problem 3: Multi-Norm Adversarial Training
# 
# Use Google Colab with a GPU: **Runtime > Change runtime type > T4 GPU**.
# 
# All the code is in this notebook. Complete each place that has a `TODO` and `raise NotImplementedError`.
# Do not change the data, the model, the attacks, the evaluation or the default hyperparameters,
# unless the homework tells you to change them.
# 
# **Threat model.** For each test image $x$, the perturbation region is
# $B_\infty(x, 8/255) \cup B_2(x, 0.5) \cup B_1(x, 12)$.
# The union accuracy is the percentage of test images that are robust against all three attacks simultaneously.

# # Set up for dataset and model
# 
# Download the base model, a subset of 10,000 CIFAR-10 training images (1,000 for each class),
# and the first 1,000 CIFAR-10 test images from Google Drive.

# In[1]:


get_ipython().system('pip install gdown')


# In[2]:


get_ipython().system('gdown 17vWkPpVRFUYAxWv3_NEj1gV7Lxh-Cnes -O linf_base.pt')
get_ipython().system('gdown 1LoY_o51zNQkrkfvAlGE-44g9G-c38OJe -O cifar10_train_10k.pt')
get_ipython().system('gdown 1bElxIZ6A708IOGaTMcDq_AsCpC9QQseA -O cifar10_test_1k.pt')


# In[3]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install torch')


# In[4]:


get_ipython().system('pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128')


# In[5]:


import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

NORMS = ("linf", "l2", "l1")
EPS = {"linf": 8 / 255, "l2": 0.5, "l1": 12.0}


# In[6]:


def load_data(path):
    # The files store images as uint8. Convert them to floats in [0, 1] on the GPU.
    x, y = torch.load(path)
    return x.float().div(255).to(device), y.to(device)


# Training data: 10,000 images. Test data: the first 1,000 CIFAR-10 test images.
x_train, y_train = load_data("cifar10_train_10k.pt")
x_test, y_test = load_data("cifar10_test_1k.pt")
print("train:", tuple(x_train.shape), "test:", tuple(x_test.shape))


def augment(x):
    # Random horizontal flip and random crop (padding 4) for each image.
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n, 1, 1), device=x.device) + torch.arange(32, device=x.device).view(1, 32, 1)
    j = torch.randint(0, 9, (n, 1, 1), device=x.device) + torch.arange(32, device=x.device).view(1, 1, 32)
    return padded[torch.arange(n, device=x.device).view(n, 1, 1), :, i, j].permute(0, 3, 1, 2).contiguous()


def batches(x, y, batch_size, shuffle):
    idx = torch.randperm(len(x), device=x.device) if shuffle else torch.arange(len(x), device=x.device)
    for k in range(0, len(x), batch_size):
        j = idx[k:k + batch_size]
        yield x[j], y[j]


# In[7]:


# PreActResNet-18 with half width (2.8M parameters). The input is an image in [0, 1].
class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet18(nn.Module):
    def __init__(self, width=32, num_classes=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        layers, in_planes = [], width
        for mult, stride in [(1, 1), (1, 1), (2, 2), (2, 1), (4, 2), (4, 1), (8, 2), (8, 1)]:
            layers.append(PreActBlock(in_planes, mult * width, stride))
            in_planes = mult * width
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_planes)
        self.linear = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        out = self.layers(self.conv1((x - self.mean) / self.std))
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1).flatten(1)
        return self.linear(out)


def make_model():
    return PreActResNet18().to(device)


def load_base_model():
    model = make_model()
    model.load_state_dict(torch.load("linf_base.pt", map_location=device))
    return model


# # Implement the Attacks
# 
# Each attack is untargeted, uses the cross-entropy loss, and returns `x_adv` with
# $\|x_{adv} - x\|_p \leq \epsilon$ and $x_{adv} \in [0, 1]$. The attacks must not change the mode (`train` or `eval`) of the model.
# 
# * `pgd_linf`: start at a uniformly random point in the $\ell_\infty$ ball. For each step, add `alpha * sign(grad)`,
#   clip the perturbation to $[-\epsilon, \epsilon]$, and clip the image to $[0, 1]$.
# * `pgd_l2`: start at a random point in the $\ell_2$ ball. For each step, add `alpha * grad / ||grad||_2`,
#   scale the perturbation back to the $\ell_2$ ball if its norm is larger than $\epsilon$, and clip the image to $[0, 1]$.
# 
# You can use your code from HW1. The $\ell_1$ attack (sparse $\ell_1$ descent, Tramer & Boneh 2019) is complete.

# In[8]:


def clip_box(x, delta):
    # Make x + delta an image in [0, 1]. This never increases |delta_i|.
    return (x + delta).clamp(0, 1) - x


def input_grad(model, x, delta, y):
    delta = delta.detach().requires_grad_(True)
    loss = F.cross_entropy(model(x + delta), y, reduction="sum")
    (grad,) = torch.autograd.grad(loss, delta)
    return grad.detach()


def l2_norm(v):
    return v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)


def pgd_linf(model, x, y, eps, alpha, steps, rand_init=True):
    # TODO: implement L-infinity PGD (use input_grad and clip_box)
    x = x.detach()
    if rand_init:
        delta = (torch.rand_like(x) * 2 - 1) * eps
    else:
        delta = torch.zeros_like(x)
    delta = clip_box(x, delta) 

    for _ in range(steps):
        grad = input_grad(model, x, delta, y)
        delta = delta + alpha * grad.sign()
        delta = delta.clamp(-eps, eps)
        delta = clip_box(x, delta)
    return (x + delta).detach()


def pgd_l2(model, x, y, eps, alpha, steps, rand_init=True):
    # TODO: implement L2 PGD (use input_grad, l2_norm and clip_box)
    x = x.detach()
    if rand_init:
        delta = torch.randn_like(x)
        delta = delta / (l2_norm(delta) + 1e-12)
        r = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)
        delta = delta * r * eps
    else:
        delta = torch.zeros_like(x)
    delta = clip_box(x, delta)

    for _ in range(steps):
        grad = input_grad(model, x, delta, y)
        delta = delta + alpha * grad / (l2_norm(grad) + 1e-12)
        scale = torch.clamp(eps / (l2_norm(delta) + 1e-12), max=1.0)
        delta = delta * scale
        delta = clip_box(x, delta)

    return (x + delta).detach()


# In[9]:


def project_l1(delta, eps):
    # Projection of each image of delta onto the L1 ball of radius eps (Duchi et al., 2008).
    v = delta.flatten(1)
    a = v.abs()
    outside = a.sum(dim=1) > eps
    if not outside.any():
        return delta
    mu, _ = a.sort(dim=1, descending=True)
    cumsum = mu.cumsum(dim=1)
    idx = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)
    rho = ((mu - (cumsum - eps) / idx > 0) * idx).argmax(dim=1, keepdim=True)
    theta = (cumsum.gather(1, rho) - eps) / (rho + 1).to(v.dtype)
    w = (a - theta).clamp_min(0) * v.sign()
    return torch.where(outside.unsqueeze(1), w, v).view(delta.shape)


def pgd_l1(model, x, y, eps, alpha, steps, rand_init=True, k=20):
    # Each step changes only the k coordinates with the largest gradient.
    # Coordinates at the image boundary that point outward are skipped.
    delta = torch.zeros_like(x)
    if rand_init:
        delta = torch.distributions.Laplace(0.0, 1.0).sample(x.shape).to(x.device)
        r = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        delta = delta / delta.flatten(1).abs().sum(1).view(-1, 1, 1, 1) * eps * r
    delta = clip_box(x, delta)
    for _ in range(steps):
        grad = input_grad(model, x, delta, y).flatten(1)
        x_cur = (x + delta).flatten(1)
        blocked = ((grad > 0) & (x_cur >= 1)) | ((grad < 0) & (x_cur <= 0))
        grad = grad.masked_fill(blocked, 0)
        top = grad.abs().topk(k, dim=1).indices
        step = torch.zeros_like(grad).scatter(1, top, grad.gather(1, top).sign())
        delta = clip_box(x, delta + alpha * (step / k).view_as(x))
        delta = project_l1(delta, eps)
    return (x + delta).detach()


ATTACKS = {"linf": pgd_linf, "l2": pgd_l2, "l1": pgd_l1}


def attack(model, x, y, norm, steps=10):
    # The attack for one norm with the default eps and step size.
    alpha = EPS["l1"] / steps ** 0.5 if norm == "l1" else 2.5 * EPS[norm] / steps
    return ATTACKS[norm](model, x, y, EPS[norm], alpha, steps)


# # Evaluate Single and Multi-Norm Robust Accuracy
# 
# `test_model` gives the clean accuracy, the robust accuracy for each norm, and the union accuracy (in %) on the
# 1,000 test images. An image is robust for a norm if the model is correct on the clean image and on the result
# of each restart of the attack. An image is robust for the union if it is robust for all three norms simultaneously.
# Use the default settings (`steps=50, restarts=2`) for all the numbers in your report.

# In[10]:


def test_model(model, steps=50, restarts=2, batch_size=250):
    model.eval()
    torch.manual_seed(0)  # the same random starts of the attacks for each model, so the results are comparable
    start = time.time()
    correct = {k: 0 for k in ("clean", *NORMS, "union")}
    for i in range(0, len(x_test), batch_size):
        x, y = x_test[i:i + batch_size], y_test[i:i + batch_size]
        with torch.no_grad():
            clean = model(x).argmax(1) == y
        union = clean.clone()
        for norm in NORMS:
            robust = clean.clone()
            for _ in range(restarts):
                x_adv = attack(model, x, y, norm, steps)
                with torch.no_grad():
                    robust &= model(x_adv).argmax(1) == y
            correct[norm] += robust.sum().item()
            union &= robust
        correct["clean"] += clean.sum().item()
        correct["union"] += union.sum().item()
    acc = {k: 100.0 * v / len(y_test) for k, v in correct.items()}
    print("  ".join(f"{k} {v:.1f}" for k, v in acc.items()) + f"   ({time.time() - start:.0f} s)")
    return acc


# In[11]:


# The base model: L-infinity PGD adversarial training for 30 epochs on all 50,000 training images.
test_model(load_base_model())


# # Adversarial Training
# 
# `adv_train(model, loss_fn, ...)` fine-tunes `model` for `epochs` epochs on the training subset.
# For each batch, it calls `loss_fn(model, x, y, **kwargs)` and does one optimizer step on the loss.
# 
# A loss function makes the adversarial examples itself. Put the model in `eval` mode for the attack
# (so that the attack does not change the BatchNorm statistics), and in `train` mode before you calculate the loss.
# 
# If `gp_beta` is set, `adv_train` does the gradient projection of RAMP (part (b)) before the optimizer step.

# In[12]:


def adv_train(model, loss_fn, epochs=10, lr=0.01, batch_size=128, gp_beta=None, seed=0, **kwargs):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(x_train) // batch_size))
    params = list(model.parameters())
    start = time.time()
    for epoch in range(epochs):
        losses = []
        for x, y in batches(x_train, y_train, batch_size, shuffle=True):
            if len(y) < batch_size:  # skip the last, smaller batch
                continue
            x = augment(x)
            loss = loss_fn(model, x, y, **kwargs)
            opt.zero_grad()
            if gp_beta is None:
                loss.backward()
            else:
                # RAMP (part (b)): combine the adversarial gradient g_a and the natural gradient g_n
                # for each parameter tensor. gradient_projection is defined in part (b).
                g_a = torch.autograd.grad(loss, params)
                g_n = torch.autograd.grad(F.cross_entropy(model(x), y), params)
                for p, ga, gn in zip(params, g_a, g_n):
                    p.grad = gp_beta * gradient_projection(gn, ga) + (1 - gp_beta) * ga
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.3f} ({time.time() - start:.0f} s)")
    model.eval()
    return model


# # Part (a): Single-Norm Adversarial Training
# 
# Complete `single_norm_loss`: attack the batch with the norm `norm` (use `attack`), and return the
# cross-entropy loss of the model on the adversarial images.

# In[13]:


def single_norm_loss(model, x, y, norm):
    # TODO: put the model in eval mode, attack with `norm`, put the model in train mode, and return the loss
    model.eval()
    x_adv = attack(model, x, y, norm)
    model.train()
    return F.cross_entropy(model(x_adv), y)


# In[14]:


model_linf = adv_train(load_base_model(), single_norm_loss, norm="linf")
test_model(model_linf)


# In[15]:


model_l2 = adv_train(load_base_model(), single_norm_loss, norm="l2")
test_model(model_l2)


# In[16]:


model_l1 = adv_train(load_base_model(), single_norm_loss, norm="l1")
test_model(model_l1)


# # Part (b): Multi-Norm Adversarial Training (RAMP)
# 
# * `ramp_loss`: attack with $q = \ell_\infty$ and $r = \ell_1$. The loss is the mean of
#   $\max(\mathcal{L}_{CE}(x^q), \mathcal{L}_{CE}(x^r))$ plus `lam` times $\mathcal{L}_{KL} = KL(p_q \| p_r)$.
#   Calculate $\mathcal{L}_{KL}$ only on the images that the model classifies correctly under the $\ell_q$ attack.
# * `gradient_projection(g_n, g_a)`: $\cos(g_n, g_a) \cdot g_n$ if $\cos(g_n, g_a) > 0$, else $0$.
# 
# `adv_train(..., gp_beta=0.5)` then uses `p.grad = gp_beta * GP(g_n, g_a) + (1 - gp_beta) * g_a` for each parameter
# tensor, where $g_a$ is the gradient of `ramp_loss` and $g_n$ is the gradient of the cross-entropy loss on the clean images.

# In[17]:


def ramp_loss(model, x, y, lam=0.5):
    # TODO: in eval mode, attack with linf and l1; in train mode, return the max of the two losses plus lam * KL
    model.eval()
    x_q = attack(model, x, y, "linf")
    x_r = attack(model, x, y, "l1")
    model.train()

    out_q = model(x_q)
    out_r = model(x_r)

    ce_q = F.cross_entropy(out_q, y, reduction="none")
    ce_r = F.cross_entropy(out_r, y, reduction="none")
    ce_max = torch.max(ce_q, ce_r).mean()

    log_p_q = F.log_softmax(out_q, dim=1)
    log_p_r = F.log_softmax(out_r, dim=1)
    kl = (log_p_q.exp() * (log_p_q - log_p_r)).sum(dim=1)
    mask = (out_q.argmax(dim=1) == y).float()
    kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)

    return ce_max + lam * kl

def gradient_projection(g_n, g_a):
    # TODO: GP(g_n, g_a) for one parameter tensor
    a, b = g_n.flatten(), g_a.flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    return cos.clamp(min=0.0) * g_n


# In[18]:


model_ramp = adv_train(load_base_model(), ramp_loss, gp_beta=0.5)
test_model(model_ramp)


# # Part (c): Your Method
# 
# Write the training loss of your method in `my_loss`, and train with `adv_train(load_base_model(), my_loss)`.
# If your method needs a different update (not the gradient of one loss), write your own copy of `adv_train`.
# Run your method and RAMP with `seed=0` and `seed=1`.
# Then evaluate your best model again with stronger attacks: `test_model(model, steps=200, restarts=5)`.

# In[23]:


def my_loss(model, x, y, q = 0.3):
    # randomly drop each norm with probability q
    # and train on the max loss over the norms that survive.
    model.eval()
    advs = [attack(model, x, y, n) for n in ("linf", "l2", "l1")]
    model.train()

    ces = torch.stack([F.cross_entropy(model(a), y, reduction="none") for a in advs], dim=1)

    keep = torch.rand_like(ces) > q
    forced = torch.randint(0, 3, (len(y), 1), device=y.device)
    keep.scatter_(1, forced, True)

    return ces.masked_fill(~keep, float("-inf")).max(dim=1).values.mean()


# In[25]:


model_mine = adv_train(load_base_model(), my_loss)
test_model(model_mine)


# In[ ]:





# In[21]:


# problem 1
def standard_loss(model, x, y):
    return F.cross_entropy(model(x), y)

model_std = adv_train(make_model(), standard_loss, epochs=5)
model_adv = adv_train(make_model(), single_norm_loss, epochs=5, norm="linf")

# -eval aross epsilons
def robust_acc(model, eps, steps, bs=250):
    model.eval()
    correct = 0
    for x, y in batches(x_test, y_test, bs, shuffle=False):
        if eps == 0:
            x_adv = x
        elif steps == 1:   # untargeted FGSM = one PGD step, no random start
            x_adv = pgd_linf(model, x, y, eps, alpha=eps, steps=1, rand_init=False)
        else:              # PGD
            x_adv = pgd_linf(model, x, y, eps, alpha=eps / 4, steps=steps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
    return 100 * correct / len(x_test)

eps_list = [0, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255]
print("eps*255 | std FGSM | std PGD | adv FGSM | adv PGD")
for e in eps_list:
    print(f"{e*255:6.0f}  | {robust_acc(model_std, e, 1):7.1f}  | {robust_acc(model_std, e, 20):6.1f}  | "
          f"{robust_acc(model_adv, e, 1):7.1f}  | {robust_acc(model_adv, e, 20):6.1f}")


# In[ ]:





# In[26]:


# problem 3
model_mine_s1 = adv_train(load_base_model(), my_loss, seed=1)
res_mine_s1 = test_model(model_mine_s1)

model_ramp_s1 = adv_train(load_base_model(), ramp_loss, gp_beta=0.5, seed=1)
res_ramp_s1 = test_model(model_ramp_s1)


# In[27]:


# ablation
model_max = adv_train(load_base_model(), my_loss, seed=0, p=0.0)
res_max = test_model(model_max)


# In[28]:


# RAMP
model_ramp_13 = adv_train(load_base_model(), ramp_loss, gp_beta=0.5, seed=0, epochs=13)
res_ramp_13 = test_model(model_ramp_13)


# In[31]:


# strong evaluation
res_strong = test_model(model_mine_s1, steps=200, restarts=5)


# In[30]:


# means over the two seeds
res_mine_s0  = {'clean': 78.0, 'linf': 42.7, 'l2': 62.4, 'l1': 47.1, 'union': 40.7}
res_ramp_s0  = {'clean': 79.3, 'linf': 41.0, 'l2': 61.5, 'l1': 44.5, 'union': 38.9}

def mean_res(a, b):
    return {k: round((a[k] + b[k]) / 2, 1) for k in a}

print("mine mean:", mean_res(res_mine_s0, res_mine_s1))
print("ramp mean:", mean_res(res_ramp_s0, res_ramp_s1))


# In[ ]:





# In[ ]:




