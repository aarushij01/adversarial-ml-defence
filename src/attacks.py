"""
attacks.py
----------
Implements adversarial attacks from scratch.

Three attacks:
1. FGSM — Fast Gradient Sign Method (Goodfellow et al., 2014)
2. PGD  — Projected Gradient Descent (Madry et al., 2017)
3. CW   — Carlini-Wagner L2 simplified (Carlini & Wagner, 2017)
"""

import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inputs are normalised with MNIST mean/std (0.1307, 0.3081), so the valid
# pixel range [0, 1] maps to [PIX_MIN, PIX_MAX] in model-input space.
# Adversarial images are clamped to this range so they stay real images.
PIX_MIN = (0.0 - 0.1307) / 0.3081
PIX_MAX = (1.0 - 0.1307) / 0.3081
MNIST_STD = 0.3081


def to_input_eps(pixel_eps):
    """Convert an epsilon in [0,1] pixel units to model-input (normalised) units."""
    return pixel_eps / MNIST_STD


def fgsm_attack(model, images, labels, epsilon=0.3):
    """
    Fast Gradient Sign Method (FGSM)
    x_adv = x + epsilon * sign(∇_x L(model(x), y))

    Single-step attack — fast but suboptimal.
    """
    images = images.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)
    images.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = model(images)
    loss = criterion(output, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # FGSM step
    gradient = images.grad.data
    perturbation = epsilon * gradient.sign()
    adversarial = (images + perturbation).detach()
    adversarial = torch.clamp(adversarial, PIX_MIN, PIX_MAX)

    return adversarial


def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_steps=40, random_start=True):
    """
    Projected Gradient Descent (PGD)
    Iterative attack — stronger than FGSM.

    x_t = Proj_{B(x,eps)}(x_{t-1} + alpha * sign(grad_x L))
    """
    images = images.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Random start within epsilon-ball
    if random_start:
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(images + delta, PIX_MIN, PIX_MAX).detach()
    else:
        x_adv = images.clone().detach()

    for step in range(num_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # Forward pass — ensure gradients are enabled
        output = model(x_adv)
        loss = criterion(output, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Check gradient exists
        if x_adv.grad is None:
            break

        gradient = x_adv.grad.data.clone()

        # Gradient step
        with torch.no_grad():
            x_adv = x_adv + alpha * gradient.sign()
            # Project back into epsilon-ball
            delta = torch.clamp(x_adv - images, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(images + delta, PIX_MIN, PIX_MAX)

    return x_adv.detach()


def cw_attack_l2(model, images, labels, c=10.0, kappa=0, max_iter=100, lr=0.05):
    """
    Carlini-Wagner L2 Attack (simplified)
    Minimises ||delta||_2 + c * max(Z_y - max_{j!=y} Z_j + kappa, 0)

    For each image we keep the SMALLEST perturbation that actually fools
    the model. Images the attack never fools are returned unchanged
    (counted as attack failures).
    """
    images = images.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)

    delta = torch.zeros_like(images, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_adv = images.clone()
    best_l2 = torch.full((images.size(0),), float("inf"), device=DEVICE)

    for iteration in range(max_iter):
        x_adv = torch.clamp(images + delta, PIX_MIN, PIX_MAX)
        output = model(x_adv)

        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        correct_logit = (output * one_hot).sum(dim=1)
        wrong_logit = (output * (1 - one_hot) - one_hot * 1e4).max(dim=1)[0]

        l2 = ((x_adv - images) ** 2).flatten(1).sum(dim=1)
        attack_loss = torch.clamp(correct_logit - wrong_logit + kappa, min=0)
        loss = (l2 + c * attack_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            fooled = output.argmax(dim=1) != labels
            improved = fooled & (l2 < best_l2)
            best_l2[improved] = l2[improved]
            best_adv[improved] = x_adv[improved].detach()

    return best_adv


def evaluate_attack(model, clean_images, adv_images, labels):
    """Evaluate attack effectiveness."""
    model.eval()
    clean_images = clean_images.clone().detach().to(DEVICE)
    adv_images = adv_images.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)

    with torch.no_grad():
        clean_output = model(clean_images)
        adv_output = model(adv_images)

        clean_pred = clean_output.argmax(dim=1)
        adv_pred = adv_output.argmax(dim=1)

        clean_correct = clean_pred.eq(labels)
        adv_correct = adv_pred.eq(labels)

        clean_acc = clean_correct.float().mean().item()
        adv_acc = adv_correct.float().mean().item()
        attack_success = (clean_correct & ~adv_correct).float().mean().item()

        perturbation = (adv_images - clean_images).abs()
        mean_linf = perturbation.max(dim=-1)[0].max(dim=-1)[0].mean().item()
        mean_l2 = perturbation.pow(2).sum(dim=(1, 2, 3)).sqrt().mean().item()

    return {
        "clean_accuracy": round(clean_acc, 4),
        "adversarial_accuracy": round(adv_acc, 4),
        "attack_success_rate": round(attack_success, 4),
        "mean_linf_perturbation": round(mean_linf, 4),
        "mean_l2_perturbation": round(mean_l2, 4),
    }
