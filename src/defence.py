"""
defence.py
----------
Implements defences against adversarial attacks.

Three defences:
1. Adversarial Training — retrain the model on adversarial examples
   - Most effective known defence
   - Trade-off: slightly lower clean accuracy, much higher robust accuracy

2. Input Preprocessing — denoise the input before classification
   - Gaussian smoothing reduces high-frequency perturbations
   - No retraining needed, works on any model
   - Less effective than adversarial training

3. Ensemble Defence — use multiple models and take majority vote
   - Harder to fool all models simultaneously
   - Computationally expensive
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from src.model import MnistCNN, DEVICE, save_model
from src.attacks import fgsm_attack, pgd_attack


def adversarial_training(model, train_loader, epochs=5, epsilon=0.3, lr=0.001):
    """
    Adversarial Training (Madry et al., 2017)
    ------------------------------------------
    During training, for each batch:
    1. Generate adversarial examples using PGD
    2. Train on the MIXTURE of clean and adversarial examples

    This teaches the model to be robust to perturbations.
    The resulting model sacrifices ~1-2% clean accuracy
    for dramatically improved adversarial robustness.

    This is currently the GOLD STANDARD defence.
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Adversarial training for {epochs} epochs...")
    print("(Slower than standard training — generating attacks per batch)")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_clean = 0
        correct_adv = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Generate PGD adversarial examples (7 steps for speed).
            # Step size 2.5 * eps / steps (Madry et al.) lets the attack
            # actually reach the edge of the epsilon-ball in 7 steps.
            model.eval()
            adv_data = pgd_attack(model, data, target, epsilon=epsilon,
                                  alpha=2.5 * epsilon / 7, num_steps=7)
            model.train()

            # Train on mix of clean and adversarial
            optimizer.zero_grad()

            # Clean loss
            clean_output = model(data)
            clean_loss = criterion(clean_output, target)

            # Adversarial loss
            adv_output = model(adv_data)
            adv_loss = criterion(adv_output, target)

            # Combined loss (equal weight)
            loss = 0.5 * clean_loss + 0.5 * adv_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_clean += clean_output.argmax(1).eq(target).sum().item()
            correct_adv += adv_output.argmax(1).eq(target).sum().item()
            total += len(target)

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Clean={correct_clean/total:.1%}, Adv={correct_adv/total:.1%}")

    return model


def gaussian_smoothing_defence(images, sigma=1.0):
    """
    Gaussian Smoothing Preprocessing Defence
    -----------------------------------------
    Apply Gaussian blur to input images before classification.

    Why it works: adversarial perturbations are often high-frequency noise.
    Smoothing removes high-frequency components, reducing perturbation effectiveness.

    Why it fails against adaptive attacks: an attacker who knows you're using
    smoothing can generate perturbations that survive smoothing. This is why
    preprocessing defences are generally considered weaker than adversarial training.

    Args:
        images: input tensor
        sigma: smoothing strength (higher = more blurring)
    """
    import torch.nn.functional as F

    # Create Gaussian kernel
    kernel_size = 5
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # shape: 1x1xKxK

    # Apply to each channel
    images_np = images.cpu()
    smoothed = F.conv2d(images_np, kernel_2d, padding=kernel_size//2)
    return smoothed.to(DEVICE)


def evaluate_defence(model, clean_images, adv_images, labels,
                     defence_fn=None, defence_name="None"):
    """
    Evaluate a defence strategy:
    - Apply defence preprocessing (if any)
    - Measure accuracy on clean and adversarial images
    """
    model.eval()
    clean_images = clean_images.to(DEVICE)
    adv_images = adv_images.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():
        # Apply defence preprocessing
        if defence_fn is not None:
            clean_defended = defence_fn(clean_images)
            adv_defended = defence_fn(adv_images)
        else:
            clean_defended = clean_images
            adv_defended = adv_images

        clean_pred = model(clean_defended).argmax(dim=1)
        adv_pred = model(adv_defended).argmax(dim=1)

        clean_acc = clean_pred.eq(labels).float().mean().item()
        adv_acc = adv_pred.eq(labels).float().mean().item()

    print(f"\nDefence: {defence_name}")
    print(f"  Clean accuracy   : {clean_acc:.1%}")
    print(f"  Adversarial acc  : {adv_acc:.1%}")
    print(f"  Recovery         : +{(adv_acc - (1-clean_acc)):.1%} from no-defence baseline")

    return {"defence": defence_name, "clean_acc": clean_acc, "adv_acc": adv_acc}
