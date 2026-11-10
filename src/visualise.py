"""
visualise.py
------------
Creates all visualisations for the adversarial ML project.

Key plots:
1. Attack gallery — clean vs adversarial images side by side
2. Epsilon sweep — how attack strength affects accuracy
3. Defence comparison — bar chart of all defences
4. Perturbation magnification — amplified view of what changed
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

FIGURES_DIR = "reports/figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(FIGURES_DIR, exist_ok=True)


def denormalize(tensor):
    """Convert from normalised MNIST tensor back to [0,1] for display."""
    mean = 0.1307
    std = 0.3081
    return tensor * std + mean


def plot_attack_gallery(model, clean_images, adv_images_dict, labels, n=8):
    """
    Show clean images vs adversarial images from each attack side by side.
    This is the most visually impactful plot — shows how invisible the
    perturbations are to humans but how devastating they are to the model.
    """
    model.eval()
    attacks = list(adv_images_dict.keys())
    n_rows = len(attacks) + 2   # clean + perturbation + one row per attack
    n_cols = n

    fig = plt.figure(figsize=(n_cols * 1.5, n_rows * 1.8))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.1)

    clean_display = denormalize(clean_images[:n].cpu())

    with torch.no_grad():
        clean_pred = model(clean_images[:n].to(DEVICE)).argmax(dim=1).cpu()

    # Row 0: clean images
    for j in range(n):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(clean_display[j, 0], cmap="gray", vmin=0, vmax=1)
        true_label = labels[j].item()
        pred_label = clean_pred[j].item()
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True:{true_label}\nPred:{pred_label}", fontsize=7, color=color)
        ax.axis("off")
        if j == 0:
            ax.set_ylabel("Clean", fontsize=9, rotation=0, labelpad=40)

    # Rows 1+: adversarial images for each attack
    for i, attack_name in enumerate(attacks):
        adv_images = adv_images_dict[attack_name]
        adv_display = denormalize(adv_images[:n].cpu())

        with torch.no_grad():
            adv_pred = model(adv_images[:n].to(DEVICE)).argmax(dim=1).cpu()

        for j in range(n):
            ax = fig.add_subplot(gs[i + 1, j])
            ax.imshow(adv_display[j, 0], cmap="gray", vmin=0, vmax=1)
            true_label = labels[j].item()
            pred_label = adv_pred[j].item()
            color = "green" if true_label == pred_label else "red"
            ax.set_title(f"True:{true_label}\nPred:{pred_label}", fontsize=7, color=color)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(attack_name, fontsize=9, rotation=0, labelpad=40)

    # Last row: magnified perturbation (x10)
    for j in range(n):
        ax = fig.add_subplot(gs[-1, j])
        last_attack = attacks[-1]
        perturb = (adv_images_dict[last_attack][j] - clean_images[j]).cpu()
        perturb_display = (perturb * 10 + 0.5).clamp(0, 1)
        ax.imshow(perturb_display[0], cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_title("Δx10", fontsize=7)
        ax.axis("off")
        if j == 0:
            ax.set_ylabel("Perturbation\n(×10)", fontsize=9, rotation=0, labelpad=40)

    plt.suptitle("Adversarial Attack Gallery\n(Red = fooled model, Green = correct)",
                 fontsize=13, fontweight="bold")
    path = f"{FIGURES_DIR}/attack_gallery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_epsilon_sweep(results_by_epsilon):
    """
    Show how attack strength (epsilon) affects accuracy.
    As epsilon increases, adversarial accuracy drops dramatically.
    """
    epsilons = sorted(results_by_epsilon.keys())
    clean_accs = [results_by_epsilon[e]["clean_accuracy"] for e in epsilons]
    adv_accs = [results_by_epsilon[e]["adversarial_accuracy"] for e in epsilons]
    success_rates = [results_by_epsilon[e]["attack_success_rate"] for e in epsilons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epsilons, clean_accs, "b-o", label="Clean Accuracy", linewidth=2)
    ax1.plot(epsilons, adv_accs, "r-o", label="Adversarial Accuracy", linewidth=2)
    ax1.fill_between(epsilons, adv_accs, clean_accs, alpha=0.15, color="red",
                     label="Accuracy Drop")
    ax1.set_xlabel("Epsilon (perturbation magnitude)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("FGSM Attack: Accuracy vs Epsilon", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epsilons, success_rates, "r-o", linewidth=2, label="Attack Success Rate")
    ax2.set_xlabel("Epsilon (perturbation magnitude)", fontsize=12)
    ax2.set_ylabel("Attack Success Rate", fontsize=12)
    ax2.set_title("FGSM Attack: Success Rate vs Epsilon", fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    path = f"{FIGURES_DIR}/epsilon_sweep.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_defence_comparison(defence_results):
    """
    Bar chart comparing clean vs adversarial accuracy for each defence.
    Shows the trade-off between clean accuracy and adversarial robustness.
    """
    names = [r["defence"] for r in defence_results]
    clean_accs = [r["clean_acc"] for r in defence_results]
    adv_accs = [r["adv_acc"] for r in defence_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, clean_accs, width, label="Clean Accuracy",
                   color="#5DCAA5", edgecolor="white")
    bars2 = ax.bar(x + width/2, adv_accs, width, label="Adversarial Accuracy",
                   color="#E24B4A", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.set_title("Defence Comparison under FGSM (ε=0.3)\n(Higher adversarial = better defence)",
                 fontsize=13)
    ax.legend(fontsize=11, loc="upper center", ncol=2, frameon=False)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    path = f"{FIGURES_DIR}/defence_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_attack_comparison(attack_results):
    """Compare attack strength across FGSM, PGD, and CW."""
    attacks = [r["attack"] for r in attack_results]
    adv_accs = [r["adversarial_accuracy"] for r in attack_results]
    success_rates = [r["attack_success_rate"] for r in attack_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["#EF9F27", "#E24B4A", "#7B68EE"]

    bars = ax1.bar(attacks, adv_accs, color=colors, edgecolor="white")
    ax1.set_ylabel("Model Accuracy Under Attack", fontsize=12)
    ax1.set_title("Adversarial Accuracy by Attack\n(Lower = stronger attack)", fontsize=12)
    ax1.set_ylim(0, 1.1)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.0%}", ha="center", fontsize=11, fontweight="bold")

    bars2 = ax2.bar(attacks, success_rates, color=colors, edgecolor="white")
    ax2.set_ylabel("Attack Success Rate", fontsize=12)
    ax2.set_title("Attack Success Rate\n(Higher = stronger attack)", fontsize=12)
    ax2.set_ylim(0, 1.1)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.0%}", ha="center", fontsize=11, fontweight="bold")

    plt.suptitle("Attack Comparison: FGSM vs PGD vs CW", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/attack_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
