"""
run_experiments.py
------------------
Runs the full adversarial ML experiment pipeline:

1. Load trained model
2. Run FGSM, PGD, CW attacks
3. Epsilon sweep (how attack strength affects accuracy)
4. Compare defences
5. Generate all plots

Run this after training the model with: python src/model.py
"""

import json
import os

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from src.model import MnistCNN, load_model, DEVICE
from src.attacks import fgsm_attack, pgd_attack, cw_attack_l2, evaluate_attack, to_input_eps
from src.defence import gaussian_smoothing_defence, evaluate_defence
from src.visualise import (
    plot_attack_gallery, plot_epsilon_sweep,
    plot_defence_comparison, plot_attack_comparison
)

import warnings
warnings.filterwarnings("ignore")

ROBUST_PATH = "models/mnist_cnn_robust.pth"


def get_test_batch(n=256):
    """Get a small batch of test images for experiments."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = torchvision.datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    subset = Subset(test_set, range(n))
    loader = DataLoader(subset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.to(DEVICE)


def run_all_experiments():
    print("=" * 60)
    print("ADVERSARIAL ML EXPERIMENT SUITE")
    print("=" * 60)

    # Load model
    print("\n[1/5] Loading trained model...")
    model = load_model()
    model.eval()

    # Get test batch
    images, labels = get_test_batch(n=1000)
    print(f"Test batch: {images.shape}, device: {DEVICE}")

    # --- EXPERIMENT 1: Run all attacks ---
    print("\n[2/5] Running adversarial attacks...")

    # Standard MNIST budget: 0.3 in [0,1] pixel units (= ~1.0 in normalised input units)
    pixel_eps = 0.3
    epsilon = to_input_eps(pixel_eps)
    pgd_alpha = 2.5 * epsilon / 40

    print("  Running FGSM...")
    fgsm_adv = fgsm_attack(model, images, labels, epsilon=epsilon)
    fgsm_results = evaluate_attack(model, images, fgsm_adv, labels)
    fgsm_results["attack"] = "FGSM"
    print(f"  FGSM: clean={fgsm_results['clean_accuracy']:.1%}, "
          f"adv={fgsm_results['adversarial_accuracy']:.1%}, "
          f"success={fgsm_results['attack_success_rate']:.1%}")

    print("  Running PGD (40 steps)...")
    pgd_adv = pgd_attack(model, images, labels, epsilon=epsilon,
                         alpha=pgd_alpha, num_steps=40)
    pgd_results = evaluate_attack(model, images, pgd_adv, labels)
    pgd_results["attack"] = "PGD"
    print(f"  PGD:  clean={pgd_results['clean_accuracy']:.1%}, "
          f"adv={pgd_results['adversarial_accuracy']:.1%}, "
          f"success={pgd_results['attack_success_rate']:.1%}")

    print("  Running CW (L2)...")
    cw_adv = cw_attack_l2(model, images[:200], labels[:200])
    cw_results = evaluate_attack(model, images[:200], cw_adv, labels[:200])
    cw_results["attack"] = "CW (L2)"
    print(f"  CW:   clean={cw_results['clean_accuracy']:.1%}, "
          f"adv={cw_results['adversarial_accuracy']:.1%}, "
          f"success={cw_results['attack_success_rate']:.1%}")

    # --- EXPERIMENT 2: Epsilon sweep ---
    print("\n[3/5] Running epsilon sweep...")
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    epsilon_results = {}

    for eps in epsilons:
        adv = fgsm_attack(model, images, labels, epsilon=to_input_eps(eps))
        results = evaluate_attack(model, images, adv, labels)
        epsilon_results[eps] = results
        print(f"  epsilon={eps:.2f}: adv_acc={results['adversarial_accuracy']:.1%}")

    # --- EXPERIMENT 3: Defence evaluation ---
    print("\n[4/5] Evaluating defences...")

    defence_results = []

    # Baseline (no defence)
    baseline = evaluate_defence(model, images, fgsm_adv, labels,
                                defence_fn=None, defence_name="No Defence")
    defence_results.append(baseline)

    # Gaussian smoothing
    smooth_light = evaluate_defence(
        model, images, fgsm_adv, labels,
        defence_fn=lambda x: gaussian_smoothing_defence(x, sigma=0.5),
        defence_name="Smoothing (σ=0.5)"
    )
    defence_results.append(smooth_light)

    smooth_strong = evaluate_defence(
        model, images, fgsm_adv, labels,
        defence_fn=lambda x: gaussian_smoothing_defence(x, sigma=1.5),
        defence_name="Smoothing (σ=1.5)"
    )
    defence_results.append(smooth_strong)

    # Adversarial training (real run — see src/train_robust.py).
    # The robust model is attacked with FGSM and PGD generated against ITSELF
    # (white-box), which is the fair test of a trained defence.
    robust = None
    if os.path.exists(ROBUST_PATH):
        robust = load_model(ROBUST_PATH)
        robust.eval()
        r_fgsm = fgsm_attack(robust, images, labels, epsilon=epsilon)
        r_fgsm_res = evaluate_attack(robust, images, r_fgsm, labels)
        r_pgd = pgd_attack(robust, images, labels, epsilon=epsilon, alpha=pgd_alpha, num_steps=40)
        r_pgd_res = evaluate_attack(robust, images, r_pgd, labels)
        print(f"  Adv. training: clean={r_fgsm_res['clean_accuracy']:.1%}, "
              f"FGSM={r_fgsm_res['adversarial_accuracy']:.1%}, PGD-40={r_pgd_res['adversarial_accuracy']:.1%}")
        defence_results.append({
            "defence": "Adv. Training",
            "clean_acc": r_fgsm_res["clean_accuracy"],
            "adv_acc": r_fgsm_res["adversarial_accuracy"],
            "pgd_acc": r_pgd_res["adversarial_accuracy"],
        })
    else:
        print(f"  {ROBUST_PATH} not found — run `make robust` to include adversarial training.")

    # --- EXPERIMENT 4: Generate visualisations ---
    print("\n[5/5] Generating visualisations...")

    plot_attack_gallery(
        model,
        images[:8].cpu(),
        {
            "FGSM (ε=0.3)": fgsm_adv[:8],
            "PGD (ε=0.3, 40 steps)": pgd_adv[:8],
        },
        labels[:8].cpu()
    )

    plot_epsilon_sweep(epsilon_results)

    plot_attack_comparison([fgsm_results, pgd_results, cw_results])

    plot_defence_comparison(defence_results)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Clean accuracy       : {fgsm_results['clean_accuracy']:.1%}")
    print(f"FGSM attack (ε=0.3) : {fgsm_results['adversarial_accuracy']:.1%} "
          f"(↓{fgsm_results['clean_accuracy'] - fgsm_results['adversarial_accuracy']:.1%})")
    print(f"PGD attack  (ε=0.3) : {pgd_results['adversarial_accuracy']:.1%} "
          f"(↓{pgd_results['clean_accuracy'] - pgd_results['adversarial_accuracy']:.1%})")
    print(f"CW attack   (L2)    : {cw_results['adversarial_accuracy']:.1%}")
    print(f"\nAll figures saved to reports/figures/")

    os.makedirs("results", exist_ok=True)
    out = {
        "n_test_images": int(len(labels)),
        "n_cw_images": 200,
        "epsilon_pixel_space": pixel_eps,
        "epsilon_normalised": round(epsilon, 4),
        "attacks": [fgsm_results, pgd_results, cw_results],
        "epsilon_sweep": {str(k): v for k, v in epsilon_results.items()},
        "defences": defence_results,
    }
    with open("results/results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("Results saved to results/results.json")

    return {
        "clean_accuracy": fgsm_results["clean_accuracy"],
        "fgsm": fgsm_results,
        "pgd": pgd_results,
        "cw": cw_results,
        "epsilon_sweep": epsilon_results,
        "defences": defence_results,
    }


if __name__ == "__main__":
    run_all_experiments()
