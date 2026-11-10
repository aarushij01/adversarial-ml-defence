"""
train_robust.py
---------------
Trains an adversarially robust MNIST CNN using PGD adversarial training
(Madry et al., 2017) and saves it to models/mnist_cnn_robust.pth.

The training loop itself lives in src/defence.py (adversarial_training).
Each batch is attacked with PGD-7 and the model is trained on an equal
mix of clean and adversarial examples.

Run:
    PYTHONPATH=. python src/train_robust.py --epochs 2
"""

import argparse
import time

import torch

from src.model import MnistCNN, get_dataloaders, evaluate, save_model, DEVICE
from src.defence import adversarial_training
from src.attacks import to_input_eps


def main():
    parser = argparse.ArgumentParser(description="PGD adversarial training on MNIST")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="L-inf budget in [0,1] pixel units (standard MNIST setting is 0.3)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_loader, test_loader = get_dataloaders(batch_size=128)

    model = MnistCNN().to(DEVICE)
    start = time.time()
    model = adversarial_training(model, train_loader, epochs=args.epochs, epsilon=to_input_eps(args.epsilon))
    print(f"Adversarial training took {(time.time() - start) / 60:.1f} min on {DEVICE}")

    acc = evaluate(model, test_loader)
    print(f"Robust model clean test accuracy: {acc}")
    save_model(model, path="models/mnist_cnn_robust.pth")


if __name__ == "__main__":
    main()
