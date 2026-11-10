"""
model.py
--------
Defines and trains a CNN on MNIST.

We use MNIST (handwritten digits) because:
- Small and fast to train (2-3 min on CPU)
- 99%+ accuracy makes the adversarial drop dramatic
- Visualisations are intuitive — everyone understands digits
- Standard benchmark used in adversarial ML papers

Architecture: Simple CNN with 2 conv layers + 2 FC layers
Achieves ~99% accuracy on clean test data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistCNN(nn.Module):
    """
    Simple CNN for MNIST digit classification.

    Architecture:
    Conv(1→32) → ReLU → MaxPool
    Conv(32→64) → ReLU → MaxPool
    Flatten → FC(1024→128) → ReLU → Dropout
    FC(128→10) → LogSoftmax
    """
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_dataloaders(batch_size=64):
    """Download MNIST and create train/test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_set = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_set)} images | Test: {len(test_set)} images")
    return train_loader, test_loader


def train(model, train_loader, epochs=5, lr=0.001):
    """Train the CNN on MNIST."""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {DEVICE}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.4f}")

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1%}")

    return model


def evaluate(model, test_loader):
    """Evaluate model accuracy on clean test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

    acc = correct / total
    print(f"Clean test accuracy: {acc:.2%}")
    return acc


def save_model(model, path="models/mnist_cnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path="models/mnist_cnn.pth"):
    model = MnistCNN()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


if __name__ == "__main__":
    print("=== Training CNN on MNIST ===")
    train_loader, test_loader = get_dataloaders()
    model = MnistCNN()
    model = train(model, train_loader, epochs=5)
    acc = evaluate(model, test_loader)
    save_model(model)
    print(f"\nBaseline accuracy: {acc:.2%}")
    print("Model ready for adversarial attack experiments.")
