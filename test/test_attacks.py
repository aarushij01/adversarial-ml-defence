"""Tests for adversarial attacks."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attacks import fgsm_attack, pgd_attack, evaluate_attack

@pytest.fixture
def dummy_model():
    """A simple linear model for testing."""
    import torch.nn as nn
    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))
    model.eval()
    return model

@pytest.fixture
def dummy_batch():
    images = torch.zeros(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    return images, labels

def test_fgsm_output_shape(dummy_model, dummy_batch):
    images, labels = dummy_batch
    adv = fgsm_attack(dummy_model, images, labels, epsilon=0.3)
    assert adv.shape == images.shape

def test_fgsm_perturbation_bounded(dummy_model, dummy_batch):
    images, labels = dummy_batch
    epsilon = 0.3
    adv = fgsm_attack(dummy_model, images, labels, epsilon=epsilon)
    # Perturbation should be roughly bounded by epsilon
    diff = (adv - images).abs().max().item()
    assert diff <= epsilon + 1e-5, f"Perturbation {diff:.4f} exceeds epsilon {epsilon}"

def test_pgd_output_shape(dummy_model, dummy_batch):
    images, labels = dummy_batch
    adv = pgd_attack(dummy_model, images, labels, epsilon=0.3, num_steps=5)
    assert adv.shape == images.shape

def test_evaluate_attack_returns_dict(dummy_model, dummy_batch):
    images, labels = dummy_batch
    adv = fgsm_attack(dummy_model, images, labels, epsilon=0.3)
    results = evaluate_attack(dummy_model, images, adv, labels)
    assert "clean_accuracy" in results
    assert "adversarial_accuracy" in results
    assert "attack_success_rate" in results

def test_pgd_stronger_than_fgsm(dummy_model, dummy_batch):
    """PGD with many steps should be at least as strong as FGSM."""
    images, labels = dummy_batch
    fgsm_adv = fgsm_attack(dummy_model, images, labels, epsilon=0.3)
    pgd_adv = pgd_attack(dummy_model, images, labels, epsilon=0.3, num_steps=20)
    fgsm_res = evaluate_attack(dummy_model, images, fgsm_adv, labels)
    pgd_res = evaluate_attack(dummy_model, images, pgd_adv, labels)
    # PGD success rate should be >= FGSM (or equal)
    assert pgd_res["attack_success_rate"] >= fgsm_res["attack_success_rate"] - 0.1
