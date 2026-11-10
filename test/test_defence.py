"""Tests for defence mechanisms."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.defence import gaussian_smoothing_defence

def test_smoothing_preserves_shape():
    images = torch.randn(4, 1, 28, 28)
    smoothed = gaussian_smoothing_defence(images, sigma=1.0)
    assert smoothed.shape == images.shape

def test_smoothing_reduces_variance():
    """Smoothing should reduce pixel variance (noise reduction)."""
    noise = torch.randn(4, 1, 28, 28) * 0.5
    smoothed = gaussian_smoothing_defence(noise, sigma=2.0)
    assert smoothed.var() < noise.var()

def test_smoothing_sigma_effect():
    """Higher sigma should produce more smoothing (lower variance)."""
    images = torch.randn(4, 1, 28, 28)
    smooth_low = gaussian_smoothing_defence(images, sigma=0.5)
    smooth_high = gaussian_smoothing_defence(images, sigma=2.0)
    assert smooth_high.var() <= smooth_low.var()
