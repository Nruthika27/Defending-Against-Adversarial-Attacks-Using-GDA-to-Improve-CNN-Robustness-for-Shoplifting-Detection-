# Defending Against Adversarial Attacks Using GDA for Shoplifting Detection

This repository contains a deep learning pipeline focused on improving CNN robustness for **shoplifting detection** under adversarial attack conditions.

The work uses the **UCF-Crime dataset** (Shoplifting and NormalVideos classes), evaluates baseline vulnerability, and applies **Generative Data Augmentation (GDA)** and adversarial training to improve resilience.

## Project Objective

Build and evaluate a surveillance classification pipeline that:

1. Trains a baseline CNN classifier for shoplifting detection.
2. Measures robustness degradation under adversarial attacks (FGSM/PGD).
3. Uses GAN-based synthetic augmentation as a defense strategy.
4. Compares multiple defended configurations under clean and adversarial settings.

## Repository Structure

- `Phase_1_3.ipynb`  
  Covers baseline training, adversarial attack evaluation, and GAN-related workflow components.

- `ucf-nruthika-phase3.ipynb`  
  Focuses on GAN training and synthetic sample generation for augmentation.

- `nruthika-phase45-2.ipynb`  
  Runs defense comparison experiments across model configurations (including adversarial and GAN-augmented training).

- `Nruthika_Complete_Pipeline.ipynb`  
  Consolidated end-to-end notebook integrating all stages.

- `Nruthik_UFC_Technical Documentation.docx`  
  Detailed technical documentation of methodology, rationale, and experiment design.

## Dataset

- **Dataset:** UCF-Crime
- **Kaggle source:** https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
- **Classes used:**
  - `NormalVideos` → label `0`
  - `Shoplifting` → label `1`

The workflow assumes a video-aware split strategy (train/validation/test) to avoid leakage across frames from the same source video.

## Core Methods

- **Backbone model:** ResNet-50 (PyTorch/timm-based workflow)
- **Adversarial attacks:** FGSM and PGD
- **Defense strategies:** standard augmentation, adversarial training, and GAN-augmented training
- **GAN setup:** conditional generator/discriminator for class-aware synthetic image generation

## Typical Environment

The notebooks are designed primarily for **Kaggle** execution, with paths commonly set to Kaggle mounted datasets (for example `/kaggle/input/...`).

Main libraries used across notebooks include:

- `torch`, `torchvision`, `torchaudio`
- `timm`
- `opencv-python`
- `scikit-learn`
- `adversarial-robustness-toolbox`
- `numpy`, `Pillow`

## How to Use

1. Open the notebooks in Kaggle (recommended) or adapt local paths.
2. Mount/download the UCF-Crime dataset and update dataset paths.
3. Run notebook cells in order:
   - Baseline training and evaluation
   - Adversarial attack testing
   - GAN augmentation and defended training
   - Final comparison under clean and adversarial conditions
4. Review saved checkpoints/metrics from each phase for analysis.

## Notes

- These experiments can be computationally intensive (GPU recommended).
- Some cells install packages inline (notebook-style workflow).
- For reproducibility, keep split logic and random seeds consistent across runs.
