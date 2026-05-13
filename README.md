# Defending Against Adversarial Attacks Using GDA for Shoplifting Detection

This project studies how adversarial attacks affect CNN-based shoplifting detection and evaluates a defense strategy based on **Generative Data Augmentation (GDA)**.

## Project Overview

The pipeline is built around the UCF-Crime dataset (Shoplifting vs NormalVideos) and is organized into phased notebooks:

- Train a baseline ResNet-50 classifier
- Evaluate robustness under FGSM and PGD attacks
- Train a conditional WGAN-GP to generate adversarial-like shoplifting samples
- Compare defense configurations (clean baseline, stronger augmentation, PGD adversarial training, GAN-augmented training)

## Repository Structure

- `Phase_1_3.ipynb` — Early phase notebook (data prep/training components)
- `ucf-nruthika-phase3.ipynb` — Conditional GAN training and synthetic sample generation
- `nruthika-phase45-2.ipynb` — Defense comparison and final analysis
- `Nruthika_Complete_Pipeline.ipynb` — Consolidated end-to-end pipeline notebook
- `Nruthik_UFC_Technical Documentation.docx` — Detailed technical documentation for the UCF-Crime pipeline (legacy filename kept as-is)

## Dataset

- **Dataset**: UCF-Crime (Shoplifting + NormalVideos subsets)
- **Kaggle source**: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
- **Labels**:
  - `1` = Shoplifting
  - `0` = NormalVideos

## Methods and Models

- **Classifier backbone**: ResNet-50 (PyTorch/timm)
- **Attacks**: FGSM and PGD (IBM Adversarial Robustness Toolbox)
- **Generator/defense**: Conditional WGAN-GP for adversarial-style sample synthesis
- **Evaluation metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix, robustness curves

## Environment and Main Dependencies

This project is notebook-driven and was developed in Kaggle-style environments. Common dependencies used across notebooks include:

- `torch`, `torchvision`
- `timm`
- `numpy`, `matplotlib`, `scikit-learn`
- `Pillow`, `opencv-python`
- `adversarial-robustness-toolbox` (`art`)

## How to Run

1. Download and mount the UCF-Crime dataset.
2. Open Jupyter/Kaggle notebooks in this order:
   - `Phase_1_3.ipynb` (or baseline training/attack notebook)
   - `ucf-nruthika-phase3.ipynb`
   - `nruthika-phase45-2.ipynb`
   - or run `Nruthika_Complete_Pipeline.ipynb` for an integrated workflow
3. Update dataset/output paths in notebook config cells before execution.
4. Run cells sequentially and save produced checkpoints/JSON/plots.

## Typical Outputs

Depending on notebook and phase, outputs include:

- Model checkpoints (`*.pth`)
- Attack/metric reports (`*.json`)
- Robustness and comparison figures (`*.png`)
- Final summary artifacts for clean vs adversarial performance comparison

## Notes

- The implementation emphasizes **video-level train/validation/test splitting** to reduce frame leakage across splits.
- Notebook names reflect project phases and iterative development.
