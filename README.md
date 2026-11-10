# Skin Lesion Screening: Fine-Tuning on PAD-UFES-20

Binary task: malignant (BCC, SCC, MEL) vs non-malignant (ACK, NEV, SEK). Run `notebook.ipynb` top to bottom on a Colab T4. Data is downloaded from Mendeley Data (CC BY 4.0), no credentials needed.

## Choices
- **Split:** 80/10/10 (1,820 / 221 / 257 images), stratified on the label and grouped by patient so no patient appears in two splits. Seed 42.
- **Model:** ViT-Small/16 (ImageNet-21k pretrained), 512 px input, all layers unfrozen. Picked because it had the best validation PR-AUC among five setups (see ablation).
- **Layers:** nothing frozen. Partial and full fine-tuning performed about the same (differences within noise), so I went with the best on validation.
- **LR and schedule:** Adam, backbone 5e-5, head 1e-3, weight decay 1e-3, OneCycleLR (20% warmup), 20 epochs, batch 16, mixed precision. Best epoch chosen on validation PR-AUC.
- **Augmentations (Albumentations):** rotation, random-resized crop, flips, brightness/contrast/saturation/hue jitter, Gaussian blur. Validation and test are only resized and center-cropped.
- **Class distribution:** close to balanced (47% malignant in train), so a mild `pos_weight` of 1.13 in the loss. No resampling.

## Results (test set: 257 images, 127 malignant)
**Threshold:** the highest threshold with at least 95% validation sensitivity, but never below 0.05. The 95% target is only reached at 0.001, which would refer 84% of validation cases, so the threshold fell back to **0.05** (validation: sensitivity 0.887, specificity 0.643). A missed cancer is far worse than an unnecessary referral, so I favored sensitivity, with the floor limiting referrals.

| Sensitivity | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.953 [0.904, 0.992] | 0.792 [0.711, 0.868] | 0.818 [0.738, 0.885] | 0.880 [0.827, 0.922] | 0.939 [0.902, 0.970] | 0.926 [0.860, 0.970] |

Missed malignant lesions: 6 of 127 (TP 121, FN 6, FP 27, TN 103). Intervals are 95% patient-level bootstrap.

**Screening 1,000 people at 5% prevalence (50 cancers, 950 healthy), using test sensitivity and specificity at this threshold:** about 48 cancers caught and 2 missed (range 0 to 5), 197 unnecessary referrals (range 126 to 274), 245 total referrals (24.5% of people). About 19% of referrals are real cancers.

## Ablation (same data, seed, schedule and LRs)
| Setup | Trainable | Val PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|
| ResNet50, frozen backbone | head only | 0.789 | 0.806 | 0.844 |
| ResNet50, stem + stage 1 frozen | 23.3M | 0.866 | 0.894 | 0.904 |
| ResNet50, full | 23.5M | 0.868 | 0.891 | 0.913 |
| ViT-S, first 6 blocks frozen | 10.7M | 0.851 | 0.939 | 0.948 |
| **ViT-S, full** | 22.0M | **0.890** | 0.926 | 0.939 |

Adapting the backbone clearly beats a frozen one. The rest are within noise (single seed, 106 malignant validation cases), so I selected on validation and did not switch based on test.

## One thing that didn't work
Overfitting. Train loss falls to about 0.02 while validation loss rises from 0.5 to 1.1. Lowering the LR, raising weight decay and freezing half the ViT did not stop it. Ranking quality (PR-AUC, ROC-AUC) stayed stable, so I kept the checkpoint with the best validation PR-AUC. The ViT overfit more than ResNet50.

## Limitations and next steps
Melanoma is rare (41 training and 5 test images), so performance on it is unreliable. One clinic, one seed, small validation set. Next: AdamW with real weight decay, stochastic depth and layer-wise LR decay; several seeds; fusing the clinical metadata (age, body region, lesion growth); test-time augmentation or an ensemble; probability calibration; external validation.
