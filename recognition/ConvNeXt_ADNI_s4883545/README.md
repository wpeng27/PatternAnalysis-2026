# ADNI MRI classification: ConvNeXt-Tiny versus a small CNN

**Weijie Peng | s4883545 | UQ COMP3710**

## Table of contents

1. [Overview](#1-overview)
2. [Feasibility review](#2-feasibility-review)
3. [Data, preprocessing and evaluation definitions](#3-data-preprocessing-and-evaluation-definitions)


## 1. Overview

In this project, I investigate whether a modern convolutional classifier improves AD/NC recognition and confidence-based review enough to justify its computational cost. A custom PyTorch ConvNeXt-Tiny and a small CNN classify the supplied processed 2D ADNI MRI JPEG slices. Both start from random weights and use identical patient-disjoint splits. ConvNeXt applies depthwise spatial filtering, channel mixing and residual connections across four stages. I selected the ConvNeXt model using exponential moving average (EMA) weights based on validation loss.

**Final selected model:** my EMA ConvNeXt achieved **89.53% validation accuracy**, **74.40% slice-level test accuracy** and **77.33% patient-level test accuracy**. I selected epoch 29 using minimum validation loss. These test results are exploratory because earlier test evaluations informed subsequent experiments. The CNN achieved **92.19% validation and 71.31% slice-level test accuracy**. The required 80% slice-level test target remains **unmet**.

![Implemented ConvNeXt architecture and block](report/figures/architecture.png)

The supporting [measured evidence](report/results.json) and [figure provenance](report/ASSETS.md) are included in `report/`. I have excluded data, checkpoints, patient manifests and full prediction exports from Git.

## 2. Feasibility review
### User need and research question

I propose a research prototype that classifies ADNI MRI slices as Alzheimer's disease (AD) or normal control (NC) and flags uncertain cases for human review. I will investigate whether ConvNeXt improves classification and confidence reliability over a simple CNN enough to justify its computational cost. Imaging researchers are the intended users.

### Model choice and data configuration

I will implement ConvNeXt-Tiny with a two-class output and compare it with a small CNN, both trained from scratch. Convolutional features and residual connections apply the course's CNN concepts. JPEG slices in ADNI/AD\_NC will be resized to 224 by 224, with grayscale copied into three identical channels. I will randomly reserve approximately 20% of training patients for validation, stratified by AD/NC. The rest will train the model. The supplied test set will be held out for final evaluation. The training data will be normalised before training the models.

### Proposed acceptance criteria

1. Verify zero patient overlap between splits, resolve duplicates and patient IDs, and keep both classes in every split.  
2. Target at least 0.80 patient-level test accuracy. Compare both models on identical splits using per-class precision and recall, F1 and AUROC.  
3. Target at least 0.90 accuracy among accepted cases at 50% or greater coverage.  
4. Target peak training VRAM below 10 GiB on one GPU with at least 12 GiB available. Measure inference latency and parameter count for both models.  
5. Explain 3 to 5 false positives or false negatives, including confident errors, and recommend when human review remains necessary.

### Preliminary feasibility evidence

Through reviewing the dataset, there were 30,520 JPEGs. Training contains 10,400 AD and 11,120 NC. Testing contains 4,460 AD and 4,540 NC. All 10 sampled headers were 240 by 256 pixels (height by width). Two AD and two NC previews loaded with sample pixel values from 0 to 255\. Exact duplicates were not checked.

### Risks computing budget and next experiment

I propose a ceiling of 24 GPU-hours for baseline training, up to three ConvNeXt configurations and final evaluation, using mixed precision and early stopping. The ConvNeXt-Tiny model size is estimated at 28 million parameters. Peak VRAM is unmeasured. Risks include leakage, unknown patient counts and overfitting. Next, I will run a one-epoch baseline test on training-folder images to record loss, runtime and peak VRAM. For memory limits, I will reduce batch size. If accuracy or confidence targets fail, I will report the shortfall and recommend continued manual review.

## 3. Data, preprocessing and evaluation definitions

Teaching staff confirmed that `392586_92.jpeg` identifies patient `392586` and slice `92`. Volumes were already sliced and blank slices removed. Folder labels are **NC=0 and AD=1**. The task sheet's CN corresponds to the supplied NC folder.

The dataset remains at `ADNI/AD_NC` relative to the repository root, with `train/AD`, `train/NC`, `test/AD` and `test/NC`. [dataset.py](dataset.py) stratifies training patients by class with seed 42 and reserves approximately 20% for validation. All slices from one patient stay together. The original test folders are retained.

| Split | NC slices / patients | AD slices / patients | Total slices / patients |
|---|---:|---:|---:|
| Train | 8,900 / 445 | 8,320 / 416 | 17,220 / 861 |
| Validation | 2,220 / 111 | 2,080 / 104 | 4,300 / 215 |
| Test | 4,540 / 227 | 4,460 / 223 | 9,000 / 450 |

All 1,526 filename-derived patients have 20 slices. The completed audit found zero patient overlap, inconsistent patient labels, byte-identical duplicates or identical decoded-pixel duplicates within or across splits. Consistent folder labels are not independent clinical verification. The audit verified that both models used the same saved manifest SHA-256 hashes, included in [results.json](report/results.json).

Pillow converts each image to grayscale and bilinearly resizes it to **224 x 224**. Values are divided by 255 and copied into three identical channels. CNN uses [0,1] inputs. Selected ConvNeXt uses `2*x-1`, giving [-1,1]. ConvNeXt training alone uses rotation up to +/-7 degrees, translation up to 5% and scale 0.95-1.05, with bilinear interpolation and black fill, without flipping or shear. Validation, test and prediction use no random augmentation. These augmentations are experimental assumptions, not established anatomical invariances.

Each slice contributes one primary prediction. Predict AD if **P(AD)>0.5**, otherwise NC, including ties. Accuracy is the fraction correct, macro-F1 averages the two class F1 scores, and AD AUROC uses continuous **P(AD)** with AD positive. Precision and recall are reported separately for NC and AD. Cross-entropy is sample-weighted. Secondary patient metrics average P(AD) across that patient's slices after verifying consistent labels, then apply the same boundary. Patient scores and confidence-filtered scores do not replace unfiltered slice accuracy. [metrics.py](metrics.py) implements these definitions.

