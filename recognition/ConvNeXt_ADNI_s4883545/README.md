# ADNI MRI classification: ConvNeXt-Tiny versus a small CNN

**Weijie Peng | s4883545 | UQ COMP3710**

## Table of contents

1. [Overview](#1-overview)
2. [Feasibility review](#2-feasibility-review)

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