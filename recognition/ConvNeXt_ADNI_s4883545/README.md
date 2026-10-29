# ADNI MRI classification: ConvNeXt-Tiny versus a small CNN

**Weijie Peng | s4883545 | UQ COMP3710**

## Table of contents

1. [Overview](#1-overview)
2. [Feasibility review](#2-feasibility-review)
3. [Data, preprocessing and evaluation definitions](#3-data-preprocessing-and-evaluation-definitions)
4. [Architecture and training protocol](#4-architecture-and-training-protocol)
5. [Experiments and convergence](#5-experiments-and-convergence)
6. [Validation and test comparison](#6-validation-and-test-comparison)
7. [Confidence and human review](#7-confidence-and-human-review)

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

## 4. Architecture and training protocol

[modules.py](modules.py) implements the architecture in PyTorch/torchvision. The ConvNeXt block follows the [original paper](https://arxiv.org/abs/2201.03545) and [official model](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py): 7 x 7 depthwise convolution, channels-last layer normalization, fourfold linear expansion, GELU, projection, layer scale and a residual branch with stochastic depth. Local adaptations are the two-class head, native PyTorch layer normalization and torchvision stochastic depth instead of timm. Attribution and the [MIT notice](LICENSE-ConvNeXt.txt) are retained.

| Component | Output, excluding batch dimension | Blocks |
|---|---|---:|
| Stem: 4 x 4 convolution, stride 4, normalization | 96 x 56 x 56 | - |
| Stage 1 | 96 x 56 x 56 | 3 |
| Downsample + stage 2 | 192 x 28 x 28 | 3 |
| Downsample + stage 3 | 384 x 14 x 14 | 9 |
| Downsample + stage 4 | 768 x 7 x 7 | 3 |
| Global average pooling, final normalization, head | 2 logits | - |

Separate normalization-plus-convolution downsamplers halve spatial dimensions and double channels. Layer scale starts at 1e-6, normalization epsilon is 1e-6, and stochastic-depth probability increases from 0 to 0.1. Convolutional/linear weights use truncated-normal initialization with standard deviation 0.02. No pretrained weights are fetched.

The CNN has three 3 x 3 convolution blocks with 16, 32 and 64 channels, each followed by batch normalization, ReLU and max pooling. A 64 x 4 x 4 pooled representation feeds a 128-unit ReLU classifier, dropout 0.2 and two logits.

**Checkpoint selection:** I used minimum validation slice cross-entropy, keeping the earliest epoch on an exact tie. I set this rule before comparing the models. Accuracy does not override this rule. Both families use the same trainer, split manifests and metric code.

| Setting | CNN | Selected ConvNeXt EMA |
|---|---|---|
| Optimizer / initial LR | AdamW / 0.001 | AdamW / 0.0001 |
| Betas / epsilon | (0.9, 0.999) / 1e-8 | Same |
| Weight decay | 0.0001, all parameters | 0.01 on multidimensional weights, zero on biases and 1D parameters |
| Batch / CUDA precision | 32 / FP16 with GradScaler | 16 / BF16 without GradScaler |
| Scaling / augmentation | [0,1] / none | [-1,1] / mild affine |
| Other regularization | Classifier dropout 0.2 | Stochastic depth up to 0.1 |
| Gradient clipping / label smoothing | Norm 5 / 0 | Same |
| EMA | None | Decay 0.999, FP32 shadow weights, successful optimizer updates only |
| Scheduler | ReduceLROnPlateau on validation loss | Same, using EMA validation loss |
| Scheduler settings | Factor 0.5, patience 2, absolute threshold 0, minimum LR 1e-6 | Same |
| Early stopping | Patience 5, minimum improvement 0 | Same |
| Epoch budget / completed / selected | 20 / 20 / 18 | 40 / 34 / 29 |
| Seed / workers / CPU threads | 42 / 0 / 4 | Same |

Bias/1D exclusions follow the [official optimizer grouping](https://github.com/facebookresearch/ConvNeXt/blob/main/optim_factory.py). EMA follows the update-after-optimizer pattern in the [official engine](https://github.com/facebookresearch/ConvNeXt/blob/main/engine.py), with a local successful-update check and fixed decay. EMA weights drive validation, scheduling, stopping and prediction. Best/last checkpoints preserve model settings, preprocessing, class mapping, optimizer, scheduler, epoch and random state. These choices differ between pipelines, so this is not an architecture-only ablation.

## 5. Experiments and convergence

The table summarises the experiments included in this report. I report outcomes at the checkpoint selected by validation loss. The hypotheses describe my reasons for trying each strategy. Comparisons use one seed, different stopping epochs and some continued training lineages, which limits causal claims. A dash means that checkpoint was not evaluated on test.

| Strategy and hypothesis | Changed setting | Selected epoch / completed | Validation accuracy / loss | Test accuracy |
|---|---|---:|---:|---:|
| CNN: establish a low-cost reference | Three convolution blocks, LR 1e-3, FP16 | 18 / 20 | 92.19% / 0.220289 | 71.31%, frozen |
| BF16: avoid FP16 overflows | BF16, LR 3e-4, unit inputs, no augmentation | 5 / 10 | 52.09% / 0.689988 | - |
| Centering: improve optimization | Scale [0,1] to [-1,1] | 2 / 7 | 70.53% / 0.659615 | - |
| Lower LR: stabilize updates | 3e-4 to 3e-5 | 2 / 7 | 73.79% / 0.525159 | - |
| Affine: improve generalization | Add mild affine transforms | 20 / 20 | 79.00% / 0.438211 | - |
| More epochs: extend a progressing run | Continue affine lineage to 30 | 29 / 30 | 80.12% / 0.427508 | 70.72%, frozen |
| Further continuation: reduce remaining loss | Continue toward 40 after test exposure | 34 / 39 | 80.42% / 0.413108 | - |
| Intermediate LR: balance progress/stability | Fresh affine run at 1e-4 | 24 / 29 | 85.79% / 0.338677 | 72.10%, exploratory |
| Grouped decay: avoid shrinking biases/1D parameters | Decay 0.01 for multidimensional weights, otherwise zero | 21 / 26 | 86.81% / 0.318166 | 73.33%, exploratory |
| EMA: smooth noisy weight updates | Fixed decay 0.999 with grouped decay | 29 / 34 | **89.53% / 0.291337** | **74.40%, exploratory** |
| Warmup/cosine: improve LR trajectory | Three warmup epochs, start factor 0.01, 40-epoch cosine horizon | 19 / 24 | 85.63% / 0.343083 | 72.17%, exploratory |

An earlier one-epoch FP16 ConvNeXt trial skipped 33 of 1,077 attempted optimizer updates and was excluded from quality comparison. The BF16 lineage recorded zero skipped updates. Grouped decay improved selected validation loss over the all-decay LR 1e-4 control. EMA improved it further. Warmup/cosine performed worse than its control. Cosine stopped before completing its planned decay horizon. I therefore retained EMA based on validation evidence. These comparisons do not establish that each observed difference isolates one causal factor.

![Training and validation loss and accuracy, with selected checkpoints](report/figures/learning_curves.png)

CNN training loss fell from 0.5169 to 0.0541, but validation was unstable: epoch 16 reached loss 6.7379 and accuracy 49.00% before recovery. Epoch 18 minimizes loss. Epoch 20's 92.49% accuracy does not override its higher 0.233174 loss. EMA validation loss fell from 0.687097 to 0.291337 at epoch 29. Epochs 30-34 did not improve it, triggering early stopping. Epoch 34 reached 90.09% validation accuracy while online training accuracy reached 94.41%. This indicates diminishing validation-loss gains, not evidence that more epochs would improve unseen patients. Training curves use changing online weights and regularization, while EMA validation uses averaged weights. Both loss panels use a logarithmic axis to retain the CNN spike.

## 6. Validation and test comparison

The current comparison uses identical **4,300 validation slices from 215 patients** and **9,000 test slices from 450 patients**. All slices are included, without confidence filtering. EMA is my final selected ConvNeXt model for this submission. CNN test results reuse its original frozen evaluation. EMA test results are later exploratory measurements.

| Metric | CNN validation | EMA validation | CNN test, original frozen | EMA test, exploratory |
|---|---:|---:|---:|---:|
| Accuracy | 92.19% | 89.53% | 71.31% | 74.40% |
| Cross-entropy | 0.220289 | 0.291337 | 2.096713 | 0.979386 |
| Macro-F1 | 0.921831 | 0.895348 | 0.701971 | 0.738579 |
| AD AUROC | 0.977354 | 0.958063 | 0.785493 | 0.824349 |
| NC precision | 93.53% | 92.79% | 65.79% | 69.42% |
| NC recall | 91.17% | 86.44% | 89.85% | 88.02% |
| AD precision | 90.82% | 86.51% | 83.54% | 83.23% |
| AD recall | 93.27% | 92.84% | 52.44% | **60.54%** |

For historical comparison, the original freeze occurred on **25 October 2026**, before test inference. That pair was CNN epoch 18 and the earlier non-EMA ConvNeXt epoch 29 from the LR 3e-5 affine lineage. Their test accuracies remain **71.31% and 70.72%**, respectively. The original ConvNeXt's macro-F1 was 0.703283 and AD AUROC 0.784545. This earlier ConvNeXt is not my final selected model. Its checkpoint is distinct from EMA even though both selected epochs are numbered 29.

I used later test results to guide further experiments. Selecting each checkpoint by validation loss does not remove the bias introduced by this broader test-guided search. The 74.40% EMA result is descriptive exploratory evidence, not a newly frozen final estimate. Neither model meets 80%. No significance claim is made for their 3.09-point test difference, and correlated slices are not independent samples.

Test confusion matrices, with true rows and predicted columns in [NC, AD] order, are CNN `[[4079,461],[2121,2339]]` and EMA `[[3996,544],[1760,2700]]`. EMA therefore misses **1,760 of 4,460 AD-labelled slices**. These are slice errors, not counts of missed patient diagnoses. Secondary patient-mean accuracy is CNN/EMA **95.81%/93.95% on validation** and **71.33%/77.33% on test**, with the same evaluation-history limitations. Patient aggregation cannot satisfy the primary slice target.

## 7. Confidence and human review

Confidence is `max(P(AD), 1-P(AD))`, not a calibrated probability of correctness. Using validation data only, I selected the lowest threshold achieving maximum coverage subject to at least 90% accepted accuracy and 50% coverage. Predictions at or above the threshold are accepted, keeping all ties. No temperature scaling or other calibration was fitted.

| Validation statistic | CNN | EMA |
|---|---:|---:|
| Selected threshold | 0.5 | **0.5549529194831848** |
| Coverage | 100.00% | 98.16% |
| Accepted accuracy | 92.1860% | 90.0024% |
| Accepted / referred slices | 4,300 / 0 | 4,221 / 79 |
| Errors accepted / referred | 336 / 0 | 422 / 28 |
| Mean confidence on wrong predictions | 78.68% | 81.60% |
| Correctness AUROC | 0.884403 | 0.829978 |
| Binary Brier score | 0.059112 | 0.082394 |
| ECE, ten confidence bins | 0.028103 | 0.044942 |

![Validation accuracy versus coverage and confidence reliability](report/figures/confidence.png)

Correctness AUROC treats correctness as positive and confidence as the score, distinct from AD AUROC. Brier is mean `(P(AD)-label)^2`. ECE uses ten equal-width bins over [0.5,1]. CNN has better validation confidence ranking and lower Brier/ECE. EMA refers only **6.22% of its 450 errors**, retaining 139 AD false negatives. Its referred group also includes 51 correct predictions. CNN accepts everything because it already exceeds the stated 90% objective, which is not evidence that all its predictions are reliable.

At confidence at least 0.95, EMA still makes 112 validation errors across 37 patients, including 39 AD false negatives. CNN makes 85 errors across 19 patients. At threshold 0.95, EMA's accepted accuracy is 96.34% at 71.26% coverage. At 0.99 it reaches 98.07%, but coverage falls below the proposed minimum to 49.35%. The selected EMA policy's patient-bootstrap 95% interval for accepted accuracy is 87.52%-92.27%, from 2,000 replicates with seed 42. This descriptive interval does not correct for selecting the model and threshold on the same patients.

The exact [review policy](report/review_policy.json) is bound to checkpoint and split hashes. It was selected using validation data after earlier test exposure and has **not been evaluated on test for EMA**. It is a reproducible research operating point, not a guarantee. Additional disagreement/context policies below remain exploratory and have not replaced it.