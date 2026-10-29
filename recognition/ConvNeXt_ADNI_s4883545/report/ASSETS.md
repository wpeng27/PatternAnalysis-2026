# Report figures and evidence

| File | Purpose and provenance |
|---|---|
| [architecture.png](figures/architecture.png) | Diagram of the implemented stage dimensions and block operations. Drawn with Matplotlib, following the paper and official implementation cited in the report. |
| [learning_curves.png](figures/learning_curves.png) | Recorded CNN and EMA histories, with selected epochs 18 and 29. Both loss panels use logarithmic axes. Training uses online weights, while EMA validation uses averaged weights. |
| [confidence.png](figures/confidence.png) | Full validation accuracy/coverage curves and ten-bin reliability summaries. Stars mark selected policies. Curves are descriptive validation results, not independent performance guarantees. |
| [failure_cases.png](figures/failure_cases.png) | Five real, deliberately selected validation errors. Panels show the actual 224 x 224 grayscale model inputs with fixed display limits. Source hashes were checked through the audited loader. Patient identifiers and source filenames are omitted. These are limited report illustrations, not a dataset distribution. |
| [results.json](results.json) | Compact measured results, experiment identities, epoch histories, confidence summaries, diagnostic outcomes and latency summaries. Includes hashes of original local evidence inputs, without source JPEGs, patient manifests, checkpoints or complete per-image prediction exports. |
| [review_policy.json](review_policy.json) | Exact validation-selected thresholds and checkpoint/split identities, consumable by `predict.py`. Applies only to the identified saved checkpoints. New training requires a new validation policy. |

The figures were generated from existing measurements and source validation images. No generative AI image model was used, and no MRI content was synthesized. The input figure uses the loader's required resizing, without extra contrast enhancement, cropping, denoising or anatomical annotation. Labels refer to supplied folders, not diagnoses independently established by visual inspection.

Original detailed experiment records remain locally under ignored `artifacts/`. Their SHA-256 digests are recorded in `results.json`. Numerical records for the five cases use case numbers in the same order as the figure. The diagnostic case arrays also follow cases 1-5. Native-precision five-case diagnostic batches can differ slightly from the original full-validation batch probabilities. FP32 controls are separate measurements. None changes the reported original prediction or the review threshold.

To reproduce model outputs, use the commands in the project report and authorized access to the dataset and retained checkpoints, or retrain with the recorded settings. A fresh clone can display this complete report and inspect the aggregate evidence, but cannot perform MRI inference without those private inputs. The raw diagnostic arrays, source images and saved weights are intentionally not linked as required report assets.
