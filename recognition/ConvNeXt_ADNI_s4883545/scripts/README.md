# Supporting commands

Run these from the repository root. The required training and prediction entry
points remain in the parent project directory.

```powershell
$p = 'recognition/ConvNeXt_ADNI_s4883545'
py -3.13 "$p/scripts/report_run.py" --help
```

| Script | Purpose |
|---|---|
| [report_run.py](report_run.py) | Summarise a measured trial or completed training run. |
| [plot_learning_curves.py](plot_learning_curves.py) | Plot recorded training and validation histories. |
| [compare_validation.py](compare_validation.py) | Verify checkpoint provenance and compare identical validation cases. |
| [analyze_confidence.py](analyze_confidence.py) | Analyse confidence and select a validation review threshold. |
| [benchmark_latency.py](benchmark_latency.py) | Measure synchronized inference and JPEG-pipeline latency. |
| [analyze_sensitivity.py](analyze_sensitivity.py) | Investigate input perturbations and disagreement on validation. |
| [analyze_failure_reasons.py](analyze_failure_reasons.py) | Test blur, resampling and patient-context explanations. |
| [check_model.py](check_model.py) | Check model shapes, gradients, updates and checkpoint reloads. |
| [check_augmentation.py](check_augmentation.py) | Preview augmentation and check a short training pipeline. |
| [diagnose_convnext.py](diagnose_convnext.py) | Inspect a checkpoint and input/optimization diagnostics. |
| [final_evaluation.py](final_evaluation.py) | Snapshot a pre-test protocol and run its evaluation. Existing project test results remain exposed. |

Use each command's `--help` for its arguments. Commands requiring outputs should
write under ignored `artifacts/`. The parent [README](../README.md#11-reproducibility-and-usage)
contains reproduction commands for the reported models.

`_paths.py` resolves core and helper files for source hashes and frozen code
snapshots. Original archived experiments retain their original files and hashes.
