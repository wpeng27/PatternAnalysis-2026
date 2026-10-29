# Automated checks

From the repository root:

```powershell
$p = 'recognition/ConvNeXt_ADNI_s4883545'
py -3.13 -m unittest discover -s "$p/tests" -t $p -p 'test_*.py'
```

The top-level directory (`-t`) makes the core modules and `scripts/` importable.
To run a specific group, change the pattern, for example `-p 'test_predict.py'`.

Tests use synthetic images, small fixtures and temporary checkpoints. They cover
patient splits, preprocessing, architecture, gradients, training/resume, metrics,
EMA, schedulers, prediction, analysis and evaluation integrity. They do not train
on the ADNI dataset or evaluate its test set. CUDA-specific checks can be skipped
when no suitable GPU is available.
