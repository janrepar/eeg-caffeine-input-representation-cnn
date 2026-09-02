# EEG caffeine input representation CNN

This project compares four CNN architectures for binary classification of Before and After EEG epochs in a caffeine/placebo experiment: raw EEG, 1D features, 2D features, and a hybrid input. Participants remain separate across training, validation, and testing.

The active configuration is version 1.0.1 in config.yaml. [CHANGES.md](CHANGES.md) records changes since the initial release: unified experiment folders, nested Optuna optimisation, reference classifiers, statistical tests, and expanded EDA.

## Datasets and structure

The current datasets in data/processed/ contain 3,713 epochs, 32 channels, and 19 participants:

~~~
raw_dataset.npz:     (3713, 32, 900)
feature_dataset.npz: (3713, 32, 13)
~~~

experiment.analysis_type filters final analysis to Caffeine, Placebo, or both groups; the current setting uses Caffeine.

~~~
.
├── CHANGES.md
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/                         # input .mat files
│   ├── processed/                   # generated .npz datasets
│   ├── build_raw_dataset.py
│   ├── build_feature_dataset.py
│   └── load_dataset.py
├── notebooks/
│   ├── exploration.ipynb            # EDA
│   └── eda_figures/
├── src/
│   ├── feature_extraction/
│   ├── models/
│   ├── train_eval/
│   └── utils/
└── outputs/
    ├── experiments/
    ├── hyperparameter_optimization/
    └── archive/
~~~

Large/generated .mat and .npz files are ignored by Git. The input at data.data_path must contain the ALLEEG key.

## Installation

The local environment uses Python 3.14. Install requirements, including Optuna 4.x:

~~~powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
~~~

If PowerShell blocks activation:

~~~powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
~~~

## Configuration and data preparation

config.yaml defines group selection, data paths, sampling rate, features, validation, Optuna settings, training, architectures, and outputs. The current default validation is five-fold GROUPKFOLD with two validation participants. With use_validation_subject: true, validation participants come only from a fold's training portion.

~~~powershell
python src/train_eval/test_dataset_builders.py
~~~

The script loads ALLEEG, applies include_conditions and exclude_subjects, builds both datasets, verifies alignment of labels, participants, conditions, and groups, and saves them in data/processed/. Labels are Before = 0 and After = 1.

## Features

The current feature input has 13 values per channel:

- time-domain: mean, std, skewness, kurtosis, min, max, hjorth_mobility, hjorth_complexity;
- frequency-domain: log10 absolute power in theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz);
- entropy: spectral_entropy.

Welch estimation for power and entropy uses the complete epoch: welch_nperseg: 900 at 600 Hz. Delta is excluded because 1.5-second epochs lack reliable resolution. Feature groups can be disabled in configuration.

## Models and leakage prevention

| Model | Source | Input |
| --- | --- | --- |
| EEGNetLike | src/models/cnn_eegnetlike.py | (batch, 1, channels, timepoints) |
| CNNFeatures1D | src/models/cnn_features_1d.py | (batch, channels, features) |
| CNNFeatures2D | src/models/cnn_features_2d.py | (batch, 1, channels, features) |
| CNNHybrid | src/models/cnn_hybrid.py | raw EEG and features |

Models return two-class logits and use CrossEntropyLoss with optional label_smoothing; training scripts print total and trainable parameter counts.

LOSO leaves one participant out for testing in every fold; GROUPKFOLD partitions participants into n_splits groups. Within every fold, validation participants come from training participants. Features are standardised by channel and feature, raw EEG per channel over training epochs and timepoints, and validation/test use training statistics only. project.random_seed seeds Python, NumPy, and PyTorch and enables deterministic CUDA settings when available.

## Running and outputs

Run a complete experiment:

~~~powershell
python src/train_eval/run_all_models.py
~~~

It creates outputs/experiments/<timestamp>_<VALIDATION>_nvalsubj<N>/, trains all four models, then automatically runs comparison analysis and statistical tests. On failure it stops. experiment_timings/model_runtime_summary.csv records status and duration of all executed stages.

Run a single model:

~~~powershell
python src/train_eval/run_cnn_eegnetlike.py
python src/train_eval/run_cnn_features1d.py
python src/train_eval/run_cnn_features2d.py
python src/train_eval/run_cnn_hybrid.py
~~~

Standalone runs create a separate folder. Run post-processing manually:

~~~powershell
python src/train_eval/analyze_model_results.py
python src/train_eval/run_statistical_tests.py
~~~

Manual scripts locate the newest complete results in outputs/results; when called by run_all_models.py, they use the active shared folder.

~~~
outputs/experiments/<timestamp>_<VALIDATION>_nvalsubj<N>/
├── config_used.yaml
├── models/<model>/                  # .pt checkpoints
├── results/<model>/                 # CSV metrics, summaries, runtime
├── plots/<model>/                   # learning curves, ROC, confusion matrices
├── analysis/                        # model comparison
├── statistical_tests/               # baselines and tests
└── experiment_timings/
~~~

Fold metrics include test/validation participants, best epoch, best train/validation values, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, and epoch_roc_auc. analysis provides all_model_fold_results.csv, model_comparison_summary.csv, used configurations, and accuracy, F1, and ROC-AUC plots in fixed order: Raw EEGNet-like, Features1D, Features2D, Hybrid.

statistical_tests provides always Before and always After baselines plus two-sided exact sign-flip tests: each model versus chance accuracy 0.5 and all model pairs for epoch_accuracy and epoch_f1. Interpret LOSO tests by participant. Treat GroupKFold tests as exploratory because training sets overlap.

## Nested Optuna optimisation

Before final LOSO training, optimise all four models:

~~~powershell
python -u src/train_eval/optimize_hyperparameters.py --model raw
python -u src/train_eval/optimize_hyperparameters.py --model features1d
python -u src/train_eval/optimize_hyperparameters.py --model features2d
python -u src/train_eval/optimize_hyperparameters.py --model hybrid
~~~

--trials N overrides the trial count. Optimisation uses outer LOSO and inner GroupKFold, maximises balanced accuracy, and penalises instability across inner folds. Results are saved in outputs/hyperparameter_optimization/<model>/: best_params_subject_<ID>.json, trials_subject_<ID>.json, and nested_optimization_summary.json.

Important: the current optimiser always filters to Caffeine. Final scripts read its parameters only under LOSO; GROUPKFOLD uses config.yaml.

## EDA and cleanup

notebooks/exploration.ipynb covers dataset composition, signal quality, channels, representative epochs, and time/frequency views. Figures are in notebooks/eda_figures/.

~~~powershell
python src/utils/prune_output_folders.py --folder outputs/experiments --before 2026-08-01 --dry-run
~~~

Deletion requires --delete and interactive DELETE confirmation. Raw .mat and especially .npz files are large, so building and training require substantial RAM and time.

## Data contract and external preprocessing

This repository does not perform the original EEG preprocessing. The input must be a MATLAB file containing ALLEEG, whose entries have these fields:

| Field | Requirement |
| --- | --- |
| data | numeric array (channels, timepoints, epochs) |
| subject | scalar participant identifier |
| condition | Before or After; matching is case-insensitive |
| group | Caffeine or Placebo |

The current configuration assumes 600 Hz and 900 samples per epoch, i.e. 1.5 seconds. Channel order is retained during construction and must be identical for all conditions and participants. The code does not validate channel names, montage, referencing, filtering, or artefact removal; document these choices with the source data and apply them consistently.

Final model filtering expects group values written exactly as Caffeine and Placebo. Empty data entries are skipped; missing or differently named fields cause an error or an invalid dataset.

## Reproducibility and limitations

For complete technical reproduction, preserve or report:

1. the source .mat file (or an unambiguous version) and every preprocessing step;
2. the exact config.yaml and code version;
3. datasets in data/processed/, or the command used to rebuild them;
4. all four Optuna result sets when using LOSO;
5. the complete directory under outputs/experiments/.

Every completed experiment saves config_used.yaml, metrics, plots, analysis, and timings. Compare results with the configuration inside the experiment folder, not only the current root configuration.

Metrics are calculated at epoch level. Epochs from one participant are not independent observations, so mean epoch-level scores must not be interpreted as independent-person counts. LOSO is more suitable for participant-level inference. GroupKFold p-values are exploratory because fold training sets overlap. LOSO Optuna optimisation is currently implemented for Caffeine only; placebo or pooled analysis needs adaptation or separately validated optimisation.

The project classifies Before versus After and does not itself establish a causal caffeine effect. Consider preprocessing, epoch imbalance, small participant count, and individual differences when interpreting results.

## Compute, privacy, and archiving

A CPU is sufficient; training.device: auto selects CUDA when PyTorch detects it. Raw-EEG training and especially nested optimisation are computationally demanding. The compressed raw dataset is about 391 MB, while loading and standardisation require more RAM than its on-disk size. First validate the workflow with one model or fewer Optuna trials, then inspect actual timings in experiment_timings/model_runtime_summary.csv.

Raw data are not included in the repository. Before sharing .mat files, .npz datasets, checkpoints, or outputs, check permissions, participant consent, identifiers, and the institution's retention policy. For long-term archival, retain a complete experiment directory, its Optuna results, and preprocessing documentation.