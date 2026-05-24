# EEG caffeine input representation CNN

This project tests different input representations of EEG signals for classifying the `Before` and `After` states in a caffeine/placebo experiment. The main goal is to compare models that work directly on raw EEG epochs, manually extracted EEG features, or a combination of both representations.

## What The Project Includes

- data preparation from an EEGLAB/MATLAB `ALLEEG` structure;
- creation of two `.npz` datasets:
  - raw EEG: `(epochs, channels, timepoints)`;
  - features: `(epochs, channels, features)`;
- extraction of time-domain, frequency-domain, and entropy features;
- training and evaluation of multiple CNN architectures;
- subject-independent validation with `GROUPKFOLD` or `LOSO`;
- saving metrics, predictions, models, configuration copies, and plots in `outputs/`.

The currently processed datasets in `data/processed/` contain 3713 epochs, 32 EEG channels, and 19 subjects. The raw input has shape `(3713, 32, 900)`, while the feature input has shape `(3713, 32, 16)`.

## Project Structure

```text
.
|-- config.yaml
|-- requirements.txt
|-- data/
|   |-- raw/                         # raw .mat files, not intended for commits
|   |-- processed/                   # generated .npz datasets
|   |-- build_raw_dataset.py
|   |-- build_feature_dataset.py
|   `-- load_dataset.py
|-- notebooks/
|   `-- exploration.ipynb
|-- src/
|   |-- feature_extraction/          # time, frequency, and entropy features
|   |-- models/                      # CNN architectures
|   |-- train_eval/                  # preparation, training, evaluation, plots
|   `-- utils/
`-- outputs/                         # experiment results
```

The `.mat` and `.npz` files are listed in `.gitignore` because they are large or generated. To reproduce the experiments, the input `.mat` file must exist at the path configured in `config.yaml`.

## Environment Setup

A Python virtual environment is recommended. The local environment in this project was created with Python 3.14, but the code uses the standard dependencies listed in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Configuration

The main settings are stored in `config.yaml`.

Important fields:

- `experiment.analysis_type` determines which data are used:
  - `caffeine_before_vs_after` uses only the `Caffeine` group;
  - `placebo_before_vs_after` uses only the `Placebo` group;
  - `before_vs_after` uses all subjects.
- `data.data_path` points to the input `.mat` file.
- `data.raw_dataset_output` and `data.feature_dataset_output` define the output `.npz` files.
- `features` defines the feature set and frequency bands.
- `validation.method` supports `GROUPKFOLD` and `LOSO`.
- `training` defines epochs, batch size, learning rate, early stopping, and device selection.
- `model` contains hyperparameters for each architecture.
- `outputs` defines directories for models, metrics, and plots.

## Data Preparation

Before training the models, prepare both datasets:

```powershell
python src/train_eval/test_dataset_builders.py
```

The script:

- reads `config.yaml`;
- loads `ALLEEG` from `data.data_path`;
- builds the raw EEG dataset;
- builds the feature dataset;
- checks that labels, subjects, conditions, and groups match between both datasets;
- saves the results to `data/processed/raw_dataset.npz` and `data/processed/feature_dataset.npz`.

The labels are binary:

- `Before = 0`
- `After = 1`

## Features

For each channel and each epoch, the project can compute:

- time-domain features: `mean`, `std`, `var`, `skewness`, `kurtosis`, `min`, `max`, `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity`;
- frequency-domain features: band power in `delta`, `theta`, `alpha`, `beta`, and `gamma`;
- entropy feature: `spectral_entropy`.

With the current configuration, this gives 16 features per channel.

## Models

The project compares four models:

- `EEGNetLike` in `src/models/cnn_eegnetlike.py` uses raw EEG epochs with shape `(batch, 1, channels, timepoints)`;
- `CNNFeatures1D` in `src/models/cnn_features_1d.py` uses features with shape `(batch, channels, features)`;
- `CNNFeatures2D` in `src/models/cnn_features_2d.py` uses features as a 2D input with shape `(batch, 1, channels, features)`;
- `CNNHybrid` in `src/models/cnn_hybrid.py` combines the raw EEG input and the feature input.

All models use PyTorch and return logits for two classes.

## Running Experiments

You can run individual models with:

```powershell
python src/train_eval/run_cnn_eegnetlike.py
python src/train_eval/run_cnn_features1d.py
python src/train_eval/run_cnn_features2d.py
python src/train_eval/run_cnn_hybrid.py
```

To run all models sequentially:

```powershell
python src/train_eval/run_all_models.py
```

This script measures the runtime of each model and saves the summary in:

```text
outputs/experiment_timings/
```

If any model fails, the sequential run stops to avoid wasting time on further experiments.

## Result Analysis

After running the models, compare the latest results across all models with:

```powershell
python src/train_eval/analyze_model_results.py
```

The script reads the latest result directories for each model, computes metric summaries, and saves comparison tables and plots in `outputs/analysis/`.

## Outputs

Each model run creates timestamped directories:

```text
outputs/models/<model_name>_<timestamp>/
outputs/results/<model_name>_<timestamp>/
outputs/plots/<model_name>_<timestamp>/
```

Typical outputs include:

- `.pt` model checkpoints;
- `config_used.yaml`, a copy of the configuration used for the run;
- CSV files with fold-level metrics;
- validation summary reports;
- plots of training curves, confusion matrices, ROC curves, and fold-level metrics;
- `runtime.txt` with runtime information.

## Validation

Validation is subject-independent: all epochs from the same subject remain in the same split.

- `GROUPKFOLD` splits subjects into `n_splits` folds.
- `LOSO` uses Leave-One-Subject-Out validation, where one subject is used as the test subject in each fold.

If `validation.use_validation_subject` is set to `true`, validation subjects are selected only from the training portion of the fold. This prevents information from the test subject from leaking into early stopping.

## Standardization

Standardization is performed inside each fold:

- for features, the mean and standard deviation are computed from the training split;
- for raw EEG, standardization is performed per channel;
- validation and test data are transformed using training statistics.

This reduces the risk of data leakage between training and evaluation.

## Typical Workflow

1. Configure `config.yaml`, especially `data.data_path`, `experiment.analysis_type`, and the validation method.
2. Prepare the datasets:

```powershell
python src/train_eval/test_dataset_builders.py
```

3. Run one model or all models:

```powershell
python src/train_eval/run_all_models.py
```

4. Compare results:

```powershell
python src/train_eval/analyze_model_results.py
```

5. Inspect the outputs in `outputs/results/`, `outputs/plots/`, and `outputs/analysis/`.

## Notes

- The project assumes that the `.mat` file contains the key `ALLEEG`.
- The current scripts do not use command-line arguments; all settings are read from `config.yaml`.
- Large raw `.mat` and `.npz` files may require substantial RAM.
- `outputs/` contains generated results and can grow quickly.
