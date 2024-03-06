# Program Description

This program consists of three main Python scripts for data preprocessing, model training, and generating SHAP explanations.

The required Python packages are
- numpy
- pickle
- json
- rdkit
- pandas
- scikit-learn (sklearn)
- xgboost
- imbalanced-learn (imblearn)
- shap
  
## 1. compute_fingerprints.py

This script computes fingerprints from molecular structures specified in a DataFrame.

### Usage:

```bash
python compute_fingerprints.py -dfp dataframe.csv -ext ext -it smile -sc Smiles -tc AT_down_05
python compute_fingerprints.py -dfp dataframe.csv -ext ext -it mol -sc Structure -tc AT_down_05
```

### Parameters:

- `-dfp`, `--dataframe_path`: Path to the DataFrame CSV file.
- `-ext`, `--extension_name`: Experiment extension name.
- `-it`, `--input_type`: Input type, either 'smile' or 'mol'.
- `-sc`, `--structure_column`: Name of the column containing molecular structures.
- `-tc`, `--target_column`: Name of the target column.

## 2. train_models.py

This script trains machine learning models using specified fingerprints and parameters.

### Usage:

```bash
python train_models.py -ext ext
python train_models.py -ext ext -s f1 recall
python train_models.py -ext ext -fp ap dl tt
python train_models.py -ext ext -fp all_count
python train_models.py -ext ext -m rf xgb
python train_models.py -ext ext -ni 10
python train_models.py -ext ext -sp oversampling
python train_models.py -ext ext -ns 100 -fs chi2
python train_models.py -ext ext -fp all_count
```

### Parameters:

- `-fp`, `--fingerprints`: Chosen fingerprints.
- `-m`, `--models`: Selected models.
- `-s`, `--scores`: Cross-validation scores.
- `-ext`, `--extension_name`: Experiment extension name.
- `-tg`, `--target`: Target name.
- `-rs`, `--random_seed`: Random seed.
- `-sp`, `--sampling`: Over/undersampling method.
- `-ni`, `--n_iter`: Number of iterations.
- `-fs`, `--feature_sel`: Feature selection method.
- `-ns`, `--n_sel`: Number of selected features.

## 3. compute_shap.py

This script computes SHAP explanations for model predictions.

### Usage:

```bash
python compute_shap.py -ext ext
python compute_shap.py -ext ext -s f1 recall
python compute_shap.py -ext ext -fp ap dl tt
python compute_shap.py -ext ext -fp all_count
python compute_shap.py -ext ext -m rf xgb
python compute_shap.py -ext ext -ni 10
python compute_shap.py -ext ext -sp oversampling
python compute_shap.py -ext ext -ns 100 -fs chi2
python compute_shap.py -ext ext -fp all_count
```

### Parameters:

- `-fp`, `--fingerprints`: Chosen fingerprints.
- `-m`, `--models`: Selected models.
- `-s`, `--scores`: Cross-validation scores.
- `-ext`, `--extension_name`: Experiment extension name.
- `-rs`, `--random_seed`: Random seed.
- `-sp`, `--sampling`: Over/undersampling method.
- `-ni`, `--n_iter`: Number of iterations.
- `-fs`, `--feature_sel`: Feature selection method.
- `-ns`, `--n_sel`: Number of selected features.

---
