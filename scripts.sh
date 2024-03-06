python compute_fingerprints.py -dfp dataframe.csv -ext ext -it smile -sc Smiles -tc AT_down_05 
python compute_fingerprints.py -dfp dataframe.csv -ext ext -it mol -sc Structure -tc AT_down_05 

python train_models.py -ext ext
python train_models.py -ext ext -s f1 recall
python train_models.py -ext ext -fp ap dl tt
python train_models.py -ext ext -fp all_count
python train_models.py -ext ext -m rf xgb
python train_models.py -ext ext -ni 10
python train_models.py -ext ext -sp oversampling
python train_models.py -ext ext -ns 100 -fs chi2
python train_models.py -ext ext -fp all_count

python compute_shap.py -ext ext
python compute_shap.py -ext ext -s f1 recall
python compute_shap.py -ext ext -fp ap dl tt
python compute_shap.py -ext ext -fp all_count
python compute_shap.py -ext ext -m rf xgb
python compute_shap.py -ext ext -ni 10
python compute_shap.py -ext ext -sp oversampling
python compute_shap.py -ext ext -ns 100 -fs chi2
python compute_shap.py -ext ext -fp all_count
