# Import necessary libraries and utilities
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, precision_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
import shap

# Define function to get machine learning models and parameters
def get_ML_models(random_state):
    # models
    models = {
        "dt": {"name": "Decision Tree", "estimator": DecisionTreeClassifier(random_state = random_state), 
               "param": [{"max_depth": list(range(5, 25))}]},
        "rf":  {"name": "Random Forest", "estimator": RandomForestClassifier(random_state = random_state), 
                "param": [{"max_depth": list(range(5, 25))}]},
        "xgb": {"name": "XGBoost", "estimator": XGBClassifier(random_state = random_state), 
                "param": [{"learning_rate": [0.05, 0.1, 0.2]}]}, 
        "gnb": {"name": "Gaussian Naive Bayes", "estimator": GaussianNB(),
               "param": [{"var_smoothing": [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-07, 1e-8]}]},
        "bnb": {"name": "Bernoulli Naive Bayes", "estimator": BernoulliNB(),
                "param": [{"alpha" : [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}]},
        "svc": {"name": "Support Vector", "estimator": SVC(random_state = random_state),                    
                "param": [{"kernel": ["rbf", "linear"], 
                           "gamma": [1e-1, 1e-2, 1e-3, 1e-4], 
                           "C": [1, 10, 100, 1000]}]},
        "lr":  {"name": "Logistic Regression", "estimator": LogisticRegression(random_state = random_state),
                "param": [{"solver": ["liblinear"], 
                           "max_iter": [1000], 
                           "penalty": ["l2"], 
                           "C": np.logspace(-4, 4, 20)}]},     
        "mlp": {"name": "MLP", "estimator": MLPClassifier(random_state = random_state),             
                "param": [{"hidden_layer_sizes": [(64, 64), (128, 128)], 
                           "learning_rate": ["adaptive"], 
                           "batch_size": [32]}]}
    }
    return(models)

# Define a function to perform grid-search and cross-validation
def classify(X_train, X_test, y_train, y_test,
             types_of_model, models, scores,
             models_path, fingerprint, experiment, scores_df):
    # Iterate over different scoring metrics
    for score in scores:
        # Iterate over different types of models
        for mtype in types_of_model:
            # Perform grid-search and cross-validation
            clf = GridSearchCV(models[mtype]["estimator"], models[mtype]["param"], scoring="{}_macro".format(score))
            # Calculate sample weights
            weight_dict = dict(zip(np.unique(y_train), len(y_train)/(len(np.unique(y_train))*np.bincount(y_train))))
            # Fit the model
            if mtype != "mlp":
                clf.fit(X_train, y_train, sample_weight=np.array([weight_dict[yi] for yi in y_train]))
            else:    
                clf.fit(X_train, y_train) # sample weights not yet available for MLP in scikit-learn
            # Make predictions
            y_pred = clf.predict(X_test)
            # Store the results
            scores_df.loc[len(scores_df)] = [mtype, score] + get_results(models[mtype]["name"], y_test, y_pred)
            # Save the model 
            with open(f"{models_path}/{fingerprint}/{mtype}_{experiment}_{score}.pkl", "wb") as handle:
                pickle.dump(clf.best_estimator_, handle)
    return scores_df


# Define a function to compute various evaluation scores
def get_results(name, y_test, y_pred_test, verbose=True):
    # Compute test scores
    a = accuracy_score(y_test, y_pred_test)
    ba = balanced_accuracy_score(y_test, y_pred_test)
    roc = roc_auc_score(y_test, y_pred_test)
    mcc = matthews_corrcoef(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="macro")
    r = recall_score(y_test, y_pred_test, average="macro")
    p = precision_score(y_test, y_pred_test, average="macro")
    r1 = recall_score(y_test, y_pred_test)
    sp = recall_score(y_test, y_pred_test, pos_label=0)
    # Print scores if verbose is True
    if verbose:
        print(name)
        print("Test Accuracy:          {:.2f}".format(a))
        print("Test Balanced Accuracy: {:.2f}".format(ba))
        print("Test ROC AUC:           {:.2f}".format(roc))
        print("Test Matthews Corr Coef:{:.2f}".format(mcc))
        print("Test F1 Score:          {:.2f}".format(f1))
        print("Test Recall:            {:.2f}".format(r))
        print("Test Precision:         {:.2f}".format(p))
        print("Test Recall (class 1):  {:.2f}".format(r1))
        print("Test Specificity:       {:.2f}\n".format(sp))
    # Return the computed scores
    return [a, ba, roc, mcc, f1, r, p, r1, sp]

# Define a function to split the data into training and testing sets
def split_data(X, y, split, random_state):
    # Split the data into training and testing sets, preserving the class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=random_state, stratify=y)
    # Return the split data
    return X_train, X_test, y_train, y_test

# Define a function to save the dataset splits
def save_splits(X_train, X_test, y_train, y_test, path, experiment):
    # Save the splits
    with open(f"{path}/X_train_{experiment}.pkl", "wb") as handle:
        pickle.dump(X_train, handle)
    with open(f"{path}/X_test_{experiment}.pkl", "wb") as handle:
        pickle.dump(X_test, handle)
    with open(f"{path}/y_train_{experiment}.pkl", "wb") as handle:
        pickle.dump(y_train, handle)
    with open(f"{path}/y_test_{experiment}.pkl", "wb") as handle:
        pickle.dump(y_test, handle)

# Define a function to read a dataframe from a CSV file and add a molecule column
def read_dataframe(path):
    # Read the dataframe from the CSV file
    df = pd.read_csv(path, index_col=0)
    # Add a molecule column to the dataframe using SMILES strings
    PandasTools.AddMoleculeColumnToFrame(df, "Smiles")
    # Return the dataframe
    return df

# Define a function to compute fragment fingerprints for a molecule
def get_frag_fp(mol, binary = False):
    # Initialize the result array
    res = np.zeros(len(Descriptors._descList[123:]))
    # Compute the fragment fingerprints
    for i, (_, fn) in enumerate(Descriptors._descList[123:]):
        res[i] = fn(mol)
    # Convert to binary if specified
    if binary:
        return np.where(res >= 1, 1, 0)
    else:
        return res.astype(int)

# Define a function to compute functional group fingerprints for a molecule
def get_all_func_fp(DayLightFuncGroups, mol, binary=False):
    # Initialize the result array
    res = np.zeros(len(DayLightFuncGroups.keys()))
    # Compute the functional group fingerprints
    for i, key in enumerate(DayLightFuncGroups.keys()):
        patt = DayLightFuncGroups[key]
        try:
            sma = Chem.MolFromSmarts(patt)
        except:
            sma = None
        if not sma:
            # Print error if SMARTS parsing fails
            print('SMARTS parser error for key #%d: %s' % (key, patt))
        else:
            # Compute the count of substructures matching SMARTS patterns
            res[i] = len(mol.GetSubstructMatches(sma))
    # Convert to binary if specified
    if binary:
        return np.where(res >= 1, 1, 0)
    else:
        return res.astype(int)
    
# Function to perform feature selection
def feature_select(f_sel, n_sel, fingerprint, y):
    # Feature selection based on the specified method
    if f_sel == "chi2":
        fingerprint = SelectKBest(chi2, k=n_sel).fit_transform(fingerprint, y)
        selection = f"{f_sel}_{n_sel}"
    elif f_sel == "f_classif":
        fingerprint = SelectKBest(f_classif, k=n_sel).fit_transform(fingerprint, y)
        selection = f"{f_sel}_{n_sel}"
    elif f_sel == "mutual_info_classif":
        fingerprint = SelectKBest(mutual_info_classif, k=n_sel).fit_transform(fingerprint, y)
        selection = f"{f_sel}_{n_sel}"
    else:
        selection = ""
    return fingerprint, selection

# Function to perform data sampling (oversampling/undersampling)
def sample_data(X_train, y_train, sampling, random_state):
    if sampling == "oversampling":
        aug = RandomOverSampler(random_state=random_state)
        X_train, y_train = aug.fit_resample(X_train, y_train)
    elif sampling == "undersampling":
        aug = RandomUnderSampler(random_state=random_state)
        X_train, y_train = aug.fit_resample(X_train, y_train)
    return X_train, y_train

# Function to scale data (if necessary)
def scale_data(X_train, X_test):
    if X_train.max() > 1:
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test) 
    return X_train, X_test

# Function to save SHAP values and explanations
def save_SHAP(exp, shapv, full_path):
    output = open(f"{full_path}_explainer.pkl", 'wb')
    pickle.dump(exp, output)
    output = open(f"{full_path}_SHAP.pkl", 'wb')
    pickle.dump(shapv, output)

# Function to explain model predictions using SHAP values
def explain_model(model, predictor, X_train, X_test, full_path):
    if model == "dt" or model == "bnb" or model == "lr" or model == "rf" or model == "xgb" or model == "mlp":
        if model == "dt":
            explainer = shap.TreeExplainer(predictor, X_train)
            shap_values = explainer.shap_values(X_test)
            save_SHAP(explainer, shap_values, full_path)
            return shap_values
        elif model == "rf":
            explainer = shap.TreeExplainer(predictor, X_train)
            shap_values = explainer.shap_values(X_test)
            save_SHAP(explainer, shap_values, full_path)
            return shap_values
        elif model == "xgb":
            explainer = shap.TreeExplainer(predictor, X_train)
            shap_values = explainer.shap_values(X_test)
            save_SHAP(explainer, shap_values, full_path)
            return shap_values
        elif model == "lr":
            explainer = shap.LinearExplainer(predictor, X_train)
            shap_values = explainer.shap_values(X_test)
            save_SHAP(explainer, shap_values, full_path)
            return shap_values
        elif model == "bnb":
            X_test_subset = X_test[np.random.choice(len(X_test), 200, replace=False)]
            explainer = shap.Explainer(predictor.predict, X_train)
            shap_values = explainer.shap_values(X_test_subset)
            save_SHAP(explainer, shap_values, full_path)
            return shap_values
        elif model == "mlp":
            explainer = shap.Explainer(predictor.predict, X_train)
            shap_values = explainer.shap_values(X_test)
            save_SHAP(explainer, shap_values, full_path)
            return shap_values