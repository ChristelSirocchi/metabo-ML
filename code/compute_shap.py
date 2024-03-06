# Libraries.
from utilities import *
import pickle
import numpy as np
import argparse
import os

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fingerprints", nargs="+", dest="fps", default=["all"], help="chosen fingerprints")
parser.add_argument("-m", "--models", nargs="+", dest="models", default=["all"], help="selected models")
parser.add_argument("-s", "--scores", nargs="+", dest="scores", default=["recall"], help="cross-validation scores")
parser.add_argument("-ext", "--extension_name", dest="ename", default="", help="experiment extension name")
parser.add_argument("-rs", "--random_seed", dest="random_seed", default=0, help="random seed", type=int)
parser.add_argument("-sp", "--sampling", dest="sampling", default="none", help="over/under-sampler")
parser.add_argument("-ni", "--n_iter", dest="n_iter", default=1, help="number of iterations", type=int)
parser.add_argument("-fs", "--f_sel", dest="f_sel", default="none", help="feature selection method")
parser.add_argument("-ns", "--n_sel", dest="n_sel", default="200", help="feature selected number", type=int)
# Parse command-line arguments
args = parser.parse_args()

# List of all possible fingerprints
all_fps = ["mg2", "mg3", "dl", "ap", "tt", "func", "frags", "maccs"]
# Generate count fingerprints for all except the last one
all_fps_count = [f"{ff}_count" for ff in all_fps[:-1]]

# Select fingerprints based on the command-line argument
if args.fps == ["all"]:
    fps = all_fps
elif args.fps == ["all_count"]:
    fps = all_fps_count
else:
    fps = args.fps

# Possible models to use
if args.models == ["all"]:
    types_of_model = ["lr", "xgb", "rf", "dt", "bnb"]
else:
    types_of_model = args.models

# Determine feature selection method and selection string
if (args.f_sel == "chi2") | (args.f_sel == "f_classif") | (args.f_sel == "mutual_info_classif"):
    selection = f"{args.f_sel}_{args.n_sel}"
else:
    selection = ""

# Store sampling method and cross-validation scores
sampling = args.sampling
scores = args.scores

# Iterate over selected fingerprints
for fp in fps:
    if fp in all_fps + all_fps_count:
        # Define fingerprint file name
        fp_file = f"{fp}_{args.ename}"
        # Create folder for SHAP explanations
        os.makedirs(f"SHAP/{fp_file}", exist_ok=True)
        # Iterate over number of iterations
        for n in range(args.n_iter):
            print("Iteration " + str(n))
            # Define random seed
            random_state = args.random_seed + n
            np.random.seed(random_state)
            # Define experiment name
            experiment = f"{sampling}_{random_state}_{selection}"
            # Load test and train data for the experiment
            with open(f"data/{fp_file}/X_test_{experiment}.pkl", 'rb') as file:
                X_test = pickle.load(file)
            with open(f"data/{fp_file}/X_train_{experiment}.pkl", 'rb') as file:
                X_train = pickle.load(file)
            # Iterate over cross-validation scores
            for score in scores:
                # Iterate over different types of models
                for model in types_of_model:
                    # Load model from file
                    with open(f"models/{fp_file}/{model}_{experiment}_{score}.pkl", 'rb') as file:
                        predictor = pickle.load(file)
                        print(f"Model {model} loaded")
                    # Define full path for saving SHAP values
                    full_path = f"SHAP/{fp_file}/{model}_{experiment}_{score}"
                    # Explain model predictions using SHAP values
                    explain_model(model, predictor, X_train, X_test, full_path)