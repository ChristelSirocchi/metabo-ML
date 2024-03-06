# Libraries.
from utilities import *
import pandas as pd
import os
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("-fp","--fingerprints", nargs = "+", dest = "fps", default = ["all"], help = "chosen fingerprints")
parser.add_argument("-m", "--models", nargs = "+", dest = "models", default = ["all"], help= "selected models")
parser.add_argument("-s", "--scores", nargs = "+", dest = "scores", default = ["recall"], help= "cross-validation scores")
parser.add_argument("-ext","--extension_name", dest= "ename", default="", help = "experiment extension name")
parser.add_argument("-tg","--target", dest = "tg", default = "y", help = "target name")
parser.add_argument("-rs", "--random_seed", dest = "random_seed", default = 0, help = "random seed", type=int)
parser.add_argument("-sp", "--sampling", dest = "sampling", default = "none", help= "over/undersampling or none")
parser.add_argument("-ni", "--n_iter", dest = "n_iter", default = 1, help= "number of iterations", type=int)
parser.add_argument("-fs", "--feature_sel", dest = "f_sel", default = "none", help= "feature selection method")
parser.add_argument("-ns", "--n_sel", dest = "n_sel", default = "200", help= "feature selected number", type=int)


# Filter warning messages
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

args = parser.parse_args()

# Folder paths
models_path = "models"
output_path = "output"
data_path = "data"

all_fps = ["mg2", "mg3", "dl", "ap", "tt", "func", "frags", "maccs"]
all_fps_count = [f"{ff}_count" for ff in all_fps[:-1]]

# Select fingeprints
if args.fps == ["all"]:
    fps = all_fps
elif args.fps == ["all_count"]:
    fps = all_fps_count
else:
    fps = args.fps

# Possible models.
if args.models == ["all"]:
    types_of_model = ["lr", "xgb", "rf", "dt", "bnb"]#, "mlp", "xgb", "svc", "ab", "knn",
else:
    types_of_model = args.models

# Scores to be optimized.
scores = args.scores

with open(f"fingerprints/{args.tg}_{args.ename}.pkl", "rb") as input_file:
    y = pickle.load(input_file)

# Iterate over fingerprints
for fp in fps:
    if fp in all_fps + all_fps_count:
        # Load fingerprint
        fp_file = f"{fp}_{args.ename}"
        with open(f"fingerprints/{fp_file}.pkl", "rb") as input_file:
            fingerprint = pickle.load(input_file)
            print(f"{fp_file} loaded")
        # Create folders
        os.makedirs(f"{data_path}/{fp_file}", exist_ok=True)
        os.makedirs(f"{models_path}/{fp_file}", exist_ok=True)
        os.makedirs(f"{output_path}/{fp_file}", exist_ok=True)
        # Apply feature selection (if any)
        fingerprint, selection = feature_select(args.f_sel, args.n_sel, fingerprint, y)
        # Iterating n times
        for n in range(args.n_iter):
            print("Iteration " + str(n))
            # Define random seed
            random_state = args.random_seed + n
            # Model grid parameters
            models = get_ML_models(random_state)
            # Define experiment name
            experiment = f"{args.sampling}_{str(random_state)}_{selection}"
            # Print fingerprint name
            print(f"Fingerprint: {fp_file}\n")
            # Creating scores dataframe
            scores_df = pd.DataFrame(columns=["model", "score", "A", "BA", "ROC", "MCC", "F1", "R", "P", "R1", "SP"])
            # Splitting data
            X_train, X_test, y_train, y_test = split_data(fingerprint, y, 0.2, random_state)
            # Over/undersampling
            X_train, y_train = sample_data(X_train, y_train, args.sampling, random_state)
            # Scaling for count fingeprints
            X_train, X_test = scale_data(X_train, X_test)
            # Saving data splits.
            save_splits(X_train, X_test, y_train, y_test, f"{data_path}/{fp_file}", experiment)
            # Training and testing the models
            scores_df = classify(X_train, X_test, y_train, y_test,
                                types_of_model, models, scores,
                                models_path, fp_file, experiment, scores_df)
            # Saving scores
            scores_df.to_csv(f"{output_path}/{fp_file}/scores_{experiment}.csv", index=False)
    else:
        print("invalid fingeprint name")