from utilities import *
import os
import numpy as np
import pickle
import json
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
from rdkit import Chem
import argparse

# Define the argument parser
parser = argparse.ArgumentParser()
# Add arguments for dataframe path, structure column name, target column name, and extension name
parser.add_argument("-dfp", "--dataframe_path", dest="df_path", help="dataframe path")
parser.add_argument("-sc", "--structure_column", dest="st_col", default="Structure", help="structure column name")
parser.add_argument("-tc", "--target_column", dest="tg_col", default="Target", help="target column name")
parser.add_argument("-ext", "--extension_name", dest="ext_name", default="", help="experiment extension name")
parser.add_argument("-it", "--input_type", dest="input_type", default="smile", help="smile or mol")

# Parse the arguments
args = parser.parse_args()

# Filter warning messages
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# Read structures
df = pd.read_csv(args.df_path)

# If the input type is SMILES, convert them to molecules
if args.input_type == "smile":
    PandasTools.AddMoleculeColumnToFrame(df, args.st_col)
    mols = df["ROMol"]
# Otherwise, use the provided molecule column directly
else:
    mols = df[args.st_col].apply(lambda x : Chem.MolFromMolBlock(x) if pd.notnull(x) else np.nan)

# Define file extension (e.g., "ext) 
ename = args.ext_name

fp_path = "fingerprints"
os.makedirs(f"{fp_path}", exist_ok=True)

# Save target
pickle.dump(np.array(df[args.tg_col]), open(f"{fp_path}/y_{ename}.pkl",'wb'))

# Compute and save all fingeprints
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)
x_mg2 = np.array([fp_gen.GetFingerprintAsNumPy(m) for m in mols])
x_mg2_count = np.array([fp_gen.GetCountFingerprintAsNumPy(m) for m in mols])
pickle.dump(x_mg2, open(f"{fp_path}/mg2_{ename}.pkl",'wb'))
pickle.dump(x_mg2_count, open(f"{fp_path}/mg2_count_{ename}.pkl",'wb'))

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024, includeChirality=True)
x_mg3 = np.array([fp_gen.GetFingerprintAsNumPy(m) for m in mols])
x_mg3_count = np.array([fp_gen.GetCountFingerprintAsNumPy(m) for m in mols])
pickle.dump(x_mg3, open(f"{fp_path}/mg3_{ename}.pkl",'wb'))
pickle.dump(x_mg3_count, open(f"{fp_path}/mg3_count_{ename}.pkl",'wb'))

fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7, fpSize=1024) # RDKIT daylight
x_dl = np.array([fp_gen.GetFingerprintAsNumPy(m) for m in mols])
x_dl_count = np.array([fp_gen.GetCountFingerprintAsNumPy(m) for m in mols])
pickle.dump(x_dl, open(f"{fp_path}/dl_{ename}.pkl",'wb'))
pickle.dump(x_dl_count, open(f"{fp_path}/dl_count_{ename}.pkl",'wb'))

fp_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True) # Atom pairs
x_ap = np.array([fp_gen.GetFingerprintAsNumPy(m) for m in mols])
x_ap_count = np.array([fp_gen.GetCountFingerprintAsNumPy(m) for m in mols])
pickle.dump(x_ap, open(f"{fp_path}/ap_{ename}.pkl",'wb'))
pickle.dump(x_ap_count, open(f"{fp_path}/ap_count_{ename}.pkl",'wb'))

fp_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024, includeChirality=True) # Topological torsion
x_tt = np.array([fp_gen.GetFingerprintAsNumPy(m) for m in mols])
x_tt_count = np.array([fp_gen.GetCountFingerprintAsNumPy(m) for m in mols])
pickle.dump(x_tt, open(f"{fp_path}/tt_{ename}.pkl",'wb'))
pickle.dump(x_tt_count, open(f"{fp_path}/tt_count_{ename}.pkl",'wb'))

x_maccs = np.array([np.array(MACCSkeys.GenMACCSKeys(m))[1:] for m in mols])
pickle.dump(x_maccs, open(f"{fp_path}/maccs_{ename}.pkl",'wb'))

x_frags = np.array([np.array(get_frag_fp(m, True)) for m in mols])
x_frags_count = np.array([np.array(get_frag_fp(m)) for m in mols])
pickle.dump(x_frags, open(f"{fp_path}/frags_{ename}.pkl",'wb'))
pickle.dump(x_frags_count, open(f"{fp_path}/frags_count_{ename}.pkl",'wb'))

# Read fingerprint structure dictionary
with open("DayLightFuncGroups.json", "r") as f:
    DayLightFuncGroups = json.load(f)

x_func = np.array([np.array(get_all_func_fp(DayLightFuncGroups, m, True)) for m in mols])
x_func_count = np.array([np.array(get_all_func_fp(DayLightFuncGroups, m)) for m in mols])
pickle.dump(x_func, open(f"{fp_path}/func_{ename}.pkl",'wb'))
pickle.dump(x_func_count, open(f"{fp_path}/func_count_{ename}.pkl",'wb'))
