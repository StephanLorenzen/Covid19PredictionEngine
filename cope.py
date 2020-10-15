# Loads the given model and data and predicts (labels and probabilities).
import pandas as pd
import numpy as np
import pickle
import joblib
import argparse
import sys
import os
from sklearn.ensemble import RandomForestClassifier as RFC

### Configuration
_datasettypes = ['covid19_infected_0','hospital_admission_12','icu_admission_-12','icu_admission_12']
_all_targets  = ['hospital_admitted','ICU_admitted','used_ventilator','death']
_rel_targets  = {
        'covid19_infected_0':_all_targets[:],
        'hospital_admission_12':_all_targets[1:],
        'icu_admission_-12':_all_targets[2:],
        'icu_admission_12':_all_targets[2:],
        }
_feature_sets = ['basicinfo','disease','temporal_features','tests']

### Loads the model, default path is "models/"
def load_model(datasettype, feature_set, target, path="models/"):
    model = "rf"
    model_save_name = path + datasettype + "_" + model + "_" + "_".join(feature_set) + "_" + target

    # Load list of features
    [_, colnames] = pickle.load(open(model_save_name+"_dtypes_colnames.pkl", "rb"))

    # Load model
    model = joblib.load(model_save_name+".model")

    return model, colnames

### Load patients from given path
def load_data(path, features):
    if not os.path.isfile(path):
        return None
    data = pd.read_csv(path, sep=';', index_col='pid')
    return data[features].values

### API input mapping
_thresholds_map = dict(zip(['test','hospital','pre-icu','post-icu'],_datasettypes))
_target_map     = dict(zip(['hospital','icu','ventilator','death'],_all_targets))
_fset_map       = {
        'basic': _feature_sets[:1],
        'comorbidities': _feature_sets[:2],
        'temporal': _feature_sets[:3],
        'tests': _feature_sets[:],
        }


### Usage
### python cope.py <data-file> [-t <time>] [-p <predict>] [-f <features>]
parser = argparse.ArgumentParser(description='The COvid19 Prediction Engine (COPE).')
parser.add_argument('data', type=str, help='data set')
parser.add_argument('-t','--time', default='test', metavar='T', choices=list(_thresholds_map.keys()),
        help='Time of prediction. Must be one of {'+','.join(_thresholds_map.keys())+'}, default is \'test\'.')
parser.add_argument('-p','--predict', default='death', metavar='P', choices=list(_target_map.keys()),
        help='Target to predict. Must be one of {'+','.join(_target_map.keys())+'}, default is \'death\'.')
parser.add_argument('-f','--features', default='comorbidities', metavar='FS', choices=list(_fset_map.keys()),
        help='Feature set FS to use. Must be one of {'+','.join(_fset_map.keys())+'}, default is \'comorbidities\'.')

# Parse and get args
args = parser.parse_args()
data_path = args.data
threshold = _thresholds_map[args.time]
target    = _target_map[args.predict]
fset      = _fset_map[args.features]

# Check feature use at testing time
if threshold=='covid19_infected_0' and len(fset)>2:
    print("The given feature set is not applicable for time='test', as temporal features and in-hospital tests are not available at this time.")
    sys.exit(1)

# Check relevant target
if target not in _rel_targets[threshold]:
    print("Predict='"+str(args.predict)+"' is not relevant for time='"+str(args.time)+"'")
    sys.exit(1)

# Load model
model, cols = load_model(threshold, fset, target)
# Load data
data = load_data(data_path, cols)
if data is None:
    print("Data file not found!")
    sys.exit(1)

print("COvid Prediction Engine.")
print("Time              = '"+args.time+"'")
print("Prediction target = '"+args.predict+"'")
print("Features          = '"+args.features+"'")
print("")
print("Output (labels):")
print(" ".join([str(x) for x in model.predict(data)]))
print("Output (probabilities in %):")
print(" ".join([str(round(x*100,1)) for x in model.predict_proba(data)[:,1]]))

