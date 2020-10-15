# Function for generating syntetic data for tests.
# using the distribution in data/demographics.csv. Generated file is stored in data/patients.csv
import pandas as pd
import numpy as np
import argparse

demo = pd.read_csv('data/demographics.csv', index_col='Event')

parser = argparse.ArgumentParser(description='Generate data set.')
parser.add_argument('N', type=int, help='Number of patients')

args = parser.parse_args()

num = args.N

print("Generating",num,"patients...")

def _set_bin(val, p):
    return 1 if val<p else 0

df_out = pd.DataFrame(index=list(range(1,num+1)))
df_out.index = df_out.index.rename('pid')

for feat, row in demo.iterrows():
    if row['bin']==1:
        df_out[feat] = np.random.random(num)
        df_out[feat] = df_out[feat].apply(lambda r: _set_bin(r,row['mean']))
    else:
        df_out[feat] = np.random.normal(row['mean'],row['var'],num)

fname = 'patients'
df_out.to_csv('data/'+fname+'.csv', sep=';')


