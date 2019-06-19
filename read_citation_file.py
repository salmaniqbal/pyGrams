import pickle
import pandas as pd
import sys

FILE = 'data/USPTO-citation-dictionary-family-pt1.pkl.bz2'
df = pd.read_pickle(FILE)
print(df.shape)

sys.exit()

with open('data/USPTO-citation-dictionary-family-pt1.pkl.bz2', 'rb') as f:
    data = pickle.load(f)
    pass

