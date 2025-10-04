import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict

df = pd.read_csv("ghb_1/f1_strided.csv")

features_col = "last_1_read_delta"
X = df[features_col].values.reshape(-1,1)

#Quantize the deltas into buckets here 10 buckets with uniform width
n_bins = 10
discretizer= KBinsDiscretizer(n_bins=n_bins,encode="ordinal",strategy="uniform")
X_binned = discretizer.fit_transform(X).astype(int).flatten()

