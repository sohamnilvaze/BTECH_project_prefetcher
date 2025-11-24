from interpret.glassbox import ExplainableBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import json
from interpret import show
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

# Load Data
df = pd.read_csv("../mem_access_traces/reu_merged.csv")
X = df.drop("Fine_grained_Target", axis=1)
y = df["Fine_grained_Target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train EBM
ebm = ExplainableBoostingClassifier(random_state=42, interactions=50)
ebm.fit(X_train, y_train)

y_pred = ebm.predict(X_test)
print("EBM Accuracy =", accuracy_score(y_test, y_pred))

print("ebm classes:- ",ebm.classes_)
print("ebm feature types:- ",ebm.feature_types)
print("ebm features types in:- ",ebm.feature_types_in_)
print("ebm feature names:- ",ebm.feature_names)
print("ebm features_names in:- ",ebm.feature_names_in_)
print("ebm term names:- ",ebm.term_names_)
print("ebm term features:- ",ebm.term_features_)
print("ebm bins:-",ebm.bins_)
