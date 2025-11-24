import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret.provider import InlineProvider

# 1) Load data (has columns like the ones in your sample, including Target)
df = pd.read_csv("mem_access_traces/ba_merged.csv")


label_col = "Fine_grained_Target"

# Features = all columns except Target and Fine_grained_Target
drop_cols = ["Target", "Fine_grained_Target"]
X = df.drop(columns=drop_cols)
y = df[label_col].astype(int)   

seed = 42
np.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=seed,
    stratify=y     # keeps class 1..8 proportions
)

ebm = ExplainableBoostingClassifier(
    random_state=42,
    max_bins=512,          # finer binning for continuous stats
    max_interaction_bins=128,
    max_rounds=700,        # more boosting rounds
    learning_rate=0.01,    # maybe slightly lower for stability
    interactions=3   # start with 0 for pure additive, then try >0
)
ebm.fit(X_train, y_train)

print("Classes seen:", ebm.classes_)   # should be array([1,2,3,4,5,6,7,8])


from sklearn.metrics import accuracy_score

y_pred = ebm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
y_proba = ebm.predict_proba(X_test)
print("ROC AUC (ovr):", roc_auc_score(y_test, y_proba, multi_class='ovr'))


# from interpret import show
# from interpret.provider import InlineProvider

# InlineProvider()
# global_exp = ebm.explain_global()
# show(global_exp)
# local_exp = ebm.explain_local(X_test.iloc[:5], y_test.iloc[:5])
# show(local_exp)