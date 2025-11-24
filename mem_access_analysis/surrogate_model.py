import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv("../mem_access_traces/reu_merged.csv")
X = df.drop("Fine_grained_Target", axis=1)
y = df["Fine_grained_Target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train strong model (teacher)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)
xgb.fit(X_train, y_train)

print("XGBoost Test Accuracy =", accuracy_score(y_test, xgb.predict(X_test)))

# ------------------------------
# Train surrogate tree
# ------------------------------
y_train_preds = xgb.predict(X_train)  # or use predict_proba for soft surrogate

surrogate = DecisionTreeClassifier(
    max_depth=5,          # interpretable
    min_samples_leaf=20,
    random_state=42
)
surrogate.fit(X_train, y_train_preds)

print("Surrogate Tree Accuracy on Teacher =", accuracy_score(y_train_preds, surrogate.predict(X_train)))

tree_rules = export_text(surrogate, feature_names=list(X.columns))

# Save rules to text file
with open("surrogate_rules_reu.txt", "w") as f:
    f.write(tree_rules)

# Save rules to CSV (one rule per line)
rules_list = tree_rules.split("\n")
pd.DataFrame({"rule": rules_list}).to_csv("surrogate_rules.csv", index=False)
