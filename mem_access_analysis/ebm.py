from interpret.glassbox import ExplainableBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import json

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

# Evaluate
y_pred = ebm.predict(X_test)
print("EBM Accuracy =", accuracy_score(y_test, y_pred))

# Exporting Main Effects
main_effects = {}
for i, feature in enumerate(X.columns):
    main_effects[feature] = {
        "bins": list(ebm.bins_[i]),
        "scores": list(ebm.term_scores_[i])
    }

print(ebm.feature_names)

# Exporting Pairwise Interactions
interactions_export = {}
for idx, pair in enumerate(ebm.feature_names):
    if len(pair) == 2:
        fname = f"{X.columns[pair[0]]}__{X.columns[pair[1]]}"
        interactions_export[fname] = np.array(ebm.term_scores_[len(X.columns) + idx]).tolist()

# Write to JSON (best way to preserve structure)
with open("ebm_main_effects_reu_merged.json", "w") as f:
    json.dump(main_effects, f, indent=4)

with open("ebm_interactions_reu_merged.json", "w") as f:
    json.dump(interactions_export, f, indent=4)

# Also export readable TXT
with open("ebm_readable_rules_reu_merged.txt", "w") as f:
    f.write("=== EBM MAIN EFFECT FUNCTIONS ===\n\n")
    for feature, values in main_effects.items():
        f.write(f"\nFeature: {feature}\n")
        f.write("Bins & Contribution Scores:\n")
        for b, s in zip(values["bins"], values["scores"]):
            f.write(f"  Bin: {b} -> Score: {s}\n")

    f.write("\n\n=== INTERACTIONS ===\n\n")
    for feature, matrix in interactions_export.items():
        f.write(f"\nInteraction: {feature}\n")
        f.write(f"Matrix shape: {len(matrix)} x {len(matrix[0])}\n\n")
