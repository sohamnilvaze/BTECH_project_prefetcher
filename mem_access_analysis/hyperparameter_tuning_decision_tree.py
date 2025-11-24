import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("../mem_access_traces/merged.csv") 
print(df['Target'].value_counts())
print(df.info())


df_s = df.sample(frac = 1, random_state = 42).reset_index(drop = True)
df_s.drop(["start","length"],axis = 1, inplace = True)
X = df_s.iloc[:,:-1]
y = df_s["Target"]

# ----------------------------
# 2. Preprocess: scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 4. Define KNN and parameter grid
# ----------------------------
dt = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini'],   # impurity metrics, ['gini', 'entropy', 'log_loss']
    'splitter': ['best', 'random'],                  # split strategy
    'max_depth': [None, 5, 10, 20, 30, 50, 100],    # tree depth
    'min_samples_split': [2, 5, 10, 20, 50],        # min samples to split
    'min_samples_leaf': [1, 2, 4, 8, 16],           # min samples per leaf
    'max_features': [None, 'sqrt', 'log2'],         # features to consider
    'max_leaf_nodes': [None, 10, 20, 50, 100],      # max leaf nodes
    'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1],  # impurity threshold
    'class_weight': ['balanced'],              # handle class imbalance
    'random_state': [42]                             # reproducibility
}


# ----------------------------
# 5. Stratified K-Fold to preserve imbalance
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# 6. Grid Search
# ----------------------------
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring='accuracy',  # or 'f1_macro' if imbalance is strong
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# ----------------------------
# 7. Best parameters and performance
# ----------------------------
print("\nBest Parameters:")
print(grid_search.best_params_)
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# ----------------------------
# 8. Evaluate on Test Set
# ----------------------------
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))