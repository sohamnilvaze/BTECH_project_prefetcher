import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['distance'],
    'metric': ['minkowski', 'manhattan', 'euclidean', 'chebyshev'],
    'p': [1, 2],  # Manhattan = 1, Euclidean = 2
    'leaf_size': [20, 30, 40, 50],
    'algorithm': ['auto']
}

# ----------------------------
# 5. Stratified K-Fold to preserve imbalance
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# 6. Grid Search
# ----------------------------
grid_search = GridSearchCV(
    estimator=knn,
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
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

'''
Best accuracy: 0.8391
auto, leaf_size = 20, metric = minkowski, n_neighbours = 3, p = 2, weights = 'distance'
test accuracy:- 0.8261
'''