import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("../mem_access_traces/merged.csv")
print(df['Target'].value_counts())
print(df.info())


df_s = df.sample(frac = 1, random_state = 42).reset_index(drop = True)
df_s.drop(["start","length"],axis = 1, inplace = True)
X = df_s.iloc[:,:-1]
le = LabelEncoder()
y = le.fit_transform(df_s["Target"])

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

xgc = xgb.XGBClassifier()   

param_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'scale_pos_weight': [1, 2, 5]  # useful for imbalance
}

# ----------------------------
# 5. Stratified K-Fold to preserve imbalance
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# 6. Grid Search
# ----------------------------
grid_search = GridSearchCV(
    estimator=xgc,
    param_grid=param_grid,
    scoring='f1_macro',  # or 'f1_macro' if imbalance is strong
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

