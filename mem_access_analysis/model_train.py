from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import pandas as pd

# Input: feats (features DataFrame), program_labels (list or Series of labels like 'seq', 'random', ...)
df = pd.read_csv("../mem_access_traces/merged.csv")
print(df['Target'].value_counts())
print(df.info())


df_s = df.sample(frac = 1, random_state = 42).reset_index(drop = True)
df_s.drop(["start","length"],axis = 1, inplace = True)
X = df_s.iloc[:,:-1]
y = df_s["Target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42,train_size=0.8)

models = {
    #"Logistic Regression": LogisticRegression(class_weight="balanced",max_iter=2000),#90.03%
    #"KNN": KNeighborsClassifier(n_neighbors=2, weights='distance', metric='minkowski',algorithm='brute'),
    #"Random Forest": RandomForestClassifier(),
    "SVC": SVC(kernel='linear',gamma=1e-3,C = 0.1),
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} for {clf} ===")
    print(f"Accuracy: {acc:.4f}")
    #print(classification_report(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))
    if name == "Random Forest":
        print("Feature importances:", clf.feature_importances_)

'''
List of models:- logistic regression, SVC, KNClassifier, RandomForestClassifier, XGBoostClassifier, LightGBM, catboost, Gradient Boosting Classifier, MLP
'''