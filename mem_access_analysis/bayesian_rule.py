from imodels import BayesianRuleListClassifier
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


brl = BayesianRuleListClassifier(listlengthprior=3, listwidthprior=2, random_state=42)
brl.fit(X_train, y_train)

y_pred = brl.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Extract rules
print("\nExtracted Rule List:\n", brl.print_rule_list())
