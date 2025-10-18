from rulefit import RuleFit
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


rf = RuleFit(tree_size=4, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Extract top rules
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
print(rules.head(10))

