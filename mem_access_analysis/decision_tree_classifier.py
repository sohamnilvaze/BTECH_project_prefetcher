from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Input: feats (features DataFrame), program_labels (list or Series of labels like 'seq', 'random', ...)
df = pd.read_csv("../mem_access_traces/merged.csv")
print(df['Target'].value_counts())
print(df.info())


df_s = df.sample(frac = 1, random_state = 42).reset_index(drop = True)
df_s.drop(["start","length"],axis = 1, inplace = True)
X = df_s.iloc[:,:-1]
y = df_s["Target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42,train_size=0.8)

# Train
dt = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10, random_state=42,min_samples_split=10, min_samples_leaf=2,max_features='log2',max_leaf_nodes=50,min_impurity_decrease=0.01,class_weight='balanced')
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Extract rules
# rules = export_text(dt, feature_names=list(X.columns))
# print("\nExtracted Rules:\n", rules)


# plt.figure(figsize=(20,10))
# plot_tree(dt, feature_names=X.columns, class_names=[str(i) for i in sorted(y.unique())], filled=True)
# plt.show()

'''
getting accuracy of around 88.09%
'''
