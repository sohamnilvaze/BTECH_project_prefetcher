from rulefit import RuleFit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


# Load your dataset
df = pd.read_csv("../mem_access_traces/merged.csv")
X = df.drop("Target", axis=1)
y = df["Target"]

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Train RuleFit
print("Training the rulefit model")
rf = RuleFit(
    tree_generator=GradientBoostingClassifier(n_estimators=50, max_depth=4),
    rfmode='classify',
    model_type='rl',   # both rules + linear terms
    random_state=42,
    max_iter = 100
)
rf.fit(X_train, y_train, feature_names=X.columns)
print("Rulefit model training completed.")
#Evaluate
y_pred = rf.predict(X_test)
# y_pred_proba = rf.predict_proba(X_test)


print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# Extract rules
#rules = rf.get_rules()
#rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
#print(rules.head(10))
for rule in rf.rule_ensemble.rules[:10]:
    print(rule)

'''
Accuracy:- 89.5%
rules:-
Delta_with_1_last_read_zeros_frac <= -0.7625492513179779
Delta_with_1_last_read_zeros_frac > -0.7625492513179779
Delta_with_1_last_read_zeros_frac <= -0.7625492513179779 & Delta_with_1_last_write_unique <= 1.4494749903678894
Delta_with_1_last_read_zeros_frac <= -0.7625492513179779 & Delta_with_1_last_write_unique > 1.4494749903678894
Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power <= -0.7087143063545227 & Delta_with_1_last_read_mode_count <= -0.08130910992622375
Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power <= -0.7087143063545227 & Delta_with_1_last_read_mode_count > -0.08130910992622375
Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median <= -0.9000981152057648 & Delta_with_1_last_read_mean <= -0.4520825147628784
Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median <= -0.9000981152057648 & Delta_with_1_last_read_mean > -0.4520825147628784
Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median > -0.9000981152057648 & Delta_with_1_last_write_mode <= -4.3304760456085205
Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median > -0.9000981152057648 & Delta_with_1_last_write_mode > -4.3304760456085205
'''

