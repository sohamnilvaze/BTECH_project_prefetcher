# # from rulefit import RuleFit
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import classification_report, accuracy_score
# # from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# # from sklearn.preprocessing import StandardScaler

# # # Load dataset
# # df = pd.read_csv("../mem_access_traces/reu_merged.csv")
# # X = df.drop("Fine_grained_Target", axis=1)
# # y = df["Fine_grained_Target"]

# # # Split before scaling
# # X_train_raw, X_test_raw, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, stratify=y, random_state=42
# # )

# # # Scale ONLY for the linear part (RuleFit handles tree part separately)
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train_raw)
# # X_test = scaler.transform(X_test_raw)

# # # Build stronger tree generator
# # tree_gen = GradientBoostingClassifier(
# #     n_estimators=500,       # more trees → more rules
# #     max_depth=50,           # deeper rules
# #     learning_rate=0.05,
# #     subsample=0.8,
# #     random_state=42
# # )

# # # Train RuleFit
# # rf = RuleFit(
# #     tree_generator=tree_gen,
# #     rfmode='classify',
# #     model_type='rl',   # rules + linear terms
# #     random_state=42,
# #     max_iter=500       # more Lasso iterations
# # )

# # print("\nTraining RuleFit...")
# # rf.fit(X_train, y_train, feature_names=X.columns)
# # print("Training complete.")

# # # Evaluate
# # y_pred = rf.predict(X_test)
# # print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
# # print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # # Extract rules
# # rules = rf.max_rules()
# # print("Rules:-",rules)
# # rules = rules[rules.coef != 0].sort_values("importance", ascending=False)

# # # Write ALL rules to a text file
# # output_file = "rulefit_ba_merged.txt"
# # with open(output_file, "w") as f:
# #     for i, r in rules.iterrows():
# #         f.write(f"Rule: {r['rule']}\n")
# #         f.write(f"Coefficient: {r['coef']}\n")
# #         f.write(f"Importance: {r['importance']}\n")
# #         f.write("\n")

# # print(f"\nSaved ALL rules to {output_file}")


# # '''
# # merged1.csv
# # Accuracy:- 89.5%
# # rules:-
# # Delta_with_1_last_read_zeros_frac <= -0.7625492513179779
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779
# # Delta_with_1_last_read_zeros_frac <= -0.7625492513179779 & Delta_with_1_last_write_unique <= 1.4494749903678894
# # Delta_with_1_last_read_zeros_frac <= -0.7625492513179779 & Delta_with_1_last_write_unique > 1.4494749903678894
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power <= -0.7087143063545227 & Delta_with_1_last_read_mode_count <= -0.08130910992622375
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power <= -0.7087143063545227 & Delta_with_1_last_read_mode_count > -0.08130910992622375
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median <= -0.9000981152057648 & Delta_with_1_last_read_mean <= -0.4520825147628784
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median <= -0.9000981152057648 & Delta_with_1_last_read_mean > -0.4520825147628784
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median > -0.9000981152057648 & Delta_with_1_last_write_mode <= -4.3304760456085205
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_1_last_read_fft_dom_power > -0.7087143063545227 & Delta_with_1_last_read_median > -0.9000981152057648 & Delta_with_1_last_write_mode > -4.3304760456085205
# # '''

# # '''
# # merged2.csv
# # Delta_with_1_last_read_zeros_frac <= -0.7625492513179779
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779
# # Delta_with_1_last_read_zeros_frac <= -0.7625492513179779 & Delta_with_2_last_write_unique <= 1.7940125465393066
# # Delta_with_1_last_read_zeros_frac <= -0.7625492513179779 & Delta_with_2_last_write_unique > 1.7940125465393066 & Delta_with_2_last_read_std <= 0.1584453284740448
# # Delta_with_1_last_read_zeros_frac <= -0.7625492513179779 & Delta_with_2_last_write_unique > 1.7940125465393066 & Delta_with_2_last_read_std > 0.1584453284740448
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_2_last_write_ac_first_peak <= 4.249490737915039 & Delta_with_2_last_read_mean <= -1.4302083849906921 & Delta_with_2_last_read_mode_count <= 0.24077913165092468
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_2_last_write_ac_first_peak <= 4.249490737915039 & Delta_with_2_last_read_mean <= -1.4302083849906921 & Delta_with_2_last_read_mode_count > 0.24077913165092468
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_2_last_write_ac_first_peak <= 4.249490737915039 & Delta_with_2_last_read_mean > -1.4302083849906921 & Delta_with_1_last_read_median <= -0.9000981152057648
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_2_last_write_ac_first_peak <= 4.249490737915039 & Delta_with_2_last_read_mean > -1.4302083849906921 & Delta_with_1_last_read_median > -0.9000981152057648
# # Delta_with_1_last_read_zeros_frac > -0.7625492513179779 & Delta_with_2_last_write_ac_first_peak > 4.249490737915039
# # '''

# # '''
# # ba_merged.csv

# # '''


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split

# from rulefit import RuleFit

# df = pd.read_csv("../mem_access_traces/ba_merged.csv",index_col=0)
# y = df.Fine_grained_Target.values
# x = df.drop("Fine_grained_Target",axis=1)
# features = x.columns
# x = x.values


# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.1, random_state=42, stratify=y
# )


# rf = RuleFit()
# rf.fit(X_train,y_train,feature_names = features)

# # rf.predict(X_test)

# rules = list(rf.rule_ensemble.rules)
# rule_coefs = rf.coef_[-len(rules):]

# rows = []
# for rule, coef in zip(rules, rule_coefs):
#     if coef != 0:
#         rows.append([str(rule), coef, rule.support])

# df_rules = pd.DataFrame(rows, columns=["rule", "coef", "support"])
# print(df_rules.sort_values("coef", ascending=False))


from rulefit import RuleFit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# Load cleaned / discretized dataset
df = pd.read_csv("../mem_access_traces/reu_merged.csv")

X = df.drop("Fine_grained_Target", axis=1)
y = df["Fine_grained_Target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Stronger trees to generate more rules
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9
)

rf = RuleFit(
    tree_generator=gb,
    rfmode='classify',
    model_type='rl',
    random_state=42,
    max_rules=2000,
    max_iter = 100
)

print("Training...")
rf.fit(X_train.values, y_train.values, feature_names=X.columns)
print("Training completed.")

# Evaluate
y_pred = rf.predict(X_test.values)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Extract rules
rules = rf.get_rules()
rules = rules[rules.coef != 0]  # keep only active rules
rules.to_csv("rulefit_rules.csv", index=False)

print("Saved rules:", rules.shape)
