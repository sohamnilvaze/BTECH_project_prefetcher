import os
import argparse
import glob
import json
import math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def load_csvs_from_dir(input_dir, pattern="*.csv"):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    return files

def read_df(path, target_col):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in {path}")
    # drop columns that are obviously metadata if present
    for c in ["start", "length", "index"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df

# -------------------------
# Model & extractor
# -------------------------
def train_tree_for_version(df, target_col, test_size=0.2, random_state=42, max_depth=6):
    print("Training the model")
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Encode target to 0..K-1 if needed
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Keep feature names
    feat_names = X.columns.tolist()

    # Simple imputation for NaNs: fill 0 (or could use median)
    X = X.fillna(0)

    # Scaling: decision trees don't need scaling, but permutation and SHAP are fine either way.
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=feat_names)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y_enc, stratify=y_enc, test_size=test_size, random_state=random_state)

    dt = DecisionTreeClassifier()

    param_grid = {
        'criterion': ['gini','entropy','log_loss'],   # impurity metrics, ['gini', 'entropy', 'log_loss']
        'splitter': ['best', 'random'],                  # split strategy
        'max_depth': [None, 5, 10, 20],    # tree depth
        'min_samples_split': [2, 5, 10],        # min samples to split
        'min_samples_leaf': [1, 2, 4, 8, 16],           # min samples per leaf
        'max_features': [None, 'sqrt', 'log2'],         # features to consider
        'max_leaf_nodes': [None, 10, 20, 50, 100],      # max leaf nodes
        'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1],  # impurity threshold
        'class_weight': ['balanced'],              # handle class imbalance
        'random_state': [42]                             # reproducibility
    }  

    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        scoring='accuracy',  # or 'f1_macro' if imbalance is strong
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train) 
    print("Model trained")

    best_dt = grid_search.best_estimator_
    y_pred = best_dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    meta = {
        "model": best_dt,
        "best_params": grid_search.best_params_,
        "scaler": scaler,
        "feat_names": feat_names,
        "label_encoder": le,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "accuracy_test": acc
    }

    return meta

def extract_splits_and_rules(clf, feat_names,txt_file):
    """
    Return:
      - splits: list of dicts {feature, threshold, node_id, depth, left_child, right_child}
      - rules_text: list of rule strings (export_text)
    """
    tree = clf.tree_
    features = tree.feature
    thresholds = tree.threshold
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    n_node_samples = tree.n_node_samples

    splits = []
    # compute depth by traversal
    stack = [(0, 0)]  # node_id, depth
    visited = set()
    while stack:
        node_id, depth = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        if features[node_id] != -2:  # not a leaf
            feat_idx = int(features[node_id])
            splits.append({
                "node_id": int(node_id),
                "feature_idx": feat_idx,
                "feature": feat_names[feat_idx],
                "threshold": float(thresholds[node_id]),
                "left_child": int(children_left[node_id]),
                "right_child": int(children_right[node_id]),
                "depth": int(depth),
                "n_samples": int(n_node_samples[node_id])
            })
            stack.append((children_right[node_id], depth+1))
            stack.append((children_left[node_id], depth+1))

    # extract text rules
    try:
        rules_text = export_text(clf, feature_names=feat_names)
        with open(txt_file, "w") as file_object:
            file_object.write(rules_text)
    except Exception:
        rules_text = ""
    return splits, rules_text

def compute_permutation_importance(clf, X_test, y_test, n_repeats=10):
    r = permutation_importance(clf, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    # returns importances: mean, std
    return r

def compute_shap_values_if_possible(clf, X_sample):
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)
        # shap_values shape depends on multiclass: list of arrays
        return shap_values, explainer
    except Exception as e:
        print("SHAP compute failed:", e)
        return None

def thresholds_by_feature(splits):
    d = defaultdict(list)
    for s in splits:
        d[s["feature"]].append(s["threshold"])
    return d

def summarize_splits(splits):
    # per feature: min/median/max thresholds and depth statistics
    per = {}
    for feat, thr_list in thresholds_by_feature(splits).items():
        arr = np.array(thr_list)
        per[feat] = {
            "n_splits": len(arr),
            "min_thr": float(np.min(arr)),
            "median_thr": float(np.median(arr)),
            "max_thr": float(np.max(arr))
        }
    return per

def compare_thresholds_across_versions(summary_dict):
    """
    summary_dict: version -> {feature: {n_splits,min_thr,median_thr,max_thr}}
    Returns DataFrame where rows=features, cols=versions * metrics
    """
    feats = set()
    for v in summary_dict:
        feats.update(summary_dict[v].keys())
    feats = sorted(list(feats))
    rows = []
    for feat in feats:
        row = {"feature": feat}
        for v in sorted(summary_dict.keys()):
            info = summary_dict[v].get(feat, {})
            row[f"{v}_n_splits"] = info.get("n_splits", 0)
            row[f"{v}_min_thr"] = info.get("min_thr", np.nan)
            row[f"{v}_median_thr"] = info.get("median_thr", np.nan)
            row[f"{v}_max_thr"] = info.get("max_thr", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def jaccard_rule_overlap(rules_text_a, rules_text_b):
    """Compute Jaccard over clause sets extracted from rules text"""
    def normalize_rule_text(rt):
        # take each line as a rule, split conditions by '&', strip whitespace
        lines = [l.strip() for l in rt.splitlines() if l.strip()]
        clauses = set()
        for ln in lines:
            # remove trailing -> class info if present
            if "->" in ln:
                ln = ln.split("->")[0].strip()
            parts = [p.strip() for p in ln.split("and") if p.strip()]
            for p in parts:
                clauses.add(p)
        return clauses
    A = normalize_rule_text(rules_text_a)
    B = normalize_rule_text(rules_text_b)
    if len(A) == 0 and len(B) == 0:
        return 1.0
    if len(A.union(B)) == 0:
        return 0.0
    return len(A.intersection(B)) / len(A.union(B))


df_v1 = read_df("../mem_access_traces/reu_merged.csv","Fine_grained_Target")
df_v2 = read_df("../mem_access_traces/reu_merged2.csv","Fine_grained_Target")
df_v3 = read_df("../mem_access_traces/reu_merged3.csv","Fine_grained_Target")
df_v4 = read_df("../mem_access_traces/reu_merged4.csv","Fine_grained_Target")
# df_v5 = read_df("../mem_access_traces/ba_merged5.csv","Fine_grained_Target")
# df_v6 = read_df("../mem_access_traces/ba_merged6.csv","Fine_grained_Target")
# df_v7 = read_df("../mem_access_traces/ba_merged7.csv","Fine_grained_Target")
# df_v8 = read_df("../mem_access_traces/ba_merged8.csv","Fine_grained_Target")

meta_v1 = train_tree_for_version(df_v1,"Fine_grained_Target")
meta_v2 = train_tree_for_version(df_v2,"Fine_grained_Target")
meta_v3 = train_tree_for_version(df_v3,"Fine_grained_Target")
meta_v4 = train_tree_for_version(df_v4,"Fine_grained_Target")
# meta_v5 = train_tree_for_version(df_v5,"Fine_grained_Target")
# meta_v6 = train_tree_for_version(df_v6,"Fine_grained_Target")
# meta_v7 = train_tree_for_version(df_v7,"Fine_grained_Target")
# meta_v8 = train_tree_for_version(df_v8,"Fine_grained_Target")

#print(f"Metadata obtained for version 1: {meta_v1}")
#print(f"Metadata obtained for version 2: {meta_v2}")
print("Accuracy for version1: ",meta_v1["accuracy_test"])
print("Accuracy for version2: ",meta_v2["accuracy_test"])
print("Accuracy for version3: ",meta_v3["accuracy_test"])
print("Accuracy for version4: ",meta_v4["accuracy_test"])
# print("Accuracy for version5: ",meta_v5["accuracy_test"])
# print("Accuracy for version6: ",meta_v6["accuracy_test"])
# print("Accuracy for version7: ",meta_v7["accuracy_test"])
# print("Accuracy for version8: ",meta_v8["accuracy_test"])

print("Best params for version 1:- ", meta_v1["best_params"])
print("Best params for version 2q:- ", meta_v2["best_params"])
print("Best params for version 3:- ", meta_v3["best_params"])
print("Best params for version 4:- ", meta_v4["best_params"])


splits_v1, rules_v1 = extract_splits_and_rules(meta_v1["model"],meta_v1["feat_names"],"reu_v1_rules.txt")
splits_v2, rules_v2 = extract_splits_and_rules(meta_v2["model"],meta_v2["feat_names"],"reu_v2_rules.txt")
splits_v3, rules_v3 = extract_splits_and_rules(meta_v3["model"],meta_v3["feat_names"],"reu_v3_rules.txt")
splits_v4, rules_v4 = extract_splits_and_rules(meta_v4["model"],meta_v4["feat_names"],"reu_v4_rules.txt")
splits_df1 = pd.DataFrame(splits_v1)
splits_df2 = pd.DataFrame(splits_v2)
splits_df3 = pd.DataFrame(splits_v3)
splits_df4 = pd.DataFrame(splits_v4)
splits_df1.to_csv("../mem_access_analysis/tree_analysis/reu_splits_v1.csv",index=False)
splits_df2.to_csv("../mem_access_analysis/tree_analysis/reu_splits_v2.csv",index=False)
splits_df3.to_csv("../mem_access_analysis/tree_analysis/reu_splits_v3.csv",index=False)
splits_df4.to_csv("../mem_access_analysis/tree_analysis/reu_splits_v4.csv",index=False)

perm_v1 = compute_permutation_importance(meta_v1["model"],meta_v1["X_test"],meta_v1["y_test"],n_repeats=10)
perm_v2 = compute_permutation_importance(meta_v2["model"],meta_v2["X_test"],meta_v2["y_test"],n_repeats=10)
perm_v3 = compute_permutation_importance(meta_v3["model"],meta_v3["X_test"],meta_v3["y_test"],n_repeats=10)
perm_v4 = compute_permutation_importance(meta_v4["model"],meta_v4["X_test"],meta_v4["y_test"],n_repeats=10)


perm_df1 = pd.DataFrame({
            "feature": meta_v1["X_test"].columns,
            "perm_mean": perm_v1["importances_mean"],
            "perm_std": perm_v1["importances_std"]
        }).sort_values("perm_mean", ascending=False)
perm_df1.to_csv("../mem_access_analysis/tree_analysis/reu_perm_v1.csv",index=False)

perm_df2 = pd.DataFrame({
            "feature": meta_v2["X_test"].columns,
            "perm_mean": perm_v2["importances_mean"],
            "perm_std": perm_v2["importances_std"]
        }).sort_values("perm_mean", ascending=False)
perm_df2.to_csv("../mem_access_analysis/tree_analysis/reu_perm_v2.csv",index=False)

perm_df3 = pd.DataFrame({
            "feature": meta_v3["X_test"].columns,
            "perm_mean": perm_v3["importances_mean"],
            "perm_std": perm_v3["importances_std"]
        }).sort_values("perm_mean", ascending=False)
perm_df3.to_csv("../mem_access_analysis/tree_analysis/reu_perm_v3.csv",index=False)

perm_df4 = pd.DataFrame({
            "feature": meta_v4["X_test"].columns,
            "perm_mean": perm_v4["importances_mean"],
            "perm_std": perm_v4["importances_std"]
        }).sort_values("perm_mean", ascending=False)
perm_df4.to_csv("../mem_access_analysis/tree_analysis/reu_perm_v4.csv",index=False)

shap_val_v1, explainer_v1 = compute_shap_values_if_possible(meta_v1["model"],meta_v1["X_test"].iloc[:min(200,len(meta_v1["X_test"]))])
shap_val_v2, explainer_v2 = compute_shap_values_if_possible(meta_v2["model"],meta_v2["X_test"].iloc[:min(200,len(meta_v2["X_test"]))])
shap_val_v3, explainer_v3 = compute_shap_values_if_possible(meta_v3["model"],meta_v3["X_test"].iloc[:min(200,len(meta_v3["X_test"]))])
shap_val_v4, explainer_v4 = compute_shap_values_if_possible(meta_v4["model"],meta_v4["X_test"].iloc[:min(200,len(meta_v4["X_test"]))])

'''
if isinstance(shap_val_v1, list):
    mean_abs_v1 = np.mean([np.abs(sv).mean(axis=0) for sv in shap_val_v1], axis=0)
else:
    mean_abs_v1 = np.abs(shap_val_v1).mean(axis=0)
shap_df_v1 = pd.DataFrame({
                "feature": meta_v1["X_test"].columns,
                "mean_abs_shap": mean_abs_v1
            }).sort_values("mean_abs_shap", ascending=False)
shap_df_v1.to_csv("../mem_access_analysis/tree_analysis/shap_v1.csv",index=False)

if isinstance(shap_val_v2, list):
    mean_abs_v2 = np.mean([np.abs(sv).mean(axis=0) for sv in shap_val_v2], axis=0)
else:
    mean_abs_v2 = np.abs(shap_val_v2).mean(axis=0)
shap_df_v2 = pd.DataFrame({
                "feature": meta_v2["X_test"].columns,
                "mean_abs_shap": mean_abs_v2
            }).sort_values("mean_abs_shap", ascending=False)
shap_df_v2.to_csv("../mem_access_analysis/tree_analysis/shap_v2.csv",index=False)
'''

feature_thresholds_v1 = thresholds_by_feature(splits_v1)
feature_thresholds_v2 = thresholds_by_feature(splits_v2)
feature_thresholds_v3 = thresholds_by_feature(splits_v3)
feature_thresholds_v4 = thresholds_by_feature(splits_v4)

split_summary_v1 = summarize_splits(splits_v1)
split_summary_v2 = summarize_splits(splits_v2)
split_summary_v3 = summarize_splits(splits_v3)
split_summary_v4 = summarize_splits(splits_v4)

#print(f"Splits obtained for version1: {splits_v1}")
#print(f"Splits obtained for version2: {splits_v2}")

# print(f"Rules obtained for version 1: {rules_v1}")
# print(f"Rules obtained for version 2: {rules_v2}")

# print(f"Permutation importance for version1: {perm_v1}")
# print(f"Permutation importance for version2: {perm_v2}")

print("Splits summary for v1: ", split_summary_v1)
print("Splits summary for v2: ", split_summary_v2)


summary_dict = {}
summary_dict["version 1"] = split_summary_v1
summary_dict["version 2"] = split_summary_v2
summary_dict["version 3"] = split_summary_v3
summary_dict["version 4"] = split_summary_v4

thr_df = compare_thresholds_across_versions(summary_dict)

thr_df.to_csv("../mem_access_analysis/tree_analysis/reu_threshold_v1_v2_v3_v4.csv",index=False)

jaccard_sim_12 = jaccard_rule_overlap(rules_v1,rules_v2)
jaccard_sim_13 = jaccard_rule_overlap(rules_v1,rules_v3)
jaccard_sim_14 = jaccard_rule_overlap(rules_v1,rules_v4)
jaccard_sim_23 = jaccard_rule_overlap(rules_v2,rules_v2)
jaccard_sim_24 = jaccard_rule_overlap(rules_v2,rules_v4)
jaccard_sim_34 = jaccard_rule_overlap(rules_v3,rules_v4)


perm_ranks = {}
perm_df1["rank"] = perm_df1["perm_mean"].rank(ascending=False, method="min")
perm_ranks["version 1"] = perm_df1.set_index("feature")["rank"].to_dict()

perm_df2["rank"] = perm_df2["perm_mean"].rank(ascending=False, method="min")
perm_ranks["version 2"] = perm_df2.set_index("feature")["rank"].to_dict()

perm_df3["rank"] = perm_df3["perm_mean"].rank(ascending=False, method="min")
perm_ranks["version 3"] = perm_df3.set_index("feature")["rank"].to_dict()

perm_df4["rank"] = perm_df4["perm_mean"].rank(ascending=False, method="min")
perm_ranks["version 4"] = perm_df4.set_index("feature")["rank"].to_dict()

versions = {}
versions["version 1"] = {"meta": meta_v1, "splits": splits_v1, "rules_text": rules_v1, "perm_importance": perm_v1, "shap_value": shap_val_v1, "explainer": explainer_v1}
versions["version 2"] = {"meta": meta_v2, "splits": splits_v2, "rules_text": rules_v2, "perm_importance": perm_v2, "shap_value": shap_val_v2, "explainer": explainer_v2}
versions["version 3"] = {"meta": meta_v3, "splits": splits_v3, "rules_text": rules_v3, "perm_importance": perm_v3, "shap_value": shap_val_v3, "explainer": explainer_v3}
versions["version 4"] = {"meta": meta_v4, "splits": splits_v4, "rules_text": rules_v4, "perm_importance": perm_v4, "shap_value": shap_val_v4, "explainer": explainer_v4}

info = {}

features = sorted(list(set().union(*[set(d["meta"]["feat_names"]) for d in versions.values()])))
rank_mat = pd.DataFrame(index=features, columns=sorted(versions.keys()))
for vname in sorted(versions.keys()):
    for feat in features:
        rank_mat.loc[feat, vname] = perm_ranks.get(vname, {}).get(feat, np.nan)
rank_mat.to_csv("../mem_access_analysis/tree_analysis/reu_feature_rank.csv",index=False)

vlist = sorted(versions.keys())
corr_res = []
for i in range(len(vlist)):
    for j in range(i+1, len(vlist)):
        a = []
        b = []
        for feat in features:
            ra = perm_ranks.get(vlist[i], {}).get(feat, np.nan)
            rb = perm_ranks.get(vlist[j], {}).get(feat, np.nan)
            a.append(np.nan if math.isnan(ra) else ra)
            b.append(np.nan if math.isnan(rb) else rb)
        # convert to numpy and drop nan pairs
        arr = np.array([a,b])
        valid = ~np.isnan(arr).any(axis=0)
        if valid.sum() < 2:
            rho = np.nan
        else:
            rho, _ = spearmanr(arr[0,valid], arr[1,valid])
        corr_res.append({"v1": vlist[i], "v2": vlist[j], "spearman_rho": float(rho)})
pd.DataFrame(corr_res).to_csv("../mem_access_analysis/tree_analysis/reu_rank_correlation_between_versions.csv",index=False)

topk = 20
top_features = set()
top_features.update(list(perm_df1.head(topk)["feature"]))
top_features.update(list(perm_df2.head(topk)["feature"]))
top_features = sorted(list(top_features))
# Plot heatmap of perm_mean across versions for top features
heat = pd.DataFrame(index=top_features, columns=sorted(versions.keys()))
perm_map_1 = dict(zip(perm_df1["feature"], perm_df1["perm_mean"]))
for f in top_features:
    heat.loc[f, "version 1"] = perm_map_1.get(f, 0.0)

perm_map_2 = dict(zip(perm_df2["feature"], perm_df2["perm_mean"]))
for f in top_features:
    heat.loc[f, "version 2"] = perm_map_2.get(f, 0.0)

perm_map_3 = dict(zip(perm_df3["feature"], perm_df3["perm_mean"]))
for f in top_features:
    heat.loc[f, "version 3"] = perm_map_3.get(f, 0.0)

perm_map_4 = dict(zip(perm_df4["feature"], perm_df4["perm_mean"]))
for f in top_features:
    heat.loc[f, "version 4"] = perm_map_4.get(f, 0.0)
heat = heat.fillna(0.0).astype(float)
plt.figure(figsize=(max(6, len(versions)*1.2), max(6, len(top_features)*0.25)))
sns.heatmap(heat, annot=False, cmap="viridis")
plt.title("Permutation Importance (mean) across versions (top features union)")
plt.tight_layout()
plt.savefig("../mem_access_analysis/tree_analysis/reu_perm_importance_heatmap.png")
plt.close()

thr_comp_df = thr_df.copy()
# We'll plot per-feature median threshold across versions (for features that appear)
features_with_splits = thr_comp_df["feature"].tolist()
for feat in features_with_splits:
    vals = []
    versions_sorted = sorted(versions.keys())
    for v in versions_sorted:
        vals.append(thr_comp_df.loc[thr_comp_df["feature"]==feat, f"{v}_median_thr"].values)
    # flatten and check exist
    vs = []
    for v in versions_sorted:
        vv = thr_comp_df.loc[thr_comp_df["feature"]==feat, f"{v}_median_thr"].values
        vs.append(vv[0] if len(vv)>0 and not np.isnan(vv[0]) else np.nan)
    if np.nansum(np.isfinite(vs)) >= 2:
        plt.figure(figsize=(6,3))
        plt.plot(versions_sorted, vs, marker='o')
        plt.title(f"Threshold (median) drift for feature: {feat}")
        plt.xlabel("version")
        plt.ylabel("median threshold (scaled)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        plt.savefig(f"../mem_access_analysis/tree_analysis/reu_thr_drift_{safe_name}.png")
        plt.close()

vlist = sorted(versions.keys())
jacc_rows = []
for i in range(len(vlist)):
    for j in range(i, len(vlist)):
        a = versions[vlist[i]]["rules_text"]
        b = versions[vlist[j]]["rules_text"]
        score = jaccard_rule_overlap(a,b)
        jacc_rows.append({"v1": vlist[i], "v2": vlist[j], "jaccard": score})
pd.DataFrame(jacc_rows).to_csv("../mem_access_analysis/tree_analysis/reu_rule_jaccard.csv", index=False)

'''reu_merged:-
v1- 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 4, 'min_samples_split': 2, 'random_state': 42, 'splitter': 'random'
v2- 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 10, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 10, 'random_state': 42, 'splitter': 'random'
v3- 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'min_impurity_decrease': 0.01, 'min_samples_leaf': 4, 'min_samples_split': 2, 'random_state': 42, 'splitter': 'best'
v4- 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.01, 'min_samples_leaf': 4, 'min_samples_split': 2, 'random_state': 42, 'splitter': 'best'
'''