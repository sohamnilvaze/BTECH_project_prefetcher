#!/usr/bin/env python3
"""
CN2Continuous: CN2 rule learner extended for numeric (continuous) features.

Provides two workflows:
  1) Direct continuous support (selectors like attr <= thr and attr > thr).
  2) Discretize numeric columns to categorical bins and run original CN2 logic.

Outputs:
  - rules: list of (complex, predicted_class, coverage, precision)
  - utilities to save rules to txt/csv

Author: ChatGPT (adapted for your memory-access dataset)
"""

import numpy as np
import pandas as pd
import collections
import time
import pickle
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Any, Union, Optional

Selector = Tuple[str, Any]  # for categorical: (attr, val) ; for numeric: (attr, '<=', thr) or (attr, '>', thr)
Complex = List[Selector]
RuleT = Tuple[Optional[Complex], Any, float, float]


def discretize_df(df: pd.DataFrame, n_bins: int = 5, method: str = "quantile", drop_original: bool = False):
    """
    Discretize numeric columns in-place returning a new dataframe with categorical bins.
    - method: 'quantile' (qcut) or 'uniform' (cut)
    - n_bins: number of bins
    - drop_original: if True, original numeric columns are dropped and replaced by bin labels
    """
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # skip integer-encoded classes if named 'target' etc. Caller ensures class column is excluded
        try:
            if method == "quantile":
                df2[col] = pd.qcut(df2[col], q=n_bins, duplicates="drop", labels=False)
            else:
                df2[col] = pd.cut(df2[col], bins=n_bins, duplicates="drop", labels=False)
        except Exception:
            # fallback to naive uniform
            df2[col] = pd.cut(df2[col], bins=n_bins, labels=False, duplicates="drop")
    if drop_original:
        return df2
    else:
        return df2


class CN2Continuous:
    """
    CN2 algorithm with numeric selector support.
    Basic hyperparameters:
      - star_max_size: how many best complexes to keep each iteration
      - min_significance: threshold for significance (higher -> fewer complexes accepted)
      - numeric_thresholds: how to choose thresholds for numeric features:
            * 'quantiles' (use unique quantile-based thresholds)
            * 'midpoints' (use midpoints between sorted unique values)
            * list/dict: pass a dict {col: [list of thresholds]} to use custom thresholds
      - n_quantiles: number of quantiles if numeric_thresholds == 'quantiles'
    """
    def __init__(
        self,
        star_max_size: int = 5,
        min_significance: float = 0.5,
        numeric_thresholds: Union[str, dict] = "quantiles",
        n_quantiles: int = 5,
        class_col: Optional[str] = None,
    ):
        self.star_max_size = star_max_size
        self.min_significance = min_significance
        self.numeric_thresholds = numeric_thresholds
        self.n_quantiles = n_quantiles
        self.class_col = class_col  # name of class column; if None uses last column
        # internals
        self.data: Optional[pd.DataFrame] = None
        self._E: Optional[pd.DataFrame] = None
        self._selectors: List[Selector] = []

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, df: pd.DataFrame) -> List[RuleT]:
        """
        Fit CN2 on a pandas DataFrame (df). The last column is assumed to be the class unless self.class_col provided.
        Returns list of rules (complex, class, coverage, precision).
        """
        self.data = df.copy().reset_index(drop=True)
        if self.class_col is None:
            class_col = self.data.columns[-1]
        else:
            class_col = self.class_col
        # ensure class_col is last column for internal convenience
        if list(self.data.columns)[-1] != class_col:
            cols = [c for c in self.data.columns if c != class_col] + [class_col]
            self.data = self.data[cols]

        self._E = self.data.copy()
        self.compute_selectors()

        rule_list: List[RuleT] = []
        classes = self.data.iloc[:, -1]
        classes_count = classes.value_counts().to_dict()

        while len(self._E) > 0:
            # debug progress
            # print(f"Remaining examples: {len(self._E)}")
            best_cpx = self.find_best_complex()
            if best_cpx is not None:
                covered_examples = self.get_covered_examples(self._E, best_cpx)
                if len(covered_examples) == 0:
                    # no examples covered (rare) -> break
                    break
                most_common_class, count = self.get_most_common_class(covered_examples)
                self._E = self.remove_examples(self._E, covered_examples)

                total = classes_count.get(most_common_class, 0)
                coverage = count / total if total > 0 else 0.0
                precision = count / len(covered_examples) if len(covered_examples) > 0 else 0.0

                rule_list.append((best_cpx, most_common_class, coverage, precision))
            else:
                break

        # default rule (fallback to global majority)
        most_common_class, count = self.get_most_common_class(self.data.index)
        total = classes_count.get(most_common_class, 0)
        coverage = count / total if total > 0 else 0.0
        precision = count / len(self.data) if len(self.data) > 0 else 0.0
        rule_list.append((None, most_common_class, coverage, precision))

        return rule_list

    def predict(self, df: pd.DataFrame, rule_list: List[RuleT]):
        """
        Predict classes for df (last column ignored). Returns (rules_performance, accuracy).
        """
        test_data = df.copy().reset_index(drop=True)
        if self.class_col is None:
            class_col = test_data.columns[-1]
        else:
            class_col = self.class_col

        # if class column exists in test df, separate it
        if class_col in test_data.columns:
            test_classes = test_data[class_col].values
            test_data = test_data.drop(columns=[class_col])
        else:
            test_classes = None

        n = len(test_data)
        predicted_classes = [None] * n
        rules_performance = []
        remaining_examples = test_data.copy()

        for rule in rule_list:
            rule_complex = rule[0]
            if rule_complex is not None:
                covered_examples = self.get_covered_examples(remaining_examples, rule_complex)
                remaining_examples = self.remove_examples(remaining_examples, covered_examples)
                indexes = list(covered_examples)
            elif len(remaining_examples) > 0:
                indexes = list(remaining_examples.index)
            else:
                indexes = []

            predicted_class = rule[1]
            correct_predictions = 0
            wrong_predictions = 0
            for idx in indexes:
                predicted_classes[idx] = predicted_class
                if test_classes is not None and test_classes[idx] == predicted_class:
                    correct_predictions += 1
                elif test_classes is not None and test_classes[idx] != predicted_class:
                    wrong_predictions += 1
            sums = correct_predictions + wrong_predictions
            accuracy = (correct_predictions / sums) if sums > 0 else '-'
            perf = {
                'rule': rule,
                'predicted class': predicted_class,
                'covered examples': len(indexes),
                'correct predictions': correct_predictions,
                'wrong predictions': wrong_predictions,
                'rule accuracy': accuracy
            }
            rules_performance.append(perf)

        overall_acc = None
        if test_classes is not None:
            overall_acc = accuracy_score(test_classes, predicted_classes)
        return rules_performance, overall_acc

    # -------------------------
    # Selector generation & core CN2 functions
    # -------------------------
    def compute_selectors(self):
        """
        Compute selectors list from self.data (attribute,value) for categorical or (attribute,op,threshold) for numeric.
        """
        self._selectors = []
        df = self.data
        attr_names = list(df.columns)[:-1]  # exclude target
        for attr in attr_names:
            col = df[attr]
            if pd.api.types.is_numeric_dtype(col):
                # numeric candidate thresholds
                if isinstance(self.numeric_thresholds, dict):
                    thr_list = self.numeric_thresholds.get(attr, [])
                elif self.numeric_thresholds == "quantiles":
                    # unique quantiles excluding 0 and 1
                    quantiles = np.linspace(0, 1, self.n_quantiles + 1)[1:-1]
                    thr_list = sorted(list(set(np.nanpercentile(col.dropna(), quantiles * 100).tolist())))
                elif self.numeric_thresholds == "midpoints":
                    uniq = np.unique(col.dropna())
                    if len(uniq) <= 1:
                        thr_list = []
                    else:
                        thr_list = []
                        for i in range(len(uniq)-1):
                            thr_list.append((uniq[i] + uniq[i+1]) / 2.0)
                        thr_list = sorted(list(set(thr_list)))
                else:
                    # fallback: quantiles
                    quantiles = np.linspace(0, 1, self.n_quantiles + 1)[1:-1]
                    thr_list = sorted(list(set(np.nanpercentile(col.dropna(), quantiles * 100).tolist())))

                # create selectors for <= and >
                for thr in thr_list:
                    self._selectors.append((attr, "<=", float(thr)))
                    self._selectors.append((attr, ">", float(thr)))
            else:
                # categorical: pair attribute,value
                possible_values = sorted(list(set(col.dropna().astype(str))))
                for val in possible_values:
                    self._selectors.append((attr, val))

    def specialize_star(self, star: List[Complex], selectors: List[Selector]) -> List[Complex]:
        new_star: List[Complex] = []
        if len(star) > 0:
            for complex_ in star:
                for selector in selectors:
                    new_complex = [s for s in complex_]
                    new_complex.append(selector)
                    # invalid if same attribute repeated
                    attrs = [s[0] for s in new_complex]
                    if len(attrs) != len(set(attrs)):
                        continue
                    new_star.append(new_complex)
        else:
            for selector in selectors:
                new_star.append([selector])
        return new_star

    def get_covered_examples(self, all_examples: pd.DataFrame, best_cpx: Complex) -> pd.Index:
        """
        Return indices of all_examples satisfying all selectors in best_cpx.
        Handles numeric selectors (<=, >) and categorical selectors (attr, value).
        all_examples is a DataFrame WITHOUT the class column (if original used, pass appropriate).
        """
        if best_cpx is None or len(best_cpx) == 0:
            return all_examples.index

        df = all_examples.copy()
        # Ensure we operate on columns typed consistently (string for categorical selectors)
        mask = pd.Series([True] * len(df), index=df.index)
        for sel in best_cpx:
            if len(sel) == 2:
                # categorical selector: (attr, val)
                attr, val = sel
                # val may be string or number; compare accordingly
                mask = mask & (df[attr].astype(str) == str(val))
            elif len(sel) == 3:
                attr, op, thr = sel
                if op == "<=":
                    mask = mask & (pd.to_numeric(df[attr], errors="coerce") <= thr)
                elif op == ">":
                    mask = mask & (pd.to_numeric(df[attr], errors="coerce") > thr)
                else:
                    raise ValueError(f"Unknown numeric operator: {op}")
            else:
                raise ValueError(f"Unknown selector format: {sel}")
        return df[mask].index

    def find_best_complex(self) -> Optional[Complex]:
        """
        Grow star of complexes and keep best by entropy while meeting min_significance.
        """
        best_complex = None
        best_complex_entropy = float('inf')
        best_complex_significance = 0
        star: List[Complex] = []

        while True:
            entropy_measures = {}
            new_star = self.specialize_star(star, self._selectors)
            for idx, tested_complex in enumerate(new_star):
                significance = self.significance(tested_complex)
                if significance > self.min_significance:
                    entropy = self.entropy(tested_complex)
                    entropy_measures[idx] = entropy
                    if entropy < best_complex_entropy:
                        best_complex = tested_complex.copy()
                        best_complex_entropy = entropy
                        best_complex_significance = significance
            # keep top-k best complexes
            if len(entropy_measures) == 0:
                break
            top_complexes = sorted(entropy_measures.items(), key=lambda x: x[1])[:self.star_max_size]
            star = [new_star[x[0]] for x in top_complexes]
            if len(star) == 0 or best_complex_significance < self.min_significance:
                break

        return best_complex

    def remove_examples(self, all_examples: pd.DataFrame, indexes: Union[pd.Index, List[int]]):
        remaining_examples = all_examples.drop(indexes)
        return remaining_examples

    def get_most_common_class(self, covered_examples: Union[pd.Index, pd.DataFrame, List[int]]):
        if isinstance(covered_examples, (list, np.ndarray, pd.Index)):
            idx = covered_examples
        else:
            idx = covered_examples.index
        classes = self.data.loc[idx, [list(self.data)[-1]]]
        if len(classes) == 0:
            return None, 0
        vc = classes.iloc[:, 0].value_counts()
        most_common = vc.index[0]
        count = int(vc.iloc[0])
        return most_common, count

    def significance(self, tested_complex: Complex) -> float:
        covered_examples = self.get_covered_examples(self._E, tested_complex)
        if len(covered_examples) == 0:
            return 0.0
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        covered_num_instances = len(classes)
        covered_counts = classes.iloc[:,0].value_counts()
        covered_probs = covered_counts.divide(covered_num_instances)

        train_classes = self.data.iloc[:,-1]
        train_num_instances = len(train_classes)
        train_counts = train_classes.value_counts()
        train_probs = train_counts.divide(train_num_instances)

        # KL-like measure used in original CN2
        with np.errstate(divide='ignore', invalid='ignore'):
            term = covered_probs * np.log(np.divide(covered_probs, train_probs, out=np.ones_like(covered_probs), where=train_probs>0))
        significance = 2 * term.sum()
        # handle NaNs
        if np.isnan(significance):
            return 0.0
        return float(significance)

    def entropy(self, tested_complex: Complex) -> float:
        covered_examples = self.get_covered_examples(self._E, tested_complex)
        if len(covered_examples) == 0:
            return float('inf')
        classes = self.data.loc[covered_examples, [list(self.data)[-1]]]
        num_instances = len(classes)
        class_counts = classes.iloc[:,0].value_counts()
        class_probabilities = class_counts.divide(num_instances)
        with np.errstate(divide='ignore', invalid='ignore'):
            plog2p = class_probabilities * np.log2(class_probabilities)
        entropy = -1.0 * plog2p.sum()
        if np.isnan(entropy):
            return float('inf')
        return float(entropy)

    # -------------------------
    # Utilities for printing/saving rules
    # -------------------------
    @staticmethod
    def rule_to_string(complex_: Optional[Complex], predicted_class: Any, coverage: float, precision: float) -> str:
        if complex_ is None:
            return f"Default -> class={predicted_class} (coverage={coverage:.4f}, precision={precision:.4f})"
        parts = []
        for sel in complex_:
            if len(sel) == 2:
                parts.append(f"{sel[0]} = {sel[1]}")
            elif len(sel) == 3:
                parts.append(f"{sel[0]} {sel[1]} {sel[2]}")
        return f"If {' AND '.join(parts)} THEN class={predicted_class} (coverage={coverage:.4f}, precision={precision:.4f})"

    def print_rules(self, rules: List[RuleT]):
        for r in rules:
            print(self.rule_to_string(r[0], r[1], r[2], r[3]))

    def save_rules_txt(self, rules: List[RuleT], fname: str):
        with open(fname, "w") as f:
            for r in rules:
                f.write(self.rule_to_string(r[0], r[1], r[2], r[3]) + "\n")

    def save_rules_csv(self, rules: List[RuleT], fname: str):
        rows = []
        for idx, r in enumerate(rules):
            complex_ = r[0]
            pred = r[1]
            coverage = r[2]
            precision = r[3]
            rows.append({
                "rule_index": idx,
                "rule": self.rule_to_string(complex_, pred, coverage, precision),
                "predicted_class": pred,
                "coverage": coverage,
                "precision": precision
            })
        pd.DataFrame(rows).to_csv(fname, index=False)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Small demo pipeline using your ba_merged.csv (or any features CSV)
    # 1) load CSV with the class as last column (or pass class_col param)
    src_csv = "../mem_access_traces/reu_merged.csv"
    print("Loading:", src_csv)
    df_all = pd.read_csv(src_csv)
    '''
    # Option A: direct numeric CN2 (continuous selectors)
    print("\n--- Running CN2Continuous (numeric selectors) ---")
    cn2c = CN2Continuous(star_max_size=6, min_significance=0.7, numeric_thresholds="quantiles", n_quantiles=4)
    t0 = time.time()
    rules = cn2c.fit(df_all)
    t1 = time.time()
    print("Training time (numeric selectors):", t1 - t0)
    cn2c.print_rules(rules)
    cn2c.save_rules_txt(rules, "cn2_cont_reu.txt")
    cn2c.save_rules_csv(rules, "cn2_cont_reu.csv")

    # Evaluate (if CSV contains labels)
    perf, acc = cn2c.predict(df_all, rules)
    print("Overall accuracy on training set using learned rules:", acc)
    '''
    # Option B: discretize numeric columns to bins and run (original selector style)
    print("\n--- Running CN2 on discretized data (qcut bins) ---")
    df_disc = discretize_df(df_all, n_bins=6, method="quantile")
    cn2d = CN2Continuous(star_max_size=6, min_significance=0.7, numeric_thresholds={}, n_quantiles=4)
    # recompute selectors will now produce categorical selectors because df_disc numeric columns are ints representing bins
    t0 = time.time()
    rules2 = cn2d.fit(df_disc)
    t1 = time.time()
    print("Training time (discretized):", t1 - t0)
    cn2d.print_rules(rules2)
    cn2d.save_rules_txt(rules2, "cn2_cont_reu.txt")
    cn2d.save_rules_csv(rules2, "cn2_cont_reu.csv")
    
