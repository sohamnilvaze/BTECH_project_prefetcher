import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict

def train_HMM_models(df_path):
    df = pd.read_csv(df_path)
    features_col = "Delta_with_1_last_read"
    df[features_col] = df[features_col].fillna(0)
    X = df[features_col].values.reshape(-1,1)

    #Quantize the deltas into buckets here 10 buckets with uniform width
    n_bins = 10
    discretizer= KBinsDiscretizer(n_bins=n_bins,encode="ordinal",strategy="uniform")
    X_binned = discretizer.fit_transform(X).astype(int).flatten()

    #Group sequences by pattern_type
    sequences = defaultdict(list)
    for pattern, group in df.groupby("pattern_type"):
        seq = group[features_col].values.reshape(-1,1)
        seq_binned = discretizer.transform(seq).astype(int)
        sequences[pattern].append(seq_binned)

    #Train one HMM model per pattern
    models = {}
    n_states = 4
    for pattern, seq_list in sequences.items():
        X_concat = np.concatenate(seq_list)
        lenghts = [len(s) for s in seq_list]

        model = hmm.MultinomialHMM(n_components=n_states, n_iter = 50,random_state=42)
        model.fit(X_concat,lenghts)
        models[pattern] = model
        print(f"Trained model for {pattern}")
    return models, discretizer

def classify_trace(trace_deltas,discretizer,models):
    #Discretize the trace
    trace = np.array(trace_deltas).reshape(-1,1)
    trace_binned = discretizer.transform(trace).astype(int)

    #Evaluate the log-likelihood under each HMM
    scores = {}
    for pattern,model in models.items():
        log_likelihood = model.score(trace_binned)
        scores[pattern] = log_likelihood

    best_pattern = max(scores,key = scores.get)
    return best_pattern, scores
 
def main():
    input_csv = input("Enter input lower GHB CSV filename: ").strip()
    models , discretizer = train_HMM_models(input_csv)
    traces_elements = input("Enter the trace elements seperated by spaces:\n")
    traces = traces_elements.split()
    best_pattern, scores = classify_trace(traces,discretizer,models)
    print(f"Best pattern: {best_pattern} | scores : {scores}")

if __name__ == '__main__':
    main()
