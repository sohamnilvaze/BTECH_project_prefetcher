import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict
import joblib

def train_HMM_models(df_path):
    df = pd.read_csv(df_path)
    features_col = "Delta_with_1_last_read"
    df[features_col] = df[features_col].fillna(0)
    vals = df[features_col].values
    lo, hi = np.percentile(vals, [1,99])
    vals_clipped = np.clip(vals,lo,hi)
    df[features_col] = vals_clipped
    X = df[features_col].values.reshape(-1, 1)

    # Quantize the deltas into 10 uniform-width buckets
    n_bins = 10
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X_binned = discretizer.fit_transform(X).astype(int).flatten()
    unique, counts = np.unique(X_binned,return_counts=True)
    print("Bins present: ",unique)
    print("Bin count: ",counts)

    # Group sequences by pattern_type
    sequences = defaultdict(list)
    for pattern, group in df.groupby("pattern_type"):
        seq = group[features_col].values.reshape(-1, 1)
        seq_binned = discretizer.transform(seq).astype(int)
        sequences[pattern].append(seq_binned)

    # Train one HMM model per pattern
    models = {}
    n_states = 4
    for pattern, seq_list in sequences.items():
        X_concat = np.concatenate(seq_list)
        lengths = [len(s) for s in seq_list]

        model = hmm.MultinomialHMM(n_components=n_states, n_iter=50, random_state=42)
        model.fit(X_concat, lengths)
        models[pattern] = model
        print(f"Trained model for {pattern}")

    return models, discretizer


def classify_trace(trace_deltas, discretizer, models):
    # Discretize the input trace
    trace = np.array(trace_deltas, dtype=float).reshape(-1, 1)
    trace_binned = discretizer.transform(trace).astype(int)

    # Evaluate log-likelihood under each HMM
    scores = {}
    for pattern, model in models.items():
        try:
            log_likelihood = model.score(trace_binned)
        except:
            log_likelihood = -np.inf  # handle very short sequences safely
        scores[pattern] = log_likelihood

    best_pattern = max(scores, key=scores.get)
    return best_pattern, scores, trace_binned


def predict_next_delta(trace_binned, model, discretizer):
    # Decode most likely hidden states via Viterbi
    logprob, states = model.decode(trace_binned, algorithm="viterbi")
    last_state = states[-1]

    # Compute probability distribution for next observation
    next_obs_prob = model.transmat_[last_state].dot(model.emissionprob_)

    # Get predicted bin (most probable)
    predicted_bin = np.argmax(next_obs_prob)

    # Convert bin index → approximate delta value
    # We use the lower edge of the bin range for interpretability
    predicted_delta = discretizer.bin_edges_[0][predicted_bin]

    return predicted_delta, predicted_bin, next_obs_prob


def main():
    input_csv = input("Enter input lower GHB CSV filename: ").strip()
    models, discretizer = train_HMM_models(input_csv)

    traces_elements = input("Enter the trace deltas separated by spaces:\n")
    traces = list(map(float, traces_elements.split()))

    # Step 1: Classify pattern
    best_pattern, scores, trace_binned = classify_trace(traces, discretizer, models)
    print(f"\nBest pattern: {best_pattern}")
    print(f"Log-likelihood scores: {scores}")

    # Step 2: Predict next delta using best model
    best_model = models[best_pattern]
    predicted_delta, predicted_bin, prob_dist = predict_next_delta(trace_binned, best_model, discretizer)

    print(f"\nPredicted next delta bin: {predicted_bin}")
    print(f"Approximate predicted delta value: {predicted_delta:.4f}")
    print(f"Probability distribution over next bins: {np.round(prob_dist, 3)}")


if __name__ == '__main__':
    main()

